"""Pass 1 segment analyzer.

Sends each chunk's multimodal data to the reasoning model for
segment-level analysis. Returns structured JSON output with
summaries, key moments, tags, quotes, issues, and detected actions.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field

from ..reasoning.llama_cpp import LlamaCppClient
from ..reasoning.server import ChatMessage, CompletionResult
from .chunker import ChunkSpec
from .prompts import PASS1_SYSTEM_PROMPT, PASS1_USER_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

MAX_RETRIES = 3


@dataclass
class ChunkAnalysis:
    """Result of analyzing a single chunk."""
    chunk_id: str
    chunk_index: int
    start_seconds: float
    end_seconds: float
    summary: str = ""
    key_moments: list[dict] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    quotes: list[dict] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    detected_actions: list[str] = field(default_factory=list)
    speaker_labels: list[str] = field(default_factory=list)
    # Structured plot facts (Phase 1 enrichment): who is on-screen and the causal
    # events in this segment, so synthesis reasons over facts, not just prose.
    characters_present: list[str] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)  # {actor, action, target}
    error: str | None = None


def parse_key_moment_time(time_val, default: float) -> float:
    """Coerce a model-supplied key-moment timestamp to seconds.

    The reasoning model returns "time" in inconsistent shapes: a number, a
    unit-suffixed string ("12.5s"), a range ("30.0s - 45.0s"), or a *list* of
    those (["0.0s", "5.0s"]). Extract the first numeric value; fall back to
    ``default`` (typically the chunk start) when nothing parseable is found.
    """
    if isinstance(time_val, list):
        time_val = time_val[0] if time_val else default
    if isinstance(time_val, str):
        nums = re.findall(r"[\d.]+", time_val)
        return float(nums[0]) if nums else default
    try:
        return float(time_val)
    except (TypeError, ValueError):
        return default


async def analyze_chunk(chunk: ChunkSpec, client: LlamaCppClient,
                        video_duration: float) -> ChunkAnalysis:
    """Analyze a single chunk using the reasoning model.

    Args:
        chunk: Chunk specification with multimodal data.
        client: Reasoning model client.
        video_duration: Total video duration (for context).

    Returns:
        ChunkAnalysis with structured results.
    """
    analysis = ChunkAnalysis(
        chunk_id=chunk.chunk_id,
        chunk_index=chunk.chunk_index,
        start_seconds=chunk.start_seconds,
        end_seconds=chunk.end_seconds,
    )

    # Build prompt content. Feed the transcript as timestamped, speaker-labelled
    # lines — NOT one flattened blob — so per-chunk analysis can attribute who
    # said what and cite precise times, and so plot-bearing dialogue keeps its
    # structure through to synthesis. Speaker labels are present when diarization
    # ran; omitted otherwise.
    transcript_lines = []
    for seg in chunk.transcript_segments:
        text = (seg.text or "").strip()
        if not text:
            continue
        speaker = (getattr(seg, "speaker", "") or "").strip()
        prefix = f"[{seg.start_seconds:.1f}s]" + (f" {speaker}:" if speaker else "")
        transcript_lines.append(f"{prefix} {text}")
    transcript_text = "\n".join(transcript_lines)
    visual_text = "\n".join(f"[{ts:.1f}s] {cap}" for ts, cap in
                             zip(chunk.frame_timestamps, chunk.visual_captions))
    ocr_text = ""  # Placeholder — filled when OCR is enabled
    audio_text = "\n".join(f"[{evt['timestamp']:.1f}s] {evt['type']}: {evt['description']}"
                           for evt in chunk.audio_events)
    scenes_text = "\n".join(
        f"[{s.start_seconds:.1f}s - {s.end_seconds:.1f}s] Scene {s.scene_index}"
        for s in chunk.scene_boundaries
    )

    user_prompt = PASS1_USER_PROMPT_TEMPLATE.format(
        video_duration=video_duration,
        segment_time=f"{chunk.start_seconds:.1f}s - {chunk.end_seconds:.1f}s",
        transcript=transcript_text or "(no transcript)",
        visual_captions=visual_text or "(no visual captions)",
        ocr_text=ocr_text or "(none)",
        audio_events=audio_text or "(none)",
        scenes=scenes_text or "(none detected)",
    )

    messages = [
        ChatMessage(role="system", content=PASS1_SYSTEM_PROMPT),
        ChatMessage(role="user", content=user_prompt),
    ]

    # A chunk with substantial dialogue should yield a non-empty summary and,
    # usually, some structured facts. Treat a valid-but-empty result on a
    # dialogue-heavy chunk as low-signal and worth one stronger retry, rather
    # than silently keeping a weak analysis (a source of the plot-comprehension
    # misses).
    has_dialogue = len(transcript_text.strip()) >= 120

    result = None
    # Retry with JSON extraction
    for attempt in range(MAX_RETRIES):
        msgs = messages
        if attempt > 0:
            # Nudge harder on retries triggered by low signal.
            msgs = messages + [ChatMessage(
                role="system",
                content="Your previous analysis was empty or missing detail for a "
                        "segment that clearly contains dialogue. Re-read the "
                        "transcript and extract the summary, characters_present, "
                        "events, and pivotal quotes it actually contains.")]
        result = await client.chat(msgs, temperature=0.3, max_tokens=2048)

        if result.is_error:
            analysis.error = result.error
            logger.error("Chunk %s analysis attempt %d failed: %s",
                         chunk.chunk_id, attempt + 1, result.error)
            return analysis

        # Extract JSON from model response
        json_text = _extract_json(result.text)
        if json_text:
            try:
                data = json.loads(json_text)
                analysis.summary = data.get("summary", "")
                analysis.key_moments = data.get("key_moments", [])
                analysis.tags = data.get("tags", [])
                analysis.quotes = data.get("quotes", [])
                analysis.issues = data.get("issues", [])
                analysis.detected_actions = data.get("detected_actions", [])
                analysis.speaker_labels = data.get("speaker_labels", [])
                analysis.characters_present = data.get("characters_present", []) or []
                analysis.events = data.get("events", []) or []

                low_signal = (has_dialogue and not analysis.summary.strip()
                              and not analysis.events and not analysis.quotes)
                if low_signal and attempt < MAX_RETRIES - 1:
                    logger.info("Chunk %s: low-signal result on a dialogue-heavy "
                                "segment; re-analyzing (attempt %d).",
                                chunk.chunk_id, attempt + 1)
                    continue
                return analysis
            except json.JSONDecodeError:
                logger.warning("Chunk %s: failed to parse JSON on attempt %d",
                               chunk.chunk_id, attempt + 1)

        logger.warning("Chunk %s: model didn't return valid JSON on attempt %d",
                       chunk.chunk_id, attempt + 1)

    # If all retries failed, store raw text as summary
    analysis.summary = result.text[:500] if not result.is_error else "Analysis failed"  # type: ignore[union-attr]
    analysis.error = "Failed to parse model output as JSON after 3 attempts"
    return analysis


async def analyze_chunks(chunks: list[ChunkSpec], client: LlamaCppClient,
                         video_duration: float,
                         max_concurrent: int = 8) -> list[ChunkAnalysis]:
    """Analyze all chunks concurrently, up to max_concurrent in flight.

    Chunk analyses are independent, so they're dispatched in parallel (bounded
    by a semaphore) to keep a multi-slot reasoning server busy. Results are
    returned in chunk order regardless of completion order.

    Args:
        chunks: List of chunk specifications.
        client: Reasoning model client.
        video_duration: Total video duration.
        max_concurrent: Max concurrent reasoning requests. Set to the server's
            parallel-slot count; higher values just queue server-side.

    Returns:
        List of ChunkAnalysis results, one per chunk, in input order.
    """
    if not chunks:
        return []

    sem = asyncio.Semaphore(max(1, max_concurrent))
    done = 0

    async def _one(chunk: ChunkSpec) -> ChunkAnalysis:
        nonlocal done
        async with sem:
            analysis = await analyze_chunk(chunk, client, video_duration)
        done += 1
        logger.info("Completed %d/%d chunks", done, len(chunks))
        return analysis

    logger.info("Analyzing %d chunks (up to %d concurrent)...",
                len(chunks), max_concurrent)
    return list(await asyncio.gather(*(_one(c) for c in chunks)))


def _extract_json(text: str) -> str | None:
    """Extract JSON from a model response that may contain markdown fences."""
    # Try direct parse first
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    # Look for JSON within markdown code fences
    import re
    match = re.search(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
    if match:
        return match.group(1)

    # Look for first { ... } block
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start:end + 1]

    return None
