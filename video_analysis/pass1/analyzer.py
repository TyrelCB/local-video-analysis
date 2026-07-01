"""Pass 1 segment analyzer.

Sends each chunk's multimodal data to the reasoning model for
segment-level analysis. Returns structured JSON output with
summaries, key moments, tags, quotes, issues, and detected actions.
"""

from __future__ import annotations

import json
import logging
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
    error: str | None = None


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

    # Build prompt content
    transcript_text = " ".join(seg.text for seg in chunk.transcript_segments)
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

    result = None
    # Retry with JSON extraction
    for attempt in range(MAX_RETRIES):
        result = await client.chat(messages, temperature=0.3, max_tokens=2048)

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
                         batch_size: int = 10) -> list[ChunkAnalysis]:
    """Analyze all chunks, processing in batches.

    Args:
        chunks: List of chunk specifications.
        client: Reasoning model client.
        video_duration: Total video duration.
        batch_size: Number of chunks to analyze in each batch.

    Returns:
        List of ChunkAnalysis results, one per chunk.
    """
    results: list[ChunkAnalysis] = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        logger.info("Analyzing chunks %d-%d of %d...",
                     i + 1, min(i + batch_size, len(chunks)), len(chunks))

        for chunk in batch:
            analysis = await analyze_chunk(chunk, client, video_duration)
            results.append(analysis)

        logger.info("Completed %d/%d chunks", len(results), len(chunks))

    return results


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
