"""Pass 2 global synthesizer.

Reads all chunk-level analysis results and produces a coherent full-video
artifact: executive summary, chapters, merged key moments, highlight
ranking, action items, and search tags.
"""

from __future__ import annotations

import json
import logging
import re

from ..models.analysis import AnalysisResult, Chapter, KeyMoment
from ..models.chunk import Chunk
from ..reasoning.llama_cpp import LlamaCppClient
from ..reasoning.server import ChatMessage
from .prompts import PASS2_SYSTEM_PROMPT, PASS2_USER_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

MAX_RETRIES = 3


async def synthesize(chunks: list[Chunk], analysis_results: list[dict],
                     video_duration: float, client: LlamaCppClient,
                     user_prompt: str | None = None,
                     mode: str = "deep",
                     transcript_segments: list | None = None) -> AnalysisResult:
    """Run Pass 2 global synthesis over all chunk analyses.

    Args:
        chunks: List of chunk specifications.
        analysis_results: List of chunk analysis results from Pass 1.
        video_duration: Total video duration in seconds.
        client: Reasoning model client.
        user_prompt: Optional user-specific analysis request.
        mode: Analysis mode (quick/deep/forensic) — recorded on the result.
        transcript_segments: Optional raw transcript segments (objects with
            ``start_seconds`` and ``text``). When provided, the actual dialogue
            is included in the prompt so synthesis reasons over verbatim lines,
            not just lossy per-chunk summaries — this is what recovers plot facts
            (who kills whom, who a character is) that summarization discards.

    Returns:
        AnalysisResult with full synthesis.
    """
    # Build summary strings for the prompt
    chunk_summaries = _build_chunk_summaries(chunks, analysis_results)
    all_key_moments = _build_key_moments_list(analysis_results)
    all_tags = _build_tags_list(analysis_results)
    transcript_block = _build_transcript_block(transcript_segments)

    user_prompt_text = PASS2_USER_PROMPT_TEMPLATE.format(
        num_chunks=len(chunks),
        video_duration=int(video_duration),
        chunk_summaries=chunk_summaries,
        all_key_moments=all_key_moments,
        all_tags=all_tags,
        transcript=transcript_block,
    )

    messages = [
        ChatMessage(role="system", content=PASS2_SYSTEM_PROMPT),
        ChatMessage(role="user", content=user_prompt_text),
    ]
    if user_prompt:
        messages.append(ChatMessage(
            role="system",
            content=f"User request: {user_prompt}. Focus your analysis on this.",
        ))

    # Retry with JSON extraction
    for attempt in range(MAX_RETRIES):
        result = await client.chat(messages, temperature=0.3, max_tokens=8192)

        if result.is_error:
            logger.error("Pass 2 synthesis attempt %d failed: %s",
                         attempt + 1, result.error)
            continue

        json_text = _extract_json(result.text)
        if json_text:
            try:
                data = json.loads(json_text)
                return _build_analysis_result(
                    data, chunks, video_duration, len(analysis_results), mode,
                    analysis_results)
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning("Synthesis attempt %d: JSON parse error: %s",
                               attempt + 1, e)

        logger.warning("Synthesis attempt %d: model didn't return valid JSON",
                       attempt + 1)

    # Fallback: build a minimal result from chunk summaries
    logger.error("All synthesis attempts failed. Building fallback result.")
    return _build_fallback(chunks, analysis_results, video_duration, mode)


def _build_chunk_summaries(chunks: list[Chunk],
                           analysis_results: list[dict]) -> str:
    """Format chunk summaries for the Pass 2 prompt."""
    lines = []
    for i, (chunk, result) in enumerate(zip(chunks, analysis_results)):
        summary = result.get("summary", "No summary available")
        lines.append(
            f"Chunk {i:03d} [{chunk.start_timestamp} - {chunk.end_timestamp}]: "
            f"{summary}"
        )
    return "\n".join(lines)


def _build_transcript_block(transcript_segments: list | None) -> str:
    """Format the raw transcript (timestamped dialogue) for the Pass 2 prompt.

    Per-chunk summaries lose the exact dialogue that carries the plot (who kills
    whom, confessions, identity reveals). Feeding the verbatim transcript lets
    synthesis reason over what was actually said. Speaker labels are included
    when present (diarization) so lines can be attributed correctly.
    """
    if not transcript_segments:
        return "(transcript not provided)"
    lines = []
    for seg in transcript_segments:
        start = getattr(seg, "start_seconds", 0.0)
        text = (getattr(seg, "text", "") or "").strip()
        if not text:
            continue
        speaker = (getattr(seg, "speaker", "") or "").strip()
        prefix = f"[{int(start)}s]" + (f" {speaker}:" if speaker else "")
        lines.append(f"{prefix} {text}")
    return "\n".join(lines) if lines else "(transcript empty)"


def _build_key_moments_list(analysis_results: list[dict]) -> str:
    """Format all key moments for the Pass 2 prompt."""
    lines = []
    for i, result in enumerate(analysis_results):
        for km in result.get("key_moments", []):
            time = km.get("time", "")
            desc = km.get("description", "")
            lines.append(f"  [{time}] {desc}")
    return "\n".join(lines) if lines else "(no key moments)"


def _build_tags_list(analysis_results: list[dict]) -> list[str]:
    """Collect all tags from chunk analyses."""
    tags: list[str] = []
    for result in analysis_results:
        for tag in result.get("tags", []):
            if tag and tag not in tags:
                tags.append(tag)
    return tags


def _parse_time_value(t, default: float = 0.0) -> float:
    """Parse a time value that may be a str ('12.3s', '1:05'), list, or number."""
    if isinstance(t, str):
        # Support mm:ss / hh:mm:ss as well as plain/suffixed seconds.
        if ":" in t:
            parts = t.split(":")
            try:
                secs = 0.0
                for p in parts:
                    secs = secs * 60 + float(re.findall(r"[\d.]+", p)[0])
                return secs
            except (ValueError, IndexError):
                return default
        nums = re.findall(r"[\d.]+", t)
        return float(nums[0]) if nums else default
    if isinstance(t, list):
        return _parse_time_value(t[0], default) if t else default
    try:
        return float(t)
    except (TypeError, ValueError):
        return default


def _build_pass1_moment_times(analysis_results: list[dict] | None) -> dict[str, float]:
    """Map normalized key-moment descriptions to their real Pass-1 timestamps.

    Pass 2 often re-emits key moments without their timestamps; the original
    Pass-1 chunk analyses carry real times, so we recover them by description.
    """
    index: dict[str, float] = {}
    for result in analysis_results or []:
        for km in result.get("key_moments", []):
            desc = (km.get("description") or km.get("title") or "").strip().lower()
            if not desc:
                continue
            t = _parse_time_value(km.get("time"), default=-1.0)
            if t >= 0 and desc not in index:
                index[desc] = t
    return index


def _as_text(v) -> str:
    """Coerce a model-returned value to a string. Models sometimes return a
    list of bullet strings (or a dict) where the schema wants prose; join rather
    than fail validation."""
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        return "; ".join(_as_text(x) for x in v)
    if isinstance(v, dict):
        return "; ".join(f"{k}: {_as_text(val)}" for k, val in v.items())
    return "" if v is None else str(v)


def _coerce_str_map(v) -> dict:
    """Coerce a mapping whose values may be lists/objects into dict[str, str].
    Some models return speaker_summary values as lists of topics; the schema
    expects a single string per speaker."""
    if not isinstance(v, dict):
        return {}
    return {str(k): _as_text(val) for k, val in v.items()}


def _build_analysis_result(data: dict, chunks: list[Chunk],
                            video_duration: float,
                            num_chunks: int,
                            mode: str = "deep",
                            analysis_results: list[dict] | None = None) -> AnalysisResult:
    """Build an AnalysisResult from parsed JSON data."""
    # Detect when the model returned chapter times in minutes instead of seconds
    # (a common small-model mistake) and convert to seconds. We only rescale if
    # the largest end time is well under the real duration but matches it once
    # multiplied by 60 — i.e. unambiguously minutes.
    raw_chapters = data.get("chapters", [])
    chapter_scale = 1.0
    if raw_chapters and video_duration > 0:
        max_end = max(
            (float(ch.get("end_seconds", 0) or 0) for ch in raw_chapters),
            default=0.0,
        )
        if 0 < max_end <= video_duration / 30 and max_end * 60 <= video_duration * 1.5:
            chapter_scale = 60.0
            logger.info(
                "Chapter timestamps look like minutes (max end %.1f vs %.0fs "
                "duration); scaling by 60.", max_end, video_duration)

    # Build chapters
    chapters = []
    for ch in raw_chapters:
        start = float(ch.get("start_seconds", 0) or 0) * chapter_scale
        end_raw = ch.get("end_seconds")
        end = float(end_raw) * chapter_scale if end_raw is not None else video_duration
        chapters.append(Chapter(
            chapter_id=f"chapter_{len(chapters)}",
            video_id="",  # Set by caller
            start_seconds=min(start, video_duration),
            end_seconds=min(end, video_duration),
            title=ch.get("title", ""),
            summary=ch.get("summary", ""),
            tags=ch.get("tags", []),
        ))

    # Build key moments. Pass 2 frequently zeroes/omits timestamps, so when a
    # moment's time is missing or 0 we recover it from the matching Pass-1 key
    # moment (which carries a real time), and failing that from the chapter it
    # falls under. As a last resort spread moments evenly across the video so
    # they never all collapse onto 0.
    pass1_times = _build_pass1_moment_times(analysis_results)

    def _resolve_time(km: dict) -> float:
        t = _parse_time_value(km.get("time"), default=0.0)
        if t > 0:
            return min(t, video_duration)
        desc = (km.get("description") or km.get("title") or "").strip().lower()
        if desc and desc in pass1_times and pass1_times[desc] > 0:
            return min(pass1_times[desc], video_duration)
        return -1.0  # signal: needs positional fallback

    raw_moments = data.get("key_moments", [])
    resolved = [_resolve_time(km) for km in raw_moments]
    # Positional fallback for any still-unresolved (-1) moments: distribute them
    # evenly over the video so ordering is preserved and none stack on 0.
    n = len(raw_moments)
    for idx, t in enumerate(resolved):
        if t < 0:
            resolved[idx] = round(video_duration * (idx + 0.5) / n, 1) if n else 0.0

    key_moments = []
    for km, ts in zip(raw_moments, resolved):
        key_moments.append(KeyMoment(
            moment_id=f"km_{len(key_moments)}",
            video_id="",  # Set by caller
            timestamp_seconds=ts,
            title=km.get("title", km.get("description", "")),
            description=km.get("description", ""),
            importance=int(km.get("importance", 1)),
            tags=km.get("tags", []),
            quote=km.get("quote"),
        ))

    # Sort key moments by importance desc
    key_moments.sort(key=lambda m: m.importance, reverse=True)

    # Highlights: top 10 by importance
    highlights = key_moments[:10]

    return AnalysisResult(
        video_id="",  # Set by caller
        executive_summary=_as_text(data.get("executive_summary", "")),
        detailed_summary=_as_text(data.get("detailed_summary", "")),
        chapters=chapters,
        key_moments=key_moments,
        highlights=highlights,
        speaker_summary=_coerce_str_map(data.get("speaker_summary", {})),
        visual_summary=_as_text(data.get("visual_summary", "")),
        audio_summary=_as_text(data.get("audio_summary", "")),
        characters=data.get("characters", []) or [],
        key_events=data.get("key_events", []) or [],
        tags=data.get("tags", _collect_all_tags(data)),
        action_items=data.get("action_items", []),
        analysis_mode=mode,
        num_chunks=num_chunks,
        num_key_moments=len(key_moments),
    )


def _build_fallback(chunks: list[Chunk], analysis_results: list[dict],
                     video_duration: float, mode: str = "deep") -> AnalysisResult:
    """Build a minimal result when synthesis fails."""
    summaries = [r.get("summary", "No summary") for r in analysis_results]
    all_tags = set()
    for r in analysis_results:
        all_tags.update(r.get("tags", []))

    return AnalysisResult(
        video_id="",
        executive_summary="Analysis could not be fully synthesized. "
                          + " ".join(summaries[:3]),
        detailed_summary="\n\n".join(summaries[:5]),
        key_moments=[],
        highlights=[],
        tags=list(all_tags),
        analysis_mode=mode,
        num_chunks=len(chunks),
    )


def _collect_all_tags(data: dict) -> list[str]:
    """Extract tags from any nested structure in the parsed data."""
    tags = []
    for key in ("tags", "keywords"):
        if key in data and isinstance(data[key], list):
            tags.extend(t for t in data[key] if t)
    return tags


def _extract_json(text: str) -> str | None:
    """Extract JSON from a model response with optional markdown fences."""
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    match = re.search(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
    if match:
        return match.group(1)

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start:end + 1]

    return None
