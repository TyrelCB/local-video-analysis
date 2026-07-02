"""Markdown export for analysis reports.

Produces a human-readable markdown report with metadata, summary,
chapters, key moments, speaker info, and tags.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models.analysis import AnalysisResult
    from ..stage0.metadata import VideoMetadata


def export_markdown(result: object, metadata: object, video_id: str,
                    audio_events: list[dict] | None = None) -> str:
    """Export analysis result as a Markdown report.

    Args:
        result: AnalysisResult object or dict.
        metadata: VideoMetadata object or dict.
        video_id: Video identifier.
        audio_events: Optional list of detected audio-event dicts
            ({timestamp, type, description}) to summarize in the report.

    Returns:
        Markdown-formatted report string.
    """
    # Support both Pydantic models and dicts
    if hasattr(result, "to_dict"):
        data = result.to_dict()
    elif isinstance(result, dict):
        data = result
    else:
        data = _obj_to_dict(result)

    if hasattr(metadata, "duration_human"):
        duration = metadata.duration_human
    elif hasattr(metadata, "duration_seconds"):
        duration = _seconds_to_human(getattr(metadata, "duration_seconds", 0))
    elif isinstance(metadata, dict):
        duration = _seconds_to_human(metadata.get("duration_seconds", 0))
    else:
        duration = "unknown"

    lines = []
    lines.append(f"# Video Analysis Report")
    lines.append("")
    lines.append(f"**Video ID:** {video_id}")
    lines.append(f"**Duration:** {duration}")

    # Metadata
    meta = data.get("metadata", {})
    lines.append(f"**Mode:** {meta.get('analysis_mode', 'deep')}")
    lines.append(f"**Chunks analyzed:** {meta.get('num_chunks', 'N/A')}")
    lines.append("")

    # Executive summary
    summary = data.get("summary", {})
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(summary.get("executive", "N/A"))
    lines.append("")

    # Detailed summary
    if summary.get("detailed"):
        lines.append("## Detailed Summary")
        lines.append("")
        lines.append(summary["detailed"])
        lines.append("")

    # Chapters
    chapters = data.get("chapters", [])
    if chapters:
        lines.append("## Chapters")
        lines.append("")
        for i, ch in enumerate(chapters, 1):
            start = _ts(ch.get("start_seconds", 0))
            end = _ts(ch.get("end_seconds", 0))
            title = ch.get("title", f"Chapter {i}")
            lines.append(f"### {i}. {title} (`{start}` — `{end}`)")
            lines.append("")
            lines.append(ch.get("summary", ""))
            lines.append("")

    # Key moments
    key_moments = data.get("key_moments", [])
    if key_moments:
        lines.append("## Key Moments")
        lines.append("")
        for i, km in enumerate(key_moments, 1):
            time = _ts(km.get("time", km.get("timestamp_seconds", 0)))
            title = km.get("title", km.get("description", "Moment"))
            desc = km.get("description", "")
            importance = km.get("importance", 1)
            stars = "★" * importance + "☆" * (5 - importance)
            lines.append(f"{i}. **{title}** (`{time}`) {stars}")
            if desc:
                lines.append(f"   {desc}")
            lines.append("")

    # Characters / entities (with resolved aliases)
    characters = data.get("characters", [])
    if characters:
        lines.append("## Characters")
        lines.append("")
        for ch in characters:
            name = ch.get("name", "Unknown")
            aliases = ch.get("aliases", []) or []
            alias_str = f" (also: {', '.join(aliases)})" if aliases else ""
            desc = ch.get("description", "")
            lines.append(f"- **{name}**{alias_str}" + (f" — {desc}" if desc else ""))
        lines.append("")

    # Key plot/causal events
    key_events = data.get("key_events", [])
    if key_events:
        lines.append("## Key Events")
        lines.append("")
        for ev in key_events:
            time = _ts(ev.get("time", 0))
            desc = ev.get("description", "")
            parts = ev.get("participants", []) or []
            part_str = f" _({', '.join(parts)})_" if parts else ""
            lines.append(f"- (`{time}`) {desc}{part_str}")
        lines.append("")

    # Action items
    action_items = data.get("action_items", [])
    if action_items:
        lines.append("## Action Items")
        lines.append("")
        for item in action_items:
            lines.append(f"- {item}")
        lines.append("")

    # Speaker summary
    speaker_summary = data.get("speaker_summary", {})
    if speaker_summary:
        lines.append("## Speakers")
        lines.append("")
        for speaker, topics in speaker_summary.items():
            lines.append(f"**{speaker}:** {topics}")
        lines.append("")

    # Audio events (structural acoustic analysis: silence / music-or-noise)
    if audio_events:
        counts: dict[str, int] = {}
        for ev in audio_events:
            counts[ev.get("type", "?")] = counts.get(ev.get("type", "?"), 0) + 1
        lines.append("## Audio Events")
        lines.append("")
        lines.append(", ".join(f"{n} {t}" for t, n in sorted(counts.items())))
        lines.append("")
        # List the most notable non-silence events (music/noise), timestamped.
        notable = [e for e in audio_events if e.get("type") != "silence"]
        for ev in notable[:15]:
            lines.append(f"- `{_ts(ev.get('timestamp', 0))}` {ev.get('type', '')}: "
                         f"{ev.get('description', '')}")
        if notable:
            lines.append("")

    # Tags
    tags = data.get("tags", [])
    if tags:
        lines.append("## Tags")
        lines.append("")
        lines.append(", ".join(tags))
        lines.append("")

    return "\n".join(lines)


def _seconds_to_human(seconds: float) -> str:
    total_secs = int(seconds)
    hours = total_secs // 3600
    minutes = (total_secs % 3600) // 60
    secs = total_secs % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _ts(seconds: float) -> str:
    """Convert seconds to timestamp string."""
    total_secs = int(seconds)
    hours = total_secs // 3600
    minutes = (total_secs % 3600) // 60
    secs = total_secs % 60
    ms = int((seconds - total_secs) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def _obj_to_dict(obj: object) -> dict:
    """Convert an object to dict via model_dump or __dict__."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return vars(obj)
