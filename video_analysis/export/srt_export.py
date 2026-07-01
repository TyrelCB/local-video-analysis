"""SRT subtitle export.

Converts transcript segments to SRT subtitle format.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..stage0.audio import TranscriptionSegment


def export_srt(segments: list) -> str:
    """Export transcript segments as SRT subtitle format.

    Args:
        segments: List of TranscriptionSegment objects or dicts.

    Returns:
        SRT-formatted string.
    """
    lines = []
    for i, seg in enumerate(segments, 1):
        if isinstance(seg, dict):
            start = _seconds_to_srt(seg.get("start_seconds", 0))
            end = _seconds_to_srt(seg.get("end_seconds", 0))
            text = seg.get("text", "")
        else:
            start = _seconds_to_srt(seg.start_seconds)
            end = _seconds_to_srt(seg.end_seconds)
            text = seg.text

        lines.append(str(i))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")

    return "\n".join(lines)


def _seconds_to_srt(seconds: float) -> str:
    """Convert seconds to SRT timestamp (HH:MM:SS,mmm)."""
    total_secs = int(seconds)
    hours = total_secs // 3600
    minutes = (total_secs % 3600) // 60
    secs = total_secs % 60
    ms = int((seconds - total_secs) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"
