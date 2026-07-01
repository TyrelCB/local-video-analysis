"""Chunk model for Pass 1 segment analysis."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """A timestamped segment of video for Pass 1 analysis."""

    video_id: str = Field(description="Parent video ID")
    chunk_id: str = Field(description="Unique chunk identifier")
    chunk_index: int = Field(description="Zero-based chunk index")
    start_seconds: float = Field(description="Start time in seconds")
    end_seconds: float = Field(description="End time in seconds")

    # Stage 0 inputs
    transcript: str = Field(default="", description="Transcribed text for this chunk")
    visual_captions: list[str] = Field(default_factory=list, description="Visual descriptions from vision model")
    ocr_text: list[str] = Field(default_factory=list, description="On-screen text from OCR")
    audio_events: list[str] = Field(default_factory=list, description="Detected audio events")
    scene_boundaries: list[float] = Field(default_factory=list, description="Scene change timestamps")
    speaker_labels: list[str] = Field(default_factory=list, description="Speaker labels present")

    # Pass 1 output
    summary: str = Field(default="", description="2-3 sentence summary of this segment")
    key_moments: list[dict] = Field(default_factory=list, description="Important moments with timestamps")
    tags: list[str] = Field(default_factory=list, description="Topic/action tags")
    quotes: list[dict] = Field(default_factory=list, description="Notable quotes")
    issues: list[str] = Field(default_factory=list, description="Errors, warnings, problems")
    detected_actions: list[str] = Field(default_factory=list, description="Physical actions or UI interactions")

    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds

    @property
    def start_timestamp(self) -> str:
        return _seconds_to_timestamp(self.start_seconds)

    @property
    def end_timestamp(self) -> str:
        return _seconds_to_timestamp(self.end_seconds)

    def get_content(self) -> str:
        """Get the full text content for this chunk (used for search indexing)."""
        parts = [self.summary]
        if self.transcript:
            parts.append(self.transcript)
        if self.visual_captions:
            parts.append(" ".join(self.visual_captions))
        if self.tags:
            parts.append(" ".join(self.tags))
        return " ".join(filter(None, parts))


def _seconds_to_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.mmm."""
    total_secs = int(seconds)
    hours = total_secs // 3600
    minutes = (total_secs % 3600) // 60
    secs = total_secs % 60
    ms = int((seconds - total_secs) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"
