"""Chapter model for Pass 2 global synthesis."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Chapter(BaseModel):
    """A chapter marker in the video."""

    chapter_id: str = Field(description="Unique chapter identifier")
    video_id: str = Field(description="Parent video ID")
    start_seconds: float = Field(description="Start time in seconds")
    end_seconds: float = Field(description="End time in seconds")
    title: str = Field(description="Chapter title")
    summary: str = Field(description="One-sentence summary of what happens in this chapter")
    key_moment_ids: list[str] = Field(default_factory=list, description="Related key moments")
    tags: list[str] = Field(default_factory=list)

    @property
    def start_timestamp(self) -> str:
        return _seconds_to_timestamp(self.start_seconds)

    @property
    def end_timestamp(self) -> str:
        return _seconds_to_timestamp(self.end_seconds)

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


def _seconds_to_timestamp(seconds: float) -> str:
    total_secs = int(seconds)
    hours = total_secs // 3600
    minutes = (total_secs % 3600) // 60
    secs = total_secs % 60
    ms = int((seconds - total_secs) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"
