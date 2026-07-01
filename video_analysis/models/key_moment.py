"""Key moment model."""

from __future__ import annotations

from pydantic import BaseModel, Field


class KeyMoment(BaseModel):
    """A notable moment in the video with timestamp and importance."""

    moment_id: str = Field(description="Unique key moment identifier")
    video_id: str = Field(description="Parent video ID")
    chunk_id: str | None = Field(default=None, description="Originating chunk ID")
    timestamp_seconds: float = Field(description="Timestamp in seconds")
    title: str = Field(description="Short title for this moment")
    description: str = Field(description="Detailed description")
    importance: int = Field(default=1, ge=1, le=5, description="Importance rating 1-5")
    tags: list[str] = Field(default_factory=list)
    quote: str | None = Field(default=None, description="Associated quote, if any")

    @property
    def timestamp(self) -> str:
        """Format timestamp as HH:MM:SS.mmm."""
        total_secs = int(self.timestamp_seconds)
        hours = total_secs // 3600
        minutes = (total_secs % 3600) // 60
        secs = total_secs % 60
        ms = int((self.timestamp_seconds - total_secs) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"
