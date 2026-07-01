"""Video metadata model.

Stores information extracted from the source video file via ffprobe.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class VideoMetadata(BaseModel):
    """Metadata extracted from a video file."""

    filename: str = Field(description="Base filename of the video")
    path: str = Field(description="Full path to the video file")
    video_id: str = Field(description="Unique identifier for this video")
    duration_seconds: float = Field(description="Total duration in seconds")
    fps: float = Field(description="Frames per second")
    width: int = Field(description="Video width in pixels")
    height: int = Field(description="Video height in pixels")
    codec: str = Field(default="", description="Video codec name")
    audio_codec: str = Field(default="", description="Audio codec name")
    audio_sample_rate: int = Field(default=0, description="Audio sample rate in Hz")
    audio_channels: int = Field(default=0, description="Number of audio channels")
    file_size_bytes: int = Field(default=0, description="File size in bytes")
    container: str = Field(default="", description="Container format (mp4, mkv, etc.)")
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    @property
    def resolution(self) -> str:
        """Resolution string, e.g. '1920x1080'."""
        return f"{self.width}x{self.height}"

    @property
    def duration_human(self) -> str:
        """Duration string, e.g. '01:30:45'."""
        total_secs = int(self.duration_seconds)
        hours = total_secs // 3600
        minutes = (total_secs % 3600) // 60
        secs = total_secs % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    @property
    def analysis_id(self) -> str:
        """Alias for video_id."""
        return self.video_id
