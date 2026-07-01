"""Video metadata extraction via ffprobe.

Extracts duration, resolution, FPS, codec, audio stream info, file size,
and other metadata from a video file.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class AudioStreamInfo:
    """Information about an audio stream."""
    codec: str = ""
    sample_rate: int = 0
    channels: int = 0
    bit_rate: int = 0


@dataclass
class VideoStreamInfo:
    """Information about a video stream."""
    codec: str = ""
    width: int = 0
    height: int = 0
    fps: float = 0.0


@dataclass
class VideoMetadata:
    """Complete video metadata."""
    filename: str = ""
    path: str = ""
    duration_seconds: float = 0.0
    container: str = ""
    file_size_bytes: int = 0
    video_stream: VideoStreamInfo = field(default_factory=VideoStreamInfo)
    audio_stream: AudioStreamInfo = field(default_factory=AudioStreamInfo)

    @property
    def resolution(self) -> str:
        return f"{self.video_stream.width}x{self.video_stream.height}"

    @property
    def duration_human(self) -> str:
        hours = int(self.duration_seconds) // 3600
        minutes = (int(self.duration_seconds) % 3600) // 60
        secs = int(self.duration_seconds) % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def extract_metadata(video_path: str) -> VideoMetadata:
    """Extract metadata from a video file using ffprobe.

    Args:
        video_path: Absolute path to the video file.

    Returns:
        VideoMetadata with all extracted fields.
    """
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        video_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffprobe failed: {e.stderr}") from e
    except FileNotFoundError:
        raise RuntimeError("ffprobe not found in PATH") from None

    meta = VideoMetadata()

    # Format-level info
    fmt = data.get("format", {})
    meta.filename = fmt.get("filename", "")
    meta.container = fmt.get("format_name", "")
    meta.duration_seconds = float(fmt.get("duration", 0))

    # Find video and audio streams
    for stream in data.get("streams", []):
        codec_type = stream.get("codec_type", "")
        if codec_type == "video":
            meta.video_stream = VideoStreamInfo(
                codec=stream.get("codec_name", ""),
                width=stream.get("width", 0),
                height=stream.get("height", 0),
                fps=_parse_fps(stream.get("r_frame_rate", ""),
                              stream.get("avg_frame_rate", "")),
            )
        elif codec_type == "audio":
            meta.audio_stream = AudioStreamInfo(
                codec=stream.get("codec_name", ""),
                sample_rate=stream.get("sample_rate", 0),
                channels=stream.get("channels", 0),
                bit_rate=stream.get("bit_rate", 0),
            )

    # File size
    meta.file_size_bytes = fmt.get("size", 0)

    return meta


def _parse_fps(frame_rate_str: str, avg_frame_rate_str: str = "") -> float:
    """Parse a frame rate string like '30000/1001' into a float."""
    for s in (frame_rate_str, avg_frame_rate_str):
        if not s:
            continue
        try:
            if "/" in s:
                num, den = s.split("/")
                num, den = float(num), float(den)
                if den > 0:
                    return round(num / den, 3)
            else:
                return float(s)
        except (ValueError, ZeroDivisionError):
            continue
    return 0.0
