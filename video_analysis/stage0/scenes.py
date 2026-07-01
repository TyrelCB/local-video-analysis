"""Scene detection using FFmpeg's built-in scene filter.

Detects shot/scene changes by analyzing visual differences between
consecutive frames. Returns scene boundaries with timestamps.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SceneBoundary:
    """A detected scene change."""
    scene_index: int
    start_seconds: float
    end_seconds: float
    confidence: float = 0.0  # Not available from FFmpeg scene filter


def detect_scenes(video_path: str, threshold: float = 0.3,
                  output_dir: str | None = None) -> list[SceneBoundary]:
    """Detect scene boundaries using FFmpeg's scene change filter.

    Uses the select='gt(scene,threshold)' filter to identify shot changes.
    Scene frames are optionally saved to disk for later visual captioning.

    Args:
        video_path: Path to the video file.
        threshold: Scene change sensitivity (0.1–1.0). Higher = fewer scenes.
        output_dir: Optional directory to save representative scene frames.

    Returns:
        List of SceneBoundary objects sorted by start time.
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # FFmpeg scene detection: output scene frames and log timestamps
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"select='gt(scene,{threshold})',showinfo",
        "-vsync", "vfr",
        "-y",
        os.path.join(output_dir, "scene_%06d.jpg") if output_dir else "/dev/null",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        logger.error("ffmpeg not found in PATH")
        return []

    # Parse showinfo output for scene timestamps
    timestamps = _parse_scene_timestamps(result.stderr)

    if not timestamps:
        logger.info("No scene changes detected with threshold %.2f", threshold)
        return []

    # Build scene boundaries from timestamps
    boundaries = []
    prev_time = 0.0
    for i, scene_time in enumerate(timestamps):
        boundaries.append(SceneBoundary(
            scene_index=i,
            start_seconds=round(prev_time, 3),
            end_seconds=round(scene_time, 3),
        ))
        prev_time = scene_time

    # Add final scene (last detected scene to end of video)
    if boundaries:
        boundaries[-1].end_seconds = None  # Will be filled by pipeline

    return boundaries


def _parse_scene_timestamps(stderr: str) -> list[float]:
    """Parse FFmpeg showinfo output to extract scene change timestamps.

    Looks for lines containing 'n:' (frame number) and 'pts_time:'
    """
    timestamps = []
    # Pattern: pts_time: XX.XXX in the showinfo output
    for line in stderr.split("\n"):
        if "pts_time:" in line:
            match = re.search(r"pts_time: ([\d.]+)", line)
            if match:
                ts = float(match.group(1))
                # Avoid duplicate timestamps (multiple showinfo lines per frame)
                if not timestamps or ts > timestamps[-1]:
                    timestamps.append(ts)
    return timestamps
