"""Frame sampling from video.

Supports fixed-interval and scene-change-based sampling.
Saves frames as JPEG with timestamp metadata.
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SampledFrame:
    """A sampled frame with its timestamp."""
    timestamp_seconds: float
    frame_path: str


def sample_frames_fixed(video_path: str, fps: float,
                         output_dir: str) -> list[SampledFrame]:
    """Sample frames at a fixed interval.

    Args:
        video_path: Path to the video file.
        fps: Frames per second to sample (e.g., 0.5 = 1 frame every 2 seconds).
        output_dir: Directory to save frames.

    Returns:
        List of SampledFrame objects.
    """
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"fps={fps}",
        "-q:v", "2",
        "-y",
        os.path.join(output_dir, "frame_%06d.jpg"),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        logger.error("Frame sampling failed: %s", e.stderr)
        return []

    # Collect all saved frames with their indices
    frames = []
    for f in sorted(os.listdir(output_dir)):
        if f.startswith("frame_") and f.endswith(".jpg"):
            idx = int(f.replace("frame_", "").replace(".jpg", ""))
            ts = (idx - 1) / fps  # 0-based index
            frames.append(SampledFrame(
                timestamp_seconds=round(ts, 3),
                frame_path=os.path.join(output_dir, f),
            ))

    return frames


def sample_frames_scenes(video_path: str, scene_timestamps: list[float],
                          frames_per_scene: int = 1,
                          output_dir: str | None = None) -> list[SampledFrame]:
    """Sample frames around scene change boundaries.

    Picks frames at the start of each scene and optionally at evenly
    spaced intervals within the scene.

    Args:
        video_path: Path to the video file.
        scene_timestamps: List of scene start timestamps.
        frames_per_scene: Number of frames to extract per scene.
        output_dir: Optional directory to save frames.

    Returns:
        List of SampledFrame objects.
    """
    frames: list[SampledFrame] = []
    scene_idx = 0

    for i, start_ts in enumerate(scene_timestamps):
        end_ts = scene_timestamps[i + 1] if i + 1 < len(scene_timestamps) else None

        # Pick evenly spaced times within the scene
        count = min(frames_per_scene, 3)  # Cap at 3 per scene
        for j in range(count):
            if end_ts:
                # Spread frames across the scene
                offset = (end_ts - start_ts) * j / max(count - 1, 1)
                ts = start_ts + offset
            else:
                ts = start_ts

            # Extract a single frame at this timestamp
            frame_path = _extract_single_frame(video_path, ts, scene_idx, j, output_dir)
            if frame_path:
                frames.append(SampledFrame(
                    timestamp_seconds=round(ts, 3),
                    frame_path=frame_path,
                ))
            scene_idx += 1

    return frames


def _extract_single_frame(video_path: str, timestamp: float,
                           scene_idx: int, frame_idx: int,
                           output_dir: str | None) -> str | None:
    """Extract a single frame at a given timestamp."""
    if not output_dir:
        return ""

    os.makedirs(output_dir, exist_ok=True)
    filename = f"frame_scene{scene_idx:03d}_f{frame_idx:02d}.jpg"
    path = os.path.join(output_dir, filename)

    cmd = [
        "ffmpeg",
        "-ss", str(timestamp),
        "-i", video_path,
        "-vf", "scale=1280:720:force_original_aspect_ratio=decrease",
        "-frames:v", "1",
        "-q:v", "2",
        "-y",
        path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return path
    except subprocess.CalledProcessError:
        logger.warning("Failed to extract frame at %.1f seconds", timestamp)
        return None
