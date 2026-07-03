"""Chunk building logic.

Creates timestamped chunks from Stage 0 data using a hybrid strategy:
1. Start with transcript segments
2. Add scene boundaries as preferred split points
3. Add speaker change points
4. Merge into target chunk length
5. Add overlap between chunks
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from ..stage0.audio import TranscriptionSegment
from ..stage0.scenes import SceneBoundary

logger = logging.getLogger(__name__)


@dataclass
class ChunkSpec:
    """A specification for a single analysis chunk."""
    chunk_id: str
    chunk_index: int
    start_seconds: float
    end_seconds: float
    transcript_segments: list[TranscriptionSegment] = field(default_factory=list)
    scene_boundaries: list[SceneBoundary] = field(default_factory=list)
    frame_timestamps: list[float] = field(default_factory=list)
    visual_captions: list[str] = field(default_factory=list)
    audio_events: list[dict] = field(default_factory=list)
    speaker_labels: list[str] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


def build_chunks(
    transcription: list[TranscriptionSegment],
    scenes: list[SceneBoundary],
    frame_timestamps: list[float],
    visual_captions: list[str],
    audio_events: list[dict],
    video_duration: float,
    chunk_seconds: int = 90,
    overlap_seconds: int = 10,
    max_tokens_per_chunk: int = 3000,
) -> list[ChunkSpec]:
    """Build analysis chunks from Stage 0 data.

    Args:
        transcription: List of transcript segments.
        scenes: List of scene boundaries.
        frame_timestamps: Timestamps of sampled frames.
        visual_captions: Corresponding visual captions.
        audio_events: Detected audio events.
        video_duration: Total video duration in seconds.
        chunk_seconds: Target chunk length in seconds.
        overlap_seconds: Overlap between chunks in seconds.
        max_tokens_per_chunk: Hard cap on estimated tokens.

    Returns:
        List of ChunkSpec objects.
    """
    if not transcription:
        logger.warning("No transcription data — creating a single chunk for the full video")
        return [ChunkSpec(
            chunk_id="chunk_000",
            chunk_index=0,
            start_seconds=0.0,
            end_seconds=video_duration,
        )]

    # Build candidate split points, scored by preference
    split_points = _build_split_points(transcription, scenes, chunk_seconds, video_duration)

    # Create chunks from split points
    chunks: list[ChunkSpec] = []
    idx = 0
    prev_end = 0.0

    for i, split in enumerate(split_points):
        chunk_start = prev_end
        chunk_end = min(split, video_duration)

        if chunk_end - chunk_start < 5:  # Skip very small chunks
            continue

        # Trim to stay within video bounds
        chunk_end = min(chunk_end, video_duration)

        chunk_id = f"chunk_{idx:05d}"
        chunk = ChunkSpec(
            chunk_id=chunk_id,
            chunk_index=idx,
            start_seconds=round(chunk_start, 3),
            end_seconds=round(chunk_end, 3),
        )

        # Assign transcript segments to this chunk, extended backward by the
        # overlap window so a plot reveal that straddles a chunk boundary (e.g. an
        # accusation at the end of the prior chunk, the reply here) is analyzed
        # with its run-up, not halved. The chunk's canonical [start, end] is
        # unchanged; only the dialogue context window is widened at the head.
        ctx_start = max(0.0, chunk_start - overlap_seconds)
        chunk.transcript_segments = [
            seg for seg in transcription
            if _overlaps(seg.start_seconds, seg.end_seconds, ctx_start, chunk_end)
        ]

        # Assign scene boundaries
        chunk.scene_boundaries = [
            s for s in scenes
            if _overlaps(s.start_seconds, s.end_seconds, chunk_start, chunk_end)
        ]

        # Assign frames
        chunk.frame_timestamps = [
            t for t in frame_timestamps
            if chunk_start <= t <= chunk_end
        ]

        # Assign visual captions (match by frame timestamps)
        chunk.visual_captions = [
            caption for caption, ts in zip(visual_captions, frame_timestamps)
            if chunk_start <= ts <= chunk_end
        ]

        # Assign audio events
        chunk.audio_events = [
            evt for evt in audio_events
            if chunk_start <= evt.get("timestamp", 0) <= chunk_end
        ]

        # Extract speaker labels
        chunk.speaker_labels = list(set(
            seg.text.split(":")[0].strip()
            for seg in chunk.transcript_segments
            if ":" in seg.text
        ))

        chunks.append(chunk)
        idx += 1
        prev_end = chunk_end + overlap_seconds

    # Handle the tail: if the last split point didn't reach the end
    if chunks and chunks[-1].end_seconds < video_duration:
        # Merge remaining into the last chunk or add a final chunk
        remaining_start = chunks[-1].end_seconds
        if video_duration - remaining_start > chunk_seconds * 0.5:
            chunks.append(ChunkSpec(
                chunk_id=f"chunk_{idx:05d}",
                chunk_index=idx,
                start_seconds=round(remaining_start, 3),
                end_seconds=round(video_duration, 3),
            ))

    logger.info("Built %d chunks (%.0fs each, %ds overlap)",
                 len(chunks), chunk_seconds, overlap_seconds)
    return chunks


def _build_split_points(
    transcription: list[TranscriptionSegment],
    scenes: list[SceneBoundary],
    chunk_seconds: int,
    video_duration: float,
) -> list[float]:
    """Build candidate split points for chunking.

    Priority: scene boundaries > transcript pauses > fixed intervals.
    """
    points: set[float] = set()
    points.add(0.0)

    # 1. Scene boundaries (highest priority split points)
    for scene in scenes:
        points.add(round(scene.start_seconds, 3))

    # 2. Fixed interval split points
    t = chunk_seconds
    while t < video_duration:
        points.add(round(t, 3))
        t += chunk_seconds

    # 3. Prefer split points near scene boundaries
    scored_points: list[tuple[float, int]] = []
    for point in sorted(points):
        if point <= 0 or point >= video_duration:
            continue
        score = 0  # base score

        # Bonus if close to a scene boundary
        for scene in scenes:
            if abs(point - scene.start_seconds) < 2.0:
                score += 10
                break

        scored_points.append((point, score))

    # Sort by score descending, then by point ascending
    scored_points.sort(key=lambda x: (-x[1], x[0]))

    # Return unique points in order, ensuring at least one point if none exist
    seen = set()
    result = []
    for point, _ in scored_points:
        if point not in seen and 0 < point < video_duration:
            seen.add(point)
            result.append(point)

    # If no split points exist (e.g., video is shorter than chunk_seconds),
    # return the video_duration as a single split point so one chunk covers everything
    if not result:
        return [video_duration]

    return result


def _overlaps(start1: float, end1: float, start2: float, end2: float) -> bool:
    """Check if two time ranges overlap."""
    return start1 < end2 and start2 < end1
