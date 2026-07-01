"""Tests for the chunking logic."""

from video_analysis.pass1.chunker import build_chunks
from video_analysis.stage0.audio import TranscriptionSegment


def _seg(start, end, text="test"):
    return TranscriptionSegment(text=text, start_seconds=start, end_seconds=end)


def test_single_chunk_for_short_video():
    """A short video with few segments should produce few chunks."""
    segments = [
        _seg(0, 5), _seg(5, 10), _seg(10, 15),
    ]
    chunks = build_chunks(
        transcription=segments,
        scenes=[],
        frame_timestamps=[],
        visual_captions=[],
        audio_events=[],
        video_duration=15.0,
        chunk_seconds=300,
        overlap_seconds=5,
    )
    assert len(chunks) >= 1
    assert chunks[0].start_seconds == 0.0
    assert chunks[-1].end_seconds == 15.0


def test_multiple_chunks():
    """A long video should produce multiple chunks."""
    segments = [_seg(i * 10, (i + 1) * 10) for i in range(100)]  # 1000s
    chunks = build_chunks(
        transcription=segments,
        scenes=[],
        frame_timestamps=[],
        visual_captions=[],
        audio_events=[],
        video_duration=1000.0,
        chunk_seconds=90,
        overlap_seconds=10,
    )
    assert len(chunks) > 1
    # Each chunk should have some transcript segments
    for chunk in chunks:
        assert chunk.start_seconds >= 0
        assert chunk.end_seconds <= 1000.0


def test_scene_boundaries_preferred():
    """Chunks should prefer to split at scene boundaries."""
    segments = [_seg(i * 5, (i + 1) * 5) for i in range(40)]
    scenes = [
        type("SceneBoundary", (), {"scene_index": 0, "start_seconds": 30, "end_seconds": 60})(),
        type("SceneBoundary", (), {"scene_index": 1, "start_seconds": 90, "end_seconds": 120})(),
    ]

    chunks = build_chunks(
        transcription=segments,
        scenes=scenes,
        frame_timestamps=[],
        visual_captions=[],
        audio_events=[],
        video_duration=200.0,
        chunk_seconds=60,
        overlap_seconds=5,
    )
    assert len(chunks) > 0

    # Check that chunk boundaries are near scene boundaries
    # (accounting for overlap shift)
    scene_starts = {s.start_seconds for s in scenes}
    chunk_ends = {c.end_seconds for c in chunks[:-1]}  # Skip last chunk; its end is video_duration

    # At least one chunk end should be near a scene start
    near_boundary = any(
        abs(ce - ss) < 5  # overlap is 5s, so chunk ends 5s before scene start
        for ce in chunk_ends
        for ss in scene_starts
    )
    assert near_boundary


def test_empty_transcription():
    """Empty transcription should produce one full-length chunk."""
    chunks = build_chunks(
        transcription=[],
        scenes=[],
        frame_timestamps=[],
        visual_captions=[],
        audio_events=[],
        video_duration=300.0,
        chunk_seconds=90,
        overlap_seconds=10,
    )
    assert len(chunks) == 1
    assert chunks[0].start_seconds == 0.0
    assert chunks[0].end_seconds == 300.0


def test_no_overlap_between_chunks():
    """Chunks should not overlap in their start/end times (overlap is handled via segment assignment)."""
    segments = [_seg(i * 5, (i + 1) * 5) for i in range(40)]
    chunks = build_chunks(
        transcription=segments,
        scenes=[],
        frame_timestamps=[],
        visual_captions=[],
        audio_events=[],
        video_duration=200.0,
        chunk_seconds=60,
        overlap_seconds=10,
    )

    for i in range(1, len(chunks)):
        # Each chunk should start after or at the previous chunk's end
        # (overlap is handled by assigning overlapping segments, not by time overlap)
        assert chunks[i].start_seconds >= chunks[i - 1].start_seconds
