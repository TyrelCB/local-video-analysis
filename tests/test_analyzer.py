"""Tests for Pass 1 analyzer: key-moment time parsing and concurrent dispatch."""

import asyncio

import pytest

from video_analysis.pass1.analyzer import (
    ChunkAnalysis,
    analyze_chunks,
    parse_key_moment_time,
)
from video_analysis.pass1.chunker import ChunkSpec


# --- parse_key_moment_time -------------------------------------------------
# Regression: a key-moment "time" of ["0.0s", ...] (a list of unit-suffixed
# strings) crashed Pass 2 with ValueError: could not convert string to float.

@pytest.mark.parametrize("time_val,expected", [
    (["0.0s", "5.0s"], 0.0),          # the regression case
    (["90.0"], 90.0),                 # list of plain numeric string
    ("12.5s", 12.5),                  # unit-suffixed string
    ("30.0s - 45.0s", 30.0),          # range string -> first value
    (42.0, 42.0),                     # already a float
    (7, 7.0),                         # int
])
def test_parse_key_moment_time_shapes(time_val, expected):
    assert parse_key_moment_time(time_val, default=100.0) == pytest.approx(expected)


@pytest.mark.parametrize("time_val", [[], "", "garbage", None])
def test_parse_key_moment_time_falls_back(time_val):
    """Unparseable / empty inputs return the supplied default, never raise."""
    assert parse_key_moment_time(time_val, default=100.0) == 100.0


# --- analyze_chunks concurrency -------------------------------------------

def _chunk(idx: int) -> ChunkSpec:
    return ChunkSpec(
        chunk_id=f"chunk_{idx:03d}",
        chunk_index=idx,
        start_seconds=float(idx * 10),
        end_seconds=float(idx * 10 + 10),
        transcript_segments=[],
        frame_timestamps=[],
        visual_captions=[],
        audio_events=[],
        scene_boundaries=[],
    )


class _FakeClient:
    """Records peak concurrency; each chat() sleeps briefly then returns JSON."""

    def __init__(self):
        self.inflight = 0
        self.peak = 0

    async def chat(self, messages, temperature=0.3, max_tokens=2048, **kwargs):
        from video_analysis.reasoning.server import CompletionResult
        self.inflight += 1
        self.peak = max(self.peak, self.inflight)
        await asyncio.sleep(0.02)
        self.inflight -= 1
        return CompletionResult(text='{"summary": "ok", "key_moments": [], "tags": []}')


def test_analyze_chunks_preserves_order_and_bounds_concurrency():
    chunks = [_chunk(i) for i in range(8)]
    client = _FakeClient()

    results = asyncio.run(analyze_chunks(chunks, client, video_duration=80.0,
                                         max_concurrent=4))

    # One result per chunk, in input order.
    assert [r.chunk_index for r in results] == list(range(8))
    assert all(isinstance(r, ChunkAnalysis) for r in results)
    assert all(r.summary == "ok" for r in results)
    # Ran concurrently (>1 in flight) but never exceeded the cap.
    assert 1 < client.peak <= 4


def test_analyze_chunks_empty():
    client = _FakeClient()
    assert asyncio.run(analyze_chunks([], client, video_duration=0.0)) == []
