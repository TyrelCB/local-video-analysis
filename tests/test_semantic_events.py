"""Tests for semantic audio-event merging (no model / GPU required)."""

from video_analysis.stage0.semantic_events import SemanticEvent, _merge_runs


def test_merges_consecutive_same_label():
    # Three contiguous windows of "Applause" (2s window, 1s hop) -> one event.
    raw = [(10.0, "Applause", 0.6), (11.0, "Applause", 0.8), (12.0, "Applause", 0.5)]
    events = _merge_runs(raw, window_seconds=2.0, hop_seconds=1.0)
    assert len(events) == 1
    ev = events[0]
    assert ev.label == "Applause"
    assert ev.start_seconds == 10.0
    assert ev.end_seconds == 14.0            # 12.0 + 2.0 window
    assert ev.confidence == 0.8              # max confidence in the run


def test_separates_different_labels():
    raw = [(10.0, "Applause", 0.6), (11.0, "Laughter", 0.7)]
    events = _merge_runs(raw, window_seconds=2.0, hop_seconds=1.0)
    assert [e.label for e in events] == ["Applause", "Laughter"]


def test_gap_breaks_run():
    # Same label but a non-contiguous gap -> two separate events.
    raw = [(10.0, "Music", 0.6), (30.0, "Music", 0.6)]
    events = _merge_runs(raw, window_seconds=2.0, hop_seconds=1.0)
    assert len(events) == 2


def test_empty():
    assert _merge_runs([], 2.0, 1.0) == []


def test_semantic_event_timestamp_alias():
    ev = SemanticEvent(start_seconds=5.0, end_seconds=7.0, label="Dog", confidence=0.5)
    assert ev.timestamp_seconds == 5.0
