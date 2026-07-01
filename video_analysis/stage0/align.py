"""Timeline alignment.

Merges all timestamped Stage 0 signals into a unified timeline
suitable for chunking and reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TimelineEvent:
    """A single event on the unified timeline."""
    timestamp_seconds: float
    event_type: str              # transcript, scene_change, speaker_change, ocr, caption, audio_event
    data: dict = field(default_factory=dict)


class Timeline:
    """Unified timeline of all timestamped signals."""

    def __init__(self):
        self.events: list[TimelineEvent] = []

    def add_events(self, events: list[TimelineEvent]) -> None:
        """Add events to the timeline."""
        self.events.extend(events)

    def sort(self) -> None:
        """Sort all events by timestamp."""
        self.events.sort(key=lambda e: e.timestamp_seconds)

    def get_events_at(self, timestamp: float, window: float = 0.5) -> list[TimelineEvent]:
        """Get all events within a time window of a timestamp."""
        return [
            e for e in self.events
            if abs(e.timestamp_seconds - timestamp) <= window
        ]

    def get_event_types(self) -> set[str]:
        """Get unique event types on this timeline."""
        return {e.event_type for e in self.events}

    def to_dict(self) -> dict:
        """Serialize timeline for JSON export."""
        self.sort()
        return {
            "total_events": len(self.events),
            "event_types": list(self.get_event_types()),
            "events": [
                {
                    "timestamp": e.timestamp_seconds,
                    "type": e.event_type,
                    "data": e.data,
                }
                for e in self.events
            ],
        }
