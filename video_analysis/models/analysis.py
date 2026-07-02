"""Analysis result model — the complete output of Pass 2 global synthesis."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from .chapter import Chapter
from .key_moment import KeyMoment


class AnalysisResult(BaseModel):
    """Complete analysis result for a video, produced by Pass 2."""

    video_id: str = Field(description="Parent video ID")
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    # Summary
    executive_summary: str = Field(description="One-paragraph executive summary")
    detailed_summary: str = Field(description="3-5 paragraph detailed summary by theme")

    # Chapters
    chapters: list[Chapter] = Field(default_factory=list)

    # Key moments
    key_moments: list[KeyMoment] = Field(default_factory=list)

    # Highlights (ranked key moments)
    highlights: list[KeyMoment] = Field(default_factory=list,
                                         description="Top key moments ranked by importance")

    # Speaker summary (for when diarization is enabled)
    speaker_summary: dict[str, str] = Field(default_factory=dict,
                                              description="Speaker label -> topic summary")

    # Visual summary
    visual_summary: str = Field(default="",
                                 description="How visuals progress throughout the video")

    # Audio summary
    audio_summary: str = Field(default="",
                                description="Audio events and patterns throughout the video")

    # Characters / entities resolved across the whole video. Each entry:
    # {name, aliases: [..], description}. Lets synthesis state that e.g.
    # "Bradley" and "Mr. Preston" are the same person.
    characters: list[dict] = Field(default_factory=list,
                                    description="Resolved characters/entities with aliases")

    # Key plot/causal events across the video: who did what to whom, when.
    # Each entry: {time, description, participants: [..]}.
    key_events: list[dict] = Field(default_factory=list,
                                   description="Causal/plot events (who did what to whom)")

    # Tags
    tags: list[str] = Field(default_factory=list)

    # Action items
    action_items: list[str] = Field(default_factory=list)

    # Pipeline info
    analysis_mode: str = Field(default="deep", description="Analysis mode used")
    num_chunks: int = Field(default=0, description="Number of chunks analyzed")
    num_key_moments: int = Field(default=0, description="Total key moments before merging")

    def to_dict(self) -> dict:
        """Serialize to a dictionary for JSON export."""
        return {
            "video_id": self.video_id,
            "metadata": {
                "created_at": self.created_at,
                "analysis_mode": self.analysis_mode,
                "num_chunks": self.num_chunks,
                "num_key_moments": self.num_key_moments,
            },
            "summary": {
                "executive": self.executive_summary,
                "detailed": self.detailed_summary,
            },
            "chapters": [c.model_dump() for c in self.chapters],
            "key_moments": [m.model_dump() for m in self.key_moments],
            "highlights": [h.model_dump() for h in self.highlights],
            "speaker_summary": self.speaker_summary,
            "visual_summary": self.visual_summary,
            "audio_summary": self.audio_summary,
            "characters": self.characters,
            "key_events": self.key_events,
            "tags": self.tags,
            "action_items": self.action_items,
        }
