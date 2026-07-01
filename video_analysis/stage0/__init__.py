"""Stage 0: Extraction + Alignment orchestrator.

Coordinates all Stage 0 tasks in sequence:
1. Extract video metadata
2. Extract audio
3. Transcribe audio
4. Detect scenes
5. Sample frames
6. Caption frames (vision model)
7. Detect audio events
8. Build unified timeline
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .align import Timeline, TimelineEvent
from .audio import extract_audio, transcribe_audio, TranscriptionResult
from .audio_events import detect_audio_events
from .frames import SampledFrame, sample_frames_fixed
from .metadata import VideoMetadata as _VideoMetadata, extract_metadata
from .scenes import SceneBoundary, detect_scenes
from .vision import caption_video_frames, VisualCaption

logger = logging.getLogger(__name__)


@dataclass
class Stage0Output:
    """Complete output from Stage 0."""
    video_id: str
    metadata: _VideoMetadata
    audio_path: str
    transcription: TranscriptionResult
    scenes: list[SceneBoundary]
    frames: list[SampledFrame]
    visual_captions: list[VisualCaption]
    audio_events: list[dict]
    timeline: dict
    artifacts_dir: str

    def get_transcript_text(self) -> str:
        """Get full transcript text."""
        return " ".join(seg.text for seg in self.transcription.segments)

    def get_transcript_timestamps(self) -> list[dict]:
        """Get transcript segments as timestamped dicts."""
        return [
            {
                "start": seg.start_seconds,
                "end": seg.end_seconds,
                "text": seg.text,
                "confidence": seg.confidence,
            }
            for seg in self.transcription.segments
        ]


async def run_stage_0(video_path: str, output_dir: str,
                       model_size: str = "medium", language: str = "auto",
                       transcription_backend: str = "faster_whisper",
                       asr_python: str = "",
                       frame_fps: float = 0.5, scene_threshold: float = 0.3,
                       enable_audio_events: bool = True,
                       vision_client=None,
                       caption_max_concurrent: int = 4,
                       caption_dedup: bool = True,
                       caption_max_frames: int | None = None) -> Stage0Output:
    """Run the complete Stage 0 pipeline.

    Args:
        video_path: Path to the input video.
        output_dir: Directory for all Stage 0 artifacts.
        model_size: Whisper model size.
        language: Language code for transcription.
        transcription_backend: 'faster_whisper' or 'torch_whisper' (GPU).
        frame_fps: Frame sampling rate.
        scene_threshold: Scene detection sensitivity.
        enable_audio_events: Whether to detect audio events.
        vision_client: Optional pre-initialized vision client.

    Returns:
        Stage0Output with all extracted and aligned data.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create subdirectories
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # 1. Extract metadata
    logger.info("Extracting video metadata...")
    metadata = extract_metadata(video_path)

    # 2. Extract audio
    logger.info("Extracting audio...")
    audio_path = os.path.join(output_dir, "audio.wav")
    extract_audio(video_path, audio_path)

    # 3. Transcribe
    logger.info("Transcribing audio...")
    transcription = transcribe_audio(audio_path, model_size=model_size, language=language,
                                     backend=transcription_backend,
                                     asr_python=asr_python or None)
    logger.info("Transcription complete: %d segments, language=%s",
                 len(transcription.segments), transcription.language)

    # 4. Scene detection
    logger.info("Detecting scenes...")
    scenes = detect_scenes(video_path, threshold=scene_threshold)

    # Fill in the end time of the last scene
    if scenes:
        scenes[-1].end_seconds = metadata.duration_seconds

    logger.info("Detected %d scene changes", len(scenes))

    # 5. Frame sampling
    logger.info("Sampling frames at %.1f fps...", frame_fps)
    frames = sample_frames_fixed(video_path, frame_fps, frames_dir)
    logger.info("Sampled %d frames", len(frames))

    # 6. Visual captioning
    logger.info("Generating visual captions...")
    if frames and vision_client:
        timestamps = [f.timestamp_seconds for f in frames]
        frame_paths = [f.frame_path for f in frames]
        visual_captions = await caption_video_frames(
            video_path, frame_paths, timestamps,
            llm_client=vision_client, deduplicate=caption_dedup,
            max_concurrent=caption_max_concurrent,
            max_frames=caption_max_frames,
        )
    else:
        visual_captions = []
    logger.info("Generated %d visual captions", len(visual_captions))

    # 7. Audio events
    audio_events = []
    if enable_audio_events:
        logger.info("Detecting audio events...")
        events = detect_audio_events(audio_path)
        audio_events = [
            {"timestamp": e.timestamp_seconds, "type": e.event_type,
             "description": e.description, "confidence": e.confidence}
            for e in events
        ]
        logger.info("Detected %d audio events", len(audio_events))

    # 8. Build timeline
    timeline = _build_timeline(transcription, scenes, visual_captions, audio_events)

    return Stage0Output(
        video_id=str(uuid.uuid4())[:8],
        metadata=metadata,
        audio_path=audio_path,
        transcription=transcription,
        scenes=scenes,
        frames=frames,
        visual_captions=visual_captions,
        audio_events=audio_events,
        timeline=timeline,
        artifacts_dir=output_dir,
    )


def _build_timeline(transcription: TranscriptionResult, scenes: list[SceneBoundary],
                     visual_captions: list[VisualCaption],
                     audio_events: list[dict]) -> dict:
    """Build a unified timeline from all Stage 0 signals."""
    events = []

    # Transcript segments
    for seg in transcription.segments:
        events.append(TimelineEvent(
            timestamp_seconds=seg.start_seconds,
            event_type="transcript",
            data={"text": seg.text, "end": seg.end_seconds},
        ))

    # Scene changes
    for scene in scenes:
        events.append(TimelineEvent(
            timestamp_seconds=scene.start_seconds,
            event_type="scene_change",
            data={"scene_index": scene.scene_index},
        ))

    # Visual captions
    for caption in visual_captions:
        events.append(TimelineEvent(
            timestamp_seconds=caption.timestamp_seconds,
            event_type="visual_caption",
            data={"caption": caption.caption[:100]},  # Truncated for timeline view
        ))

    # Audio events
    for event in audio_events:
        events.append(TimelineEvent(
            timestamp_seconds=event["timestamp"],
            event_type=event["type"],
            data=event,
        ))

    timeline_obj = Timeline()
    timeline_obj.add_events(events)
    timeline_obj.sort()
    return timeline_obj.to_dict()
