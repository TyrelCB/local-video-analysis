"""Core pipeline orchestrator.

Coordinates Stage 0 → Pass 1 → Pass 2 in a resumable pipeline.
Tracks progress, handles errors, and stores results in SQLite.
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Callable

from .config import AppConfig
from .models.analysis import AnalysisResult
from .models.chunk import Chunk
from .pass1.chunker import build_chunks
from .pass1.chunker import ChunkSpec as _ChunkSpec
from .pass1.analyzer import analyze_chunks as _analyze_chunks
from .pass1.analyzer import ChunkAnalysis, parse_key_moment_time
from .pass2.synthesizer import synthesize as _synthesize
from .reasoning.llama_cpp import LlamaCppClient, LlamaCppConfig
from .stage0 import run_stage_0, Stage0Output
from .storage.database import Database

logger = logging.getLogger(__name__)


async def run_pipeline(
    video_path: str,
    config: AppConfig,
    user_prompt: str | None = None,
    mode: str | None = None,
    output_dir: str | None = None,
    on_progress: Callable[[str, float, str], None] | None = None,
) -> dict:
    """Run the complete analysis pipeline on a video.

    Args:
        video_path: Path to the input video file.
        config: Application configuration.
        user_prompt: Optional user request for focused analysis.
        mode: Analysis mode (quick/deep/forensic). Uses config default if None.
        output_dir: Override for output directory.
        on_progress: Optional callback(stage, percent, message).

    Returns:
        Dict with video_id, status, analysis result, and file paths.
    """
    mode = mode or config.analysis.default_mode
    analysis_cfg = config.resolve_analysis_config(mode)
    video_cfg = config.resolve_video_config(mode)

    if output_dir is None:
        output_dir = os.path.join(config.storage.output_dir, str(uuid.uuid4())[:8])
    os.makedirs(output_dir, exist_ok=True)

    # Initialize database
    db_path = os.path.join(output_dir, "analysis.db")
    db = Database(db_path)
    db.init_schema()

    # Create video ID
    video_id = str(uuid.uuid4())[:8]
    video_filename = os.path.basename(video_path)

    # Create video record first (job has FK to videos)
    # We create with placeholder values and update after Stage 0 extracts real metadata
    db.create_video(
        video_id=video_id,
        filename=video_filename,
        path=video_path,
        duration=0.0,
        fps=0.0,
        width=0,
        height=0,
    )

    # Create job
    import json
    db.create_job(video_id, video_id, config_json=json.dumps({"default_mode": config.analysis.default_mode}))

    def progress(stage: str, pct: float, msg: str) -> None:
        logger.info("[%s] %.0f%% — %s", stage, pct, msg)
        db.update_job(video_id, current_stage=stage, progress_percent=pct)
        if on_progress:
            on_progress(stage, pct, msg)

    try:
        # ===== Stage 0: Extraction + Alignment =====
        progress("stage0", 5, f"Starting extraction for {video_filename}")

        vision_client = LlamaCppClient(LlamaCppConfig(
            url=config.vision_server.url,
            model=config.vision_server.model,
            timeout=config.vision_server.timeout,
        ))

        stage0 = await run_stage_0(
            video_path=video_path,
            output_dir=os.path.join(output_dir, "stage0"),
            model_size=config.audio.whisper_model,
            language=config.audio.language,
            transcription_backend=config.audio.transcription_backend,
            asr_python=config.audio.asr_python,
            frame_fps=video_cfg.frame_sampling_fps,
            scene_threshold=config.video.scene_threshold,
            enable_audio_events=analysis_cfg.enable_audio_events,
            audio_events_backend=config.audio.audio_events_backend,
            vision_client=vision_client,
            caption_max_concurrent=video_cfg.caption_max_concurrent,
            caption_dedup=video_cfg.caption_dedup,
        )

        progress("stage0", 40, f"Extracted {len(stage0.transcription.segments)} transcript segments, "
                 f"{len(stage0.frames)} frames, {len(stage0.visual_captions)} captions")

        # Store video metadata (update placeholder record created earlier)
        db.update_video(
            video_id=video_id,
            duration=stage0.metadata.duration_seconds,
            fps=stage0.metadata.video_stream.fps,
            width=stage0.metadata.video_stream.width,
            height=stage0.metadata.video_stream.height,
        )
        db.update_video_status(video_id, "running")

        # Store transcript segments
        segments_data = [
            {
                "id": f"seg_{i:05d}",
                "video_id": video_id,
                "start_seconds": seg.start_seconds,
                "end_seconds": seg.end_seconds,
                "text": seg.text,
                "speaker": "",
                "words": str(seg.words),
            }
            for i, seg in enumerate(stage0.transcription.segments)
        ]
        db.insert_transcript_segments(segments_data)

        # Store scenes
        scenes_data = [
            {
                "id": f"scene_{s.scene_index:03d}",
                "video_id": video_id,
                "scene_index": s.scene_index,
                "start_seconds": s.start_seconds,
                "end_seconds": s.end_seconds or stage0.metadata.duration_seconds,
                "frame_path": "",
            }
            for s in stage0.scenes
        ]
        db.insert_scenes(scenes_data)

        # Store audio events
        if stage0.audio_events:
            db.insert_audio_events([
                (f"ae_{i:04d}", video_id, None,
                 ev.get("timestamp", 0.0), ev.get("type", ""), ev.get("description", ""))
                for i, ev in enumerate(stage0.audio_events)
            ])

        # ===== Pass 1: Segment Analysis =====
        progress("pass1", 45, "Building chunks...")

        frames = stage0.frames
        frame_timestamps = [f.timestamp_seconds for f in frames]
        frame_captions = [c.caption for c in stage0.visual_captions]

        chunk_specs = build_chunks(
            transcription=stage0.transcription.segments,
            scenes=stage0.scenes,
            frame_timestamps=frame_timestamps,
            visual_captions=frame_captions,
            audio_events=stage0.audio_events,
            video_duration=stage0.metadata.duration_seconds,
            chunk_seconds=analysis_cfg.chunk_seconds,
            overlap_seconds=analysis_cfg.overlap_seconds,
        )

        progress("pass1", 50, f"Analyzing {len(chunk_specs)} chunks...")

        reasoning_client = LlamaCppClient(LlamaCppConfig(
            url=config.reasoning_server.url,
            model=config.reasoning_server.model,
            timeout=config.reasoning_server.timeout,
        ))

        chunk_analyses = await _analyze_chunks(
            chunk_specs, reasoning_client,
            video_duration=stage0.metadata.duration_seconds,
            max_concurrent=analysis_cfg.reasoning_max_concurrent,
        )

        # Store chunk results
        for i, (chunk_spec, analysis) in enumerate(zip(chunk_specs, chunk_analyses)):
            chunk_id = f"chunk_{i:05d}"
            db.insert_chunk(
                chunk_id=chunk_id,
                video_id=video_id,
                chunk_index=i,
                start=chunk_spec.start_seconds,
                end=chunk_spec.end_seconds,
                transcript=analysis.summary[:200],  # Store summary preview
                visual_summary="",
                ocr_text="",
                audio_events=str(analysis.key_moments),
                summary=analysis.summary,
                tags=str(analysis.tags),
            )

            # Store key moments
            for j, km in enumerate(analysis.key_moments):
                time_val = km.get("time", chunk_spec.start_seconds)
                time_seconds = parse_key_moment_time(time_val, chunk_spec.start_seconds)

                db.insert_key_moments([
                    (f"km_{i:03d}_{j:02d}", video_id, chunk_id,
                     time_seconds,
                     km.get("description", "")[:100],
                     km.get("description", ""),
                     km.get("importance", 1),
                     )
                ])

            # Index for search
            from .storage.search import VideoSearch
            search = VideoSearch(db_path)
            search.add_chunk_content(
                chunk_id=chunk_id,
                video_id=video_id,
                start_seconds=chunk_spec.start_seconds,
                transcript=analysis.summary,
                visual_summary="",
                tags=" ".join(analysis.tags),
            )

        progress("pass1", 80, "Pass 1 complete")

        # ===== Pass 2: Global Synthesis =====
        progress("pass2", 85, "Running global synthesis...")

        # Build Chunk objects for synthesizer
        chunks_for_synthesis = []
        for i, (chunk_spec, analysis) in enumerate(zip(chunk_specs, chunk_analyses)):
            chunk_obj = Chunk(
                video_id=video_id,
                chunk_id=f"chunk_{i:05d}",
                chunk_index=i,
                start_seconds=chunk_spec.start_seconds,
                end_seconds=chunk_spec.end_seconds,
                summary=analysis.summary,
                key_moments=analysis.key_moments,
                tags=analysis.tags,
                quotes=analysis.quotes,
                issues=analysis.issues,
                detected_actions=analysis.detected_actions,
            )
            chunks_for_synthesis.append(chunk_obj)

        # Build analysis results dict for synthesizer
        analysis_dicts = []
        for analysis in chunk_analyses:
            analysis_dicts.append({
                "summary": analysis.summary,
                "key_moments": analysis.key_moments,
                "tags": analysis.tags,
                "quotes": analysis.quotes,
                "issues": analysis.issues,
                "detected_actions": analysis.detected_actions,
            })

        result = await _synthesize(
            chunks=chunks_for_synthesis,
            analysis_results=analysis_dicts,
            video_duration=stage0.metadata.duration_seconds,
            client=reasoning_client,
            user_prompt=user_prompt,
            mode=mode,
        )

        result.video_id = video_id

        progress("pass2", 95, "Synthesis complete")

        # ===== Mark complete =====
        db.update_video_status(video_id, "completed")
        db.update_job(video_id, status="completed", progress_percent=100)

        output_paths = _save_outputs(result, output_dir, stage0, chunk_specs,
                                      chunk_analyses, video_id, video_filename)

        return {
            "video_id": video_id,
            "status": "complete",
            "duration": stage0.metadata.duration_human,
            "summary": {
                "executive": result.executive_summary,
                "detailed": result.detailed_summary,
            },
            "chapters": [c.model_dump() for c in result.chapters],
            "key_moments": [m.model_dump() for m in result.key_moments[:20]],
            "tags": result.tags,
            "action_items": result.action_items,
            "report_path": output_paths.get("markdown"),
            "json_path": output_paths.get("json"),
            "output_dir": output_dir,
        }

    except Exception as e:
        logger.exception("Pipeline failed")
        db.update_video_status(video_id, "failed")
        db.update_job(video_id, status="failed", error_message=str(e))
        return {
            "video_id": video_id,
            "status": "failed",
            "error": str(e),
        }


def _save_outputs(result: AnalysisResult, output_dir: str,
                   stage0: Stage0Output,
                   chunk_specs: list,
                   chunk_analyses: list[ChunkAnalysis],
                   video_id: str,
                   video_filename: str) -> dict[str, str]:
    """Save analysis outputs to the output directory."""
    outputs = {}
    os.makedirs(output_dir, exist_ok=True)

    # JSON export
    json_path = os.path.join(output_dir, f"{video_id}_analysis.json")
    with open(json_path, "w") as f:
        import json
        json.dump(result.to_dict(), f, indent=2)
    outputs["json"] = json_path

    # Markdown report
    from .export.markdown_export import export_markdown
    md_path = os.path.join(output_dir, f"{video_id}_report.md")
    md_content = export_markdown(result, stage0.metadata, video_id,
                                 audio_events=stage0.audio_events)
    with open(md_path, "w") as f:
        f.write(md_content)
    outputs["markdown"] = md_path

    # SRT export
    from .export.srt_export import export_srt
    srt_path = os.path.join(output_dir, f"{video_id}_transcript.srt")
    srt_content = export_srt(stage0.transcription.segments)
    with open(srt_path, "w") as f:
        f.write(srt_content)
    outputs["srt"] = srt_path

    return outputs
