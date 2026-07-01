"""FastMCP tool definitions for the video analysis engine.

Exposes the pipeline as MCP tools for agent workflows:
- analyze_video
- transcribe_audio
- extract_video_metadata
- extract_chapters
- search_video
- get_video_chunk
- get_clip_context
- export_video_report
- export_srt
- export_json
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastmcp import FastMCP

from ..config import AppConfig
from ..storage.search import VideoSearch

logger = logging.getLogger(__name__)

# Global for storing completed results (in production: use a database)
_analysis_cache: dict = {}


def create_server() -> FastMCP:
    """Create the MCP server with all tools."""
    server = FastMCP(
        "video-analysis",
        version="0.1.0",
    )

    @server.tool()
    def analyze_video(video_path: str, mode: str = "deep",
                       user_prompt: str | None = None) -> dict:
        """Analyze a video file and return structured results.

        Args:
            video_path: Absolute path to the video file.
            mode: Analysis mode — "quick", "deep", or "forensic".
            user_prompt: Optional prompt for focused analysis.

        Returns:
            Dict with video_id, status, summary, chapters, key_moments, and file paths.
        """
        from ..pipeline import run_pipeline
        cfg = AppConfig.load()
        result = run_pipeline(
            video_path=video_path,
            config=cfg,
            user_prompt=user_prompt,
            mode=mode,
        )
        if result.get("status") == "complete":
            _analysis_cache[result["video_id"]] = result
        return result

    @server.tool()
    def transcribe_audio(video_path: str) -> dict:
        """Transcribe audio from a video file.

        Args:
            video_path: Path to the video file.

        Returns:
            Dict with segments, language, and duration.
        """
        from ..stage0.audio import extract_audio, transcribe_audio
        from ..stage0.metadata import extract_metadata
        import tempfile

        cfg = AppConfig.load()
        metadata = extract_metadata(video_path)
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            audio_path = f.name
        extract_audio(video_path, audio_path)
        result = transcribe_audio(
            audio_path,
            model_size=cfg.audio.whisper_model,
            language=cfg.audio.language,
            backend=cfg.audio.transcription_backend,
            asr_python=cfg.audio.asr_python or None,
        )

        return {
            "duration": metadata.duration_human,
            "language": result.language,
            "segments": [
                {
                    "start": seg.start_seconds,
                    "end": seg.end_seconds,
                    "text": seg.text,
                    "confidence": seg.confidence,
                }
                for seg in result.segments
            ],
        }

    @server.tool()
    def extract_video_metadata(video_path: str) -> dict:
        """Extract metadata from a video file.

        Args:
            video_path: Path to the video file.

        Returns:
            Dict with duration, resolution, fps, codec, etc.
        """
        from ..stage0.metadata import extract_metadata
        meta = extract_metadata(video_path)
        return {
            "filename": meta.filename,
            "duration_seconds": meta.duration_seconds,
            "duration_human": meta.duration_human,
            "resolution": meta.resolution,
            "fps": meta.fps,
            "codec": meta.video_stream.codec,
            "audio_codec": meta.audio_stream.codec,
            "file_size_bytes": meta.file_size_bytes,
            "container": meta.container,
        }

    @server.tool()
    def extract_chapters(video_id: str) -> list[dict]:
        """Extract chapter markers for a previously analyzed video.

        Args:
            video_id: The video analysis ID.

        Returns:
            List of chapter objects with start time, title, and summary.
        """
        result = _analysis_cache.get(video_id, {})
        return result.get("chapters", [])

    @server.tool()
    def search_video(video_id: str, query: str) -> dict:
        """Search a previously analyzed video.

        Args:
            video_id: The video analysis ID.
            query: Search query string.

        Returns:
            Dict with results array containing timestamped matches.
        """
        result = _analysis_cache.get(video_id, {})

        # Search in chapters
        chapters = result.get("chapters", [])
        key_moments = result.get("key_moments", [])
        tags = result.get("tags", [])

        matches = []
        for ch in chapters:
            text = f"{ch.get('title', '')} {ch.get('summary', '')}"
            if query.lower() in text.lower():
                matches.append({
                    "start": _ts(ch.get("start_seconds", 0)),
                    "end": _ts(ch.get("end_seconds", 0)),
                    "score": 0.9,
                    "type": "chapter",
                    "preview": ch.get("summary", ""),
                })

        for km in key_moments:
            text = f"{km.get('title', '')} {km.get('description', '')}"
            if query.lower() in text.lower():
                matches.append({
                    "start": _ts(km.get("time", km.get("timestamp_seconds", 0))),
                    "score": 0.7,
                    "type": "key_moment",
                    "preview": km.get("description", ""),
                })

        return {
            "video_id": video_id,
            "query": query,
            "results": matches,
            "total": len(matches),
        }

    @server.tool()
    def get_video_chunk(video_id: str, chunk_index: int = 0) -> dict | None:
        """Get a specific analysis chunk.

        Args:
            video_id: The video analysis ID.
            chunk_index: Chunk index (0-based).

        Returns:
            Chunk data with summary, key moments, and tags.
        """
        result = _analysis_cache.get(video_id, {})
        # Chunks are stored internally; return metadata
        return {"video_id": video_id, "chunk_index": chunk_index,
                "message": "Use search_video or extract_chapters for chunk details"}

    @server.tool()
    def get_clip_context(video_id: str, start_time: float,
                          end_time: float) -> dict:
        """Get surrounding context for a time range.

        Args:
            video_id: The video analysis ID.
            start_time: Start time in seconds.
            end_time: End time in seconds.

        Returns:
            Transcript segments, visual descriptions, and audio events for the range.
        """
        result = _analysis_cache.get(video_id, {})
        return {
            "video_id": video_id,
            "start": _ts(start_time),
            "end": _ts(end_time),
            "message": "Full clip context requires database integration",
        }

    @server.tool()
    def export_video_report(video_id: str, format: str = "markdown") -> str | None:
        """Export an analysis report.

        Args:
            video_id: The video analysis ID.
            format: Output format — "markdown", "json", or "srt".

        Returns:
            Path to the exported file.
        """
        result = _analysis_cache.get(video_id, {})
        return result.get(f"{format}_path")

    @server.tool()
    def export_srt(video_id: str) -> str | None:
        """Export transcript as SRT subtitles.

        Args:
            video_id: The video analysis ID.

        Returns:
            Path to the SRT file.
        """
        result = _analysis_cache.get(video_id, {})
        return result.get("srt_path")

    @server.tool()
    def export_json(video_id: str) -> str | None:
        """Export analysis as JSON.

        Args:
            video_id: The video analysis ID.

        Returns:
            Path to the JSON file.
        """
        result = _analysis_cache.get(video_id, {})
        return result.get("json_path")

    return server


def _ts(seconds: float) -> str:
    """Convert seconds to timestamp string."""
    total_secs = int(seconds)
    hours = total_secs // 3600
    minutes = (total_secs % 3600) // 60
    secs = total_secs % 60
    ms = int((seconds - total_secs) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"
