"""SQLite database connection and management.

Provides a context-managed connection with schema initialization,
and utility methods for common operations.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from .schema import SCHEMA_SQL


class Database:
    """SQLite database wrapper with schema management."""

    def __init__(self, db_path: str | Path | None = None):
        if db_path is None:
            db_path = Path("data/video_analysis.db")
        self.db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None

    @contextmanager
    def connection(self):
        """Context manager for a database connection."""
        conn = None
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            yield conn
        finally:
            if conn:
                conn.close()

    def init_schema(self) -> None:
        """Initialize all tables and indexes from SCHEMA_SQL."""
        with self.connection() as conn:
            conn.executescript(SCHEMA_SQL)
            conn.commit()

    @contextmanager
    def cursor(self):
        """Context manager that yields a cursor and commits on success."""
        with self.connection() as conn:
            cur = conn.cursor()
            try:
                yield cur
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def get_video(self, video_id: str) -> dict | None:
        """Get a video record by ID."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM videos WHERE id = ?", (video_id,)
            ).fetchone()
            return dict(row) if row else None

    def create_video(self, video_id: str, filename: str, path: str,
                     duration: float, fps: float, width: int, height: int) -> None:
        """Insert a new video record."""
        with self.cursor() as cur:
            cur.execute(
                """INSERT INTO videos (id, filename, path, duration_seconds, fps, width, height)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (video_id, filename, path, duration, fps, width, height),
            )

    def update_video(self, video_id: str, **kwargs) -> None:
        """Update video metadata fields."""
        allowed = {"filename", "path", "duration_seconds", "fps", "width", "height", "status"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [video_id]
        with self.cursor() as cur:
            cur.execute(f"UPDATE videos SET {set_clause} WHERE id = ?", values)

    def update_video_status(self, video_id: str, status: str) -> None:
        """Update a video's analysis status."""
        with self.cursor() as cur:
            cur.execute(
                "UPDATE videos SET status = ? WHERE id = ?",
                (status, video_id),
            )

    def create_job(self, job_id: str, video_id: str, config_json: str = "{}") -> None:
        """Create a new pipeline job record."""
        with self.cursor() as cur:
            cur.execute(
                """INSERT INTO jobs (id, video_id, status, config_json)
                   VALUES (?, ?, 'pending', ?)""",
                (job_id, video_id, config_json),
            )

    def update_job(self, job_id: str, **kwargs) -> None:
        """Update a job's status, progress, current stage/step, etc."""
        allowed = {"status", "current_stage", "current_step",
                   "progress_percent", "error_message", "completed_at"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [job_id]
        with self.cursor() as cur:
            cur.execute(
                f"UPDATE jobs SET {set_clause} WHERE id = ?", values,
            )

    def get_active_job(self, video_id: str) -> dict | None:
        """Get a job that is still running (not completed or failed)."""
        with self.connection() as conn:
            row = conn.execute(
                """SELECT * FROM jobs
                   WHERE video_id = ? AND status IN ('pending', 'running')
                   ORDER BY started_at DESC LIMIT 1""",
                (video_id,),
            ).fetchone()
            return dict(row) if row else None

    def get_chunks(self, video_id: str) -> list[dict]:
        """Get all chunks for a video, ordered by start time."""
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM chunks WHERE video_id = ? ORDER BY start_seconds",
                (video_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def insert_chunk(self, chunk_id: str, video_id: str, chunk_index: int,
                     start: float, end: float, transcript: str,
                     visual_summary: str, ocr_text: str, audio_events: str,
                     summary: str, tags: str) -> None:
        """Insert a chunk record."""
        with self.cursor() as cur:
            cur.execute(
                """INSERT INTO chunks (id, video_id, chunk_index, start_seconds, end_seconds,
                   transcript, visual_summary, ocr_text, audio_events, summary, tags)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (chunk_id, video_id, chunk_index, start, end,
                 transcript, visual_summary, ocr_text, audio_events, summary, tags),
            )

    def insert_key_moments(self, moments: list[tuple]) -> None:
        """Batch insert key moments. Each tuple: (id, video_id, chunk_id, timestamp_seconds, title, description, importance)."""
        with self.cursor() as cur:
            cur.executemany(
                """INSERT INTO key_moments (id, video_id, chunk_id, timestamp_seconds, title, description, importance)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                moments,
            )

    def get_key_moments(self, video_id: str) -> list[dict]:
        """Get all key moments for a video, sorted by importance desc."""
        with self.connection() as conn:
            rows = conn.execute(
                """SELECT * FROM key_moments WHERE video_id = ?
                   ORDER BY importance DESC, timestamp_seconds""",
                (video_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def insert_audio_events(self, events: list[tuple]) -> None:
        """Batch insert audio events. Each tuple: (id, video_id, chunk_id, timestamp_seconds, event_type, description)."""
        with self.cursor() as cur:
            cur.executemany(
                """INSERT INTO audio_events (id, video_id, chunk_id, timestamp_seconds, event_type, description)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                events,
            )

    def get_audio_events(self, video_id: str) -> list[dict]:
        """Get all audio events for a video, sorted by timestamp."""
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM audio_events WHERE video_id = ? ORDER BY timestamp_seconds",
                (video_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def insert_transcript_segments(self, segments: list[dict]) -> None:
        """Batch insert transcript segments. Each dict: {id, video_id, chunk_id, start_seconds, end_seconds, text, speaker, words}."""
        with self.cursor() as cur:
            cur.executemany(
                """INSERT INTO transcript_segments (id, video_id, chunk_id, start_seconds, end_seconds, text, speaker, words)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                [(s.get("id"), s.get("video_id"), s.get("chunk_id"),
                  s.get("start_seconds"), s.get("end_seconds"),
                  s.get("text"), s.get("speaker"), s.get("words"))
                 for s in segments],
            )

    def insert_scenes(self, scenes: list[dict]) -> None:
        """Batch insert scene boundaries. Each dict: {id, video_id, scene_index, start_seconds, end_seconds, frame_path}."""
        with self.cursor() as cur:
            cur.executemany(
                """INSERT INTO scenes (id, video_id, scene_index, start_seconds, end_seconds, frame_path)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                [(s.get("id"), s.get("video_id"), s.get("scene_index"),
                  s.get("start_seconds"), s.get("end_seconds"),
                  s.get("frame_path")) for s in scenes],
            )

    def get_scenes(self, video_id: str) -> list[dict]:
        """Get all scenes for a video."""
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM scenes WHERE video_id = ? ORDER BY start_seconds",
                (video_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_all_video_ids(self) -> list[str]:
        """Get all analyzed video IDs."""
        with self.connection() as conn:
            rows = conn.execute("SELECT id FROM videos WHERE status = 'completed'").fetchall()
            return [r["id"] for r in rows]
