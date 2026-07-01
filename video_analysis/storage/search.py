"""FTS5 full-text search over analyzed video content.

Supports keyword search across transcripts, chunk summaries, visual
captions, OCR text, and tags. Returns timestamped results with
highlighted previews.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SearchResult:
    """A single search result."""
    video_id: str
    chunk_id: str | None
    timestamp_seconds: float
    content_type: str          # chunk_summary, transcript, visual_caption, ocr
    score: float
    preview: str
    matched_fields: list[str] = field(default_factory=list)


class VideoSearch:
    """FTS5-based search over the analysis database."""

    def __init__(self, db_path: str | Path | None = None):
        if db_path is None:
            db_path = Path("data/video_analysis.db")
        self.db_path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def search(self, query: str, video_id: str | None = None,
               top_k: int = 20) -> list[ResultItem]:
        """Search content matching a query.

        Args:
            query: Free-text search query.
            video_id: Optional filter to a single video.
            top_k: Maximum results to return.

        Returns:
            List of results sorted by relevance.
        """
        conn = self._connect()
        try:
            where_clause = f"video_id = '{video_id}'" if video_id else "1=1"

            # FTS5 MATCH requires the query to be embedded in SQL, not parameterized
            safe_query = query.replace("'", "''")
            sql = f"""
                SELECT video_id, chunk_id, timestamp_seconds, content_type,
                       substr(content, 1, 300) AS preview
                FROM search_content
                WHERE search_content MATCH '{safe_query}' AND {where_clause}
                ORDER BY rank
                LIMIT {int(top_k)}
            """
            rows = conn.execute(sql).fetchall()

            return [
                ResultItem(
                    video_id=str(r["video_id"]),
                    chunk_id=r["chunk_id"],
                    timestamp_seconds=float(r["timestamp_seconds"]),
                    content_type=str(r["content_type"]),
                    preview=str(r["preview"]),
                )
                for r in rows
            ]
        finally:
            conn.close()

    def search_chunks(self, video_id: str, query: str,
                      top_k: int = 20) -> list[ResultItem]:
        """Search only within a single video's chunks."""
        return self.search(query, video_id=video_id, top_k=top_k)

    def add_chunk_content(self, chunk_id: str, video_id: str,
                          start_seconds: float, transcript: str,
                          visual_summary: str, tags: str) -> None:
        """Index a chunk's content for search."""
        conn = self._connect()
        try:
            content = " ".join(filter(None, [transcript, visual_summary, tags]))
            conn.execute(
                """INSERT INTO search_content (content, video_id, chunk_id, timestamp_seconds, content_type)
                   VALUES (?, ?, ?, ?, 'chunk_summary')""",
                (content, video_id, chunk_id, start_seconds),
            )
            conn.commit()
        finally:
            conn.close()

    def reindex(self, video_id: str) -> None:
        """Rebuild FTS index for a specific video.

        Called when chunks are updated or after analysis completes.
        """
        conn = self._connect()
        try:
            # Delete old entries for this video
            conn.execute(
                "DELETE FROM search_content WHERE video_id = ?",
                (video_id,),
            )
            # Re-insert all chunks for this video
            rows = conn.execute(
                "SELECT id, start_seconds, transcript, visual_summary, tags "
                "FROM chunks WHERE video_id = ?",
                (video_id,),
            ).fetchall()
            for row in rows:
                content = " ".join(filter(None, [
                    row["transcript"], row["visual_summary"], row["tags"],
                ]))
                conn.execute(
                    """INSERT INTO search_content (content, video_id, chunk_id, timestamp_seconds, content_type)
                       VALUES (?, ?, ?, ?, 'chunk_summary')""",
                    (content, video_id, row["id"], float(row["start_seconds"])),
                )
            conn.commit()
        finally:
            conn.close()

    def delete_video_index(self, video_id: str) -> None:
        """Remove all indexed content for a deleted video."""
        conn = self._connect()
        try:
            conn.execute(
                "DELETE FROM search_content WHERE video_id = ?",
                (video_id,),
            )
            conn.commit()
        finally:
            conn.close()


@dataclass
class ResultItem:
    """A single FTS search result."""
    video_id: str
    chunk_id: str | None
    timestamp_seconds: float
    content_type: str
    preview: str

    @property
    def timestamp(self) -> str:
        """Format timestamp as HH:MM:SS.mmm."""
        total_secs = int(self.timestamp_seconds)
        hours = total_secs // 3600
        minutes = (total_secs % 3600) // 60
        secs = total_secs % 60
        ms = int((self.timestamp_seconds - total_secs) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"
