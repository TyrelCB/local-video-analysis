"""Storage layer: SQLite database + FTS5 search."""

from .database import Database
from .schema import SCHEMA_SQL
from .search import VideoSearch

__all__ = ["Database", "SCHEMA_SQL", "VideoSearch"]
