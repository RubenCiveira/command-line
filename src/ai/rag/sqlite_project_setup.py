"""SQLite setup helpers for project detail embeddings."""

from __future__ import annotations

import sqlite3
from pathlib import Path


class SqliteProjectSetup:
    """Validate and configure the SQLite schema for project details."""

    VECTOR_DIMENSIONS = 384  # sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def configure(self) -> None:
        """Create the database, load sqlite-vec, and ensure schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = self.connect()
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._ensure_details_table(conn)
            self._ensure_chunks_table(conn)
            self._ensure_vec_table(conn)
            self._ensure_indexes(conn)
            conn.commit()
        finally:
            conn.close()

    def connect(self) -> sqlite3.Connection:
        """Return a ready connection with sqlite-vec loaded."""
        import sqlite_vec

        conn = sqlite3.connect(str(self.db_path))
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return conn

    def _ensure_details_table(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS project_details (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_name TEXT NOT NULL UNIQUE,
                content TEXT NOT NULL,
                updated_at REAL NOT NULL DEFAULT (unixepoch('now'))
            )
            """
        )

    def _ensure_chunks_table(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS project_detail_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                detail_id INTEGER NOT NULL
                    REFERENCES project_details(id) ON DELETE CASCADE,
                content TEXT NOT NULL
            )
            """
        )

    def _ensure_vec_table(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_project_detail_chunks USING vec0(
                chunk_id INTEGER PRIMARY KEY,
                embedding float[{self.VECTOR_DIMENSIONS}]
            )
            """
        )

    def _ensure_indexes(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_project_details_name
            ON project_details(project_name)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_project_detail_chunks_detail
            ON project_detail_chunks(detail_id)
            """
        )
