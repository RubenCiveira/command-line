"""SQLite setup helpers for RAG storage with sqlite-vec."""

from __future__ import annotations

import sqlite3
from pathlib import Path


class SqliteRagSetup:
    """Validate and configure the SQLite schema for RAG."""

    VECTOR_DIMENSIONS = 384  # paraphrase-multilingual-MiniLM-L12-v2

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def configure(self) -> None:
        """Create the database, load sqlite-vec, and ensure schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = self.connect()
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._ensure_documents_table(conn)
            self._ensure_chunks_table(conn)
            self._ensure_vec_table(conn)
            self._ensure_document_categories_table(conn)
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

    def _ensure_documents_table(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL,
                topic TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at REAL NOT NULL DEFAULT (unixepoch('now')),
                UNIQUE(topic, path)
            )
            """
        )

    def _ensure_chunks_table(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL
                    REFERENCES documents(id) ON DELETE CASCADE,
                content TEXT NOT NULL
            )
            """
        )

    def _ensure_vec_table(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
                chunk_id INTEGER PRIMARY KEY,
                embedding float[{self.VECTOR_DIMENSIONS}]
            )
            """
        )

    def _ensure_document_categories_table(
        self, conn: sqlite3.Connection,
    ) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS document_categories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL
                    REFERENCES documents(id) ON DELETE CASCADE,
                level INTEGER NOT NULL,
                category TEXT NOT NULL,
                score REAL NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_doc_categories_category
            ON document_categories(category)
            """
        )
