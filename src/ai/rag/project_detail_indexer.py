"""Index project details into a local SQLite vector store."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from sentence_transformers import SentenceTransformer

from ai.rag.sqlite_project_setup import SqliteProjectSetup
from ai.user.project_topic import ProjectTopic

logger = logging.getLogger(__name__)


class ProjectDetailIndexer:
    """Index project detail text for semantic retrieval."""

    MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._model: SentenceTransformer | None = None

    def index_projects(self, projects: list[ProjectTopic]) -> int:
        """Index project details. Returns number of projects updated."""
        setup = SqliteProjectSetup(self.db_path)
        setup.configure()
        updated = 0

        for project in projects:
            detail = (project.detail or "").strip()
            if not detail:
                continue

            conn = setup.connect()
            try:
                detail_id, changed = self._upsert_detail(
                    conn, project.name, detail,
                )
                if not changed:
                    continue

                chunks = self._split_lines(detail)
                if chunks:
                    model = self._get_model()
                    embeddings = model.encode(chunks, normalize_embeddings=True)
                    for chunk_text, embedding in zip(chunks, embeddings):
                        self._insert_chunk_and_embedding(
                            conn, detail_id, chunk_text, embedding,
                        )
                conn.commit()
                updated += 1
            except Exception as exc:
                logger.error(
                    "Error indexing project %s: %s",
                    project.name,
                    exc,
                )
                try:
                    conn.rollback()
                except Exception:
                    pass
            finally:
                conn.close()

        return updated

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.MODEL_NAME)
        return self._model

    def _split_lines(self, text: str) -> list[str]:
        lines = [line.strip() for line in text.splitlines()]
        return [line for line in lines if line]

    def _upsert_detail(
        self,
        conn: sqlite3.Connection,
        project_name: str,
        content: str,
    ) -> tuple[int, bool]:
        row = conn.execute(
            "SELECT id, content FROM project_details WHERE project_name = ?",
            (project_name,),
        ).fetchone()

        if row:
            detail_id, existing = row
            if existing == content:
                return int(detail_id), False
            conn.execute(
                "UPDATE project_details SET content = ?, updated_at = unixepoch('now') "
                "WHERE id = ?",
                (content, detail_id),
            )
            self._delete_chunks(conn, detail_id)
            return int(detail_id), True

        cursor = conn.execute(
            "INSERT INTO project_details (project_name, content) VALUES (?, ?)",
            (project_name, content),
        )
        return int(cursor.lastrowid), True

    def _delete_chunks(self, conn: sqlite3.Connection, detail_id: int) -> None:
        chunk_ids = conn.execute(
            "SELECT id FROM project_detail_chunks WHERE detail_id = ?",
            (detail_id,),
        ).fetchall()
        if chunk_ids:
            ids = [row[0] for row in chunk_ids]
            placeholders = ",".join("?" * len(ids))
            conn.execute(
                "DELETE FROM vec_project_detail_chunks "
                f"WHERE chunk_id IN ({placeholders})",
                ids,
            )
        conn.execute(
            "DELETE FROM project_detail_chunks WHERE detail_id = ?",
            (detail_id,),
        )

    def _insert_chunk_and_embedding(
        self,
        conn: sqlite3.Connection,
        detail_id: int,
        content: str,
        embedding,
    ) -> None:
        from sqlite_vec import serialize_float32

        cursor = conn.execute(
            "INSERT INTO project_detail_chunks (detail_id, content) VALUES (?, ?)",
            (detail_id, content),
        )
        chunk_id = cursor.lastrowid
        vec_blob = serialize_float32(embedding.tolist())
        conn.execute(
            "INSERT INTO vec_project_detail_chunks (chunk_id, embedding) VALUES (?, ?)",
            (chunk_id, vec_blob),
        )
