"""Retrieve RAG context from the local SQLite knowledge base."""

from __future__ import annotations

from pathlib import Path
from typing import List

from sentence_transformers import SentenceTransformer

from ai.rag.sqlite_rag_setup import SqliteRagSetup
from ai.user.user_config import UserConfig


class RagRetriever:
    """Query the local vector store for relevant context."""

    MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self, config: UserConfig) -> None:
        self.db_path = config.ragDatabasePath()
        self._model: SentenceTransformer | None = None

    def retrieve(
        self,
        question: str,
        topics: list[str] | None = None,
        categories: list[str] | None = None,
        k: int = 5,
    ) -> str:
        """Embed the question and return the top-k matching chunks.

        Args:
            question: The user's query text.
            topics: Optional topic names to filter by.
            categories: Optional category names to filter by.
                        Matches documents that have any of the
                        given categories at any level.
            k: Number of results to return.

        Returns:
            Concatenated text of matching chunks separated by
            double newlines. Empty string if no results.
        """
        model = self._get_model()
        query_vec = model.encode([question], normalize_embeddings=True)[0]

        setup = SqliteRagSetup(self.db_path)
        conn = setup.connect()
        try:
            results = self._search(conn, query_vec, topics, categories, k)
            return "\n\n".join(results)
        finally:
            conn.close()

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.MODEL_NAME)
        return self._model

    def _search(
        self,
        conn,
        query_embedding,
        topics: list[str] | None,
        categories: list[str] | None,
        k: int,
    ) -> List[str]:
        from sqlite_vec import serialize_float32

        vec_blob = serialize_float32(query_embedding.tolist())

        has_topics = bool(topics)
        has_categories = bool(categories)

        # Build WHERE clauses for post-KNN filtering
        filters: list[str] = []
        params: list = [vec_blob, k * 5 if (has_topics or has_categories) else k]

        if has_topics:
            placeholders = ",".join("?" * len(topics))
            filters.append(f"d.topic IN ({placeholders})")
            params.extend(topics)

        if has_categories:
            placeholders = ",".join("?" * len(categories))
            filters.append(
                f"d.id IN ("
                f"SELECT document_id FROM document_categories "
                f"WHERE category IN ({placeholders}))"
            )
            params.extend(categories)

        if filters:
            where_clause = "WHERE " + " AND ".join(filters)
            sql = f"""
                SELECT c.content
                FROM (
                    SELECT rowid, distance
                    FROM vec_chunks
                    WHERE embedding MATCH ? AND k = ?
                ) AS vc
                JOIN chunks c ON c.id = vc.rowid
                JOIN documents d ON d.id = c.document_id
                {where_clause}
                ORDER BY vc.distance
                LIMIT ?
            """
            params.append(k)
        else:
            sql = """
                SELECT c.content
                FROM (
                    SELECT rowid, distance
                    FROM vec_chunks
                    WHERE embedding MATCH ? AND k = ?
                ) AS vc
                JOIN chunks c ON c.id = vc.rowid
                ORDER BY vc.distance
            """

        rows = conn.execute(sql, params).fetchall()
        return [row[0] for row in rows]
