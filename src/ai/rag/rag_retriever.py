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
        query_classification: list[dict] | None = None,
        k: int = 5,
    ) -> str:
        """Embed the question and return the top-k matching chunks.

        Args:
            question: The user's query text.
            topics: Optional topic names to filter by.
            categories: Optional category names to filter by.
                        Matches documents that have any of the
                        given categories at any level.
            query_classification: Classification path from
                Classificator.classify(). Level 0 filters exclusively;
                deeper levels re-rank by category proximity.
            k: Number of results to return.

        Returns:
            Concatenated text of matching chunks separated by
            double newlines. Empty string if no results.
        """
        results = self.retrieve_with_sources(
            question, topics, categories, query_classification, k,
        )
        return "\n\n".join(r["content"] for r in results)

    def retrieve_with_sources(
        self,
        question: str,
        topics: list[str] | None = None,
        categories: list[str] | None = None,
        query_classification: list[dict] | None = None,
        k: int = 5,
    ) -> list[dict]:
        """Embed the question and return top-k chunks with source info.

        Args:
            query_classification: Classification path from
                Classificator.classify(). Level 0 is used as a hard
                filter (exclusive). Deeper levels boost documents that
                share more of the classification path.

        Returns a list of dicts with keys:
            content, path, topic, distance, doc_id.
            When query_classification is used, also: category_overlap.
        """
        model = self._get_model()
        query_vec = model.encode([question], normalize_embeddings=True)[0]

        # Extract level-0 category for hard filtering
        root_category = None
        if query_classification:
            level0 = [s for s in query_classification if s["level"] == 0]
            if level0:
                root_category = level0[0]["label"]

        setup = SqliteRagSetup(self.db_path)
        conn = setup.connect()
        try:
            fetch_k = k * 3 if query_classification else k
            results = self._search(
                conn, query_vec, topics, categories, fetch_k,
                root_category=root_category,
            )

            if (
                query_classification
                and len(query_classification) > 1
                and len(results) > 0
            ):
                results = self._rerank_by_categories(
                    conn, results, query_classification,
                )

            return results[:k]
        finally:
            conn.close()

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.MODEL_NAME)
        return self._model

    # ── KNN search ────────────────────────────────────────────────

    def _search(
        self,
        conn,
        query_embedding,
        topics: list[str] | None,
        categories: list[str] | None,
        k: int,
        root_category: str | None = None,
    ) -> list[dict]:
        from sqlite_vec import serialize_float32

        vec_blob = serialize_float32(query_embedding.tolist())

        has_topics = bool(topics)
        has_categories = bool(categories)
        has_root = bool(root_category)

        # Build WHERE clauses for post-KNN filtering
        filters: list[str] = []
        needs_overfetch = has_topics or has_categories or has_root
        params: list = [vec_blob, k * 5 if needs_overfetch else k]

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

        if has_root:
            filters.append(
                "d.id IN ("
                "SELECT document_id FROM document_categories "
                "WHERE category = ? AND level = 0)"
            )
            params.append(root_category)

        where_clause = ""
        if filters:
            where_clause = "WHERE " + " AND ".join(filters)

        sql = f"""
            SELECT c.content, d.path, d.topic, vc.distance, d.id
            FROM (
                SELECT chunk_id, distance
                FROM vec_chunks
                WHERE embedding MATCH ? AND k = ?
            ) AS vc
            JOIN chunks c ON c.id = vc.chunk_id
            JOIN documents d ON d.id = c.document_id
            {where_clause}
            ORDER BY vc.distance
            LIMIT ?
        """
        params.append(k)

        rows = conn.execute(sql, params).fetchall()
        return [
            {
                "content": row[0],
                "path": row[1],
                "topic": row[2],
                "distance": row[3],
                "doc_id": row[4],
            }
            for row in rows
        ]

    # ── Category re-ranking ───────────────────────────────────────

    def _rerank_by_categories(
        self,
        conn,
        results: list[dict],
        query_classification: list[dict],
    ) -> list[dict]:
        """Re-rank results by category overlap with query classification.

        Level 0 is already filtered. Deeper levels (1, 2, ...) are used
        to boost documents that share more of the classification path.
        """
        query_cats = {
            s["level"]: s["label"]
            for s in query_classification
            if s["level"] > 0
        }
        if not query_cats:
            return results

        # Fetch categories for result documents
        doc_ids = list(set(r["doc_id"] for r in results))
        placeholders = ",".join("?" * len(doc_ids))
        rows = conn.execute(
            f"SELECT document_id, level, category "
            f"FROM document_categories "
            f"WHERE document_id IN ({placeholders}) AND level > 0",
            doc_ids,
        ).fetchall()

        # Build doc_id -> {level: category}
        doc_cats: dict[int, dict[int, str]] = {}
        for doc_id, level, category in rows:
            if doc_id not in doc_cats:
                doc_cats[doc_id] = {}
            doc_cats[doc_id][level] = category

        # Score each result by overlap with query categories
        for result in results:
            cats = doc_cats.get(result["doc_id"], {})
            overlap = sum(
                1 for level, label in query_cats.items()
                if cats.get(level) == label
            )
            result["category_overlap"] = overlap

        # Sort: more overlap first, then by vector distance
        results.sort(
            key=lambda r: (-r.get("category_overlap", 0), r["distance"]),
        )

        return results
