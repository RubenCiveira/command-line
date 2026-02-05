"""Retrieve project details by semantic similarity."""

from __future__ import annotations

from pathlib import Path

from sentence_transformers import SentenceTransformer

from ai.rag.sqlite_project_setup import SqliteProjectSetup
from ai.user.project_topic import ProjectTopic


class ProjectDetailRetriever:
    """Rank projects by similarity between query and project detail."""

    MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._model: SentenceTransformer | None = None

    def rank_projects(
        self,
        query: str,
        candidates: list[ProjectTopic],
        k: int = 8,
    ) -> list[dict]:
        if not candidates:
            return []

        model = self._get_model()
        query_vec = model.encode([query], normalize_embeddings=True)[0]

        setup = SqliteProjectSetup(self.db_path)
        conn = setup.connect()
        try:
            names = [p.name for p in candidates]
            results = self._search(conn, query_vec, names, k)
        finally:
            conn.close()

        if not results:
            return []

        project_map = {p.name: p for p in candidates}
        by_project: dict[str, float] = {}
        for row in results:
            name = row["project_name"]
            distance = row["distance"]
            current = by_project.get(name)
            if current is None or distance < current:
                by_project[name] = distance

        ranked = []
        for name, distance in by_project.items():
            score = max(0.0, 1.0 - float(distance))
            ranked.append({
                "project": project_map[name],
                "score": score,
                "distance": float(distance),
            })

        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.MODEL_NAME)
        return self._model

    def _search(
        self,
        conn,
        query_embedding,
        project_names: list[str],
        k: int,
    ) -> list[dict]:
        from sqlite_vec import serialize_float32

        vec_blob = serialize_float32(query_embedding.tolist())
        if not project_names:
            return []

        placeholders = ",".join("?" * len(project_names))
        params: list = [vec_blob, k * 3, *project_names, k]

        sql = f"""
            SELECT c.content, d.project_name, vc.distance
            FROM (
                SELECT chunk_id, distance
                FROM vec_project_detail_chunks
                WHERE embedding MATCH ? AND k = ?
            ) AS vc
            JOIN project_detail_chunks c ON c.id = vc.chunk_id
            JOIN project_details d ON d.id = c.detail_id
            WHERE d.project_name IN ({placeholders})
            ORDER BY vc.distance
            LIMIT ?
        """

        rows = conn.execute(sql, params).fetchall()
        return [
            {
                "content": row[0],
                "project_name": row[1],
                "distance": row[2],
            }
            for row in rows
        ]
