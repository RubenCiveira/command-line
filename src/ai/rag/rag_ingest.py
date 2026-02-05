"""Ingest topic documents into SQLite RAG tables."""

from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path
from typing import Iterable, List

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ai.rag.sqlite_rag_setup import SqliteRagSetup
from ai.rag.content_extractor import RagContentExtractor
from ai.classificator import Classificator
from ai.user.user_config import UserConfig

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai.model_pool import ModelPool

logger = logging.getLogger(__name__)


class RagIngest:
    """Ingest topic documents into the local SQLite RAG database."""

    MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self, config: UserConfig, model_pool: ModelPool | None = None) -> None:
        self.config = config
        self.db_path = config.ragDatabasePath()
        self.topics = config.ragTopics()
        self._model: SentenceTransformer | None = None
        self._classificator: Classificator | None = None
        self.extractor = RagContentExtractor()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=900,
            chunk_overlap=150,
        )
        categories_path = config.ragCategoriesPath()
        if categories_path.exists():
            self._classificator = Classificator(categories_path, model_pool=model_pool)

    def ingest(self) -> int:
        """Process all configured topic directories. Returns count of new docs."""
        setup = SqliteRagSetup(self.db_path)
        setup.configure()

        files = self._collect_files()
        created = 0

        for topic_name, topic_path, file_path in files:
            relative_path = file_path.relative_to(topic_path).as_posix()
            conn = setup.connect()
            try:
                if self._document_exists(conn, topic_name, relative_path):
                    continue

                content = self._read_text(file_path)
                if not content.strip():
                    continue
                if not self._is_text_like(content):
                    continue

                document_id = self._insert_document(
                    conn, topic_name, relative_path, content,
                )
                chunks = self._split_text(content)
                if chunks:
                    model = self._get_model()
                    embeddings = model.encode(chunks, normalize_embeddings=True)
                    for chunk_text, embedding in zip(chunks, embeddings):
                        self._insert_chunk_and_embedding(
                            conn, document_id, chunk_text, embedding,
                        )
                self._classify_and_store(conn, document_id, content)
                conn.commit()
                created += 1
                logger.info("Ingested: %s/%s", topic_name, relative_path)
            except Exception as exc:
                logger.error(
                    "Error ingesting %s/%s: %s",
                    topic_name, relative_path, exc,
                )
                try:
                    conn.rollback()
                except Exception:
                    pass
            finally:
                conn.close()

        return created

    def ingest_background(self) -> threading.Thread:
        """Start ingestion in a background daemon thread."""
        thread = threading.Thread(target=self._safe_ingest, daemon=True)
        thread.start()
        return thread

    def _safe_ingest(self) -> None:
        try:
            count = self.ingest()
            logger.info("Background RAG ingestion finished: %d new documents", count)
        except Exception as exc:
            logger.error("Background RAG ingestion failed: %s", exc)

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.MODEL_NAME)
        return self._model

    # ── File collection ──────────────────────────────────────────

    def _collect_files(self) -> list[tuple[str, Path, Path]]:
        files: list[tuple[str, Path, Path]] = []
        for topic in self.topics:
            name = topic.get("name", "")
            path_str = topic.get("path", "")
            if not name or not path_str:
                continue
            topic_path = Path(path_str).expanduser()
            if not topic_path.exists():
                continue
            for file_path in self._iter_files(topic_path):
                files.append((name, topic_path, file_path))
        return files

    def _iter_files(self, root: Path) -> Iterable[Path]:
        for path in root.rglob("*"):
            if path.is_file():
                yield path

    # ── Text extraction and validation ───────────────────────────

    def _read_text(self, path: Path) -> str:
        try:
            extracted = self.extractor.extract(path)
            if extracted:
                return extracted
            if not self._is_probably_text_file(path):
                return ""
            return path.read_text(encoding="utf-8", errors="ignore")
        except (OSError, UnicodeError) as exc:
            raise RuntimeError(f"Failed to read {path}") from exc

    def _is_probably_text_file(self, path: Path) -> bool:
        try:
            with path.open("rb") as handle:
                chunk = handle.read(2048)
        except OSError:
            return False

        if not chunk:
            return False
        if b"\x00" in chunk:
            return False

        non_text = 0
        for byte in chunk:
            if byte in (9, 10, 13):
                continue
            if 32 <= byte <= 126:
                continue
            non_text += 1

        return (non_text / len(chunk)) < 0.3

    def _is_text_like(self, content: str) -> bool:
        sample = content[:4000]
        if "\x00" in sample:
            return False

        printable = 0
        meaningful = 0
        for ch in sample:
            if ch.isprintable() or ch in "\n\r\t":
                printable += 1
            if ch.isalnum():
                meaningful += 1

        if not sample:
            return False

        printable_ratio = printable / len(sample)
        meaningful_ratio = meaningful / len(sample)
        return printable_ratio >= 0.95 and meaningful_ratio >= 0.2

    def _split_text(self, content: str) -> list[str]:
        return self.splitter.split_text(content)

    # ── Database operations ──────────────────────────────────────

    def _document_exists(
        self, conn: sqlite3.Connection, topic: str, path: str,
    ) -> bool:
        row = conn.execute(
            "SELECT 1 FROM documents WHERE topic = ? AND path = ?",
            (topic, path),
        ).fetchone()
        return row is not None

    def _insert_document(
        self,
        conn: sqlite3.Connection,
        topic: str,
        path: str,
        content: str,
    ) -> int:
        cursor = conn.execute(
            "INSERT INTO documents (path, topic, content) VALUES (?, ?, ?)",
            (path, topic, content),
        )
        return cursor.lastrowid

    def _insert_chunk_and_embedding(
        self,
        conn: sqlite3.Connection,
        document_id: int,
        content: str,
        embedding,
    ) -> None:
        from sqlite_vec import serialize_float32

        cursor = conn.execute(
            "INSERT INTO chunks (document_id, content) VALUES (?, ?)",
            (document_id, content),
        )
        chunk_id = cursor.lastrowid

        vec_blob = serialize_float32(embedding.tolist())
        conn.execute(
            "INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?, ?)",
            (chunk_id, vec_blob),
        )

    # ── Classification ───────────────────────────────────────────

    def _classify_and_store(
        self,
        conn: sqlite3.Connection,
        document_id: int,
        content: str,
    ) -> None:
        if self._classificator is None:
            return

        # Use the first 2000 chars as sample for classification
        sample = content[:2000]
        try:
            steps = self._classificator.classify(sample)
        except Exception as exc:
            logger.warning("Classification failed for doc %d: %s", document_id, exc)
            return

        for step in steps:
            conn.execute(
                "INSERT INTO document_categories "
                "(document_id, level, category, score) VALUES (?, ?, ?, ?)",
                (document_id, step["level"], step["label"], step["score"]),
            )
