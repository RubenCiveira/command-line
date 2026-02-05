"""Data transfer object for a RAG ingestion topic."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RagTopic:
    """A directory to watch for RAG ingestion."""

    name: str
    path: str
