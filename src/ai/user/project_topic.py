"""Data transfer object for a project workspace."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProjectTopic:
    """A project directory the assistant can work with."""

    name: str
    path: str
    description: str
