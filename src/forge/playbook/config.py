from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union


@dataclass
class PlaybookStep:
    name: str
    agent: str
    prompt: str = "{input}"


ExecutionUnit = Union[PlaybookStep, list[PlaybookStep]]


@dataclass
class PlaybookConfig:
    name: str
    description: str
    agents: dict[str, str]
    units: list[ExecutionUnit]
