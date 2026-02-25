from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class AgentConfig:
    name: str
    description: str
    mode: str
    model: str | None
    temperature: float | None
    tool_filter: Dict[str, Any] = field(default_factory=dict)
    permission: Dict[str, Any] = field(default_factory=dict)
    prompt: str = ""
    tool_context_prompt: str = ""
    steps: int = 10
    top_p: float | None = None
    hidden: bool = False
    disable: bool = False
    color: str | None = None
    extras: Dict[str, Any] = field(default_factory=dict)
