from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class AgentConfig:
    name: str
    description: str
    mode: str
    model: str | None
    temperature: float | None
    tools: Dict[str, Any]
    permission: Dict[str, Any]
    prompt: str