from pathlib import Path
from ai.agent.agent_config import AgentConfig
from abc import ABC, abstractmethod

class Parser(ABC):

    @abstractmethod
    def parse(self, path: Path) -> AgentConfig:
        return None