from __future__ import annotations

from abc import ABC, abstractmethod

from ai.agent.thought.conclusion import Conclusion
from ai.agent.thought.response import Response


class Thought(ABC):
    def __init__(self, action: str):
        self.action = action

    @abstractmethod
    def resolve(self, response: Response | None = None) -> Conclusion:
        pass
