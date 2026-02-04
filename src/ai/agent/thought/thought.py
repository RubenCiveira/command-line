
from abc import ABC, abstractmethod

from ai.agent.thought.conclusion import Conclusion

class Thought(ABC):
    def __init__(self, action: str):
        self.action = action

    @abstractmethod
    def resolve(self) -> Conclusion:
        pass

