
from abc import ABC, abstractmethod

# from ai.agent.thought.thought import Thought

class Conclusion(ABC):
    def __init__(self, proposal: str):
        self.proposal = proposal
    
    def and_then(self):
        pass