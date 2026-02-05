from __future__ import annotations

from abc import ABC


class Conclusion(ABC):
    def __init__(self, proposal: str, doubts: dict | None = None):
        self.proposal = proposal
        self.doubts = doubts

    def and_then(self):
        pass
