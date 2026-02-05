from __future__ import annotations

from ai.agent.thought.thought import Thought
from ai.agent.thought.conclusion import Conclusion
from ai.agent.thought.response import Response


class LLMThought(Thought):
    def __init__(self, action: str, prompt, llm):
        super().__init__(action)
        self._llm = llm
        self._prompt = prompt

    def resolve(self, response: Response | None = None) -> Conclusion:
        content = self._llm.invoke(self._prompt).content
        return Conclusion(content)
