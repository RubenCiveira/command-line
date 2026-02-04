
from typing import Any

from ai.agent.agent_config import AgentConfig
from ai.agent.thought.thought import Thought
from ai.agent.thought.llm_thought import LLMThought

from langchain_core.messages import SystemMessage, HumanMessage

class Agent:
    llm: Any
    config: AgentConfig

    def __init__(self, config: AgentConfig, llm: Any):
        self.llm = llm
        self.config = config

    def invoke(self, message: str) -> Thought:
        prompt = [
            SystemMessage(content=self.config.prompt),
            HumanMessage(content=message),
        ]
        return LLMThought("thinking", prompt, self.llm)
