from pathlib import Path

from ai.agent.agent import Agent
from ai.user.user_config import UserConfig
from ai.agent.agent_config import AgentConfig
from ai.agent.parser.parser import Parser
from ai.agent.parser.open_code_markdown_parser import OpenCodeMarkdownParser

from ai.llm.llm_factory import LLMFactory

class AgentFactory:
    _parsers = {
        ".md": OpenCodeMarkdownParser(),
    }
    root = Path.home() / ".config" / "asistente" / "agents"

    def __init__(self, user_config: UserConfig):
        self.user_config = user_config

    def load(self, name: str) -> Agent | None:
        path = self._find_agent_file(name)
        parser = self._find_parser(path)
        config = parser.parse(path)
        return self.from_config( config )

    def _find_parser(self, path: Path) -> Parser:
        parser = self._parsers.get(path.suffix)
        if not parser:
            raise ValueError(f"No parser for extension {path.suffix}")
        return parser

    def _find_agent_file(self, name: str) -> Path:
        for ext in (".md", ".json", ".yaml"):
            candidate = self.root / f"{name}{ext}"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Agent '{name}' not found in {self.root}")

    def from_config(self, config: AgentConfig) -> "Agent":
        llm = LLMFactory.create(config.model if config.model else self.user_config.currentAgentModel(), config.temperature)

        # tools = ToolRegistry.build_tools(config.tools)

        # chain = initialize_agent(
        #     tools,
        #     llm,
        #     agent=AgentType.OPENAI_FUNCTIONS,
        #     verbose=True,
        # )

        return Agent(
            config=config,
            llm=llm,
            # tools=tools,
            # chain=chain,
        )
