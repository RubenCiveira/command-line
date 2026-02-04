from pathlib import Path
from typing import Dict, Any
import yaml
import re

from ai.agent.parser.parser import Parser
from ai.agent.agent_config import AgentConfig

class OpenCodeMarkdownParser(Parser):
    FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)

    def parse(self, path: Path) -> AgentConfig:
        text = path.read_text(encoding="utf-8")

        match = self.FRONTMATTER_RE.match(text)
        if not match:
            raise ValueError(f"{path} missing YAML frontmatter")

        yaml_text, body = match.groups()

        cfg = yaml.safe_load(yaml_text) or {}

        return AgentConfig(
            name=path.stem,
            description=cfg.get("description", ""),
            mode=cfg.get("mode", "all"),
            model=cfg.get("model"),
            temperature=cfg.get("temperature"),
            tools=cfg.get("tools", {}),
            permission=cfg.get("permission", {}),
            prompt=body.strip(),
        )