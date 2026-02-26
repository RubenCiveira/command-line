from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict

import yaml

from forge.agent.config import AgentConfig


class ForgeMarkdownAgentParser:
    FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)
    FILE_REF_RE = re.compile(r"^\{file:(.+)\}$")

    def parse(self, path: Path) -> AgentConfig:
        text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        match = self.FRONTMATTER_RE.match(text)
        if not match:
            raise ValueError(f"{path} missing YAML frontmatter")

        yaml_text, body = match.groups()
        try:
            cfg = yaml.safe_load(yaml_text) or {}
        except yaml.YAMLError as exc:
            raise ValueError(f"{path}: invalid YAML frontmatter — {exc}") from exc
        prompt = self._resolve_prompt(cfg.get("prompt"), body, path)
        tool_context_prompt = self._resolve_value(cfg.get("tool_context_prompt"), path)

        extras = self._extract_extras(cfg)

        return AgentConfig(
            name=path.stem,
            description=cfg.get("description", ""),
            mode=cfg.get("mode", "all"),
            model=cfg.get("model"),
            temperature=cfg.get("temperature"),
            tool_filter=cfg.get("tool_filter", {}),
            permission=cfg.get("permission", {}),
            prompt=prompt,
            tool_context_prompt=tool_context_prompt,
            language=cfg.get("language", ""),
            steps=cfg.get("steps", 10),
            top_p=cfg.get("top_p"),
            hidden=bool(cfg.get("hidden", False)),
            disable=bool(cfg.get("disable", False)),
            color=cfg.get("color"),
            extras=extras,
        )

    def _resolve_prompt(self, prompt_value: Any, body: str, path: Path) -> str:
        body = (body or "").strip()
        if body:
            return body
        return self._resolve_value(prompt_value, path)

    def _resolve_value(self, value: Any, path: Path) -> str:
        """Resolve a string value that may be a literal or a {file:...} reference."""
        if not isinstance(value, str):
            return ""
        match = self.FILE_REF_RE.match(value.strip())
        if not match:
            return value
        relative = match.group(1).strip()
        base_dir = path.parent.resolve()
        target = (path.parent / relative).resolve()
        if not str(target).startswith(str(base_dir) + "/") and target != base_dir:
            raise ValueError(f"File path escapes agent directory: {target}")
        return target.read_text(encoding="utf-8")

    def _extract_extras(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        reserved = {
            "description",
            "mode",
            "model",
            "temperature",
            "tool_filter",
            "permission",
            "prompt",
            "tool_context_prompt",
            "language",
            "steps",
            "top_p",
            "hidden",
            "disable",
            "color",
        }
        return {key: value for key, value in cfg.items() if key not in reserved}
