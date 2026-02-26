from __future__ import annotations

import re
from pathlib import Path

import yaml

from forge.playbook.config import ExecutionUnit, PlaybookConfig, PlaybookStep


class ForgePlaybookParser:
    FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n?", re.DOTALL)

    def parse(self, path: Path) -> PlaybookConfig:
        text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        match = self.FRONTMATTER_RE.match(text)
        if not match:
            raise ValueError(f"{path} missing YAML frontmatter")

        try:
            cfg = yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError as exc:
            raise ValueError(f"{path}: invalid YAML — {exc}") from exc

        units: list[ExecutionUnit] = []
        for entry in cfg.get("steps", []):
            if "parallel" in entry:
                units.append([self._parse_step(s) for s in entry["parallel"]])
            else:
                units.append(self._parse_step(entry))

        return PlaybookConfig(
            name=path.stem,
            description=cfg.get("description", ""),
            agents=cfg.get("agents", {}),
            units=units,
        )

    def _parse_step(self, d: dict) -> PlaybookStep:
        if "name" not in d:
            raise ValueError(f"Playbook step missing required 'name' field: {d}")
        if "agent" not in d:
            raise ValueError(f"Playbook step '{d['name']}' missing required 'agent' field")
        return PlaybookStep(
            name=d["name"],
            agent=d["agent"],
            prompt=d.get("prompt", "{input}"),
        )
