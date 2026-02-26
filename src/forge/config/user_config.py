from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol


@dataclass
class UserConfig:
    language: str = ""


class UserConfigStore(Protocol):
    def load(self) -> UserConfig: ...
    def save(self, config: UserConfig) -> None: ...


class FileUserConfigStore:
    def __init__(self, path: Path | None = None) -> None:
        self._path = path or (Path.home() / ".config" / "opencode" / "user.json")

    def load(self) -> UserConfig:
        if not self._path.exists():
            return UserConfig()
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return UserConfig(language=data.get("language", ""))
        except (OSError, json.JSONDecodeError):
            return UserConfig()

    def save(self, config: UserConfig) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(asdict(config), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
