from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


class FilePermissionStore:
    def __init__(self, path: Path | None = None) -> None:
        self._path = path or (Path.home() / ".config" / "opencode" / "permissions.json")

    def load(self) -> Mapping[str, Any]:
        if not self._path.exists():
            return {}
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
