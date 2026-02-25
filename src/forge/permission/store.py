from __future__ import annotations

from typing import Any, Mapping, Protocol


class PermissionStore(Protocol):
    def load(self) -> Mapping[str, Any]:
        ...
