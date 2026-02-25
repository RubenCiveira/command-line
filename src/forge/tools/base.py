from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from forge.permission.manager import PermissionManager


class ForgeTool(ABC):
    name: str
    description: str
    permission_key: str | None = None

    def __init__(self, permissions: PermissionManager | None = None) -> None:
        self._permissions = permissions or PermissionManager({})

    def to_langchain(self) -> Any:
        try:
            from langchain_core.tools import StructuredTool
        except ImportError:
            from langchain.tools import StructuredTool  # type: ignore[no-redef]

        # Wrap _invoke in a plain function with an explicit `str` annotation so
        # StructuredTool.from_function builds a clean Pydantic schema.  Using
        # langchain_core.tools.Tool instead would trigger a langchain_core bug
        # where its own `list` shadows the builtin, causing `list[str]` to fail
        # schema generation when bind_tools() converts to OpenAI format.
        _invoke = self._invoke

        def _call(tool_input: str = "") -> str:
            result = _invoke(tool_input)
            return str(result) if result is not None else ""

        _call.__name__ = self.name

        return StructuredTool.from_function(
            func=_call,
            name=self.name,
            description=self.description,
        )

    def _invoke(self, tool_input: Any) -> Any:
        key = self.permission_key or self.name
        mode = self._permissions.mode_for(key, tool_input)
        if mode == "deny":
            return f"Tool '{self.name}' denied by configuration."
        if mode == "ask":
            # Pass the already-computed mode to avoid recalculating it inside the manager.
            allowed = self._permissions._ask_broker(key, tool_input, mode)
            if allowed is None:
                return f"Tool '{self.name}' requires confirmation."
            if not allowed:
                return f"Tool '{self.name}' denied by user."
        return self.run(tool_input)

    @abstractmethod
    def run(self, tool_input: Any) -> Any:
        raise NotImplementedError
