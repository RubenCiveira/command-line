from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from forge.permission.manager import PermissionManager


class ForgeTool(ABC):
    name: str
    description: str
    permission_key: str | None = None
    # Subclasses may override this to provide a richer description for the
    # tool_input field in the generated schema.  Small LLMs use this hint to
    # understand they must supply a plain string value, not a schema type.
    input_description: str = ""

    def __init__(self, permissions: PermissionManager | None = None) -> None:
        self._permissions = permissions or PermissionManager({})

    def to_langchain(self) -> Any:
        try:
            from langchain_core.tools import StructuredTool
        except ImportError:
            from langchain.tools import StructuredTool  # type: ignore[no-redef]
        from pydantic import BaseModel, Field  # noqa: PLC0415

        # Wrap _invoke in a plain function with an explicit `str` annotation so
        # StructuredTool.from_function builds a clean Pydantic schema.  Using
        # langchain_core.tools.Tool instead would trigger a langchain_core bug
        # where its own `list` shadows the builtin, causing `list[str]` to fail
        # schema generation when bind_tools() converts to OpenAI format.
        _invoke = self._invoke
        _field_desc = self.input_description or self.description

        # Build a Pydantic schema with a descriptive field so small LLMs
        # understand the parameter expects a plain string value.
        _Schema = type(
            "_Schema",
            (BaseModel,),
            {
                "__annotations__": {"tool_input": str},
                "tool_input": Field(default="", description=_field_desc),
            },
        )

        def _call(tool_input: str = "") -> str:
            result = _invoke(tool_input)
            return str(result) if result is not None else ""

        _call.__name__ = self.name

        return StructuredTool.from_function(
            func=_call,
            name=self.name,
            description=self.description,
            args_schema=_Schema,
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
