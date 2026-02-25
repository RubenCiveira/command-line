from __future__ import annotations

import json
from fnmatch import fnmatch
from typing import Any, Mapping

from forge.permission.broker import PermissionBroker


class PermissionManager:
    def __init__(
        self,
        permissions: Mapping[str, Any] | None = None,
        broker: PermissionBroker | None = None,
    ) -> None:
        self._permissions = permissions or {}
        self._broker = broker

    def mode_for(self, tool_name: str, tool_input: Any) -> str:
        spec = self._permissions.get(tool_name)
        if spec is None:
            return "allow"
        if isinstance(spec, str):
            return spec
        if isinstance(spec, Mapping):
            return self._match_pattern(spec, tool_input)
        return "allow"

    def request(self, tool_name: str, tool_input: Any) -> bool | None:
        """Evaluate permissions and ask the broker when mode is 'ask'.

        Returns True (allow), False (deny), or None (no broker to confirm).
        Can be called directly; also used internally by ForgeTool._invoke().
        """
        mode = self.mode_for(tool_name, tool_input)
        return self._ask_broker(tool_name, tool_input, mode)

    async def request_async(self, tool_name: str, tool_input: Any) -> bool | None:
        """Async variant of request()."""
        mode = self.mode_for(tool_name, tool_input)
        if mode == "allow":
            return True
        if mode == "deny":
            return False
        if mode != "ask":
            return True
        if not self._broker:
            return None
        req = self._broker.create_request(tool_name, tool_input)
        decision = await self._broker.wait_async(req)
        if decision is None:
            return None
        return decision == "allow"

    def _ask_broker(self, tool_name: str, tool_input: Any, mode: str) -> bool | None:
        """Dispatch to broker for 'ask' mode; resolve allow/deny directly otherwise."""
        if mode == "allow":
            return True
        if mode == "deny":
            return False
        if mode != "ask":
            return True
        if not self._broker:
            return None
        req = self._broker.create_request(tool_name, tool_input)
        decision = self._broker.wait(req)
        if decision is None:
            return None
        return decision == "allow"

    def _match_pattern(self, rules: Mapping[str, Any], tool_input: Any) -> str:
        # Extract a canonical string for pattern matching.
        # For dict inputs, prefer well-known keys; fall back to stable JSON representation.
        if tool_input is None:
            text = ""
        elif isinstance(tool_input, dict):
            text = str(
                tool_input.get("path")
                or tool_input.get("command")
                or tool_input.get("cmd")
                or json.dumps(tool_input, sort_keys=True, ensure_ascii=False)
            )
        else:
            text = str(tool_input)

        # First matching pattern wins (deterministic, easy to reason about).
        for pattern, value in rules.items():
            if fnmatch(text, pattern):
                return value if isinstance(value, str) else "allow"
        return "allow"
