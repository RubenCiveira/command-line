from __future__ import annotations

from typing import Any, Callable

from forge.permission.broker import PermissionBroker, PermissionRequest


class ConsolePermissionBroker(PermissionBroker):
    def __init__(self, prompt: Callable[[str], str] | None = None) -> None:
        self._prompt = prompt or (lambda text: input(text))
        super().__init__(on_request=self._on_request)

    def _on_request(self, request: PermissionRequest) -> None:
        tool_input = "" if request.tool_input is None else str(request.tool_input)
        message = (
            f"Permission required for tool '{request.tool}'.\n"
            f"Input: {tool_input}\n"
            "Allow? [y/N]: "
        )
        response = self._prompt(message).strip().lower()
        decision = "allow" if response in {"y", "yes"} else "deny"
        request.resolve(decision)
