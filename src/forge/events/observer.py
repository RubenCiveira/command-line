from __future__ import annotations

import sys
from typing import Any


class ForgeObserver:
    """Base class for Forge event observers.

    Override only the methods you care about — all methods are no-ops by default,
    so subclasses only need to implement the events they are interested in.
    """

    # ── Agent lifecycle ────────────────────────────────────────────────────────

    def on_agent_start(self, agent_name: str, message: str) -> None:
        """Called when an agent begins processing a message."""

    def on_tool_call(self, agent_name: str, tool_name: str, tool_input: Any) -> None:
        """Called before a tool is invoked."""

    def on_tool_result(self, agent_name: str, tool_name: str, result: Any) -> None:
        """Called after a tool returns a result."""

    def on_agent_end(self, agent_name: str, response: Any) -> None:
        """Called when an agent produces its final response."""

    # ── Playbook lifecycle ─────────────────────────────────────────────────────

    def on_playbook_start(self, playbook_name: str, message: str) -> None:
        """Called when a playbook begins execution."""

    def on_step_start(self, step_name: str, agent_name: str, prompt: str) -> None:
        """Called before a sequential playbook step starts."""

    def on_step_end(self, step_name: str, result: str) -> None:
        """Called after a sequential playbook step completes."""

    def on_parallel_start(self, step_names: list[str]) -> None:
        """Called before a group of parallel steps starts."""

    def on_parallel_step_end(self, step_name: str, result: str) -> None:
        """Called after an individual parallel step completes."""

    def on_playbook_end(self, context: dict[str, str]) -> None:
        """Called when the playbook finishes all steps."""


class VerboseObserver(ForgeObserver):
    """Observer that prints detailed workflow events to stderr.

    Drop-in replacement for the legacy ``verbose=True`` flag on agents and playbooks.
    """

    def on_tool_call(self, agent_name: str, tool_name: str, tool_input: Any) -> None:
        print(f"[forge] tool call  → {tool_name}({tool_input})", file=sys.stderr)

    def on_tool_result(self, agent_name: str, tool_name: str, result: Any) -> None:
        preview = str(result)[:200].replace("\n", "\\n")
        print(f"[forge] tool result← {preview}", file=sys.stderr)

    def on_agent_end(self, agent_name: str, response: Any) -> None:
        print("[forge] final answer", file=sys.stderr)

    def on_step_start(self, step_name: str, agent_name: str, prompt: str) -> None:
        print(f"[playbook] step      → {step_name} (agent: {agent_name})", file=sys.stderr)

    def on_step_end(self, step_name: str, result: str) -> None:
        preview = result[:200].replace("\n", "\\n")
        print(f"[playbook] step done ← {step_name}: {preview}", file=sys.stderr)

    def on_parallel_start(self, step_names: list[str]) -> None:
        names = ", ".join(step_names)
        print(f"[playbook] parallel  → [{names}]", file=sys.stderr)

    def on_parallel_step_end(self, step_name: str, result: str) -> None:
        preview = result[:200].replace("\n", "\\n")
        print(f"[playbook] parallel done ← {step_name}: {preview}", file=sys.stderr)
