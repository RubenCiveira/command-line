from __future__ import annotations

import json
import sys
from typing import Any, Iterable

from forge.agent.config import AgentConfig


TOOL_CONTEXT_PROMPT = """\
You are an AI agent with access to tools. Follow these rules strictly:

1. ALWAYS call a tool when you need information you don't already have.
2. NEVER guess, invent, or assume facts — use tools to retrieve them.
3. After receiving a tool result, reason about it before calling the next tool or answering.
4. When you have enough information, give a direct and concise answer.
5. If a tool returns an error, try a different approach or explain the limitation.
"""


class ForgeAgent:
    def __init__(
        self,
        config: AgentConfig,
        llm: Any,
        tools: Iterable[Any],
        verbose: bool = False,
    ) -> None:
        self.config = config
        self.llm = llm
        self.verbose = verbose
        # Keep a name → langchain-tool mapping for dispatching tool calls.
        self.tools: dict[str, Any] = {t.name: t for t in tools}
        self._history: list[Any] = []

    def reset(self) -> None:
        """Clear conversation history."""
        self._history = []

    def invoke(self, message: str) -> Any:
        from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

        self._history.append(HumanMessage(content=message))
        system = SystemMessage(content=self._build_system_prompt())
        max_steps = self.config.steps

        for _ in range(max_steps):
            response = self.llm.invoke([system] + self._history)
            self._history.append(response)

            tool_calls = getattr(response, "tool_calls", None)
            if not tool_calls:
                # No tool calls — final answer reached.
                if self.verbose:
                    print(f"[forge] final answer", file=sys.stderr)
                return response

            # Execute each requested tool and feed results back.
            for call in tool_calls:
                tool_name = call["name"]
                tool_input = call.get("args", {})
                call_id = call.get("id", tool_name)

                if self.verbose:
                    print(f"[forge] tool call  → {tool_name}({tool_input})", file=sys.stderr)

                tool = self.tools.get(tool_name)
                if tool is None:
                    result = f"Error: unknown tool '{tool_name}'"
                else:
                    result = tool.invoke(self._normalize_args(tool_input))

                if self.verbose:
                    preview = str(result)[:200].replace("\n", "\\n")
                    print(f"[forge] tool result← {preview}", file=sys.stderr)

                self._history.append(
                    ToolMessage(content=str(result), tool_call_id=call_id)
                )

        # Reached step limit — return whatever the last LLM response was.
        return self._history[-1] if self._history else None

    def _build_system_prompt(self) -> str:
        """Combine tool context (framework) + agent prompt (user) + tool list.

        Order matters: framework instructions come first so they are always
        in context, then the agent-specific persona and constraints.
        """
        parts: list[str] = []

        # 1. Framework-level tool context — instructs the model on HOW to use tools.
        #    Agent-level config takes priority; fall back to the module constant.
        if self.tools:
            tool_ctx = self.config.tool_context_prompt.strip() or TOOL_CONTEXT_PROMPT.strip()
            parts.append(tool_ctx)

        # 2. Agent-specific prompt — role, domain, personality, constraints.
        if self.config.prompt.strip():
            parts.append(self.config.prompt.strip())

        # 3. Available tools — name + description so the model knows WHAT exists.
        if self.tools:
            tool_lines = "\n".join(
                f"- {name}: {tool.description}" for name, tool in self.tools.items()
            )
            parts.append(f"Available tools:\n{tool_lines}")

        return "\n\n".join(parts)

    @staticmethod
    def _normalize_args(args: Any) -> dict[str, str]:
        """Convert LLM tool args to {'tool_input': str} expected by ForgeTool schema.

        The LLM may produce:
        - A dict with a single 'tool_input' key  → use its value as-is (str-cast).
        - Any other dict                          → JSON-serialize the whole dict.
        - A plain string                          → wrap in the expected key.
        """
        if isinstance(args, str):
            return {"tool_input": args}
        if isinstance(args, dict):
            if "tool_input" in args:
                val = args["tool_input"]
                if isinstance(val, dict):
                    # Small models sometimes echo back the JSON Schema type definition
                    # (e.g. {"type": "string"}) instead of the actual value.
                    # If the dict looks like a schema node, discard it; otherwise
                    # JSON-serialize it so structured tools can parse it back.
                    _SCHEMA_TYPE_KEYS = {"type", "description", "enum", "default"}
                    if set(val.keys()) <= _SCHEMA_TYPE_KEYS:
                        return {"tool_input": ""}
                    return {"tool_input": json.dumps(val, ensure_ascii=False)}
                return {"tool_input": str(val)}
            return {"tool_input": json.dumps(args, ensure_ascii=False)}
        return {"tool_input": str(args)}
