from __future__ import annotations

import json
from typing import Any, Iterable

import forge.brain
from forge.agent.config import AgentConfig
from forge.events.observer import ForgeObserver


TOOL_CONTEXT_PROMPT = forge.brain.load("tool_context")


class ForgeAgent:
    def __init__(
        self,
        config: AgentConfig,
        llm: Any,
        tools: Iterable[Any],
        observers: list[ForgeObserver] | None = None,
    ) -> None:
        self.config = config
        self.llm = llm
        # Keep a name → langchain-tool mapping for dispatching tool calls.
        self.tools: dict[str, Any] = {t.name: t for t in tools}
        self._history: list[Any] = []
        self._observers: list[ForgeObserver] = list(observers or [])

    def reset(self) -> None:
        """Clear conversation history."""
        self._history = []

    def _emit(self, event: str, *args: Any, **kwargs: Any) -> None:
        for obs in self._observers:
            getattr(obs, event)(*args, **kwargs)

    def invoke(self, message: str, language: str = "") -> Any:
        from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

        self._history.append(HumanMessage(content=message))
        system = SystemMessage(content=self._build_system_prompt(language=language))
        max_steps = self.config.steps

        self._emit("on_agent_start", self.config.name, message)

        for _ in range(max_steps):
            response = self.llm.invoke([system] + self._history)
            self._history.append(response)

            tool_calls = getattr(response, "tool_calls", None)
            if not tool_calls:
                # No tool calls — final answer reached.
                self._emit("on_agent_end", self.config.name, response)
                return response

            # Execute each requested tool and feed results back.
            for call in tool_calls:
                tool_name = call["name"]
                tool_input = call.get("args", {})
                call_id = call.get("id", tool_name)

                self._emit("on_tool_call", self.config.name, tool_name, tool_input)

                tool = self.tools.get(tool_name)
                if tool is None:
                    result = f"Error: unknown tool '{tool_name}'"
                else:
                    result = tool.invoke(self._normalize_args(tool_input))

                self._emit("on_tool_result", self.config.name, tool_name, result)

                self._history.append(
                    ToolMessage(content=str(result), tool_call_id=call_id)
                )

        # Reached step limit — return whatever the last LLM response was.
        return self._history[-1] if self._history else None

    def _build_system_prompt(self, language: str = "") -> str:
        """Combine tool context (framework) + agent prompt (user) + tool list.

        Order matters: framework instructions come first so they are always
        in context, then the agent-specific persona and constraints.
        """
        parts: list[str] = []

        # 1. Framework-level tool context — instructs the model on HOW to use tools.
        #    Agent-level config takes priority; fall back to the module constant.
        if self.tools:
            tool_ctx = self.config.tool_context_prompt.strip() or TOOL_CONTEXT_PROMPT.strip()
            lang = language or self.config.language or "the language of the user's input message"
            tool_ctx = tool_ctx.replace("{language}", lang)
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
                    # Detect this by checking whether the dict only contains schema
                    # meta-keys AND the 'type' value is a real JSON Schema type keyword.
                    # If 'type' contains an actual value (e.g. "." or "src/**"), treat
                    # that value as the intended tool_input instead of discarding it.
                    _SCHEMA_TYPE_KEYS = {"type", "description", "enum", "default"}
                    _JSON_SCHEMA_TYPES = {
                        "string", "integer", "number", "boolean",
                        "object", "array", "null",
                    }
                    if set(val.keys()) <= _SCHEMA_TYPE_KEYS:
                        type_val = val.get("type")
                        if type_val in _JSON_SCHEMA_TYPES or type_val is None:
                            # Schema echo detected. The model may have put the real
                            # value in another key (e.g. {'tool_input': {type:string}, 's': 'path'}).
                            other_strings = [
                                v for k, v in args.items()
                                if k != "tool_input" and isinstance(v, str)
                            ]
                            if len(other_strings) == 1:
                                return {"tool_input": other_strings[0]}
                            return {"tool_input": ""}
                        return {"tool_input": str(type_val)}
                    # The dict has keys beyond the schema meta-keys.
                    # Common pattern: {"type": "string", "value": "actual content"}
                    # where the LLM adds a "value" or "content" key with the real input.
                    for key in ("value", "content", "text", "input", "prompt"):
                        candidate = val.get(key)
                        if isinstance(candidate, str) and candidate:
                            return {"tool_input": candidate}
                    return {"tool_input": json.dumps(val, ensure_ascii=False)}
                return {"tool_input": str(val)}
            # No 'tool_input' key. If there is exactly one string value the LLM
            # put the actual input under a non-standard key (e.g. 'f', 's', 'file').
            string_vals = [v for v in args.values() if isinstance(v, str)]
            if len(string_vals) == 1:
                return {"tool_input": string_vals[0]}
            return {"tool_input": json.dumps(args, ensure_ascii=False)}
        return {"tool_input": str(args)}
