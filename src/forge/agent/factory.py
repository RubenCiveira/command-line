from __future__ import annotations

import dataclasses
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Callable, Mapping

from forge.agent.agent import ForgeAgent
from forge.agent.config import AgentConfig
from forge.agent.markdown_parser import ForgeMarkdownAgentParser
from forge.config.user_config import UserConfig, UserConfigStore
from forge.events.observer import ForgeObserver, VerboseObserver
from forge.llm.llm_factory import LLMFactory
from forge.permission.broker import PermissionBroker
from forge.permission.console_broker import ConsolePermissionBroker
from forge.permission.manager import PermissionManager
from forge.permission.store import PermissionStore
from forge.permission.util import merge_permissions
from forge.tools.base import ForgeTool
from forge.tools.registry import ToolRegistry

LlmFactory = Callable[[str | None, float | None, Mapping[str, Any]], Any]


class ForgeAgentFactory:
    def __init__(
        self,
        llm_factory: LlmFactory | None = None,
        tool_registry: ToolRegistry | None = None,
        permission_store: PermissionStore | None = None,
        permission_broker: PermissionBroker | None = None,
        permission_manager_factory: Callable[[Mapping[str, Any]], PermissionManager] | None = None,
        tool_context_prompt: str | Path | None = None,
        user_config_store: UserConfigStore | None = None,
        observers: list[ForgeObserver] | None = None,
        verbose: bool = False,
    ) -> None:
        self._llm_factory = llm_factory or LLMFactory.create
        self._tool_registry = tool_registry
        self._permission_store = permission_store
        self._permission_broker = permission_broker or ConsolePermissionBroker()
        self._permission_manager_factory = permission_manager_factory or (
            lambda permissions: PermissionManager(permissions, self._permission_broker)
        )
        if isinstance(tool_context_prompt, Path):
            self._tool_context_prompt: str = tool_context_prompt.read_text(encoding="utf-8")
        else:
            self._tool_context_prompt = tool_context_prompt or ""
        user_config: UserConfig = user_config_store.load() if user_config_store else UserConfig()
        self._user_language: str = user_config.language
        self._user_model: str = user_config.model
        self._user_temperature: float | None = user_config.temperature
        self._observers: list[ForgeObserver] = list(observers or [])
        if verbose:
            self._observers.append(VerboseObserver())

    def add_observer(self, observer: ForgeObserver) -> None:
        """Register an additional observer. All subsequently created agents will use it."""
        self._observers.append(observer)

    def from_markdown(self, path: Path) -> ForgeAgent:
        parser = ForgeMarkdownAgentParser()
        config = parser.parse(path)
        return self.from_config(config)

    def from_config(self, config: AgentConfig) -> ForgeAgent:
        merged_permissions = self._merge_permissions(config.permission)
        permission_manager = self._permission_manager_factory(merged_permissions)

        # Apply factory-level fallbacks when the agent definition doesn't specify its own.
        if self._tool_context_prompt and not config.tool_context_prompt:
            config = dataclasses.replace(config, tool_context_prompt=self._tool_context_prompt)
        if self._user_language and not config.language:
            config = dataclasses.replace(config, language=self._user_language)
        if self._user_model and not config.model:
            config = dataclasses.replace(config, model=self._user_model)
        if self._user_temperature is not None and config.temperature is None:
            config = dataclasses.replace(config, temperature=self._user_temperature)

        tools = self._build_tools(config.tool_filter, permission_manager)
        llm = self._llm_factory(config.model, config.temperature, config.extras)
        # In react mode the LLM uses text-based tool calling — do NOT bind the
        # tool schemas via the API (that confuses small models even further).
        llm_with_tools = llm if config.react else (llm.bind_tools(tools) if tools else llm)
        return ForgeAgent(config=config, llm=llm_with_tools, tools=tools, observers=self._observers)

    def _merge_permissions(self, agent_permissions: Mapping[str, Any]) -> Mapping[str, Any]:
        if not self._permission_store:
            return agent_permissions
        global_permissions = self._permission_store.load()
        return merge_permissions(global_permissions, agent_permissions)

    def _build_tools(
        self,
        tool_config: Mapping[str, Any],
        permission_manager: PermissionManager,
    ) -> list[Any]:
        tool_instances = self._instantiate_tools(permission_manager)
        enabled = [tool for tool in tool_instances if self._tool_enabled(tool, tool_config)]
        return [tool.to_langchain() for tool in enabled]

    def _instantiate_tools(self, permission_manager: PermissionManager) -> list[ForgeTool]:
        registry = self._tool_registry or ToolRegistry.default()
        return registry.build_all(permission_manager)

    def _tool_enabled(self, tool: ForgeTool, tool_config: Mapping[str, Any]) -> bool:
        if not tool_config:
            return True
        # Whitelist mode: if any pattern is explicitly enabled (true), tools that
        # don't match any pattern are disabled by default.  This lets agents say
        # "only image" without having to enumerate every other tool as false.
        has_whitelist = any(bool(v) for v in tool_config.values())
        default = not has_whitelist
        decision = None
        for pattern, value in tool_config.items():
            if fnmatch(tool.name, pattern):
                decision = bool(value)
        return decision if decision is not None else default
