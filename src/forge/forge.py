"""Forge — top-level entry point for the framework.

Scans one or more directories for agent/playbook definitions, builds an internal
registry, and can route a free-form message to the most appropriate handler.

Directories
-----------
working_dir  : project-specific agents and playbooks (e.g. a project's .forge/ folder).
global_dir   : shared/reusable definitions (e.g. ~/.config/forge/).
brain_dir    : core prompt files — intent.txt, tool_context.txt, etc.
               Defaults to the built-in forge/brain/ directory.

Both *working_dir* and *global_dir* are scanned recursively for ``*.md`` files.
Working-dir definitions take priority over global ones when names clash.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from forge.agent.factory import ForgeAgentFactory
from forge.config.user_config import UserConfig, UserConfigStore
from forge.events.observer import ForgeObserver, VerboseObserver
from forge.llm.llm_factory import LLMFactory
from forge.permission.console_broker import ConsolePermissionBroker
from forge.permission.file_store import FilePermissionStore
from forge.permission.manager import PermissionManager
from forge.playbook.playbook import ForgePlaybookFactory

_DEFAULT_BRAIN_DIR = Path(__file__).parent / "brain"
_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---", re.DOTALL)


class Forge:
    """Top-level entry point for the Forge framework.

    Args:
        working_dir:       Project-specific agents and playbooks.
                           Defaults to the current working directory.
        global_dir:        Shared/global agents and playbooks
                           (e.g. ``~/.config/forge``). Optional.
        brain_dir:         Directory with core prompt files.
                           Defaults to the built-in ``forge/brain/``.
        user_config_store: Provides user preferences (language, default
                           model/temperature). Uses ``FileUserConfigStore``
                           with default path when omitted.
        observers:         Event observers attached to every agent/playbook.
        verbose:           Shortcut to add a ``VerboseObserver``.
    """

    def __init__(
        self,
        working_dir: Path | str | None = None,
        global_dir: Path | str | None = None,
        brain_dir: Path | str | None = None,
        user_config_store: UserConfigStore | None = None,
        observers: list[ForgeObserver] | None = None,
        verbose: bool = False,
    ) -> None:
        self._working_dir: Path | None = Path(working_dir).resolve() if working_dir else None
        self._global_dir: Path | None = Path(global_dir).resolve() if global_dir else None
        self._brain_dir: Path = Path(brain_dir).resolve() if brain_dir else _DEFAULT_BRAIN_DIR

        self._verbose = verbose
        self._observers: list[ForgeObserver] = list(observers or [])
        if verbose:
            self._observers.append(VerboseObserver())

        user_config: UserConfig = user_config_store.load() if user_config_store else UserConfig()

        # Shared infrastructure
        permission_store = FilePermissionStore()
        broker = ConsolePermissionBroker()

        self._agent_factory = ForgeAgentFactory(
            permission_store=permission_store,
            permission_manager_factory=lambda perms: PermissionManager(perms, broker),
            user_config_store=user_config_store,
            observers=self._observers,
        )
        self._playbook_factory = ForgePlaybookFactory(
            agent_factory=self._agent_factory,
            user_config_store=user_config_store,
            observers=self._observers,
        )

        # Routing prompt from brain dir
        intent_path = self._brain_dir / "intent.txt"
        self._intent_prompt: str = (
            intent_path.read_text(encoding="utf-8") if intent_path.exists() else ""
        )

        # Registries: name → path  (working_dir wins over global_dir)
        self._agent_paths: dict[str, Path] = {}
        self._playbook_paths: dict[str, Path] = {}
        self._descriptions: dict[str, str] = {}
        self._first_model: str = ""  # populated during _scan
        self._scan()

        # Router LLM: prefer explicit user config model, fall back to first
        # model discovered while scanning agents/playbooks.
        self._router_llm: Any = None
        routing_model = user_config.model or self._first_model
        if routing_model:
            self._router_llm = LLMFactory.create(
                routing_model,
                user_config.temperature,
                {},
            )

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def agents(self) -> dict[str, Path]:
        """Discovered agent definitions: ``name → path``."""
        return dict(self._agent_paths)

    @property
    def playbooks(self) -> dict[str, Path]:
        """Discovered playbook definitions: ``name → path``."""
        return dict(self._playbook_paths)

    # ------------------------------------------------------------------
    # Invocation
    # ------------------------------------------------------------------

    def invoke(self, message: str) -> str:
        """Route *message* to the best agent or playbook and return the output.

        Falls back to the first available agent (then playbook) when routing is
        unavailable or yields no match.
        """
        name = self._route(message)
        if name is None:
            if self._agent_paths:
                name = next(iter(self._agent_paths))
            elif self._playbook_paths:
                name = next(iter(self._playbook_paths))
            else:
                return "No agents or playbooks discovered."

        if name in self._playbook_paths:
            return self.invoke_playbook(name, message)
        if name in self._agent_paths:
            return self.invoke_agent(name, message)
        return f"Unknown handler: '{name}'"

    def invoke_agent(self, name: str, message: str) -> str:
        """Directly invoke the named agent with *message*."""
        path = self._agent_paths[name]
        agent = self._agent_factory.from_markdown(path)
        result = agent.invoke(message)
        return result.content if hasattr(result, "content") else str(result)

    def invoke_playbook(self, name: str, message: str) -> str:
        """Directly invoke the named playbook with *message*."""
        path = self._playbook_paths[name]
        playbook = self._playbook_factory.from_markdown(path)
        context = playbook.invoke(message)
        last_unit = playbook.config.units[-1]
        if isinstance(last_unit, list):
            return "\n\n".join(context.get(s.name, "") for s in last_unit)
        return context.get(last_unit.name, "")

    # ------------------------------------------------------------------
    # Directory scanning
    # ------------------------------------------------------------------

    def _scan(self) -> None:
        """Discover agents and playbooks from working and global directories.

        Working-dir files take priority; global-dir fills in any missing names.
        """
        # Process working_dir first so it wins name conflicts
        dirs: list[Path] = []
        if self._working_dir and self._working_dir.exists():
            dirs.append(self._working_dir)
        if self._global_dir and self._global_dir.exists():
            dirs.append(self._global_dir)

        for directory in dirs:
            for md_path in sorted(directory.rglob("*.md")):
                result = self._classify(md_path)
                if result is None:
                    continue
                kind, description, model = result
                name = md_path.stem
                if kind == "playbook" and name not in self._playbook_paths:
                    self._playbook_paths[name] = md_path
                    self._descriptions[name] = description
                elif kind == "agent" and name not in self._agent_paths:
                    self._agent_paths[name] = md_path
                    self._descriptions[name] = description
                if model and not self._first_model:
                    self._first_model = model

    def _classify(self, path: Path) -> tuple[str, str, str] | None:
        """Return ``(kind, description, model)`` for a markdown file, or ``None``.

        *model* is the agent's configured model string, or ``""`` for playbooks
        and agents that inherit their model from user config.
        """
        try:
            text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
            match = _FRONTMATTER_RE.match(text)
            if not match:
                return None
            cfg = yaml.safe_load(match.group(1)) or {}
            if cfg.get("hidden") or cfg.get("disable"):
                return None
            desc: str = cfg.get("description", "")
            model: str = cfg.get("model", "") or ""
            if "agents" in cfg and "steps" in cfg:
                return "playbook", desc, model
            if "model" in cfg or "tool_filter" in cfg or "mode" in cfg:
                return "agent", desc, model
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _route(self, message: str) -> str | None:
        """Ask the brain router to pick the best handler name, or return None."""
        if not self._router_llm or not self._intent_prompt:
            if self._verbose:
                print("[forge] no routing model available — using fallback")
            return None
        all_options = {**self._playbook_paths, **self._agent_paths}
        if not all_options:
            return None
        options_text = "\n".join(
            f"- {name}: {self._descriptions.get(name, '')}"
            for name in all_options
        )
        prompt = self._intent_prompt.format(options=options_text, input=message)
        try:
            from langchain_core.messages import HumanMessage

            response = self._router_llm.invoke([HumanMessage(content=prompt)])
            raw = response.content.strip() if hasattr(response, "content") else str(response).strip()
            if self._verbose:
                print(f"[forge] router raw response: {raw!r}")
            chosen = self._match_option(raw, all_options)
            if self._verbose:
                print(f"[forge] router selected: {chosen!r}")
            return chosen
        except Exception as exc:
            if self._verbose:
                print(f"[forge] routing failed: {exc}")
            return None

    def _match_option(self, raw: str, options: dict[str, Any]) -> str | None:
        """Match a raw LLM response to one of the known option names.

        Tries in order:
        1. Exact match after stripping whitespace and quotes.
        2. Case-insensitive exact match.
        3. Any option name that appears as a substring of the response.
        """
        candidate = raw.strip().strip("\"'").strip()
        if candidate.lower() == "none":
            return None
        # Pass 1 — exact
        if candidate in options:
            return candidate
        # Pass 2 — case-insensitive
        candidate_lower = candidate.lower()
        for name in options:
            if name.lower() == candidate_lower:
                return name
        # Pass 3 — option name contained anywhere in the response
        raw_lower = raw.lower()
        for name in options:
            if name.lower() in raw_lower:
                return name
        return None
