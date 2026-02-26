from __future__ import annotations

import dataclasses
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from forge.agent.agent import ForgeAgent
from forge.agent.factory import ForgeAgentFactory
from forge.config.user_config import UserConfig, UserConfigStore
from forge.events.observer import ForgeObserver, VerboseObserver
from forge.playbook.config import PlaybookConfig, PlaybookStep
from forge.playbook.parser import ForgePlaybookParser


class ForgePlaybook:
    def __init__(
        self,
        config: PlaybookConfig,
        agents: dict[str, ForgeAgent],
        observers: list[ForgeObserver] | None = None,
    ) -> None:
        self.config = config
        self._agents = agents
        self._observers: list[ForgeObserver] = list(observers or [])

    def _emit(self, event: str, *args: Any, **kwargs: Any) -> None:
        for obs in self._observers:
            getattr(obs, event)(*args, **kwargs)

    def invoke(self, message: str) -> dict[str, str]:
        context: dict[str, str] = {
            "input": message,
            "language": self.config.language or _detect_language(message),
        }

        self._emit("on_playbook_start", self.config.name, message)

        for unit in self.config.units:
            if isinstance(unit, list):
                self._run_parallel(unit, context)
            else:
                self._run_step(unit, context)

        self._emit("on_playbook_end", context)
        return context

    def _run_step(self, step: PlaybookStep, context: dict[str, str]) -> None:
        prompt = step.prompt.format_map(context)
        self._emit("on_step_start", step.name, step.agent, prompt)
        agent = self._agents[step.agent]
        agent.reset()
        result = agent.invoke(prompt, language=context.get("language", ""))
        output = _extract_content(result)
        context[step.name] = output
        self._emit("on_step_end", step.name, output)

    def _run_parallel(self, steps: list[PlaybookStep], context: dict[str, str]) -> None:
        self._emit("on_parallel_start", [s.name for s in steps])
        snapshot = dict(context)
        with ThreadPoolExecutor(max_workers=len(steps)) as executor:
            futures = {
                executor.submit(self._invoke_step, step, snapshot): step.name
                for step in steps
            }
            for future in as_completed(futures):
                name = futures[future]
                context[name] = future.result()
                self._emit("on_parallel_step_end", name, context[name])

    def _invoke_step(self, step: PlaybookStep, context: dict[str, str]) -> str:
        """Run a single step in a thread-safe way.

        Creates a fresh ForgeAgent that shares the same stateless ``llm`` and
        ``tools`` as the template agent but has its own empty ``_history``, so
        parallel invocations never corrupt each other's conversation state.
        """
        template = self._agents[step.agent]
        agent = ForgeAgent(
            config=template.config,
            llm=template.llm,
            tools=template.tools.values(),
            observers=self._observers,
        )
        prompt = step.prompt.format_map(context)
        result = agent.invoke(prompt, language=context.get("language", ""))
        return _extract_content(result)


class ForgePlaybookFactory:
    def __init__(
        self,
        agent_factory: ForgeAgentFactory,
        user_config_store: UserConfigStore | None = None,
        observers: list[ForgeObserver] | None = None,
        verbose: bool = False,
    ) -> None:
        self._agent_factory = agent_factory
        user_config: UserConfig = user_config_store.load() if user_config_store else UserConfig()
        self._user_language: str = user_config.language
        self._observers: list[ForgeObserver] = list(observers or [])
        if verbose:
            self._observers.append(VerboseObserver())

    def add_observer(self, observer: ForgeObserver) -> None:
        """Register an additional observer for all subsequently created playbooks."""
        self._observers.append(observer)

    def from_markdown(self, path: Path) -> ForgePlaybook:
        parser = ForgePlaybookParser()
        config = parser.parse(path)

        if self._user_language and not config.language:
            config = dataclasses.replace(config, language=self._user_language)

        agents: dict[str, ForgeAgent] = {}
        for name, agent_path_str in config.agents.items():
            agent_path = (path.parent / agent_path_str).resolve()
            agents[name] = self._agent_factory.from_markdown(agent_path)

        return ForgePlaybook(config=config, agents=agents, observers=self._observers)


def _extract_content(result: Any) -> str:
    if hasattr(result, "content"):
        return str(result.content)
    return str(result)


def _detect_language(text: str) -> str:
    _SIGNATURES: list[tuple[set[str], str]] = [
        ({"el", "la", "los", "las", "un", "una", "es", "son", "en", "de", "que", "y", "por", "para", "con"}, "Spanish"),
        ({"the", "is", "are", "in", "of", "and", "to", "a", "an", "for", "with", "this", "that"}, "English"),
        ({"le", "la", "les", "un", "une", "des", "est", "sont", "et", "en", "de", "pour", "avec"}, "French"),
        ({"der", "die", "das", "ein", "eine", "ist", "sind", "und", "in", "von", "für", "mit"}, "German"),
        ({"il", "la", "i", "gli", "le", "un", "una", "è", "sono", "e", "in", "di", "per", "con"}, "Italian"),
        ({"o", "a", "os", "as", "um", "uma", "é", "são", "e", "em", "de", "para", "com"}, "Portuguese"),
    ]
    words = set(text.lower().split())
    best_lang, best_count = "the language of the user's input message", 0
    for keywords, lang in _SIGNATURES:
        count = len(words & keywords)
        if count > best_count:
            best_count = count
            best_lang = lang
    return best_lang
