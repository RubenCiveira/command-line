from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from forge.agent.agent import ForgeAgent
from forge.agent.factory import ForgeAgentFactory
from forge.playbook.config import PlaybookConfig, PlaybookStep
from forge.playbook.parser import ForgePlaybookParser


class ForgePlaybook:
    def __init__(
        self,
        config: PlaybookConfig,
        agents: dict[str, ForgeAgent],
        verbose: bool = False,
    ) -> None:
        self.config = config
        self._agents = agents
        self.verbose = verbose

    def invoke(self, message: str) -> dict[str, str]:
        context: dict[str, str] = {"input": message}

        for unit in self.config.units:
            if isinstance(unit, list):
                self._run_parallel(unit, context)
            else:
                self._run_step(unit, context)

        return context

    def _run_step(self, step: PlaybookStep, context: dict[str, str]) -> None:
        if self.verbose:
            print(f"[playbook] step      → {step.name} (agent: {step.agent})", file=sys.stderr)
        prompt = step.prompt.format_map(context)
        agent = self._agents[step.agent]
        agent.reset()
        result = agent.invoke(prompt)
        output = _extract_content(result)
        context[step.name] = output
        if self.verbose:
            preview = output[:200].replace("\n", "\\n")
            print(f"[playbook] step done ← {step.name}: {preview}", file=sys.stderr)

    def _run_parallel(self, steps: list[PlaybookStep], context: dict[str, str]) -> None:
        if self.verbose:
            names = ", ".join(s.name for s in steps)
            print(f"[playbook] parallel  → [{names}]", file=sys.stderr)
        snapshot = dict(context)
        with ThreadPoolExecutor(max_workers=len(steps)) as executor:
            futures = {
                executor.submit(self._invoke_step, step, snapshot): step.name
                for step in steps
            }
            for future in as_completed(futures):
                name = futures[future]
                context[name] = future.result()
                if self.verbose:
                    preview = context[name][:200].replace("\n", "\\n")
                    print(f"[playbook] parallel done ← {name}: {preview}", file=sys.stderr)

    def _invoke_step(self, step: PlaybookStep, context: dict[str, str]) -> str:
        prompt = step.prompt.format_map(context)
        agent = self._agents[step.agent]
        agent.reset()
        result = agent.invoke(prompt)
        return _extract_content(result)


class ForgePlaybookFactory:
    def __init__(
        self,
        agent_factory: ForgeAgentFactory,
        verbose: bool = False,
    ) -> None:
        self._agent_factory = agent_factory
        self._verbose = verbose

    def from_markdown(self, path: Path) -> ForgePlaybook:
        parser = ForgePlaybookParser()
        config = parser.parse(path)

        agents: dict[str, ForgeAgent] = {}
        for name, agent_path_str in config.agents.items():
            agent_path = (path.parent / agent_path_str).resolve()
            agents[name] = self._agent_factory.from_markdown(agent_path)

        return ForgePlaybook(config=config, agents=agents, verbose=self._verbose)


def _extract_content(result: Any) -> str:
    if hasattr(result, "content"):
        return str(result.content)
    return str(result)
