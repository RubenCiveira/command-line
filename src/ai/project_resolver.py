"""Resolve which project a user message refers to."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ai.agent.thought.conclusion import Conclusion
from ai.agent.thought.internal.project_resolver_thought import (
    ProjectResolverThought,
)
from ai.user.user_config import UserConfig

if TYPE_CHECKING:
    from ai.model_pool import ModelPool
    from ai.user.user_memory import UserMemory


class ProjectResolver:
    """Facade that maps a user message to a project.

    Wraps :class:`ProjectResolverThought` with the project list from
    ``UserConfig`` and session history from ``UserMemory``.

    For now, user confirmation is never requested (empty response) — the
    resolver always auto-selects the best candidate.  The ``doubts``
    field in the returned :class:`Conclusion` signals whether the choice
    was confident or ambiguous.
    """

    def __init__(
        self,
        config: UserConfig,
        memory: UserMemory | None = None,
        model_pool: ModelPool | None = None,
    ) -> None:
        self._config = config
        self._memory = memory
        self._model_pool = model_pool

    def resolve(self, message: str) -> Conclusion:
        """Determine which project *message* refers to.

        Returns a :class:`Conclusion` whose ``proposal`` is the project
        name (best guess) and whose ``doubts`` is a JSON schema when the
        match is ambiguous.
        """
        projects = self._config.projectTopics()

        thought = ProjectResolverThought(
            query=message,
            projects=projects,
            memory=self._memory,
            model_pool=self._model_pool,
        )

        return thought.resolve()
