"""Thought that resolves which project a user query refers to."""

from __future__ import annotations

import difflib
import logging
import re
from typing import TYPE_CHECKING, Any

from ai.agent.thought.thought import Thought
from ai.agent.thought.conclusion import Conclusion
from ai.agent.thought.response import Response
from ai.user.project_topic import ProjectTopic

if TYPE_CHECKING:
    from ai.model_pool import ModelPool
    from ai.user.user_memory import UserMemory

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "MoritzLaurer/ModernBERT-large-zeroshot-v2.0"
MIN_CONFIDENCE = 0.35
HISTORY_BOOST = 0.15
FUZZY_THRESHOLD = 0.6
AMBIGUITY_GAP = 0.15


class ProjectResolverThought(Thought):
    """Resolve which project a user query refers to.

    Uses a four-layer strategy:

      1. **Keyword / fuzzy match** on project names — handles the common
         case where the user mentions the project by name (exact, substring
         or close typo).
      2. **Zero-shot classification** with project names as candidate
         labels — semantic fallback when no keyword match is found.
      3. **Conversation history boost** — if a project was mentioned in
         recent session history (via ``UserMemory``), its score is boosted.
      4. **Doubts** — if the result is still ambiguous, the Conclusion
         carries a ``doubts`` JSON schema so the caller can build a form
         for the user.  If no user input is available (batch mode), the
         system picks the best candidate automatically.

    Usage::

        thought = ProjectResolverThought(query, projects, memory, pool)

        # Phase 1 — initial resolution (may produce doubts)
        conclusion = thought.resolve()

        if conclusion.doubts:
            # show form, collect answers …
            response = Response(answers={"project": "backend"})
            conclusion = thought.resolve(response)

        selected_project_name = conclusion.proposal
    """

    def __init__(
        self,
        query: str,
        projects: list[ProjectTopic],
        memory: UserMemory | None = None,
        model_pool: ModelPool | None = None,
        model_name: str = DEFAULT_MODEL,
        min_confidence: float = MIN_CONFIDENCE,
        history_boost: float = HISTORY_BOOST,
    ) -> None:
        super().__init__("resolving_project")
        self._query = query
        self._projects = projects
        self._memory = memory
        self._model_pool = model_pool
        self._model_name = model_name
        self._min_confidence = min_confidence
        self._history_boost = history_boost
        self._pipeline: Any | None = None

    # ── Public ────────────────────────────────────────────────────

    def resolve(self, response: Response | None = None) -> Conclusion:
        if not self._projects:
            return Conclusion(proposal="")

        # If the user already answered the doubts form → apply directly
        if response and not response.is_empty:
            return self._apply_response(response)

        # Layer 1: keyword / fuzzy match
        keyword_matches = self._keyword_match()
        if len(keyword_matches) == 1:
            return Conclusion(proposal=keyword_matches[0].name)

        # Layer 2: zero-shot classification
        candidates = keyword_matches if keyword_matches else self._projects
        scored = self._zero_shot_classify(candidates)

        # Layer 3: conversation history boost
        scored = self._apply_history_boost(scored)

        scored.sort(key=lambda e: e["score"], reverse=True)

        if not scored:
            return Conclusion(proposal="")

        top = scored[0]
        runner_up = scored[1]["score"] if len(scored) > 1 else 0.0
        gap = top["score"] - runner_up

        if top["score"] >= self._min_confidence and gap >= AMBIGUITY_GAP:
            return Conclusion(proposal=top["project"].name)

        # Layer 4: ambiguous — attach doubts for user confirmation.
        # proposal carries the best guess so batch callers can use it.
        return Conclusion(
            proposal=top["project"].name,
            doubts=self._build_doubts_schema(scored),
        )

    # ── Layer 1: Keyword / Fuzzy match ────────────────────────────

    def _keyword_match(self) -> list[ProjectTopic]:
        query_lower = self._query.lower()
        tokens = set(re.findall(r"\w+", query_lower))

        exact: list[ProjectTopic] = []
        substring: list[ProjectTopic] = []
        fuzzy: list[ProjectTopic] = []

        for project in self._projects:
            name_lower = project.name.lower()

            # Exact token match: "shell" in {"haz", "commit", "de", "shell"}
            if name_lower in tokens:
                exact.append(project)
                continue

            # Substring: project name appears somewhere in the query
            if name_lower in query_lower:
                substring.append(project)
                continue

            # Fuzzy: best SequenceMatcher ratio against any query token
            best_ratio = max(
                (
                    difflib.SequenceMatcher(None, name_lower, tok).ratio()
                    for tok in tokens
                ),
                default=0.0,
            )
            if best_ratio >= FUZZY_THRESHOLD:
                fuzzy.append(project)

        # Prefer exact > substring > fuzzy
        if exact:
            return exact
        if substring:
            return substring
        return fuzzy

    # ── Layer 2: Zero-shot classification ─────────────────────────

    def _zero_shot_classify(
        self, candidates: list[ProjectTopic],
    ) -> list[dict]:
        labels = [p.name for p in candidates]

        if len(labels) < 2:
            return (
                [{"project": candidates[0], "score": 1.0}]
                if candidates
                else []
            )

        clf = self._get_pipeline()
        result = clf(self._query, candidate_labels=labels)

        project_by_name = {p.name: p for p in candidates}
        return [
            {"project": project_by_name[label], "score": score}
            for label, score in zip(result["labels"], result["scores"])
        ]

    def _get_pipeline(self):
        if self._pipeline is None:
            if self._model_pool is not None:
                self._pipeline = self._model_pool.get_or_load(
                    "zero-shot-classification", self._model_name,
                )
            else:
                from transformers import pipeline

                logger.info("Loading zero-shot model: %s", self._model_name)
                self._pipeline = pipeline(
                    "zero-shot-classification", model=self._model_name,
                )
        return self._pipeline

    # ── Layer 3: History boost ────────────────────────────────────

    def _apply_history_boost(self, scored: list[dict]) -> list[dict]:
        if not self._memory:
            return scored

        recent = self._extract_recent_projects()
        if not recent:
            return scored

        for entry in scored:
            if entry["project"].name in recent:
                entry["score"] = min(1.0, entry["score"] + self._history_boost)

        return scored

    def _extract_recent_projects(self) -> set[str]:
        """Return project names mentioned in recent session history."""
        history = self._memory.tail_session(10)
        project_names_lower = {p.name.lower(): p.name for p in self._projects}
        mentioned: set[str] = set()

        for row in history:
            # Inspect both command and result fields
            text = " ".join(
                str(row.get(k, "")) for k in ("cmd", "result", "project")
            )
            text_lower = text.lower()
            for name_lower, name in project_names_lower.items():
                if name_lower in text_lower:
                    mentioned.add(name)

        return mentioned

    # ── Layer 4: Doubts schema ────────────────────────────────────

    def _build_doubts_schema(self, scored: list[dict]) -> dict:
        """Build a JSON Schema for user confirmation of the project."""
        options = []
        for entry in scored[:5]:
            project = entry["project"]
            score = entry["score"]
            desc = project.description or project.path
            options.append({
                "const": project.name,
                "title": f"{project.name} \u2014 {desc} ({score:.2f})",
            })

        return {
            "type": "object",
            "properties": {
                "project": {
                    "type": "string",
                    "title": "\u00bfA qu\u00e9 proyecto te refieres?",
                    "oneOf": options,
                },
            },
            "required": ["project"],
        }

    # ── Response handling ─────────────────────────────────────────

    def _apply_response(self, response: Response) -> Conclusion:
        """Select the project named in the user's response."""
        chosen = response.answers.get("project", "")
        for project in self._projects:
            if project.name == chosen:
                return Conclusion(proposal=project.name)

        # Fallback: user typed something we don't recognise
        return Conclusion(proposal=chosen)
