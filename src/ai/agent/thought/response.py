"""Carries user answers to a Conclusion's doubts."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Response:
    """Answers provided by the user for a Conclusion's doubts.

    When a Thought produces a Conclusion with a ``doubts`` JSON schema,
    the caller can collect answers and wrap them in a Response.
    An empty Response signals that no user input was provided
    (e.g. batch mode or the user declined to answer).
    """

    answers: dict = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not self.answers
