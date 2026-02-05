"""Prompt injection and jailbreak detection using Llama Prompt Guard 2."""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "protectai/deberta-v3-base-prompt-injection-v2"
MAX_TOKENS = 512


class Guardrails:
    """Detect prompt injection and jailbreak attempts.

    Uses Meta's Llama Prompt Guard 2 (86M params, mDeBERTa-based)
    as a lightweight classifier that runs before sending prompts
    to the LLM.

    The model classifies text as BENIGN or INJECTION.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        threshold: float = 0.5,
        use_heuristics: bool = True,
        heuristic_score: float = 0.75,
    ) -> None:
        self.model_name = model_name
        self.threshold = threshold
        self.use_heuristics = use_heuristics
        self.heuristic_score = heuristic_score
        self._pipeline: Any | None = None
        self._benign_labels: set[str] | None = None
        self._heuristic_patterns = self._build_heuristic_patterns()

    # ── Public API ───────────────────────────────────────────────

    def check(self, text: str) -> dict:
        """Classify a single text input.

        Returns a dict with:
            safe: bool — True if the input is considered safe
            label: str — "BENIGN" or "INJECTION"
            score: float — confidence score for the predicted label
            injection_score: float — probability of injection
        """
        clf = self._get_pipeline()
        segments = self._split_segments(text)
        worst_injection = 0.0
        worst_result = None

        for segment in segments:
            results = clf(segment)
            result = results[0] if isinstance(results, list) else results
            label = self._normalize_label(result["label"])
            score = result["score"]

            is_benign_label = label in self._benign_labels
            injection_score = score if not is_benign_label else 1.0 - score
            if is_benign_label and self.use_heuristics and self._looks_like_injection(segment):
                injection_score = max(injection_score, self.heuristic_score)
            if injection_score > worst_injection:
                worst_injection = injection_score
                worst_result = result

        if worst_result is None:
            return {
                "safe": True,
                "label": "BENIGN",
                "score": 1.0,
                "injection_score": 0.0,
            }

        label = self._normalize_label(worst_result["label"])
        score = worst_result["score"]
        injection_score = worst_injection
        safe = injection_score < self.threshold

        return {
            "safe": safe,
            "label": "BENIGN" if safe else ("INJECTION" if label in self._benign_labels else label),
            "score": score,
            "injection_score": injection_score,
        }

    def is_safe(self, text: str) -> bool:
        """Return True if the text is not detected as injection."""
        return self.check(text)["safe"]

    def check_batch(self, texts: list[str]) -> list[dict]:
        """Classify multiple texts. Returns a list of check() results."""
        return [self.check(text) for text in texts]

    # ── Pipeline loading ─────────────────────────────────────────

    def _get_pipeline(self):
        if self._pipeline is None:
            from transformers import pipeline

            logger.info("Loading guard model: %s", self.model_name)
            self._pipeline = pipeline(
                "text-classification",
                model=self.model_name,
            )
            self._benign_labels = self._build_benign_labels(self._pipeline)
        return self._pipeline

    # ── Text segmentation ────────────────────────────────────────

    def _split_segments(self, text: str) -> list[str]:
        """Split long text into segments within the model's token limit.

        Prompt Guard 2 supports 512 tokens. We split by characters
        as a conservative approximation (4 chars ~ 1 token).
        """
        max_chars = MAX_TOKENS * 4
        if len(text) <= max_chars:
            return [text]

        segments = []
        for start in range(0, len(text), max_chars):
            segments.append(text[start : start + max_chars])
        return segments

    def _normalize_label(self, label: str) -> str:
        return label.strip().upper()

    def _build_benign_labels(self, clf) -> set[str]:
        id2label = getattr(getattr(clf, "model", None), "config", None)
        labels = set()

        if id2label is not None:
            labels = {self._normalize_label(lbl) for lbl in id2label.id2label.values()}

        benign = {lbl for lbl in labels if "BENIGN" in lbl or "SAFE" in lbl}
        if benign:
            return benign

        if "LABEL_0" in labels:
            return {"LABEL_0"}

        return {"BENIGN"}

    def _build_heuristic_patterns(self) -> list[re.Pattern[str]]:
        patterns = [
            r"\b(ignore|disregard|forget)\b.*\b(instructions|rules|system|previous)\b",
            r"\b(system prompt|developer mode|jailbreak|dan)\b",
            r"\bno restrictions\b|\bwithout any restrictions\b",
            r"\b(act as|pretend|simulate)\b",
            r"\byou are now\b|\bfrom now on\b",
            r"\broleplay\b|\bimpersonate\b",
        ]
        return [re.compile(pat, re.IGNORECASE | re.DOTALL) for pat in patterns]

    def _looks_like_injection(self, text: str) -> bool:
        return any(pat.search(text) for pat in self._heuristic_patterns)
