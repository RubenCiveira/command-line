"""Intent classification using a fine-tuned text-classification model."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ai.model_pool import ModelPool

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Falconsai/intent_classification"


class Intention:
    """Detect the intention behind a user message.

    Uses a fine-tuned text-classification model from Hugging Face Hub
    to classify messages into intent categories with confidence scores.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_labels: int = 5,
        model_pool: ModelPool | None = None,
    ) -> None:
        self.model_name = model_name
        self.max_labels = max_labels
        self._model_pool = model_pool
        self._pipeline: Any | None = None

    # ── Public API ───────────────────────────────────────────────

    def classify(self, text: str) -> list[dict]:
        """Classify text into intent categories.

        Returns a list of dicts sorted by score descending:
            {"label": str, "score": float}
        Limited to max_labels entries.
        """
        clf = self._get_pipeline()
        results = clf(text)
        if results and isinstance(results[0], list):
            results = results[0]
        sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
        return sorted_results[: self.max_labels]

    def top_intent(self, text: str) -> dict:
        """Convenience: return only the top-scoring intent.

        Returns a dict with:
            label: str — the predicted intent label
            score: float — confidence score
        """
        results = self.classify(text)
        if not results:
            return {"label": "", "score": 0.0}
        return results[0]

    # ── Pipeline loading ─────────────────────────────────────────

    def _get_pipeline(self):
        if self._pipeline is None:
            if self._model_pool is not None:
                self._pipeline = self._model_pool.get_or_load(
                    "text-classification", self.model_name,
                    top_k=None,
                )
            else:
                from transformers import pipeline

                logger.info("Loading intent model: %s", self.model_name)
                self._pipeline = pipeline(
                    "text-classification",
                    model=self.model_name,
                    top_k=None,
                )
        return self._pipeline
