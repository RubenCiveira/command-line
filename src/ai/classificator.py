"""Hierarchical zero-shot text classification using IPTC categories."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "MoritzLaurer/ModernBERT-large-zeroshot-v2.0"
# "facebook/bart-large-mnli"
MIN_CONFIDENCE = 0.3


class Classificator:
    """Classify text into a hierarchical category tree via zero-shot.

    Loads an IPTC-style category tree from a JSON file and walks it
    level by level, picking the best-scoring label at each level
    until confidence drops below the threshold or a leaf is reached.
    """

    def __init__(
        self,
        categories_path: Path,
        model_name: str = DEFAULT_MODEL,
        min_confidence: float = MIN_CONFIDENCE,
    ) -> None:
        self.categories_path = categories_path
        self.model_name = model_name
        self.min_confidence = min_confidence
        self._pipeline: Any | None = None
        self._tree: dict | None = None

    # ── Public API ───────────────────────────────────────────────

    def classify(self, text: str) -> list[dict]:
        """Classify text through the category tree.

        Returns a list of dicts, one per level traversed:
            {"level": int, "label": str, "score": float}

        The list represents the path from root to the deepest
        category that exceeded min_confidence.
        """
        tree = self._get_tree()
        clf = self._get_pipeline()
        return self._classify_hierarchical(clf, text, tree)

    def classify_path(self, text: str) -> str:
        """Convenience: return the classification as a ' > ' joined path.

        Example: "sport > football > transfers"
        Returns empty string if classification fails.
        """
        steps = self.classify(text)
        if not steps:
            return ""
        return " > ".join(step["label"] for step in steps)

    # ── Tree loading ─────────────────────────────────────────────

    def _get_tree(self) -> dict:
        if self._tree is None:
            if not self.categories_path.exists():
                raise FileNotFoundError(
                    f"Categories file not found: {self.categories_path}. "
                    "Run build_iptc_tree.py and copy the output to this path."
                )
            with open(self.categories_path, encoding="utf-8") as f:
                self._tree = json.load(f)
        return self._tree

    # ── Pipeline loading ─────────────────────────────────────────

    def _get_pipeline(self):
        if self._pipeline is None:
            from transformers import pipeline

            logger.info("Loading zero-shot model: %s", self.model_name)
            self._pipeline = pipeline(
                "zero-shot-classification", model=self.model_name,
            )
        return self._pipeline

    # ── Hierarchical classification ──────────────────────────────

    def _classify_hierarchical(
        self, clf, text: str, tree: dict,
    ) -> list[dict]:
        path: list[dict] = []
        current = tree
        level = 0

        while current is not None:
            labels = list(current.keys())

            if len(labels) < 2:
                if labels:
                    only_label = labels[0]
                    path.append({
                        "level": level,
                        "label": only_label,
                        "score": 1.0,
                    })
                    current = current[only_label]
                    level += 1
                else:
                    break
                continue

            result = clf(text, candidate_labels=labels)
            best_label = result["labels"][0]
            best_score = result["scores"][0]

            path.append({
                "level": level,
                "label": best_label,
                "score": best_score,
            })

            if best_score < self.min_confidence:
                break

            subtree = current.get(best_label)
            if subtree is None:
                break

            current = subtree
            level += 1

        return path
