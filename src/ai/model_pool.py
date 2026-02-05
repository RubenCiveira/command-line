"""Central pool for transformer pipeline lifecycle management."""

from __future__ import annotations

import gc
import logging
import threading
from collections import OrderedDict
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MAX_MEMORY_BYTES: int = 2 * 1024**3  # 2 GiB


class ModelPool:
    """Cache and manage transformer pipelines with LRU eviction.

    Pipelines are keyed by (task, model_name). When the total estimated
    memory of cached models exceeds max_memory_bytes, the least-recently
    used entries are evicted until the budget is satisfied.
    """

    def __init__(self, max_memory_bytes: int = DEFAULT_MAX_MEMORY_BYTES) -> None:
        self._max_memory_bytes = max_memory_bytes
        self._cache: OrderedDict[tuple[str, str], _CacheEntry] = OrderedDict()
        self._total_bytes: int = 0
        self._lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────

    def get_or_load(
        self,
        task: str,
        model_name: str,
        **pipeline_kwargs: Any,
    ) -> Any:
        """Return a cached pipeline or load a new one.

        Parameters
        ----------
        task : str
            The transformers pipeline task string
            (e.g. "zero-shot-classification", "text-classification").
        model_name : str
            Hugging Face model identifier.
        **pipeline_kwargs
            Extra keyword arguments forwarded to
            ``transformers.pipeline()`` on first load only.
            These are NOT part of the cache key.
        """
        key = (task, model_name)

        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                logger.debug("Pool cache hit: %s", key)
                return self._cache[key].pipeline

        # Load outside the lock (loading is slow)
        from transformers import pipeline

        logger.info("Pool loading pipeline: task=%s model=%s", task, model_name)
        pipe = pipeline(task, model=model_name, **pipeline_kwargs)
        mem = self._estimate_memory(pipe)

        with self._lock:
            # Double-check: another thread may have loaded the same key
            if key in self._cache:
                self._cache.move_to_end(key)
                self._free_pipeline(pipe)
                return self._cache[key].pipeline

            self._evict_until_fits(mem)
            self._cache[key] = _CacheEntry(pipeline=pipe, memory_bytes=mem)
            self._total_bytes += mem
            logger.info(
                "Pool cached %s (%.1f MiB). Total: %.1f / %.1f MiB",
                key,
                mem / 1024**2,
                self._total_bytes / 1024**2,
                self._max_memory_bytes / 1024**2,
            )

        return pipe

    def unload(self, task: str, model_name: str) -> bool:
        """Explicitly evict a specific pipeline. Returns True if it existed."""
        key = (task, model_name)
        with self._lock:
            if key not in self._cache:
                return False
            self._evict_key(key)
            return True

    def clear(self) -> None:
        """Evict all cached pipelines and free memory."""
        with self._lock:
            for key in list(self._cache.keys()):
                self._evict_key(key)

    @property
    def total_memory_bytes(self) -> int:
        return self._total_bytes

    @property
    def max_memory_bytes(self) -> int:
        return self._max_memory_bytes

    @property
    def cached_keys(self) -> list[tuple[str, str]]:
        with self._lock:
            return list(self._cache.keys())

    def __len__(self) -> int:
        return len(self._cache)

    def __contains__(self, key: tuple[str, str]) -> bool:
        return key in self._cache

    # ── Memory estimation ─────────────────────────────────────────

    @staticmethod
    def _estimate_memory(pipe: Any) -> int:
        """Estimate pipeline memory via torch parameter counting."""
        model = getattr(pipe, "model", None)
        if model is None:
            logger.warning("Pipeline has no .model; using 100 MiB fallback")
            return 100 * 1024**2

        total_bytes = sum(
            p.nelement() * p.element_size() for p in model.parameters()
        )
        total_bytes += sum(
            b.nelement() * b.element_size() for b in model.buffers()
        )

        # 20% overhead for tokenizer and framework bookkeeping
        return int(total_bytes * 1.2)

    # ── Eviction ──────────────────────────────────────────────────

    def _evict_until_fits(self, incoming_bytes: int) -> None:
        """Evict LRU entries until incoming_bytes fits within budget."""
        while (
            self._cache
            and self._total_bytes + incoming_bytes > self._max_memory_bytes
        ):
            oldest_key = next(iter(self._cache))
            logger.info("Pool evicting LRU entry: %s", oldest_key)
            self._evict_key(oldest_key)

    def _evict_key(self, key: tuple[str, str]) -> None:
        entry = self._cache.pop(key)
        self._total_bytes -= entry.memory_bytes
        self._free_pipeline(entry.pipeline)
        logger.info(
            "Pool evicted %s, freed ~%.1f MiB", key, entry.memory_bytes / 1024**2,
        )

    @staticmethod
    def _free_pipeline(pipe: Any) -> None:
        import torch

        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class _CacheEntry:
    """Internal bookkeeping for a cached pipeline."""

    __slots__ = ("pipeline", "memory_bytes")

    def __init__(self, pipeline: Any, memory_bytes: int) -> None:
        self.pipeline = pipeline
        self.memory_bytes = memory_bytes
