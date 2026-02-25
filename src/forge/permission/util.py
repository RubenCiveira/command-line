from __future__ import annotations

from typing import Any, Dict, Mapping


def merge_permissions(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
) -> Mapping[str, Any]:
    result: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            merged = dict(result[key])
            merged.update(value)
            result[key] = merged
        else:
            result[key] = value
    return result
