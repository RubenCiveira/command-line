from __future__ import annotations

from pathlib import Path

_BRAIN_DIR = Path(__file__).parent


def load(name: str) -> str:
    path = _BRAIN_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Brain prompt not found: {path}")
    return path.read_text(encoding="utf-8")
