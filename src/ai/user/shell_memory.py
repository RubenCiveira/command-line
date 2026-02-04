from __future__ import annotations

import json
import os
import time
import hashlib
from pathlib import Path
from typing import Any



class ShellMemory:
    SESSION_TTL_SECONDS = 24 * 3600
    GLOBAL_MAX_BYTES = 10 * 1024 * 1024  # 10MB

    def __init__(self):
        self.root = Path.home() / ".config" / ".asistente" / "memory"
        self.root.mkdir(parents=True, exist_ok=True)

        self.sessions_dir = self.root / "sessions"
        self.sessions_dir.mkdir(exist_ok=True)

        # limpieza automática al iniciar
        self._cleanup_sessions()

    # ─────────────────────────────
    # Public
    # ─────────────────────────────

    def append(self, row: dict):
        """
        Añade una fila JSONL tanto a la sesión actual como al global.
        """

        # timestamp automático (muy útil para LLM)
        row = {
            "ts": time.time(),
            **row,
        }

        self._append(self._session_path(), row)
        self._append(self._global_path(), row)

        self._rotate_global()

    def tail_session(self, n=20):
        path = self._session_path()
        return self._tail_jsonl( path, n )

    def tail_global(self, n=20):
        path = self._global_path()
        return self._tail_jsonl( path, n )

    # ─────────────────────────────
    # IDs
    # ─────────────────────────────

    def _session_id(self) -> str:
        try:
            tty = os.ttyname(0)
        except OSError:
            tty = "no-tty"

        return hashlib.sha1(tty.encode()).hexdigest()[:12]

    # ─────────────────────────────
    # Paths
    # ─────────────────────────────

    def _session_path(self) -> Path:
        return self.sessions_dir / f"{self._session_id()}.jsonl"

    def _global_path(self) -> Path:
        return self.root / "global.jsonl"

    # ─────────────────────────────
    # Append JSONL
    # ─────────────────────────────

    def _append(self, path: Path, data: dict):
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    # ─────────────────────────────
    # Rotation
    # ─────────────────────────────

    def _cleanup_sessions(self):
        """
        Borra sesiones antiguas (> 24h)
        """
        now = time.time()

        for file in self.sessions_dir.glob("*.jsonl"):
            age = now - file.stat().st_mtime
            if age > self.SESSION_TTL_SECONDS:
                file.unlink(missing_ok=True)

    def _rotate_global(self):
        """
        Si el global supera tamaño máximo:
        conservar solo últimas líneas.
        """
        path = self._global_path()

        if not path.exists():
            return

        size = path.stat().st_size
        if size <= self.GLOBAL_MAX_BYTES:
            return

        # leemos desde el final conservando aprox últimas líneas
        keep_bytes = self.GLOBAL_MAX_BYTES // 2  # deja margen

        with path.open("rb") as f:
            f.seek(-keep_bytes, os.SEEK_END)
            tail = f.read()

        # asegurar que empezamos en línea completa
        first_newline = tail.find(b"\n")
        if first_newline != -1:
            tail = tail[first_newline + 1 :]

        path.write_bytes(tail)

    def _tail_jsonl(self, path: Path, n: int) -> list[dict[str, Any]]:
        if n <= 0 or not path.exists():
            return []

        # leer desde el final en bloques
        block_size = 8192
        data = bytearray()
        lines: list[bytes] = []

        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            pos = f.tell()

            while pos > 0 and len(lines) <= n:
                read_size = min(block_size, pos)
                pos -= read_size
                f.seek(pos, os.SEEK_SET)
                chunk = f.read(read_size)
                data[:0] = chunk  # prepend
                lines = data.splitlines()

        # coger las últimas n líneas y parsearlas
        tail_lines = lines[-n:]
        out: list[dict[str, Any]] = []
        for raw in tail_lines:
            raw = raw.strip()
            if not raw:
                continue
            try:
                out.append(json.loads(raw.decode("utf-8")))
            except json.JSONDecodeError:
                # línea corrupta parcial (p.ej. tras rotación) → la saltamos
                continue

        return out