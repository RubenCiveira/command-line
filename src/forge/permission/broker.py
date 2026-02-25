from __future__ import annotations

import asyncio
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict


@dataclass
class PermissionRequest:
    request_id: str
    tool: str
    tool_input: Any
    created_at: float
    decision: str | None = None
    event: threading.Event = field(default_factory=threading.Event)

    def resolve(self, decision: str) -> None:
        self.decision = decision
        self.event.set()


class PermissionBroker:
    def __init__(self, on_request: Callable[[PermissionRequest], None] | None = None) -> None:
        self._on_request = on_request
        self._pending: Dict[str, PermissionRequest] = {}
        self._lock = threading.Lock()

    def create_request(self, tool: str, tool_input: Any) -> PermissionRequest:
        request = PermissionRequest(
            request_id=str(uuid.uuid4()),
            tool=tool,
            tool_input=tool_input,
            created_at=time.time(),
            event=threading.Event(),
        )
        with self._lock:
            self._pending[request.request_id] = request
        if self._on_request:
            self._on_request(request)
        # If the callback resolved the request synchronously, clean up immediately.
        if request.decision is not None:
            with self._lock:
                self._pending.pop(request.request_id, None)
        return request

    def resolve(self, request_id: str, decision: str) -> bool:
        with self._lock:
            request = self._pending.pop(request_id, None)
        if not request:
            return False
        request.resolve(decision)
        return True

    def wait(self, request: PermissionRequest, timeout: float | None = None) -> str | None:
        request.event.wait(timeout=timeout)
        with self._lock:
            self._pending.pop(request.request_id, None)
        return request.decision

    async def wait_async(self, request: PermissionRequest, timeout: float | None = None) -> str | None:
        if timeout is None:
            await asyncio.to_thread(request.event.wait)
        else:
            await asyncio.to_thread(request.event.wait, timeout)
        with self._lock:
            self._pending.pop(request.request_id, None)
        return request.decision
