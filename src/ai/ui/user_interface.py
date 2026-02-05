"""User interface abstraction for async-friendly interactions."""

from __future__ import annotations

from typing import Any, Callable, Protocol


FormCallback = Callable[[dict | None], None]
ConfirmCallback = Callable[[bool | None], None]
NotifyCallback = Callable[[bool | None], None]


class UserInterface(Protocol):
    """Non-blocking UI interface for async or CLI sessions.

    All methods must return immediately and use the callback to deliver
    results when the user completes the interaction.
    """

    def request_form(self, schema: dict[str, Any], on_complete: FormCallback) -> None:
        """Request a JSON-schema form from the user.

        Calls on_complete(values) on accept, or on_complete(None) on cancel.
        """

    def request_confirmation(
        self, message: str, on_complete: ConfirmCallback,
    ) -> None:
        """Ask the user to confirm an action.

        Calls on_complete(True) on accept, on_complete(False or None) on cancel.
        """

    def notify(self, message: str, on_complete: NotifyCallback) -> None:
        """Notify the user about an event or exception.

        Calls on_complete(True) when acknowledged, on_complete(False or None)
        if dismissed or closed without acknowledgement.
        """
