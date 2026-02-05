"""CLI implementation of the UserInterface."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.prompt import Prompt

from ai.ui.user_interface import UserInterface


class CliUserInterface(UserInterface):
    """Blocking CLI implementation that calls the callback immediately."""

    def __init__(self, console: Console | None = None) -> None:
        self._console = console or Console()

    def request_form(self, schema: dict[str, Any], on_complete) -> None:
        try:
            values = self._prompt_schema(schema)
            on_complete(values)
        except KeyboardInterrupt:
            self._console.print("\n[dim]Cancelado.[/dim]")
            on_complete(None)

    def request_confirmation(self, message: str, on_complete) -> None:
        try:
            answer = Prompt.ask(
                f"{message} [y/N]",
                default="",
                show_default=False,
            ).strip().lower()
            if not answer:
                on_complete(None)
                return
            on_complete(answer in {"y", "yes", "s", "si"})
        except KeyboardInterrupt:
            self._console.print("\n[dim]Cancelado.[/dim]")
            on_complete(None)

    def notify(self, message: str, on_complete) -> None:
        try:
            self._console.print(message)
            Prompt.ask("[dim]Pulsa Enter para continuar[/dim]", default="")
            on_complete(True)
        except KeyboardInterrupt:
            self._console.print("\n[dim]Cerrado.[/dim]")
            on_complete(None)

    def _prompt_schema(self, schema: dict[str, Any]) -> dict | None:
        props = schema.get("properties", {})
        if not props:
            return {}

        values: dict[str, Any] = {}
        for key, spec in props.items():
            title = spec.get("title", key)
            value = self._prompt_field(title, spec)
            if value is None:
                return None
            values[key] = value
        return values

    def _prompt_field(self, title: str, spec: dict[str, Any]) -> Any | None:
        if "oneOf" in spec:
            options = spec["oneOf"]
            self._console.print(f"\n[bold]{title}[/bold]")
            for i, opt in enumerate(options, 1):
                label = opt.get("title") or opt.get("const") or str(opt)
                self._console.print(f"  [{i}] {label}")

            choice = Prompt.ask("Selecciona una opcion", default="").strip()
            if not choice:
                return None
            try:
                idx = int(choice)
            except ValueError:
                return None
            if 1 <= idx <= len(options):
                return options[idx - 1].get("const")
            return None

        text = Prompt.ask(f"{title} (vacio para cancelar)", default="").strip()
        if not text:
            return None
        return text
