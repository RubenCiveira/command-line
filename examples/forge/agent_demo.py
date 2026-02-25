from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forge.agent.factory import ForgeAgentFactory
from forge.permission.console_broker import ConsolePermissionBroker
from forge.permission.file_store import FilePermissionStore
from forge.permission.manager import PermissionManager


def main() -> None:
    parser = argparse.ArgumentParser(description="Forge agent demo")
    parser.add_argument(
        "--tool-context",
        type=Path,
        default=None,
        help="Path to a file with the tool-context system prompt "
             "(falls back to the built-in prompt when omitted)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print tool calls and results to stderr",
    )
    args = parser.parse_args()

    broker = ConsolePermissionBroker()
    permission_store = FilePermissionStore()

    factory = ForgeAgentFactory(
        permission_store=permission_store,
        permission_manager_factory=lambda permissions: PermissionManager(
            permissions, broker
        ),
        tool_context_prompt=args.tool_context,
        verbose=args.verbose,
    )

    agent_path = Path(__file__).parent / "agent.md"
    agent = factory.from_markdown(agent_path)

    response = agent.invoke("Lista los archivos del directorio actual y dime qué tipo de proyecto es.")
    print(response)


if __name__ == "__main__":
    main()
