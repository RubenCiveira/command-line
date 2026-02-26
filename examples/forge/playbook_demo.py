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
from forge.playbook.playbook import ForgePlaybookFactory


def main() -> None:
    parser = argparse.ArgumentParser(description="Forge playbook demo")
    parser.add_argument(
        "playbook",
        type=Path,
        nargs="?",
        default=Path(__file__).parent / "playbook.md",
        help="Path to the playbook markdown file",
    )
    parser.add_argument(
        "message",
        nargs="?",
        default="Analyze this project",
        help="Message to pass as {input} to the first step",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    broker = ConsolePermissionBroker()
    permission_store = FilePermissionStore()

    agent_factory = ForgeAgentFactory(
        permission_store=permission_store,
        permission_manager_factory=lambda permissions: PermissionManager(
            permissions, broker
        ),
        verbose=args.verbose,
    )

    playbook_factory = ForgePlaybookFactory(
        agent_factory=agent_factory,
        verbose=args.verbose,
    )

    playbook = playbook_factory.from_markdown(args.playbook)
    context = playbook.invoke(args.message)

    last_step = playbook.config.units[-1]
    if isinstance(last_step, list):
        for step in last_step:
            print(f"=== {step.name} ===")
            print(context.get(step.name, ""))
    else:
        print(context.get(last_step.name, ""))


if __name__ == "__main__":
    main()
