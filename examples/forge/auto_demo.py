"""Autonomous Forge demo.

Forge scans the example directory for agents and playbooks, then routes the
user's message to the best handler automatically.

Usage:
    # Auto-route a message
    .venv/bin/python examples/forge/auto_demo.py "analyze this project"

    # List all discovered agents and playbooks
    .venv/bin/python examples/forge/auto_demo.py --list

    # Invoke a specific agent directly
    .venv/bin/python examples/forge/auto_demo.py --agent agent "what files are in src?"

    # Invoke a specific playbook directly
    .venv/bin/python examples/forge/auto_demo.py --playbook playbook "analyze this project"
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forge.config.user_config import FileUserConfigStore
from forge.forge import Forge


def main() -> None:
    parser = argparse.ArgumentParser(description="Forge autonomous demo")
    parser.add_argument("message", nargs="?", default="", help="Message to process")
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all discovered agents and playbooks",
    )
    parser.add_argument(
        "--agent", "-a",
        metavar="NAME",
        help="Invoke a specific agent by name",
    )
    parser.add_argument(
        "--playbook", "-p",
        metavar="NAME",
        help="Invoke a specific playbook by name",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    user_config_store = FileUserConfigStore(Path(__file__).parent / "user.json")

    forge = Forge(
        working_dir=Path(__file__).parent,
        user_config_store=user_config_store,
        verbose=args.verbose,
    )

    if args.list:
        print("Agents:")
        for name, path in forge.agents.items():
            desc = forge._descriptions.get(name, "")
            print(f"  {name}: {desc}  ({path.relative_to(ROOT)})")
        print("\nPlaybooks:")
        for name, path in forge.playbooks.items():
            desc = forge._descriptions.get(name, "")
            print(f"  {name}: {desc}  ({path.relative_to(ROOT)})")
        return

    message = args.message or "Analyze this project"

    if args.agent:
        result = forge.invoke_agent(args.agent, message)
    elif args.playbook:
        result = forge.invoke_playbook(args.playbook, message)
    else:
        result = forge.invoke(message)

    print(result)


if __name__ == "__main__":
    main()
