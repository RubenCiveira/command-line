"""Image generation agent demo.

Usage:
    # Local Stable Diffusion (requires: pip install diffusers accelerate)
    .venv/bin/python examples/forge/image_demo.py "un atardecer sobre el mar con veleros"

    # OpenAI DALL-E (requires: OPENAI_API_KEY env var)
    .venv/bin/python examples/forge/image_demo.py --backend dalle "un gato astronauta"
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from forge.agent.factory import ForgeAgentFactory
from forge.agent.markdown_parser import ForgeMarkdownAgentParser
from forge.config.user_config import FileUserConfigStore
from forge.permission.console_broker import ConsolePermissionBroker
from forge.permission.file_store import FilePermissionStore
from forge.permission.manager import PermissionManager
from forge.tools.image import DalleImageTool, LocalImageTool
from forge.tools.registry import ToolRegistry


def main() -> None:
    parser = argparse.ArgumentParser(description="Forge image generation demo")
    parser.add_argument("prompt", nargs="?", default="a serene mountain landscape at dawn")
    parser.add_argument(
        "--backend",
        choices=["local", "dalle"],
        default="local",
        help="Image generation backend (default: local Stable Diffusion)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "local: HuggingFace model ID (default: stabilityai/sdxl-turbo)\n"
            "dalle: dall-e-3 or dall-e-2"
        ),
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    broker = ConsolePermissionBroker()
    permission_store = FilePermissionStore()
    user_config_store = FileUserConfigStore(Path(__file__).parent / "user.json")

    # Build the image tool
    if args.backend == "dalle":
        image_tool = DalleImageTool(model=args.model or "dall-e-3")
    else:
        image_tool = LocalImageTool(model_id=args.model or "stabilityai/sdxl-turbo")

    # Register the pre-configured instance so the factory patches its permissions
    # at build time and the tool_filter in image_agent.md can select it.
    registry = ToolRegistry()
    registry.register_instance(image_tool)

    factory = ForgeAgentFactory(
        tool_registry=registry,
        permission_store=permission_store,
        permission_manager_factory=lambda permissions: PermissionManager(
            permissions, broker
        ),
        user_config_store=user_config_store,
        verbose=args.verbose,
    )

    agent_path = Path(__file__).parent / "image_agent.md"
    agent = factory.from_markdown(agent_path)

    response = agent.invoke(args.prompt)
    content = response.content if hasattr(response, "content") else str(response)
    print(content)

    # Try to open the image if a path was returned
    import re
    match = re.search(r"Image saved: (.+\.png)", content)
    if match:
        img_path = match.group(1).strip()
        if sys.platform == "darwin":
            os.system(f'open "{img_path}"')
        elif sys.platform.startswith("linux"):
            os.system(f'xdg-open "{img_path}"')
        elif sys.platform == "win32":
            os.system(f'start "" "{img_path}"')


if __name__ == "__main__":
    main()
