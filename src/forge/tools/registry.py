"""Tool registry — decoupled tool management and extension point.

The ``ToolRegistry`` holds tool *classes* (instantiated fresh per agent) and
pre-configured tool *instances* (whose permissions are patched at build time).
It replaces the monolithic ``all_tools()`` call and gives callers a clean
surface to extend the tool pool without touching built-in code.

Typical usage::

    # Use all built-in tools (default)
    registry = ToolRegistry.default()

    # Add a custom tool class
    registry.register(MyCustomTool)

    # Inject a pre-configured instance (e.g. LocalImageTool with custom model)
    image_tool = LocalImageTool(model_id="my-org/my-model")
    registry.register_instance(image_tool)

    # Pass to the agent factory
    factory = ForgeAgentFactory(tool_registry=registry, ...)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Type

from forge.tools.base import ForgeTool

if TYPE_CHECKING:
    from forge.permission.manager import PermissionManager


class ToolRegistry:
    """Registry of tool classes and pre-configured instances.

    Two storage buckets:

    * ``_classes`` — ``dict[name, Type[ForgeTool]]``: classes registered by
      their ``.name`` attribute.  Instantiated fresh on every ``build_all()``
      call, receiving the current ``PermissionManager``.

    * ``_instances`` — ``dict[name, ForgeTool]``: pre-configured instances
      with custom constructor args already baked in.  ``build_all()`` patches
      their ``_permissions`` attribute before returning them.

    When the same name appears in both buckets, the *instance* wins.
    """

    def __init__(self) -> None:
        self._classes: dict[str, Type[ForgeTool]] = {}
        self._instances: dict[str, ForgeTool] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, cls: Type[ForgeTool]) -> None:
        """Register a tool class by its declared ``.name`` attribute."""
        self._classes[cls.name] = cls

    def register_instance(self, tool: ForgeTool) -> None:
        """Register a pre-configured tool instance.

        The instance's ``_permissions`` attribute is replaced with the live
        ``PermissionManager`` when ``build_all()`` is called, so permission
        checks continue to work normally.
        """
        self._instances[tool.name] = tool

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build_all(self, permissions: PermissionManager | None = None) -> list[ForgeTool]:
        """Return all registered tools with the given permission manager injected.

        1. Instantiates all registered classes.
        2. Patches ``_permissions`` on all registered instances.
        3. Instances override class-built tools of the same name.

        Args:
            permissions: Manager to inject.  Defaults to an unconstrained
                         ``PermissionManager({})`` when ``None``.
        """
        from forge.permission.manager import PermissionManager as PM  # noqa: PLC0415

        pm = permissions if permissions is not None else PM({})

        result: dict[str, ForgeTool] = {}

        for name, cls in self._classes.items():
            result[name] = cls(pm)

        for name, instance in self._instances.items():
            instance._permissions = pm
            result[name] = instance

        return list(result.values())

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def names(self) -> list[str]:
        """Return all registered tool names (instances override classes)."""
        seen: dict[str, None] = {}
        for name in self._classes:
            seen[name] = None
        for name in self._instances:
            seen[name] = None
        return list(seen)

    def copy(self) -> ToolRegistry:
        """Return a shallow copy that shares no dict state with the original.

        Use this to extend the default registry without mutating it::

            registry = ToolRegistry.default().copy()
            registry.register(MyTool)
        """
        new = ToolRegistry()
        new._classes = dict(self._classes)
        new._instances = dict(self._instances)
        return new

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def default(cls) -> ToolRegistry:
        """Return a registry pre-populated with every built-in tool.

        Image tools (``LocalImageTool``, ``DalleImageTool``) are imported
        lazily inside this method so that missing optional dependencies
        (``diffusers``, ``openai``) do not break import-time loading.
        """
        from forge.tools.builtins import (  # noqa: PLC0415
            BashTool,
            EditTool,
            GlobTool,
            GrepTool,
            ListTool,
            LspTool,
            McpTool,
            PatchTool,
            QuestionTool,
            ReadTool,
            SkillTool,
            TodoReadTool,
            TodoWriteTool,
            WebfetchTool,
            WebsearchTool,
            WriteTool,
        )
        from forge.tools.image import DalleImageTool, LocalImageTool  # noqa: PLC0415

        registry = cls()
        for tool_cls in (
            BashTool,
            EditTool,
            WriteTool,
            ReadTool,
            GrepTool,
            GlobTool,
            ListTool,
            LspTool,
            PatchTool,
            SkillTool,
            TodoWriteTool,
            TodoReadTool,
            WebfetchTool,
            WebsearchTool,
            QuestionTool,
            McpTool,
            DalleImageTool,
            LocalImageTool,  # registered last → wins for name "image"
        ):
            registry.register(tool_cls)
        return registry
