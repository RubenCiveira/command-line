from __future__ import annotations

import ipaddress
import json
import os
import re
import subprocess
import urllib.parse
import urllib.request
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

from forge.permission.manager import PermissionManager
from forge.tools.base import ForgeTool


def _to_dict(value: Any) -> Any:
    """Try to parse a JSON string into a dict/list; return the value unchanged otherwise."""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                return json.loads(stripped)
            except (json.JSONDecodeError, ValueError):
                pass
    return value


def _coerce_path(value: Any) -> str:
    value = _to_dict(value)
    if isinstance(value, dict):
        return str(value.get("path", ""))
    return str(value)


def _project_root() -> Path:
    return Path.cwd()


def _todo_store_path() -> Path:
    return _project_root() / ".opencode" / "todos.json"


class BashTool(ForgeTool):
    name: str = "bash"
    description: str = (
        "Execute shell commands. "
        "WARNING: runs with shell=True; always enable permission checks for this tool."
    )
    permission_key: str | None = "bash"

    def run(self, tool_input: Any) -> Any:
        cmd = str(tool_input)
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            out = (result.stdout or "").strip()
            err = (result.stderr or "").strip()
            return out if out else err
        except Exception as exc:
            return f"Error executing bash: {exc}"


class ReadTool(ForgeTool):
    name: str = "read"
    description: str = "Read a file from disk. Pass a file path string, e.g. 'src/forge/agent.py'. Do NOT pass a directory path."

    def run(self, tool_input: Any) -> Any:
        path = _coerce_path(tool_input)
        try:
            return Path(path).read_text(encoding="utf-8")
        except FileNotFoundError:
            return f"Error: {path} not found"
        except Exception as exc:
            return f"Error reading {path}: {exc}"


class WriteTool(ForgeTool):
    name: str = "write"
    description: str = "Create or overwrite a file."

    def run(self, tool_input: Any) -> Any:
        tool_input = _to_dict(tool_input)
        if not isinstance(tool_input, dict):
            return "Error: expected {'path': ..., 'content': ...}"
        path = Path(tool_input.get("path", ""))
        content = tool_input.get("content", "")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(content), encoding="utf-8")
            return f"Written: {path}"
        except Exception as exc:
            return f"Error writing {path}: {exc}"


class EditTool(ForgeTool):
    name: str = "edit"
    description: str = (
        "Edit a file by replacing exact text. "
        "Replaces the first occurrence by default; set replace_all=true to replace all."
    )

    def run(self, tool_input: Any) -> Any:
        tool_input = _to_dict(tool_input)
        if not isinstance(tool_input, dict):
            return "Error: expected {'path': ..., 'find': ..., 'replace': ..., 'replace_all': bool}"
        path = Path(tool_input.get("path", ""))
        to_find = tool_input.get("find", "")
        to_replace = tool_input.get("replace", "")
        replace_all = bool(tool_input.get("replace_all", False))
        try:
            text = path.read_text(encoding="utf-8")
            if to_find not in text:
                return f"Error: text not found in {path}"
            count = None if replace_all else 1
            new_text = text.replace(to_find, to_replace, count) if count else text.replace(to_find, to_replace)
            path.write_text(new_text, encoding="utf-8")
            return f"Edited {path}"
        except FileNotFoundError:
            return f"Error: file {path} not found"
        except Exception as exc:
            return f"Error editing {path}: {exc}"


class GrepTool(ForgeTool):
    name: str = "grep"
    description: str = "Search for a regex pattern in files."

    def run(self, tool_input: Any) -> Any:
        tool_input = _to_dict(tool_input)
        if not isinstance(tool_input, dict):
            return "Error: expected {'pattern': ..., 'path': ..., 'include': ...}"
        pattern = tool_input.get("pattern", "")
        root = tool_input.get("path", ".")
        include = tool_input.get("include")
        matches: list[str] = []

        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return f"Error: invalid regex: {exc}"

        for root_dir, _, files in os.walk(root):
            for fname in files:
                if include and not fnmatch(fname, include):
                    continue
                full = os.path.join(root_dir, fname)
                try:
                    with open(full, "r", encoding="utf-8") as handle:
                        for num, line in enumerate(handle, start=1):
                            if regex.search(line):
                                matches.append(f"{full}:{num}:{line.strip()}")
                except Exception:
                    continue
        return "\n".join(matches)


class GlobTool(ForgeTool):
    name: str = "glob"
    description: str = "Find files recursively by glob pattern. Pass {'pattern': '**/*.py', 'path': 'src'} or a plain pattern string like 'src/**'."

    def run(self, tool_input: Any) -> Any:
        tool_input = _to_dict(tool_input)
        if isinstance(tool_input, dict):
            pattern = tool_input.get("pattern", "")
            root = tool_input.get("path", ".")
        else:
            pattern = str(tool_input)
            root = "."
        if not pattern:
            return "Error: glob pattern is required"
        base = Path(root)
        try:
            matches = [str(p) for p in base.glob(pattern)]
        except ValueError as exc:
            return f"Error: invalid glob pattern '{pattern}': {exc}"
        return "\n".join(matches)


class ListTool(ForgeTool):
    name: str = "list"
    description: str = "List files and folders in a directory. Pass a directory path string, e.g. 'src', 'src/forge', or '.' for the current directory."

    def run(self, tool_input: Any) -> Any:
        path = _coerce_path(tool_input) or "."
        try:
            items = os.listdir(path)
            return "\n".join(items)
        except Exception as exc:
            return f"Error listing {path}: {exc}"


class LspTool(ForgeTool):
    name: str = "lsp"
    description: str = "Experimental LSP integration."

    def run(self, tool_input: Any) -> Any:
        return "LSP tool is experimental and not configured."


class PatchTool(ForgeTool):
    name: str = "patch"
    description: str = "Apply a unified diff patch."

    def run(self, tool_input: Any) -> Any:
        patch_text = str(tool_input)
        if not patch_text.strip():
            return "Error: patch input is empty"
        try:
            result = subprocess.run(
                ["patch", "-p0", "-N", "-r", "-"],
                input=patch_text,
                capture_output=True,
                text=True,
                timeout=60,
            )
            out = (result.stdout or "").strip()
            err = (result.stderr or "").strip()
            return out if out else err
        except FileNotFoundError:
            return "Error: 'patch' command not available"
        except Exception as exc:
            return f"Error applying patch: {exc}"


class SkillTool(ForgeTool):
    name: str = "skill"
    description: str = "Load a specialized skill (not implemented)."

    def run(self, tool_input: Any) -> Any:
        return "Skill tool not implemented in this environment."


class TodoWriteTool(ForgeTool):
    name: str = "todowrite"
    description: str = "Persist a TODO list to disk."

    def run(self, tool_input: Any) -> Any:
        tool_input = _to_dict(tool_input)
        if not isinstance(tool_input, dict):
            return "Error: expected {'todos': [...]}"
        todos = tool_input.get("todos", [])
        store = _todo_store_path()
        try:
            store.parent.mkdir(parents=True, exist_ok=True)
            store.write_text(json.dumps(todos, ensure_ascii=True, indent=2))
            return f"Saved {len(todos)} todo items"
        except Exception as exc:
            return f"Error saving todos: {exc}"


class TodoReadTool(ForgeTool):
    name: str = "todoread"
    description: str = "Read the stored TODO list."

    def run(self, tool_input: Any) -> Any:
        store = _todo_store_path()
        if not store.exists():
            return "[]"
        try:
            return store.read_text(encoding="utf-8")
        except Exception as exc:
            return f"Error reading todos: {exc}"


def _validate_url(url: str) -> str | None:
    """Return an error message if the URL is not safe to fetch, None otherwise."""
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return "Error: malformed URL"
    if parsed.scheme not in {"http", "https"}:
        return f"Error: URL scheme '{parsed.scheme}' is not allowed (only http/https)"
    hostname = parsed.hostname or ""
    try:
        addr = ipaddress.ip_address(hostname)
        if addr.is_loopback or addr.is_private or addr.is_link_local or addr.is_reserved:
            return f"Error: requests to private/internal addresses are not allowed ({hostname})"
    except ValueError:
        # hostname is a domain name, not a bare IP — allow it
        pass
    blocked_hosts = {"localhost"}
    if hostname.lower() in blocked_hosts:
        return f"Error: requests to '{hostname}' are not allowed"
    return None


class WebfetchTool(ForgeTool):
    name: str = "webfetch"
    description: str = "Fetch content from a URL."

    def run(self, tool_input: Any) -> Any:
        tool_input = _to_dict(tool_input)
        if not isinstance(tool_input, dict):
            return "Error: expected {'url': ..., 'format': ...}"
        url = tool_input.get("url")
        if not url:
            return "Error: missing url"
        error = _validate_url(str(url))
        if error:
            return error
        timeout = float(tool_input.get("timeout", 30))
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                content = response.read()
                return content.decode("utf-8", errors="replace")
        except Exception as exc:
            return f"Error fetching {url}: {exc}"


class WebsearchTool(ForgeTool):
    name: str = "websearch"
    description: str = "Search the web (not configured)."

    def run(self, tool_input: Any) -> Any:
        return "Websearch tool not configured."


class QuestionTool(ForgeTool):
    name: str = "question"
    description: str = "Ask a user question (interactive)."

    def run(self, tool_input: Any) -> Any:
        return "Question tool requires an interactive UI handler."


class McpTool(ForgeTool):
    name: str = "mcp"
    description: str = "Wrapper for MCP servers (not configured)."

    def run(self, tool_input: Any) -> Any:
        return "MCP tool not configured."


def all_tools(permissions: PermissionManager | None = None) -> list[ForgeTool]:
    return [
        BashTool(permissions),
        EditTool(permissions),
        WriteTool(permissions),
        ReadTool(permissions),
        GrepTool(permissions),
        GlobTool(permissions),
        ListTool(permissions),
        LspTool(permissions),
        PatchTool(permissions),
        SkillTool(permissions),
        TodoWriteTool(permissions),
        TodoReadTool(permissions),
        WebfetchTool(permissions),
        WebsearchTool(permissions),
        QuestionTool(permissions),
        McpTool(permissions),
    ]
