from __future__ import annotations

from typing import Any

from .bash import execute_bash, SCHEMA as BASH_SCHEMA
from .read import execute_read, SCHEMA as READ_SCHEMA
from .edit import execute_edit, SCHEMA as EDIT_SCHEMA
from .write import execute_write, SCHEMA as WRITE_SCHEMA
from .grep import execute_grep, SCHEMA as GREP_SCHEMA
from .find import execute_find, SCHEMA as FIND_SCHEMA
from .ls import execute_ls, SCHEMA as LS_SCHEMA

ALL_TOOL_NAMES = ["read", "bash", "edit", "write", "grep", "find", "ls"]


def _wrap_execute(fn):
    """
    Wrap a tool execute function that returns list[dict] into one that
    matches the agent ToolExecuteFn protocol:
        async (tool_call_id, params, abort_event, on_update) -> AgentToolResult
    """
    async def _execute(tool_call_id, params, abort_event=None, on_update=None):
        try:
            from pi_agent import AgentToolResult, TextContent
            raw = await fn(params)
            # Convert list[{"type":"text","text":...}] → AgentToolResult
            content = []
            for block in raw:
                if isinstance(block, dict) and block.get("type") == "text":
                    content.append(TextContent(text=block.get("text", "")))
                else:
                    content.append(TextContent(text=str(block)))
            return AgentToolResult(content=content, details={})
        except ImportError:
            # agent library not available — return raw for testing
            raw = await fn(params)
            return raw
    return _execute


def _to_agent_tool(d: dict):
    """Convert a tool descriptor dict to an AgentTool."""
    try:
        from pi_agent import AgentTool
        return AgentTool(
            name=d["name"],
            label=d.get("label", d["name"]),
            description=d["description"],
            parameters=d["parameters"],
            execute=_wrap_execute(d["_execute_raw"]),
        )
    except ImportError:
        return d


def _make_bash(cwd: str, settings) -> dict:
    async def _exec(args):
        return await execute_bash(args, cwd, settings)
    return {
        "name": "bash",
        "label": "Bash",
        "description": "Execute shell commands",
        "parameters": BASH_SCHEMA,
        "_execute_raw": _exec,
    }


def _make_read(cwd: str, settings) -> dict:
    async def _exec(args):
        return await execute_read(args, cwd, settings)
    return {
        "name": "read",
        "label": "Read",
        "description": "Read a file's contents",
        "parameters": READ_SCHEMA,
        "_execute_raw": _exec,
    }


def _make_edit(cwd: str) -> dict:
    async def _exec(args):
        return await execute_edit(args, cwd)
    return {
        "name": "edit",
        "label": "Edit",
        "description": "Apply text edits to a file",
        "parameters": EDIT_SCHEMA,
        "_execute_raw": _exec,
    }


def _make_write(cwd: str) -> dict:
    async def _exec(args):
        return await execute_write(args, cwd)
    return {
        "name": "write",
        "label": "Write",
        "description": "Write content to a file",
        "parameters": WRITE_SCHEMA,
        "_execute_raw": _exec,
    }


def _make_grep(cwd: str) -> dict:
    async def _exec(args):
        return await execute_grep(args, cwd)
    return {
        "name": "grep",
        "label": "Grep",
        "description": "Search file contents with a regex or literal pattern",
        "parameters": GREP_SCHEMA,
        "_execute_raw": _exec,
    }


def _make_find(cwd: str) -> dict:
    async def _exec(args):
        return await execute_find(args, cwd)
    return {
        "name": "find",
        "label": "Find",
        "description": "Find files matching a glob pattern",
        "parameters": FIND_SCHEMA,
        "_execute_raw": _exec,
    }


def _make_ls(cwd: str) -> dict:
    async def _exec(args):
        return await execute_ls(args, cwd)
    return {
        "name": "ls",
        "label": "LS",
        "description": "List directory contents",
        "parameters": LS_SCHEMA,
        "_execute_raw": _exec,
    }


def build_tools(
    cwd: str,
    settings,
    active_names: list[str] | None = None,
    custom_tools: list | None = None,
) -> list:
    """Build AgentTool-compatible objects for the requested tool names."""
    names = active_names if active_names is not None else ALL_TOOL_NAMES
    tools: list[Any] = []

    for name in names:
        if name == "bash":
            raw = _make_bash(cwd, settings)
        elif name == "read":
            raw = _make_read(cwd, settings)
        elif name == "edit":
            raw = _make_edit(cwd)
        elif name == "write":
            raw = _make_write(cwd)
        elif name == "grep":
            raw = _make_grep(cwd)
        elif name == "find":
            raw = _make_find(cwd)
        elif name == "ls":
            raw = _make_ls(cwd)
        else:
            continue

        tools.append(_to_agent_tool(raw))

    if custom_tools:
        tools.extend(custom_tools)

    return tools
