from __future__ import annotations

import asyncio
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable

from coding_agent.core.types import BashResult, TruncationResult
from .truncate import truncate_tail, DEFAULT_MAX_LINES, DEFAULT_MAX_BYTES

DEFAULT_TIMEOUT = 120

SCHEMA = {
    "type": "object",
    "properties": {
        "command": {"type": "string", "description": "Shell command to execute"},
        "timeout": {
            "type": "integer",
            "description": "Timeout in seconds",
            "default": DEFAULT_TIMEOUT,
        },
    },
    "required": ["command"],
}


@dataclass
class BashOperations:
    exec: Callable | None = None   # custom backend (SSH, containers, etc.)
    on_chunk: Callable | None = None


def _make_text_content(text: str) -> dict:
    return {"type": "text", "text": text}


async def execute_bash(
    args: dict,
    cwd: str,
    settings,
    operations: BashOperations | None = None,
    signal=None,
) -> list:
    """Execute a shell command and return list[TextContent]."""
    command: str = args["command"]
    timeout: int = args.get("timeout", DEFAULT_TIMEOUT)

    if getattr(settings, "shell_command_prefix", None):
        command = settings.shell_command_prefix + "\n" + command

    raw_output: str = ""
    exit_code: int = 0
    timed_out = False

    if operations and operations.exec:
        raw_output = await operations.exec(command, cwd, timeout=timeout)
    else:
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=cwd,
            )
            try:
                stdout_bytes, _ = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
                raw_output = stdout_bytes.decode("utf-8", errors="replace")
                exit_code = proc.returncode or 0
            except asyncio.TimeoutError:
                timed_out = True
                try:
                    proc.kill()
                    await proc.communicate()
                except Exception:
                    pass
                raw_output = f"[Command timed out after {timeout}s]\n"
                exit_code = 1
        except Exception as exc:
            raw_output = f"[Error running command: {exc}]\n"
            exit_code = 1

    output, trunc = truncate_tail(raw_output)

    if trunc.truncated_by is not None:
        header = (
            f"[Output truncated — showing last {trunc.output_lines} lines]\n"
        )
        output = header + output

    # Write full output to a temp file when it's large
    if len(raw_output.encode("utf-8")) > DEFAULT_MAX_BYTES or raw_output.count("\n") > DEFAULT_MAX_LINES:
        try:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".txt", mode="w", encoding="utf-8"
            ) as tmp:
                tmp.write(raw_output)
                tmp_path = tmp.name
            output += f"\n[Full output saved to: {tmp_path}]"
        except Exception:
            pass

    return [{"type": "text", "text": output, "exit_code": exit_code}]


def make_bash_tool(cwd: str, settings) -> dict:
    return {
        "name": "bash",
        "label": "Bash",
        "description": "Execute shell commands",
        "parameters": SCHEMA,
        "execute": lambda args, ctx=None: execute_bash(args, cwd, settings),
        "execution_mode": "parallel",
    }
