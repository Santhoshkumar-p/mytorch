from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from .path_utils import resolve_to_cwd
from .truncate import truncate_head

SCHEMA = {
    "type": "object",
    "properties": {
        "pattern": {"type": "string", "description": "Glob pattern e.g. *.py"},
        "path": {"type": "string", "description": "Directory to search"},
        "limit": {"type": "integer", "default": 1000},
    },
    "required": ["pattern"],
}


def _make_text_content(text: str) -> dict:
    return {"type": "text", "text": text}


async def _run_fd(pattern: str, search_path: str, limit: int) -> str | None:
    fd = shutil.which("fd")
    if not fd:
        return None

    cmd = [fd, "--glob", pattern, search_path, "--max-results", str(limit)]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
        return stdout.decode("utf-8", errors="replace")
    except (asyncio.TimeoutError, OSError):
        return None


def _python_find(pattern: str, search_path: str, limit: int) -> str:
    p = Path(search_path)
    if not p.exists():
        return f"Error: path not found: {search_path}\n"

    results = sorted(str(f) for f in p.rglob(pattern))
    if len(results) > limit:
        results = results[:limit]

    return "\n".join(results) + "\n" if results else ""


async def execute_find(args: dict, cwd: str) -> list:
    pattern: str = args["pattern"]
    path_str: str | None = args.get("path")
    limit: int = args.get("limit", 1000)

    search_path = resolve_to_cwd(path_str, cwd) if path_str else cwd

    output = await _run_fd(pattern, search_path, limit)

    if output is None:
        output = _python_find(pattern, search_path, limit)

    if not output:
        return [_make_text_content("No files found.")]

    text, trunc = truncate_head(output)
    if trunc.truncated_by is not None:
        text += "\n[Output truncated]"

    return [_make_text_content(text)]
