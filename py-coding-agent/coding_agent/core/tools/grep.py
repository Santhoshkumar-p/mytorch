from __future__ import annotations

import asyncio
import fnmatch
import os
import re
import shutil
from pathlib import Path

from .path_utils import resolve_to_cwd
from .truncate import truncate_head, truncate_line

SCHEMA = {
    "type": "object",
    "properties": {
        "pattern": {"type": "string"},
        "path": {"type": "string", "description": "Directory or file to search"},
        "glob": {"type": "string", "description": "File glob filter e.g. *.py"},
        "ignore_case": {"type": "boolean"},
        "literal": {"type": "boolean", "description": "Treat pattern as literal string"},
        "context": {"type": "integer", "description": "Lines of context around matches"},
        "limit": {"type": "integer", "description": "Max matches", "default": 100},
    },
    "required": ["pattern"],
}


def _make_text_content(text: str) -> dict:
    return {"type": "text", "text": text}


async def _run_ripgrep(
    pattern: str,
    search_path: str,
    glob: str | None,
    ignore_case: bool,
    literal: bool,
    context: int,
    limit: int,
) -> str | None:
    rg = shutil.which("rg")
    if not rg:
        return None

    cmd = [rg, "--line-number", "--no-heading", "--color=never"]
    if ignore_case:
        cmd.append("-i")
    if literal:
        cmd.append("-F")
    if context:
        cmd.extend(["-C", str(context)])
    if glob:
        cmd.extend(["--glob", glob])
    cmd.extend(["--max-count", str(limit)])
    cmd.append(pattern)
    cmd.append(search_path)

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


def _python_grep(
    pattern: str,
    search_path: str,
    glob: str | None,
    ignore_case: bool,
    literal: bool,
    context: int,
    limit: int,
) -> str:
    flags = re.IGNORECASE if ignore_case else 0
    if literal:
        compiled = re.compile(re.escape(pattern), flags)
    else:
        try:
            compiled = re.compile(pattern, flags)
        except re.error as e:
            return f"Invalid regex: {e}\n"

    matches: list[str] = []
    match_count = 0

    p = Path(search_path)
    if p.is_file():
        files = [p]
    else:
        files = [f for f in p.rglob("*") if f.is_file()]
        if glob:
            files = [f for f in files if fnmatch.fnmatch(f.name, glob)]

    for file_path in sorted(files):
        try:
            lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue

        for i, line in enumerate(lines):
            if compiled.search(line):
                match_count += 1
                rel = str(file_path)

                if context:
                    ctx_start = max(0, i - context)
                    ctx_end = min(len(lines), i + context + 1)
                    for j in range(ctx_start, ctx_end):
                        sep = ":" if j == i else "-"
                        matches.append(
                            f"{rel}:{j + 1}{sep}{truncate_line(lines[j])}"
                        )
                    matches.append("--")
                else:
                    matches.append(
                        f"{rel}:{i + 1}:{truncate_line(line)}"
                    )

                if match_count >= limit:
                    matches.append(f"[Limit of {limit} matches reached]")
                    return "\n".join(matches) + "\n"

    return "\n".join(matches) + "\n" if matches else ""


async def execute_grep(args: dict, cwd: str) -> list:
    pattern: str = args["pattern"]
    path_str: str | None = args.get("path")
    glob: str | None = args.get("glob")
    ignore_case: bool = args.get("ignore_case", False)
    literal: bool = args.get("literal", False)
    context: int = args.get("context", 0)
    limit: int = args.get("limit", 100)

    search_path = resolve_to_cwd(path_str, cwd) if path_str else cwd

    output = await _run_ripgrep(
        pattern, search_path, glob, ignore_case, literal, context, limit
    )

    if output is None:
        output = _python_grep(
            pattern, search_path, glob, ignore_case, literal, context, limit
        )

    if not output:
        return [_make_text_content("No matches found.")]

    text, trunc = truncate_head(output)
    if trunc.truncated_by is not None:
        text += "\n[Output truncated — results exceeded display limit]"

    return [_make_text_content(text)]
