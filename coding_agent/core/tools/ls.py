from __future__ import annotations

import os
from pathlib import Path

from .path_utils import resolve_to_cwd
from .truncate import truncate_head

SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "recursive": {"type": "boolean"},
        "show_hidden": {"type": "boolean"},
        "show_size": {"type": "boolean"},
    },
}


def _make_text_content(text: str) -> dict:
    return {"type": "text", "text": text}


def _fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:>6} {unit}"
        n //= 1024
    return f"{n:>6} TB"


def _list_dir(
    path: str,
    show_hidden: bool,
    show_size: bool,
) -> list[str]:
    lines: list[str] = []
    try:
        entries = list(os.scandir(path))
    except PermissionError:
        return [f"[Permission denied: {path}]"]
    except OSError as e:
        return [f"[Error: {e}]"]

    # Dirs first, then files; each group sorted case-insensitively
    dirs = sorted(
        [e for e in entries if e.is_dir(follow_symlinks=False)],
        key=lambda e: e.name.lower(),
    )
    files = sorted(
        [e for e in entries if not e.is_dir(follow_symlinks=False)],
        key=lambda e: e.name.lower(),
    )

    for entry in dirs + files:
        if not show_hidden and entry.name.startswith("."):
            continue
        is_dir = entry.is_dir(follow_symlinks=False)
        name = entry.name + ("/" if is_dir else "")
        if show_size and not is_dir:
            try:
                size = entry.stat(follow_symlinks=False).st_size
                lines.append(f"{_fmt_size(size)}  {name}")
            except OSError:
                lines.append(f"{'?':>8}  {name}")
        else:
            lines.append(name)

    return lines


async def execute_ls(args: dict, cwd: str) -> list:
    path_str: str | None = args.get("path")
    recursive: bool = args.get("recursive", False)
    show_hidden: bool = args.get("show_hidden", False)
    show_size: bool = args.get("show_size", False)

    target = resolve_to_cwd(path_str, cwd) if path_str else cwd
    p = Path(target)

    if not p.exists():
        return [_make_text_content(f"Error: path not found: {target}")]

    if p.is_file():
        lines = [p.name]
    elif not recursive:
        lines = _list_dir(target, show_hidden, show_size)
    else:
        lines = []
        for root, dirs, files in os.walk(target):
            # Skip hidden dirs unless show_hidden
            if not show_hidden:
                dirs[:] = [d for d in dirs if not d.startswith(".")]
            rel_root = os.path.relpath(root, target)
            prefix = "" if rel_root == "." else rel_root + os.sep
            for d in sorted(dirs, key=str.lower):
                lines.append(prefix + d + "/")
            for f in sorted(files, key=str.lower):
                if not show_hidden and f.startswith("."):
                    continue
                name = prefix + f
                if show_size:
                    try:
                        size = os.stat(os.path.join(root, f)).st_size
                        lines.append(f"{_fmt_size(size)}  {name}")
                    except OSError:
                        lines.append(f"{'?':>8}  {name}")
                else:
                    lines.append(name)

    text = "\n".join(lines) + "\n" if lines else "(empty)\n"
    text, trunc = truncate_head(text)
    if trunc.truncated_by is not None:
        text += "\n[Output truncated]"

    return [_make_text_content(text)]
