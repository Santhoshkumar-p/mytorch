from __future__ import annotations

from pathlib import Path

from .path_utils import resolve_to_cwd
from .file_mutation_queue import with_file_mutation_queue

SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "content": {"type": "string"},
    },
    "required": ["path", "content"],
}


def _make_text_content(text: str) -> dict:
    return {"type": "text", "text": text}


async def execute_write(args: dict, cwd: str) -> list:
    path_str: str = args["path"]
    content: str = args["content"]

    abs_path = resolve_to_cwd(path_str, cwd)

    async def _do_write():
        p = Path(abs_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return [_make_text_content(f"Written {len(content)} characters to {path_str}")]

    try:
        return await with_file_mutation_queue(abs_path, _do_write)
    except OSError as e:
        return [_make_text_content(f"Error writing file: {e}")]
