from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

from .path_utils import resolve_read_path
from .truncate import truncate_head

SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "offset": {"type": "integer", "description": "Start line (1-indexed)"},
        "limit": {"type": "integer", "description": "Max lines to read"},
    },
    "required": ["path"],
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico", ".bmp"}

_MAX_IMAGE_DIM = 2000


def _make_text_content(text: str) -> dict:
    return {"type": "text", "text": text}


def _make_image_content(data: str, mime_type: str) -> dict:
    return {"type": "image", "data": data, "mime_type": mime_type}


def _resize_image(data: bytes, mime_type: str) -> bytes:
    """Resize image to fit within _MAX_IMAGE_DIM x _MAX_IMAGE_DIM, returns bytes."""
    try:
        from PIL import Image
        import io

        img = Image.open(io.BytesIO(data))
        if img.width > _MAX_IMAGE_DIM or img.height > _MAX_IMAGE_DIM:
            img.thumbnail((_MAX_IMAGE_DIM, _MAX_IMAGE_DIM))
            buf = io.BytesIO()
            fmt = img.format or "PNG"
            img.save(buf, format=fmt)
            return buf.getvalue()
    except ImportError:
        pass
    return data


async def execute_read(args: dict, cwd: str, settings) -> list:
    path_str: str = args["path"]
    offset: int | None = args.get("offset")
    limit: int | None = args.get("limit")

    resolved = resolve_read_path(path_str, cwd)
    if resolved is None:
        return [_make_text_content(f"Error: file not found: {path_str}")]

    suffix = Path(resolved).suffix.lower()

    if suffix in IMAGE_EXTENSIONS:
        try:
            with open(resolved, "rb") as f:
                raw = f.read()
        except OSError as e:
            return [_make_text_content(f"Error reading image: {e}")]

        mime_type, _ = mimetypes.guess_type(resolved)
        if not mime_type:
            mime_type = "image/png"

        if getattr(settings, "image_auto_resize", True):
            raw = _resize_image(raw, mime_type)

        encoded = base64.b64encode(raw).decode("ascii")
        return [_make_image_content(encoded, mime_type)]

    # Text file
    try:
        with open(resolved, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
    except OSError as e:
        return [_make_text_content(f"Error reading file: {e}")]

    total_lines = len(all_lines)

    # Apply offset (1-indexed) and limit
    start = max((offset or 1) - 1, 0)
    end = start + limit if limit is not None else total_lines
    selected = all_lines[start:end]

    # Add line numbers
    numbered = [
        f"{start + i + 1}\t{line}" for i, line in enumerate(selected)
    ]
    text = "".join(numbered)

    text, trunc = truncate_head(text)

    if trunc.truncated_by is not None:
        continuation_start = start + trunc.output_lines + 1
        hint = (
            f"\n[File truncated — {total_lines} total lines. "
            f"Use offset={continuation_start} to continue reading.]"
        )
        text += hint

    return [_make_text_content(text)]
