from __future__ import annotations
import base64
import mimetypes
import os
import sys
from dataclasses import dataclass, field

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}


@dataclass
class ProcessedFiles:
    text: str | None = None
    images: list = field(default_factory=list)  # list[ImageContent]


async def process_file_arguments(
    raw_args: list[str],
    cwd: str,
    auto_resize: bool = True,
) -> ProcessedFiles:
    text_parts: list[str] = []
    images: list = []

    for arg in raw_args:
        if arg.startswith("@"):
            path = os.path.expanduser(arg[1:])
            if not os.path.isabs(path):
                path = os.path.join(cwd, path)
            # Normalize macOS narrow-no-break-space
            path = path.replace("\u202f", " ")
            if not os.path.exists(path):
                continue
            ext = os.path.splitext(path)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                data = open(path, "rb").read()
                if auto_resize:
                    data = _maybe_resize(data)
                mime = mimetypes.guess_type(path)[0] or "image/png"
                try:
                    from pi_agent.types import ImageContent
                    images.append(ImageContent(
                        data=base64.b64encode(data).decode(),
                        mime_type=mime,
                    ))
                except ImportError:
                    images.append({
                        "data": base64.b64encode(data).decode(),
                        "mime_type": mime,
                    })
            else:
                try:
                    content = open(path, errors="replace").read()
                    if content.strip():
                        text_parts.append(f'<file path="{path}">\n{content}\n</file>')
                except OSError:
                    pass
        else:
            text_parts.append(arg)

    return ProcessedFiles(
        text="\n\n".join(text_parts) if text_parts else None,
        images=images,
    )


async def read_stdin_if_piped() -> str | None:
    """Return stdin content if not a TTY, else None."""
    if sys.stdin.isatty():
        return None
    try:
        return sys.stdin.read()
    except Exception:
        return None


def _maybe_resize(data: bytes, max_dim: int = 2000) -> bytes:
    try:
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(data))
        if img.width > max_dim or img.height > max_dim:
            img.thumbnail((max_dim, max_dim))
            buf = io.BytesIO()
            img.save(buf, format=img.format or "PNG")
            return buf.getvalue()
    except (ImportError, Exception):
        # PIL not installed, or image data is corrupt/unrecognized — skip resize
        pass
    return data
