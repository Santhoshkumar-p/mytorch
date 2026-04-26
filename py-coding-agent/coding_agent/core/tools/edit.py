from __future__ import annotations

import difflib
import unicodedata
from pathlib import Path

from .path_utils import resolve_to_cwd
from .file_mutation_queue import with_file_mutation_queue

SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "edits": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                },
                "required": ["old_text", "new_text"],
            },
            "minItems": 1,
        },
    },
    "required": ["path", "edits"],
}

# Smart typographic substitutions for fuzzy matching
_SMART_QUOTE_MAP = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "--",
        "\u00a0": " ",   # non-breaking space
        "\u202f": " ",   # narrow no-break space
    }
)


def _detect_line_ending(content: str) -> str:
    if "\r\n" in content:
        return "CRLF"
    if "\r" in content:
        return "CR"
    return "LF"


def _normalize_for_fuzzy(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return text.translate(_SMART_QUOTE_MAP)


def _apply_edit(content: str, old_text: str, new_text: str) -> str:
    # Exact match
    count = content.count(old_text)
    if count == 1:
        return content.replace(old_text, new_text, 1)
    if count > 1:
        raise ValueError(
            f"old_text appears {count} times in the file — be more specific."
        )

    # Fuzzy match: normalize both and locate
    norm_content = _normalize_for_fuzzy(content)
    norm_old = _normalize_for_fuzzy(old_text)

    fuzzy_count = norm_content.count(norm_old)
    if fuzzy_count == 1:
        idx = norm_content.index(norm_old)
        return content[:idx] + new_text + content[idx + len(norm_old):]
    if fuzzy_count > 1:
        raise ValueError(
            f"old_text appears {fuzzy_count} times after fuzzy normalization — be more specific."
        )

    raise ValueError("old_text not found in the file.")


def _to_lf(content: str, line_ending: str) -> str:
    if line_ending == "CRLF":
        return content.replace("\r\n", "\n")
    if line_ending == "CR":
        return content.replace("\r", "\n")
    return content


def _from_lf(content: str, line_ending: str) -> str:
    if line_ending == "CRLF":
        return content.replace("\n", "\r\n")
    if line_ending == "CR":
        return content.replace("\n", "\r")
    return content


def _make_text_content(text: str) -> dict:
    return {"type": "text", "text": text}


async def execute_edit(args: dict, cwd: str) -> list:
    path_str: str = args["path"]
    edits: list[dict] = args["edits"]

    abs_path = resolve_to_cwd(path_str, cwd)

    async def _do_edit():
        p = Path(abs_path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {abs_path}")

        raw_bytes = p.read_bytes()
        bom = b""
        if raw_bytes.startswith(b"\xef\xbb\xbf"):
            bom = b"\xef\xbb\xbf"
            raw_bytes = raw_bytes[3:]

        original_text = raw_bytes.decode("utf-8", errors="replace")
        line_ending = _detect_line_ending(original_text)
        normalized = _to_lf(original_text, line_ending)

        content = normalized
        # Apply edits in reverse order to preserve offsets
        for edit in reversed(edits):
            old_lf = _to_lf(edit["old_text"], line_ending)
            new_lf = _to_lf(edit["new_text"], line_ending)
            content = _apply_edit(content, old_lf, new_lf)

        result_text = _from_lf(content, line_ending)
        result_bytes = bom + result_text.encode("utf-8")
        p.write_bytes(result_bytes)

        # Generate unified diff
        original_lines = original_text.splitlines(keepends=True)
        result_lines = result_text.splitlines(keepends=True)
        diff = "".join(
            difflib.unified_diff(
                original_lines,
                result_lines,
                fromfile=f"a/{path_str}",
                tofile=f"b/{path_str}",
            )
        )
        if not diff:
            diff = "[No changes — old_text and new_text were identical]\n"

        return [_make_text_content(diff)]

    try:
        return await with_file_mutation_queue(abs_path, _do_edit)
    except (FileNotFoundError, ValueError, OSError) as e:
        return [_make_text_content(f"Error: {e}")]
