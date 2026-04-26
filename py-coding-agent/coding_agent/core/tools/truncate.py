from __future__ import annotations

from coding_agent.core.types import TruncationResult

DEFAULT_MAX_LINES = 2000
DEFAULT_MAX_BYTES = 50_000
GREP_MAX_LINE_LENGTH = 500
BASH_PREVIEW_LINES = 5


def truncate_tail(
    text: str,
    max_lines: int = DEFAULT_MAX_LINES,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> tuple[str, TruncationResult]:
    """Keep the LAST N lines/bytes — used for bash command output."""
    encoded = text.encode("utf-8")
    total_lines = text.count("\n") + (1 if text and not text.endswith("\n") else 0)

    truncated_by: str | None = None

    # Byte truncation first
    if len(encoded) > max_bytes:
        truncated_by = "bytes"
        tail_bytes = encoded[-max_bytes:]
        text = tail_bytes.decode("utf-8", errors="replace")
        # Drop a potentially partial first line
        newline_pos = text.find("\n")
        if newline_pos != -1:
            text = text[newline_pos + 1 :]

    # Line truncation
    lines = text.splitlines(keepends=True)
    if len(lines) > max_lines:
        truncated_by = "lines"
        lines = lines[-max_lines:]
        text = "".join(lines)

    output_lines = text.count("\n") + (1 if text and not text.endswith("\n") else 0)
    output_bytes = len(text.encode("utf-8"))

    result = TruncationResult(
        truncated_by=truncated_by,
        total_lines=total_lines,
        output_lines=output_lines,
        output_bytes=output_bytes,
    )
    return text, result


def truncate_head(
    text: str,
    max_lines: int = DEFAULT_MAX_LINES,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> tuple[str, TruncationResult]:
    """Keep the FIRST N lines/bytes — used for read/find/ls output."""
    lines = text.splitlines(keepends=True)
    total_lines = len(lines)

    truncated_by: str | None = None
    first_line_exceeds_limit = False
    kept: list[str] = []
    byte_count = 0

    for i, line in enumerate(lines):
        line_bytes = len(line.encode("utf-8"))
        if i == 0 and line_bytes > max_bytes:
            first_line_exceeds_limit = True
            kept.append(line)
            byte_count += line_bytes
            truncated_by = "bytes"
            break
        if byte_count + line_bytes > max_bytes:
            truncated_by = "bytes"
            break
        if len(kept) >= max_lines:
            truncated_by = "lines"
            break
        kept.append(line)
        byte_count += line_bytes

    output_text = "".join(kept)
    output_lines = len(kept)
    output_bytes = len(output_text.encode("utf-8"))

    result = TruncationResult(
        truncated_by=truncated_by,
        total_lines=total_lines,
        output_lines=output_lines,
        output_bytes=output_bytes,
        first_line_exceeds_limit=first_line_exceeds_limit,
    )
    return output_text, result


def truncate_line(line: str, max_len: int = GREP_MAX_LINE_LENGTH) -> str:
    """Truncate a single long line, appending a note about omitted characters."""
    if len(line) <= max_len:
        return line
    omitted = len(line) - max_len
    return line[:max_len] + f" ... [{omitted} chars omitted]"
