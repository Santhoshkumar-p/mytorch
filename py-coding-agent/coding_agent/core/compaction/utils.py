from __future__ import annotations
import json
from ..types import FileOperations

TOOL_RESULT_MAX_CHARS = 2000


def extract_file_ops_from_message(message) -> FileOperations:
    """Scan tool calls in an AssistantMessage for read/write/edit paths."""
    ops = FileOperations()
    content = getattr(message, "content", None)
    if not content:
        return ops
    for item in content:
        name = getattr(item, "name", None)
        args = getattr(item, "args", None) or getattr(item, "input", None) or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (ValueError, TypeError):
                args = {}
        if name == "read":
            path = args.get("path")
            if path:
                ops.read.add(path)
        elif name == "write":
            path = args.get("path")
            if path:
                ops.written.add(path)
        elif name == "edit":
            path = args.get("path")
            if path:
                ops.edited.add(path)
    return ops


def compute_file_lists(ops: FileOperations) -> dict:
    """Return {"read": [...], "modified": [...]} with modified = written | edited."""
    return {
        "read": sorted(ops.read),
        "modified": sorted(ops.written | ops.edited),
    }


def format_file_operations(ops: FileOperations) -> str:
    """Format as XML <read_files>...</read_files><modified_files>...</modified_files>."""
    lists = compute_file_lists(ops)
    read_items = "\n".join(f"  <file>{p}</file>" for p in lists["read"])
    modified_items = "\n".join(f"  <file>{p}</file>" for p in lists["modified"])
    return (
        f"<read_files>\n{read_items}\n</read_files>\n"
        f"<modified_files>\n{modified_items}\n</modified_files>"
    )


def serialize_conversation(messages: list, max_chars_per_result: int = TOOL_RESULT_MAX_CHARS) -> str:
    """Serialize messages for LLM summarization. Truncates tool results."""
    lines: list[str] = []

    for message in messages:
        role = getattr(message, "role", None)
        content = getattr(message, "content", None)

        if role == "user":
            if isinstance(content, str):
                lines.append(f"User: {content}")
            elif isinstance(content, list):
                for item in content:
                    item_type = getattr(item, "type", None)
                    if item_type == "tool_result":
                        result_content = getattr(item, "content", "")
                        if isinstance(result_content, list):
                            text = " ".join(
                                getattr(block, "text", "") for block in result_content
                                if getattr(block, "type", None) == "text"
                            )
                        else:
                            text = str(result_content)
                        if len(text) > max_chars_per_result:
                            text = text[:max_chars_per_result] + "... [truncated]"
                        lines.append(f"[Tool result: {text}]")
                    elif item_type == "text":
                        lines.append(f"User: {getattr(item, 'text', '')}")
        elif role == "assistant":
            if isinstance(content, str):
                lines.append(f"Assistant: {content}")
            elif isinstance(content, list):
                for item in content:
                    item_type = getattr(item, "type", None)
                    if item_type == "text":
                        lines.append(f"Assistant: {getattr(item, 'text', '')}")
                    elif item_type == "thinking":
                        thinking_text = getattr(item, "thinking", "")
                        lines.append(f"<thinking>{thinking_text}</thinking>")
                    elif item_type == "tool_use":
                        tool_name = getattr(item, "name", "")
                        args = getattr(item, "input", {})
                        try:
                            args_str = json.dumps(args)
                        except (TypeError, ValueError):
                            args_str = str(args)
                        lines.append(f"[Tool call: {tool_name}({args_str})]")

    return "\n".join(lines)
