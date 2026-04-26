"""Tests for coding_agent.core.compaction.utils and compaction."""
import pytest


def test_extract_file_ops_empty():
    from coding_agent.core.compaction.utils import extract_file_ops_from_message

    class Msg:
        content = []

    ops = extract_file_ops_from_message(Msg())
    assert len(ops.read) == 0
    assert len(ops.written) == 0
    assert len(ops.edited) == 0


def test_extract_file_ops_from_tool_calls():
    from coding_agent.core.compaction.utils import extract_file_ops_from_message

    # utils.py checks getattr(item, "args", None) or getattr(item, "input", None)
    class ReadItem:
        name = "read"
        input = {"path": "/foo.py"}
        args = {"path": "/foo.py"}

    class WriteItem:
        name = "write"
        input = {"path": "/bar.py"}
        args = {"path": "/bar.py"}

    class EditItem:
        name = "edit"
        input = {"path": "/baz.py"}
        args = {"path": "/baz.py"}

    class Msg:
        content = [ReadItem(), WriteItem(), EditItem()]

    ops = extract_file_ops_from_message(Msg())
    assert "/foo.py" in ops.read
    assert "/bar.py" in ops.written
    assert "/baz.py" in ops.edited


def test_extract_file_ops_ignores_other_tools():
    from coding_agent.core.compaction.utils import extract_file_ops_from_message

    class BashItem:
        name = "bash"
        input = {"command": "ls -la"}
        args = {"command": "ls -la"}

    class Msg:
        content = [BashItem()]

    ops = extract_file_ops_from_message(Msg())
    assert len(ops.read) == 0
    assert len(ops.written) == 0
    assert len(ops.edited) == 0


def test_serialize_conversation():
    from coding_agent.core.compaction.utils import serialize_conversation

    class TextContent:
        def __init__(self, t):
            self.type = "text"
            self.text = t

    class UserMsg:
        role = "user"
        content = [TextContent("Hello")]

    class AsstMsg:
        role = "assistant"
        content = [TextContent("World")]

    out = serialize_conversation([UserMsg(), AsstMsg()])
    assert "Hello" in out
    assert "World" in out


def test_serialize_conversation_string_content():
    from coding_agent.core.compaction.utils import serialize_conversation

    class UserMsg:
        role = "user"
        content = "Simple string message"

    class AsstMsg:
        role = "assistant"
        content = "Simple string response"

    out = serialize_conversation([UserMsg(), AsstMsg()])
    assert "Simple string message" in out
    assert "Simple string response" in out


def test_serialize_conversation_tool_call():
    from coding_agent.core.compaction.utils import serialize_conversation

    class ToolUse:
        type = "tool_use"
        name = "bash"
        input = {"command": "ls"}

    class AsstMsg:
        role = "assistant"
        content = [ToolUse()]

    out = serialize_conversation([AsstMsg()])
    assert "bash" in out
    assert "Tool call" in out


def test_serialize_conversation_truncates_long_tool_results():
    from coding_agent.core.compaction.utils import serialize_conversation

    class ToolResult:
        type = "tool_result"
        content = "x" * 5000
        tool_use_id = "123"

    class UserMsg:
        role = "user"
        content = [ToolResult()]

    out = serialize_conversation([UserMsg()], max_chars_per_result=100)
    assert "truncated" in out
    assert len(out) < 5000


def test_format_file_operations():
    from coding_agent.core.compaction.utils import format_file_operations
    from coding_agent.core.types import FileOperations
    ops = FileOperations(read={"/a.py"}, written={"/b.py"}, edited={"/c.py"})
    xml = format_file_operations(ops)
    assert "read_files" in xml
    assert "/a.py" in xml
    assert "modified_files" in xml
    assert "/b.py" in xml
    assert "/c.py" in xml


def test_format_file_operations_empty():
    from coding_agent.core.compaction.utils import format_file_operations
    from coding_agent.core.types import FileOperations
    ops = FileOperations()
    xml = format_file_operations(ops)
    assert "read_files" in xml
    assert "modified_files" in xml


def test_compute_file_lists():
    from coding_agent.core.compaction.utils import compute_file_lists
    from coding_agent.core.types import FileOperations
    ops = FileOperations(read={"/a.py"}, written={"/b.py"}, edited={"/c.py"})
    result = compute_file_lists(ops)
    assert result["read"] == ["/a.py"]
    assert "/b.py" in result["modified"]
    assert "/c.py" in result["modified"]


def test_should_compact_true():
    from coding_agent.core.compaction.compaction import should_compact
    from coding_agent.core.types import CompactionSettings
    s = CompactionSettings(reserve_tokens=16384)
    assert should_compact(190_000, 200_000, s) is True   # 190k > 200k-16k = 184k


def test_should_compact_false():
    from coding_agent.core.compaction.compaction import should_compact
    from coding_agent.core.types import CompactionSettings
    s = CompactionSettings(reserve_tokens=16384)
    assert should_compact(100_000, 200_000, s) is False  # 100k < 184k


def test_should_compact_at_boundary():
    from coding_agent.core.compaction.compaction import should_compact
    from coding_agent.core.types import CompactionSettings
    s = CompactionSettings(reserve_tokens=16384)
    # Exactly at boundary: context_tokens == context_window - reserve_tokens → False
    assert should_compact(183_616, 200_000, s) is False
    # One over: True
    assert should_compact(183_617, 200_000, s) is True


def test_find_cut_point_empty():
    from coding_agent.core.compaction.compaction import find_cut_point
    result = find_cut_point([], keep_recent_tokens=1000)
    assert result is None


def test_find_cut_point_defaults_to_first(tmp_path):
    from coding_agent.core.compaction.compaction import find_cut_point
    from coding_agent.core.session_manager import SessionManager

    sm = SessionManager.create(str(tmp_path))
    e1 = sm.append_model_change("anthropic", "a")
    e2 = sm.append_thinking_level_change("low")

    # keep_recent_tokens very high → should return first entry
    result = find_cut_point(sm._entries, keep_recent_tokens=10_000_000)
    assert result == e1.id


def test_estimate_tokens():
    from coding_agent.core.compaction.compaction import estimate_tokens
    msg = "hello world"
    result = estimate_tokens(msg)
    assert isinstance(result, int)
    assert result > 0


def test_serialize_thinking_block():
    from coding_agent.core.compaction.utils import serialize_conversation

    class ThinkingBlock:
        type = "thinking"
        thinking = "Let me think about this..."

    class AsstMsg:
        role = "assistant"
        content = [ThinkingBlock()]

    out = serialize_conversation([AsstMsg()])
    assert "thinking" in out.lower()
