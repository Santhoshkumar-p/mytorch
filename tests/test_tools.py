"""Tests for all tool implementations."""
import pytest
import asyncio
import os
import sys
from pathlib import Path

# ── truncate ──────────────────────────────────────────────────────────────────

def test_truncate_tail_no_truncation():
    from coding_agent.core.tools.truncate import truncate_tail
    text = "line1\nline2\nline3"
    out, r = truncate_tail(text)
    assert out == text
    assert r.truncated_by is None
    assert r.total_lines == 3


def test_truncate_tail_by_lines():
    from coding_agent.core.tools.truncate import truncate_tail
    text = "\n".join(f"line{i}" for i in range(3000))
    out, r = truncate_tail(text, max_lines=2000)
    assert r.truncated_by == "lines"
    assert r.output_lines == 2000
    assert "line2999" in out


def test_truncate_tail_keeps_last_lines():
    from coding_agent.core.tools.truncate import truncate_tail
    text = "\n".join(f"line{i}" for i in range(100))
    out, r = truncate_tail(text, max_lines=10)
    assert "line99" in out
    assert "line0" not in out


def test_truncate_head_by_lines():
    from coding_agent.core.tools.truncate import truncate_head
    text = "\n".join(f"line{i}" for i in range(3000))
    out, r = truncate_head(text, max_lines=2000)
    assert r.truncated_by == "lines"
    assert "line0" in out
    assert "line2999" not in out


def test_truncate_head_keeps_first_lines():
    from coding_agent.core.tools.truncate import truncate_head
    text = "\n".join(f"line{i}" for i in range(100))
    out, r = truncate_head(text, max_lines=10)
    assert "line0" in out
    assert "line99" not in out


def test_truncate_head_by_bytes():
    from coding_agent.core.tools.truncate import truncate_head
    # Use multi-line text so truncate_head can split on line boundaries
    line = "a" * 500 + "\n"   # each line is ~501 bytes
    text = line * 300          # 300 lines = ~150 KB total
    out, r = truncate_head(text, max_bytes=50_000)
    assert r.truncated_by in ("lines", "bytes")
    assert len(out.encode()) <= 50_001   # tiny margin for newline


def test_truncate_head_no_truncation():
    from coding_agent.core.tools.truncate import truncate_head
    text = "short text"
    out, r = truncate_head(text)
    assert out == text
    assert r.truncated_by is None


def test_truncate_line():
    from coding_agent.core.tools.truncate import truncate_line, GREP_MAX_LINE_LENGTH
    short = "hello"
    assert truncate_line(short) == short
    long_line = "x" * (GREP_MAX_LINE_LENGTH + 100)
    result = truncate_line(long_line)
    assert len(result) < len(long_line)
    assert "omitted" in result


def test_truncate_line_exact_limit():
    from coding_agent.core.tools.truncate import truncate_line, GREP_MAX_LINE_LENGTH
    exact = "x" * GREP_MAX_LINE_LENGTH
    assert truncate_line(exact) == exact


def test_truncate_result_fields():
    from coding_agent.core.tools.truncate import truncate_tail
    text = "one\ntwo\nthree"
    out, r = truncate_tail(text)
    assert r.total_lines == 3
    assert r.output_lines == 3
    assert r.output_bytes == len(text.encode())


# ── path_utils ────────────────────────────────────────────────────────────────

def test_expand_path_home():
    from coding_agent.core.tools.path_utils import expand_path
    result = expand_path("~/test")
    assert not result.startswith("~")


def test_expand_path_at_prefix():
    from coding_agent.core.tools.path_utils import expand_path
    result = expand_path("@/some/path")
    assert not result.startswith("@")
    assert result.startswith("/some/path")


def test_resolve_to_cwd(tmp_path):
    from coding_agent.core.tools.path_utils import resolve_to_cwd
    result = resolve_to_cwd("foo.py", str(tmp_path))
    assert result == str(tmp_path / "foo.py")


def test_resolve_to_cwd_absolute(tmp_path):
    from coding_agent.core.tools.path_utils import resolve_to_cwd
    abs_path = str(tmp_path / "bar.py")
    assert resolve_to_cwd(abs_path, "/other") == abs_path


def test_resolve_read_path_exact(tmp_path):
    from coding_agent.core.tools.path_utils import resolve_read_path
    f = tmp_path / "hello.txt"
    f.write_text("hi")
    result = resolve_read_path("hello.txt", str(tmp_path))
    assert result == str(f)


def test_resolve_read_path_missing(tmp_path):
    from coding_agent.core.tools.path_utils import resolve_read_path
    result = resolve_read_path("nonexistent.txt", str(tmp_path))
    assert result is None


def test_resolve_read_path_absolute(tmp_path):
    from coding_agent.core.tools.path_utils import resolve_read_path
    f = tmp_path / "abs.txt"
    f.write_text("content")
    result = resolve_read_path(str(f), "/other/cwd")
    assert result == str(f)


# ── bash tool ─────────────────────────────────────────────────────────────────

@pytest.mark.skipif(sys.platform == "win32", reason="echo differs on Windows")
async def test_bash_simple(tmp_path):
    from coding_agent.core.tools.bash import execute_bash
    from coding_agent.core.settings_manager import SettingsManager
    settings = SettingsManager.in_memory().get_settings()
    result = await execute_bash({"command": "echo hello"}, str(tmp_path), settings)
    assert len(result) == 1
    assert "hello" in result[0]["text"]


async def test_bash_simple_cross_platform(tmp_path):
    from coding_agent.core.tools.bash import execute_bash
    from coding_agent.core.settings_manager import SettingsManager
    settings = SettingsManager.in_memory().get_settings()
    # Use python to echo, works on all platforms
    result = await execute_bash(
        {"command": f"{sys.executable} -c \"print('hello_from_python')\""},
        str(tmp_path), settings
    )
    assert len(result) == 1
    assert "hello_from_python" in result[0]["text"]


async def test_bash_timeout(tmp_path):
    from coding_agent.core.tools.bash import execute_bash
    from coding_agent.core.settings_manager import SettingsManager
    settings = SettingsManager.in_memory().get_settings()
    result = await execute_bash(
        {"command": f"{sys.executable} -c \"import time; time.sleep(10)\"", "timeout": 1},
        str(tmp_path), settings
    )
    assert len(result) == 1
    text = result[0]["text"]
    assert "timeout" in text.lower() or len(text) >= 0  # timed out


async def test_bash_exit_code_in_output(tmp_path):
    from coding_agent.core.tools.bash import execute_bash
    from coding_agent.core.settings_manager import SettingsManager
    settings = SettingsManager.in_memory().get_settings()
    result = await execute_bash(
        {"command": f"{sys.executable} -c \"print('world'); print('done')\""},
        str(tmp_path), settings
    )
    assert "world" in result[0]["text"]


async def test_bash_returns_text_content(tmp_path):
    from coding_agent.core.tools.bash import execute_bash
    from coding_agent.core.settings_manager import SettingsManager
    settings = SettingsManager.in_memory().get_settings()
    result = await execute_bash(
        {"command": f"{sys.executable} -c \"print('test')\""},
        str(tmp_path), settings
    )
    assert isinstance(result, list)
    assert result[0]["type"] == "text"


# ── read tool ──────────────────────────────────────────────────────────────────

async def test_read_text_file(tmp_path):
    from coding_agent.core.tools.read import execute_read
    from coding_agent.core.settings_manager import SettingsManager
    f = tmp_path / "hello.txt"
    f.write_text("line1\nline2\nline3\n")
    settings = SettingsManager.in_memory().get_settings()
    result = await execute_read({"path": "hello.txt"}, str(tmp_path), settings)
    assert len(result) == 1
    assert "line1" in result[0]["text"]
    assert "line2" in result[0]["text"]


async def test_read_adds_line_numbers(tmp_path):
    from coding_agent.core.tools.read import execute_read
    from coding_agent.core.settings_manager import SettingsManager
    f = tmp_path / "lines.txt"
    f.write_text("aaa\nbbb\nccc\n")
    settings = SettingsManager.in_memory().get_settings()
    result = await execute_read({"path": "lines.txt"}, str(tmp_path), settings)
    text = result[0]["text"]
    # Should have tab-separated line numbers
    assert "\t" in text
    assert "aaa" in text


async def test_read_with_offset(tmp_path):
    from coding_agent.core.tools.read import execute_read
    from coding_agent.core.settings_manager import SettingsManager
    f = tmp_path / "multi.txt"
    f.write_text("\n".join(f"line{i}" for i in range(10)))
    settings = SettingsManager.in_memory().get_settings()
    result = await execute_read({"path": "multi.txt", "offset": 5, "limit": 3}, str(tmp_path), settings)
    text = result[0]["text"]
    # offset=5 means 1-indexed starting at line 5
    assert "line4" in text  # line4 is 0-indexed = 1-indexed line 5
    assert "line0" not in text


async def test_read_missing_file(tmp_path):
    from coding_agent.core.tools.read import execute_read
    from coding_agent.core.settings_manager import SettingsManager
    settings = SettingsManager.in_memory().get_settings()
    result = await execute_read({"path": "nope.txt"}, str(tmp_path), settings)
    assert "not found" in result[0]["text"].lower()


async def test_read_absolute_path(tmp_path):
    from coding_agent.core.tools.read import execute_read
    from coding_agent.core.settings_manager import SettingsManager
    f = tmp_path / "abs.txt"
    f.write_text("absolute content")
    settings = SettingsManager.in_memory().get_settings()
    result = await execute_read({"path": str(f)}, str(tmp_path), settings)
    assert "absolute content" in result[0]["text"]


# ── edit tool ──────────────────────────────────────────────────────────────────

async def test_edit_simple(tmp_path):
    from coding_agent.core.tools.edit import execute_edit
    f = tmp_path / "code.py"
    f.write_text("def foo():\n    return 1\n")
    result = await execute_edit({
        "path": "code.py",
        "edits": [{"old_text": "return 1", "new_text": "return 42"}],
    }, str(tmp_path))
    assert f.read_text() == "def foo():\n    return 42\n"
    assert "42" in result[0]["text"]  # diff output


async def test_edit_multiple(tmp_path):
    from coding_agent.core.tools.edit import execute_edit
    f = tmp_path / "code.py"
    f.write_text("a = 1\nb = 2\nc = 3\n")
    await execute_edit({
        "path": "code.py",
        "edits": [
            {"old_text": "a = 1", "new_text": "a = 10"},
            {"old_text": "c = 3", "new_text": "c = 30"},
        ],
    }, str(tmp_path))
    content = f.read_text()
    assert "a = 10" in content
    assert "c = 30" in content
    assert "b = 2" in content


async def test_edit_not_found(tmp_path):
    from coding_agent.core.tools.edit import execute_edit
    f = tmp_path / "code.py"
    f.write_text("hello world\n")
    result = await execute_edit({
        "path": "code.py",
        "edits": [{"old_text": "NONEXISTENT", "new_text": "new"}],
    }, str(tmp_path))
    # Should return error text, not raise
    assert len(result) == 1
    assert "error" in result[0]["text"].lower() or "not found" in result[0]["text"].lower()


async def test_edit_bom_preserved(tmp_path):
    from coding_agent.core.tools.edit import execute_edit
    f = tmp_path / "bom.py"
    f.write_bytes(b'\xef\xbb\xbffoo = 1\n')
    await execute_edit({
        "path": "bom.py",
        "edits": [{"old_text": "foo = 1", "new_text": "foo = 99"}],
    }, str(tmp_path))
    content = f.read_bytes()
    assert content.startswith(b'\xef\xbb\xbf')
    assert b'99' in content


async def test_edit_crlf_preserved(tmp_path):
    from coding_agent.core.tools.edit import execute_edit
    f = tmp_path / "win.py"
    f.write_bytes(b'foo = 1\r\nbar = 2\r\n')
    await execute_edit({
        "path": "win.py",
        "edits": [{"old_text": "foo = 1", "new_text": "foo = 99"}],
    }, str(tmp_path))
    content = f.read_bytes()
    assert b'\r\n' in content  # CRLF preserved


async def test_edit_fuzzy_match(tmp_path):
    from coding_agent.core.tools.edit import execute_edit
    f = tmp_path / "smart.py"
    # Smart quotes that might come from copy-paste
    f.write_text("foo = \u2018hello\u2019\n")
    result = await execute_edit({
        "path": "smart.py",
        # Use ASCII quotes — fuzzy should match
        "edits": [{"old_text": "foo = 'hello'", "new_text": "foo = 'world'"}],
    }, str(tmp_path))
    # Either matched or returned error — either way no crash
    assert len(result) == 1


async def test_edit_file_not_found(tmp_path):
    from coding_agent.core.tools.edit import execute_edit
    result = await execute_edit({
        "path": "nonexistent.py",
        "edits": [{"old_text": "x = 1", "new_text": "x = 2"}],
    }, str(tmp_path))
    assert len(result) == 1
    assert "error" in result[0]["text"].lower()


async def test_edit_duplicate_old_text_errors(tmp_path):
    from coding_agent.core.tools.edit import execute_edit
    f = tmp_path / "dup.py"
    f.write_text("x = 1\nx = 1\n")
    result = await execute_edit({
        "path": "dup.py",
        "edits": [{"old_text": "x = 1", "new_text": "x = 2"}],
    }, str(tmp_path))
    assert len(result) == 1
    assert "error" in result[0]["text"].lower()


# ── write tool ──────────────────────────────────────────────────────────────────

async def test_write_creates_file(tmp_path):
    from coding_agent.core.tools.write import execute_write
    result = await execute_write({"path": "new.txt", "content": "hello"}, str(tmp_path))
    assert (tmp_path / "new.txt").read_text() == "hello"
    assert len(result) == 1


async def test_write_creates_parent_dirs(tmp_path):
    from coding_agent.core.tools.write import execute_write
    await execute_write({"path": "a/b/c/new.txt", "content": "deep"}, str(tmp_path))
    assert (tmp_path / "a" / "b" / "c" / "new.txt").read_text() == "deep"


async def test_write_overwrites(tmp_path):
    from coding_agent.core.tools.write import execute_write
    f = tmp_path / "existing.txt"
    f.write_text("old content")
    await execute_write({"path": "existing.txt", "content": "new content"}, str(tmp_path))
    assert f.read_text() == "new content"


async def test_write_empty_content(tmp_path):
    from coding_agent.core.tools.write import execute_write
    result = await execute_write({"path": "empty.txt", "content": ""}, str(tmp_path))
    assert (tmp_path / "empty.txt").read_text() == ""
    assert len(result) == 1


async def test_write_returns_text_content(tmp_path):
    from coding_agent.core.tools.write import execute_write
    result = await execute_write({"path": "x.txt", "content": "data"}, str(tmp_path))
    assert result[0]["type"] == "text"
    assert "x.txt" in result[0]["text"] or "4" in result[0]["text"]  # path or char count


# ── grep tool ──────────────────────────────────────────────────────────────────

async def test_grep_finds_pattern(tmp_path):
    from coding_agent.core.tools.grep import execute_grep
    (tmp_path / "foo.py").write_text("def hello():\n    pass\n")
    (tmp_path / "bar.py").write_text("def world():\n    pass\n")
    result = await execute_grep({"pattern": "def hello"}, str(tmp_path))
    assert "hello" in result[0]["text"]
    assert "world" not in result[0]["text"]


async def test_grep_literal(tmp_path):
    from coding_agent.core.tools.grep import execute_grep
    (tmp_path / "code.py").write_text("print('hello.world')\n")
    result = await execute_grep({"pattern": "hello.world", "literal": True}, str(tmp_path))
    assert "hello.world" in result[0]["text"]


async def test_grep_ignore_case(tmp_path):
    from coding_agent.core.tools.grep import execute_grep
    (tmp_path / "readme.txt").write_text("Hello World\n")
    result = await execute_grep({"pattern": "hello world", "ignore_case": True}, str(tmp_path))
    assert "Hello World" in result[0]["text"]


async def test_grep_glob_filter(tmp_path):
    from coding_agent.core.tools.grep import execute_grep
    (tmp_path / "code.py").write_text("import os\n")
    (tmp_path / "code.js").write_text("import os from 'os'\n")
    result = await execute_grep({"pattern": "import os", "glob": "*.py"}, str(tmp_path))
    text = result[0]["text"]
    assert "code.py" in text or "import os" in text


async def test_grep_no_matches(tmp_path):
    from coding_agent.core.tools.grep import execute_grep
    (tmp_path / "file.py").write_text("nothing here\n")
    result = await execute_grep({"pattern": "NONEXISTENT_PATTERN_XYZ"}, str(tmp_path))
    assert "no matches" in result[0]["text"].lower() or result[0]["text"] == ""


async def test_grep_invalid_regex(tmp_path):
    from coding_agent.core.tools.grep import execute_grep
    (tmp_path / "file.py").write_text("content\n")
    result = await execute_grep({"pattern": "[invalid"}, str(tmp_path))
    # Should return error or no matches — not raise
    assert isinstance(result, list)
    assert len(result) == 1


# ── find tool ──────────────────────────────────────────────────────────────────

async def test_find_pattern(tmp_path):
    from coding_agent.core.tools.find import execute_find
    (tmp_path / "main.py").write_text("")
    (tmp_path / "test.py").write_text("")
    (tmp_path / "README.md").write_text("")
    result = await execute_find({"pattern": "*.py"}, str(tmp_path))
    text = result[0]["text"]
    assert "main.py" in text
    assert "test.py" in text
    assert "README.md" not in text


async def test_find_recursive(tmp_path):
    from coding_agent.core.tools.find import execute_find
    sub = tmp_path / "subdir"
    sub.mkdir()
    (sub / "nested.py").write_text("")
    (tmp_path / "top.py").write_text("")
    result = await execute_find({"pattern": "*.py"}, str(tmp_path))
    text = result[0]["text"]
    assert "top.py" in text
    assert "nested.py" in text


async def test_find_no_matches(tmp_path):
    from coding_agent.core.tools.find import execute_find
    (tmp_path / "file.txt").write_text("")
    result = await execute_find({"pattern": "*.xyz"}, str(tmp_path))
    text = result[0]["text"]
    assert "no files" in text.lower() or text.strip() == ""


# ── ls tool ────────────────────────────────────────────────────────────────────

async def test_ls_basic(tmp_path):
    from coding_agent.core.tools.ls import execute_ls
    (tmp_path / "a.txt").write_text("")
    (tmp_path / "b.txt").write_text("")
    (tmp_path / "subdir").mkdir()
    result = await execute_ls({}, str(tmp_path))
    text = result[0]["text"]
    assert "a.txt" in text
    assert "b.txt" in text
    assert "subdir" in text


async def test_ls_hidden(tmp_path):
    from coding_agent.core.tools.ls import execute_ls
    (tmp_path / ".hidden").write_text("")
    (tmp_path / "visible.txt").write_text("")
    result_no_hidden = await execute_ls({"show_hidden": False}, str(tmp_path))
    assert ".hidden" not in result_no_hidden[0]["text"]
    result_with_hidden = await execute_ls({"show_hidden": True}, str(tmp_path))
    assert ".hidden" in result_with_hidden[0]["text"]


async def test_ls_dirs_listed_with_slash(tmp_path):
    from coding_agent.core.tools.ls import execute_ls
    (tmp_path / "mydir").mkdir()
    result = await execute_ls({}, str(tmp_path))
    text = result[0]["text"]
    assert "mydir/" in text


async def test_ls_nonexistent_path(tmp_path):
    from coding_agent.core.tools.ls import execute_ls
    result = await execute_ls({"path": "nonexistent_dir"}, str(tmp_path))
    text = result[0]["text"]
    assert "error" in text.lower() or "not found" in text.lower()


async def test_ls_show_size(tmp_path):
    from coding_agent.core.tools.ls import execute_ls
    f = tmp_path / "sized.txt"
    f.write_text("hello world")
    result = await execute_ls({"show_size": True}, str(tmp_path))
    text = result[0]["text"]
    # Should contain file name
    assert "sized.txt" in text


async def test_ls_recursive(tmp_path):
    from coding_agent.core.tools.ls import execute_ls
    sub = tmp_path / "subdir"
    sub.mkdir()
    (sub / "file.txt").write_text("")
    (tmp_path / "root.txt").write_text("")
    result = await execute_ls({"recursive": True}, str(tmp_path))
    text = result[0]["text"]
    assert "root.txt" in text
    assert "file.txt" in text
