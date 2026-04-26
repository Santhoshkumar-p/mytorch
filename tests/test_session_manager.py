"""Tests for coding_agent.core.session_manager.SessionManager."""
import pytest
import json
import asyncio
from pathlib import Path


def test_create_and_new_id(tmp_path):
    from coding_agent.core.session_manager import SessionManager
    sm = SessionManager.create(str(tmp_path))
    assert sm.get_session_id()
    assert sm.get_leaf_id() is None
    assert sm.is_persisted() is False


def test_session_id_is_uuid_like(tmp_path):
    from coding_agent.core.session_manager import SessionManager
    sm = SessionManager.create(str(tmp_path))
    sid = sm.get_session_id()
    assert len(sid) == 36  # UUID format
    assert "-" in sid


async def test_append_message_and_flush(tmp_path):
    from coding_agent.core.session_manager import SessionManager

    class FakeMsg:
        role = "user"
        content = []

    sm = SessionManager.create(str(tmp_path))
    entry = sm.append_message_entry(FakeMsg())
    assert sm.get_leaf_id() == entry.id
    await sm.flush()
    assert sm.get_session_file() is not None
    assert Path(sm.get_session_file()).exists()


async def test_flush_creates_session_dir(tmp_path):
    from coding_agent.core.session_manager import SessionManager
    session_dir = tmp_path / "mysessions"
    sm = SessionManager.create(str(tmp_path), session_dir=str(session_dir))
    sm.append_model_change("anthropic", "claude-opus-4-5")
    await sm.flush()
    assert session_dir.exists()


def test_build_session_context_empty(tmp_path):
    from coding_agent.core.session_manager import SessionManager
    sm = SessionManager.create(str(tmp_path))
    ctx = sm.build_session_context()
    assert ctx.messages == []
    assert ctx.thinking_level == "off"


async def test_open_written_session(tmp_path):
    from coding_agent.core.session_manager import SessionManager
    sm = SessionManager.create(str(tmp_path))
    sm.append_model_change("anthropic", "claude-opus-4-5")
    sm.append_thinking_level_change("medium")
    sm.append_session_info("My Test Session")
    await sm.flush()
    path = sm.get_session_file()

    sm2 = SessionManager.open(path)
    ctx = sm2.build_session_context()
    assert ctx.thinking_level == "medium"
    assert ctx.model == {"provider": "anthropic", "model_id": "claude-opus-4-5"}
    assert sm2.get_session_name() == "My Test Session"


async def test_open_session_has_same_id(tmp_path):
    from coding_agent.core.session_manager import SessionManager
    sm = SessionManager.create(str(tmp_path))
    await sm.flush()
    orig_id = sm.get_session_id()
    path = sm.get_session_file()

    sm2 = SessionManager.open(path)
    assert sm2.get_session_id() == orig_id


def test_get_branch(tmp_path):
    from coding_agent.core.session_manager import SessionManager
    sm = SessionManager.create(str(tmp_path))
    e1 = sm.append_model_change("anthropic", "claude-opus-4-5")
    e2 = sm.append_thinking_level_change("low")
    e3 = sm.append_session_info("test")
    branch = sm.get_branch(e3.id)
    assert [e.id for e in branch] == [e1.id, e2.id, e3.id]


def test_get_branch_single_entry(tmp_path):
    from coding_agent.core.session_manager import SessionManager
    sm = SessionManager.create(str(tmp_path))
    e1 = sm.append_model_change("anthropic", "claude-opus-4-5")
    branch = sm.get_branch(e1.id)
    assert len(branch) == 1
    assert branch[0].id == e1.id


async def test_create_branched_session(tmp_path):
    from coding_agent.core.session_manager import SessionManager
    sm = SessionManager.create(str(tmp_path))
    sm.append_model_change("anthropic", "claude-opus-4-5")
    e2 = sm.append_thinking_level_change("low")
    sm.append_session_info("later")
    await sm.flush()
    new_path = sm.create_branched_session(e2.id)
    assert new_path is not None
    sm2 = SessionManager.open(new_path)
    ctx = sm2.build_session_context()
    assert ctx.thinking_level == "low"
    # "later" info was after the branch point, shouldn't be in new session
    assert sm2.get_session_name() is None


async def test_create_branched_session_preserves_parent_session(tmp_path):
    from coding_agent.core.session_manager import SessionManager
    sm = SessionManager.create(str(tmp_path))
    e1 = sm.append_model_change("anthropic", "claude-opus-4-5")
    await sm.flush()
    new_path = sm.create_branched_session(e1.id)
    sm2 = SessionManager.open(new_path)
    # Parent session ID should be original session
    assert sm2._header.parent_session == sm.get_session_id()


def test_compaction_cutoff(tmp_path):
    from coding_agent.core.session_manager import SessionManager
    sm = SessionManager.create(str(tmp_path))
    e1 = sm.append_model_change("anthropic", "a")
    e2 = sm.append_thinking_level_change("high")
    e3 = sm.append_compaction("summary", first_kept_entry_id=e2.id, tokens_before=100)
    ctx = sm.build_session_context()
    # Context should start from e2, not e1
    assert ctx.thinking_level == "high"


def test_compaction_excludes_entries_before_cutoff(tmp_path):
    from coding_agent.core.session_manager import SessionManager
    sm = SessionManager.create(str(tmp_path))
    sm.append_thinking_level_change("low")   # e1 - should be excluded
    e2 = sm.append_model_change("anthropic", "claude-opus-4-5")    # e2 - kept
    sm.append_compaction("summary", first_kept_entry_id=e2.id, tokens_before=50)
    ctx = sm.build_session_context()
    # thinking_level from e1 should be gone — only model from e2 should show
    assert ctx.model == {"provider": "anthropic", "model_id": "claude-opus-4-5"}


def test_get_label(tmp_path):
    from coding_agent.core.session_manager import SessionManager
    sm = SessionManager.create(str(tmp_path))
    e1 = sm.append_model_change("anthropic", "a")
    sm.append_label_change(e1.id, "important")
    assert sm.get_label(e1.id) == "important"
    sm.append_label_change(e1.id, None)   # clear label
    assert sm.get_label(e1.id) is None


def test_get_session_name_none_when_empty(tmp_path):
    from coding_agent.core.session_manager import SessionManager
    sm = SessionManager.create(str(tmp_path))
    assert sm.get_session_name() is None


def test_get_session_name_last_one_wins(tmp_path):
    from coding_agent.core.session_manager import SessionManager
    sm = SessionManager.create(str(tmp_path))
    sm.append_session_info("First")
    sm.append_session_info("Second")
    assert sm.get_session_name() == "Second"


async def test_list_all(tmp_path):
    from coding_agent.core.session_manager import SessionManager
    session_dir = str(tmp_path / "sessions")
    sm1 = SessionManager.create(str(tmp_path), session_dir=session_dir)
    await sm1.flush()
    sm2 = SessionManager.create(str(tmp_path), session_dir=session_dir)
    await sm2.flush()
    files = SessionManager.list_all(session_dir)
    assert len(files) == 2


def test_list_all_empty_dir(tmp_path):
    from coding_agent.core.session_manager import SessionManager
    files = SessionManager.list_all(str(tmp_path / "nosessions"))
    assert files == []


def test_new_session_resets_state(tmp_path):
    from coding_agent.core.session_manager import SessionManager
    sm = SessionManager.create(str(tmp_path))
    sm.append_model_change("anthropic", "claude-opus-4-5")
    old_id = sm.get_session_id()
    sm.new_session()
    assert sm.get_session_id() != old_id
    assert sm.get_leaf_id() is None
    assert sm.is_persisted() is False


def test_get_children(tmp_path):
    from coding_agent.core.session_manager import SessionManager
    sm = SessionManager.create(str(tmp_path))
    e1 = sm.append_model_change("anthropic", "claude-opus-4-5")
    e2 = sm.append_thinking_level_change("low")
    # e2's parent is e1
    children_of_e1 = sm.get_children(e1.id)
    assert any(c.id == e2.id for c in children_of_e1)


def test_update_entry(tmp_path):
    from coding_agent.core.session_manager import SessionManager
    from coding_agent.core.types import ModelChangeEntry
    sm = SessionManager.create(str(tmp_path))
    e1 = sm.append_model_change("anthropic", "claude-opus-4-5")
    updated = ModelChangeEntry(id=e1.id, parent_id=e1.parent_id,
                                timestamp=e1.timestamp, provider="openai", model_id="gpt-4o")
    sm.update_entry(updated)
    retrieved = sm.get_entry(e1.id)
    assert retrieved.provider == "openai"


def test_custom_entry(tmp_path):
    from coding_agent.core.session_manager import SessionManager
    sm = SessionManager.create(str(tmp_path))
    e = sm.append_custom_entry("my_event", data={"key": "value"})
    assert e.custom_type == "my_event"
    assert e.data == {"key": "value"}


def test_custom_message_entry(tmp_path):
    from coding_agent.core.session_manager import SessionManager
    sm = SessionManager.create(str(tmp_path))
    e = sm.append_custom_message_entry("info", "System message", display=True)
    ctx = sm.build_session_context()
    # Custom messages with display=True show in context
    assert "System message" in ctx.messages


def test_custom_message_entry_hidden(tmp_path):
    from coding_agent.core.session_manager import SessionManager
    sm = SessionManager.create(str(tmp_path))
    sm.append_custom_message_entry("hidden", "Hidden message", display=False)
    ctx = sm.build_session_context()
    assert "Hidden message" not in ctx.messages


def test_branch_summary_entry(tmp_path):
    from coding_agent.core.session_manager import SessionManager
    sm = SessionManager.create(str(tmp_path))
    e1 = sm.append_model_change("anthropic", "claude-opus-4-5")
    e = sm.append_branch_summary(from_id=e1.id, summary="Branch summary")
    assert e.from_id == e1.id
    assert e.summary == "Branch summary"


def test_reset_leaf(tmp_path):
    from coding_agent.core.session_manager import SessionManager
    sm = SessionManager.create(str(tmp_path))
    sm.append_model_change("anthropic", "claude-opus-4-5")
    assert sm.get_leaf_id() is not None
    sm.reset_leaf()
    assert sm.get_leaf_id() is None
