"""Tests for coding_agent.core.types dataclasses."""
import pytest


def test_settings_defaults():
    from coding_agent.core.types import Settings
    s = Settings()
    assert s.default_thinking_level == "off"
    assert s.steering_mode == "all"
    assert s.enable_skill_commands is True
    assert isinstance(s.compaction.reserve_tokens, int)


def test_settings_compaction_defaults():
    from coding_agent.core.types import Settings, CompactionSettings
    s = Settings()
    assert isinstance(s.compaction, CompactionSettings)
    assert s.compaction.reserve_tokens == 16384
    assert s.compaction.keep_recent_tokens == 20000
    assert s.compaction.enabled is True


def test_settings_retry_defaults():
    from coding_agent.core.types import Settings, RetrySettings
    s = Settings()
    assert isinstance(s.retry, RetrySettings)
    assert s.retry.enabled is True
    assert s.retry.max_retries == 3


def test_settings_branch_summary_defaults():
    from coding_agent.core.types import Settings, BranchSummarySettings
    s = Settings()
    assert isinstance(s.branch_summary, BranchSummarySettings)
    assert s.branch_summary.reserve_tokens == 16384


def test_session_message_entry():
    from coding_agent.core.types import SessionMessageEntry
    e = SessionMessageEntry(id="abc", parent_id=None, timestamp="2024-01-01T00:00:00Z")
    assert e.type == "message"
    assert e.id == "abc"
    assert e.parent_id is None
    assert e.message is None


def test_model_change_entry():
    from coding_agent.core.types import ModelChangeEntry
    e = ModelChangeEntry(id="def", parent_id="abc", timestamp="2024-01-01T00:00:00Z",
                         provider="anthropic", model_id="claude-opus-4-5")
    assert e.type == "model_change"
    assert e.provider == "anthropic"
    assert e.model_id == "claude-opus-4-5"


def test_thinking_level_change_entry():
    from coding_agent.core.types import ThinkingLevelChangeEntry
    e = ThinkingLevelChangeEntry(id="x", parent_id=None, timestamp="2024-01-01T00:00:00Z",
                                  thinking_level="medium")
    assert e.type == "thinking_level_change"
    assert e.thinking_level == "medium"


def test_compaction_entry():
    from coding_agent.core.types import CompactionEntry
    e = CompactionEntry(id="c", parent_id="b", timestamp="2024-01-01T00:00:00Z",
                        summary="a summary", first_kept_entry_id="b", tokens_before=1000)
    assert e.type == "compaction"
    assert e.summary == "a summary"
    assert e.tokens_before == 1000
    assert e.from_hook is False


def test_branch_summary_entry():
    from coding_agent.core.types import BranchSummaryEntry
    e = BranchSummaryEntry(id="d", parent_id="c", timestamp="2024-01-01T00:00:00Z",
                            from_id="c", summary="branch")
    assert e.type == "branch_summary"
    assert e.from_id == "c"


def test_custom_entry():
    from coding_agent.core.types import CustomEntry
    e = CustomEntry(id="e", parent_id=None, timestamp="t", custom_type="my_type", data={"key": "val"})
    assert e.type == "custom"
    assert e.custom_type == "my_type"
    assert e.data == {"key": "val"}


def test_custom_message_entry():
    from coding_agent.core.types import CustomMessageEntry
    e = CustomMessageEntry(id="f", parent_id=None, timestamp="t",
                            custom_type="info", content="Hello!", display=True)
    assert e.type == "custom_message"
    assert e.content == "Hello!"
    assert e.display is True


def test_label_entry():
    from coding_agent.core.types import LabelEntry
    e = LabelEntry(id="g", parent_id=None, timestamp="t", target_id="abc", label="important")
    assert e.type == "label"
    assert e.target_id == "abc"
    assert e.label == "important"


def test_session_info_entry():
    from coding_agent.core.types import SessionInfoEntry
    e = SessionInfoEntry(id="h", parent_id=None, timestamp="t", name="My Session")
    assert e.type == "session_info"
    assert e.name == "My Session"


def test_bash_result():
    from coding_agent.core.types import BashResult
    r = BashResult(output="hello", exit_code=0)
    assert r.cancelled is False
    assert r.truncation is None
    assert r.full_output_path is None


def test_bash_result_with_all_fields():
    from coding_agent.core.types import BashResult, TruncationResult
    tr = TruncationResult(truncated_by="lines", total_lines=1000, output_lines=500)
    r = BashResult(output="partial", exit_code=1, cancelled=True, truncated=True, truncation=tr)
    assert r.cancelled is True
    assert r.truncated is True
    assert r.truncation.truncated_by == "lines"


def test_truncation_result():
    from coding_agent.core.types import TruncationResult
    tr = TruncationResult(truncated_by=None)
    assert tr.total_lines == 0
    assert tr.output_lines == 0
    assert tr.first_line_exceeds_limit is False


def test_skill():
    from coding_agent.core.types import Skill
    s = Skill(name="test-skill", path="/foo.md", content="# content",
              description="A skill", base_dir="/", source_info={})
    assert s.commands == []
    assert s.tags == []
    assert s.disable_model_invocation is False


def test_context_file():
    from coding_agent.core.types import ContextFile
    cf = ContextFile(path="/proj/AGENTS.md", content="Be helpful")
    assert cf.path == "/proj/AGENTS.md"
    assert cf.content == "Be helpful"


def test_file_operations_defaults():
    from coding_agent.core.types import FileOperations
    ops = FileOperations()
    assert ops.read == set()
    assert ops.written == set()
    assert ops.edited == set()


def test_agent_state_defaults():
    from coding_agent.core.types import AgentState
    s = AgentState()
    assert s.messages == []
    assert s.streaming_message is None
    assert isinstance(s.pending_tool_calls, frozenset)
    assert len(s.pending_tool_calls) == 0
    assert s.error_message is None


def test_prompt_options_defaults():
    from coding_agent.core.types import PromptOptions
    opts = PromptOptions()
    assert opts.images == []
    assert opts.expand_prompt_templates is True
    assert opts.streaming_behavior is None
    assert opts.source is None


def test_session_context():
    from coding_agent.core.types import SessionContext
    ctx = SessionContext(messages=[], thinking_level="medium",
                          model={"provider": "anthropic", "model_id": "claude-opus-4-5"})
    assert ctx.thinking_level == "medium"
    assert ctx.model["provider"] == "anthropic"


def test_session_header():
    from coding_agent.core.types import SessionHeader
    h = SessionHeader(id="abc", timestamp="2024-01-01T00:00:00Z", cwd="/proj")
    assert h.version == 3
    assert h.parent_session is None
