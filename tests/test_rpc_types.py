"""Tests for coding_agent.modes.rpc.rpc_types."""
import pytest


def test_all_commands_in_cmd_types():
    from coding_agent.modes.rpc.rpc_types import CMD_TYPES
    expected = {
        "prompt", "steer", "follow_up", "abort", "abort_retry",
        "new_session", "get_state", "set_model", "cycle_model",
        "set_thinking_level", "cycle_thinking_level", "set_steering_mode",
        "set_follow_up_mode", "compact", "set_auto_compaction", "set_auto_retry",
        "bash", "abort_bash", "get_session_stats", "export_html",
        "switch_session", "fork", "navigate_tree", "reload",
        "get_fork_messages", "get_last_assistant_text", "set_session_name",
        "get_messages", "get_commands", "import",
    }
    for name in expected:
        assert name in CMD_TYPES, f"Missing RPC command: {name}"


def test_prompt_cmd_defaults():
    from coding_agent.modes.rpc.rpc_types import PromptCmd
    cmd = PromptCmd(message="hello")
    assert cmd.type == "prompt"
    assert cmd.images == []
    assert cmd.streaming_behavior is None


def test_prompt_cmd_with_images():
    from coding_agent.modes.rpc.rpc_types import PromptCmd
    cmd = PromptCmd(message="look at this", images=["img1", "img2"])
    assert len(cmd.images) == 2


def test_steer_cmd_defaults():
    from coding_agent.modes.rpc.rpc_types import SteerCmd
    cmd = SteerCmd(message="steer me")
    assert cmd.type == "steer"
    assert cmd.images == []
    assert cmd.id is None


def test_follow_up_cmd():
    from coding_agent.modes.rpc.rpc_types import FollowUpCmd
    cmd = FollowUpCmd(message="follow up")
    assert cmd.type == "follow_up"


def test_abort_cmd():
    from coding_agent.modes.rpc.rpc_types import AbortCmd
    cmd = AbortCmd()
    assert cmd.type == "abort"
    assert cmd.id is None


def test_new_session_cmd():
    from coding_agent.modes.rpc.rpc_types import NewSessionCmd
    cmd = NewSessionCmd(parent_session="abc123")
    assert cmd.type == "new_session"
    assert cmd.parent_session == "abc123"


def test_set_model_cmd():
    from coding_agent.modes.rpc.rpc_types import SetModelCmd
    cmd = SetModelCmd(provider="anthropic", model_id="claude-opus-4-5")
    assert cmd.type == "set_model"
    assert cmd.provider == "anthropic"
    assert cmd.model_id == "claude-opus-4-5"


def test_set_thinking_level_cmd():
    from coding_agent.modes.rpc.rpc_types import SetThinkingLevelCmd
    cmd = SetThinkingLevelCmd(level="high")
    assert cmd.type == "set_thinking_level"
    assert cmd.level == "high"


def test_compact_cmd():
    from coding_agent.modes.rpc.rpc_types import CompactCmd
    cmd = CompactCmd(custom_instructions="Focus on files changed")
    assert cmd.type == "compact"
    assert cmd.custom_instructions == "Focus on files changed"


def test_set_auto_compaction_cmd():
    from coding_agent.modes.rpc.rpc_types import SetAutoCompactionCmd
    cmd = SetAutoCompactionCmd(enabled=False)
    assert cmd.type == "set_auto_compaction"
    assert cmd.enabled is False


def test_bash_cmd():
    from coding_agent.modes.rpc.rpc_types import BashCmd
    cmd = BashCmd(command="ls -la")
    assert cmd.type == "bash"
    assert cmd.command == "ls -la"


def test_export_html_cmd():
    from coding_agent.modes.rpc.rpc_types import ExportHtmlCmd
    cmd = ExportHtmlCmd(output_path="/tmp/out.html")
    assert cmd.type == "export_html"
    assert cmd.output_path == "/tmp/out.html"


def test_switch_session_cmd():
    from coding_agent.modes.rpc.rpc_types import SwitchSessionCmd
    cmd = SwitchSessionCmd(session_path="/sessions/abc.jsonl")
    assert cmd.type == "switch_session"


def test_fork_cmd():
    from coding_agent.modes.rpc.rpc_types import ForkCmd
    cmd = ForkCmd(entry_id="abc123")
    assert cmd.type == "fork"
    assert cmd.entry_id == "abc123"


def test_set_session_name_cmd():
    from coding_agent.modes.rpc.rpc_types import SetSessionNameCmd
    cmd = SetSessionNameCmd(name="My Session")
    assert cmd.type == "set_session_name"
    assert cmd.name == "My Session"


def test_import_cmd():
    from coding_agent.modes.rpc.rpc_types import ImportCmd
    cmd = ImportCmd(path="/some/file.md")
    assert cmd.type == "import"
    assert cmd.path == "/some/file.md"


def test_rpc_session_state_defaults():
    from coding_agent.modes.rpc.rpc_types import RpcSessionState
    state = RpcSessionState()
    assert state.thinking_level == "off"
    assert state.is_streaming is False
    assert state.message_count == 0
    assert state.session_id == ""
    assert state.model is None
    assert state.auto_compaction_enabled is True


def test_rpc_session_state_with_values():
    from coding_agent.modes.rpc.rpc_types import RpcSessionState
    state = RpcSessionState(
        session_id="abc",
        thinking_level="high",
        is_streaming=True,
        message_count=5,
    )
    assert state.session_id == "abc"
    assert state.thinking_level == "high"
    assert state.is_streaming is True
    assert state.message_count == 5


def test_rpc_success():
    from coding_agent.modes.rpc.rpc_types import RpcSuccess
    s = RpcSuccess(id="req-1", data={"result": "ok"})
    assert s.type == "success"
    assert s.id == "req-1"
    assert s.data == {"result": "ok"}


def test_rpc_error():
    from coding_agent.modes.rpc.rpc_types import RpcError
    e = RpcError(id="req-2", error="Something went wrong")
    assert e.type == "error"
    assert e.error == "Something went wrong"


def test_rpc_event():
    from coding_agent.modes.rpc.rpc_types import RpcEvent
    ev = RpcEvent(event="stream_start", data={"tokens": 100})
    assert ev.type == "event"
    assert ev.event == "stream_start"


def test_cmd_types_instantiable():
    from coding_agent.modes.rpc.rpc_types import CMD_TYPES
    for name, cls in CMD_TYPES.items():
        # Should be able to instantiate each command type
        instance = cls()
        assert instance.type == name
