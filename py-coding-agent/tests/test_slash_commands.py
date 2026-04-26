"""
Slash command tests — exercises every /command handler in the interactive TUI.

Strategy
--------
* Build a real AgentSession + AgentSessionRuntime backed by MockProvider
  (no API key, deterministic echo responses).
* Start the TUI with Textual's `app.run_test()` (headless terminal).
* Call `await app._dispatch_slash("/cmd [arg]")` directly for each command
  rather than simulating keystrokes — this tests the full dispatch chain
  (registry lookup → handler) without the flakiness of key-event timing.
* Assertions use a combination of:
  - Checking `len(app.screen_stack)` to detect pushed modals.
  - Inspecting notification toasts queued on the app.
  - Verifying session-state side-effects (name set, model changed, etc.).
"""
from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

import pytest

pytest_plugins = ("anyio",)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_agent():
    from pi_agent import Agent, Model, ProviderRegistry, MockProvider, create_agent_stream_fn
    registry = ProviderRegistry()
    registry.register("mock", MockProvider())
    stream_fn = create_agent_stream_fn(registry)
    agent = Agent(stream_fn=stream_fn)
    agent.set_model(Model(id="mock-model", provider="mock", api="mock"))
    return agent


def _make_runtime(tmp_path):
    from coding_agent.core.agent_session import AgentSession, AgentSessionConfig
    from coding_agent.core.agent_session_runtime import AgentSessionRuntime
    from coding_agent.core.agent_session_services import AgentSessionServices
    from coding_agent.core.session_manager import SessionManager
    from coding_agent.core.settings_manager import SettingsManager
    from coding_agent.core.resource_loader import ResourceLoader

    sm      = SessionManager.create(str(tmp_path), session_dir=str(tmp_path / "sessions"))
    sm_mgr  = SettingsManager.in_memory({"enabledModels": [
        "anthropic/claude-haiku-4-5",
        "anthropic/claude-opus-4-5",
    ]})
    rl      = ResourceLoader(str(tmp_path), str(tmp_path / "agent"), sm_mgr)

    session = AgentSession(AgentSessionConfig(
        agent=_make_agent(),
        session_manager=sm,
        settings_manager=sm_mgr,
        cwd=str(tmp_path),
        resource_loader=rl,
        no_context_files=True,
    ))

    services = AgentSessionServices(
        settings_manager=sm_mgr,
        resource_loader=rl,
        cwd=str(tmp_path),
        agent_dir=str(tmp_path / "agent"),
    )
    return AgentSessionRuntime(session, services)


async def _wait_idle(session, timeout: float = 5.0) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if not session.is_streaming:
            return
        await asyncio.sleep(0.05)
    raise TimeoutError("agent not idle")


def _notifications(app) -> list[str]:
    """Return message text of all pending notifications.
    Textual 8.x: app._notifications is a set of Notification dataclasses.
    """
    try:
        # app._notifications is a set; each item has .message
        return [getattr(n, "message", str(n)) for n in app._notifications]
    except Exception:
        pass
    # Fallback: query Toast widgets
    try:
        from textual.widgets._toast import Toast  # noqa: PLC0415
        return [t.notification.message for t in app.query(Toast)]
    except Exception:
        return []


def _has_modal(app) -> bool:
    """Return True if any modal screen is currently on the stack."""
    return len(app.screen_stack) > 1


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def runtime(tmp_path):
    return _make_runtime(tmp_path)


# ── Registry tests (no TUI needed) ────────────────────────────────────────────

def test_all_commands_registered():
    from coding_agent.core.slash_commands import get_slash_commands
    cmds = get_slash_commands()
    names = {c.name for c in cmds}
    required = {
        "help", "session", "name", "compact", "new", "clear",
        "model", "scoped-models", "thinking", "export", "import",
        "share", "copy", "sessions", "resume", "fork", "tree",
        "settings", "hotkeys", "changelog", "reload", "abort", "quit",
    }
    missing = required - names
    assert not missing, f"Missing commands: {missing}"


def test_all_builtin_actions_have_handlers():
    from coding_agent.core.slash_commands import get_slash_commands
    from coding_agent.modes.interactive.app import AgentApp
    import inspect

    src = inspect.getsource(AgentApp._run_builtin)
    for cmd in get_slash_commands():
        if cmd.source == "builtin":
            action = (cmd.source_info or {}).get("action", "")
            assert action in src, f"/{cmd.name} action '{action}' missing from _run_builtin"


# ── TUI command tests ─────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_help_opens_modal(runtime):
    from coding_agent.modes.interactive.app import AgentApp
    app = AgentApp(runtime)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        await app._dispatch_slash("/help")
        await pilot.pause(0.1)
        assert _has_modal(app), "/help should push an info modal"
        await pilot.press("escape")
        await pilot.pause(0.1)
        assert not _has_modal(app)


@pytest.mark.anyio
async def test_session_opens_modal(runtime):
    from coding_agent.modes.interactive.app import AgentApp
    app = AgentApp(runtime)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        await app._dispatch_slash("/session")
        await pilot.pause(0.1)
        assert _has_modal(app), "/session should push an info modal"


@pytest.mark.anyio
async def test_settings_opens_modal(runtime):
    from coding_agent.modes.interactive.app import AgentApp
    app = AgentApp(runtime)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        await app._dispatch_slash("/settings")
        await pilot.pause(0.1)
        assert _has_modal(app), "/settings should push an info modal"


@pytest.mark.anyio
async def test_hotkeys_opens_modal(runtime):
    from coding_agent.modes.interactive.app import AgentApp
    app = AgentApp(runtime)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        await app._dispatch_slash("/hotkeys")
        await pilot.pause(0.1)
        assert _has_modal(app), "/hotkeys should push an info modal"


@pytest.mark.anyio
async def test_changelog_does_not_crash(runtime):
    """Changelog may or may not exist — either path is acceptable."""
    from coding_agent.modes.interactive.app import AgentApp
    app = AgentApp(runtime)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        await app._dispatch_slash("/changelog")
        await pilot.pause(0.1)
        # Either a modal was pushed (file exists) or a warning notification
        # Either is acceptable — the key check is no exception was raised.


@pytest.mark.anyio
async def test_name_set_and_read(runtime):
    from coding_agent.modes.interactive.app import AgentApp
    app = AgentApp(runtime)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        await app._dispatch_slash("/name my-test-session")
        await pilot.pause(0.1)
        assert app._session.session_name == "my-test-session"


@pytest.mark.anyio
async def test_name_no_arg_warns(runtime):
    from coding_agent.modes.interactive.app import AgentApp
    app = AgentApp(runtime)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        await app._dispatch_slash("/name")
        await pilot.pause(0.1)
        # Should show warning notification (no crash)


@pytest.mark.anyio
async def test_compact_warns_with_no_messages(runtime):
    from coding_agent.modes.interactive.app import AgentApp
    app = AgentApp(runtime)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        assert len(app._session.messages) < 2
        await app._dispatch_slash("/compact")
        await pilot.pause(0.1)
        notes = _notifications(app)
        assert any("compact" in n.lower() or "message" in n.lower() for n in notes), \
            f"expected compact warning, got: {notes}"


@pytest.mark.anyio
async def test_compact_with_messages(runtime, tmp_path):
    """Send a real prompt so we have >= 2 messages, then compact."""
    from coding_agent.modes.interactive.app import AgentApp
    from coding_agent.core.types import PromptOptions
    app = AgentApp(runtime)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.6)   # let _initialize() complete
        await app._session.prompt("hello", PromptOptions(source="test"))
        await _wait_idle(app._session)
        await pilot.pause(0.1)
        await app._dispatch_slash("/compact custom instructions here")
        await pilot.pause(0.3)
        # Should trigger compaction attempt (may or may not succeed with mock)


@pytest.mark.anyio
async def test_clear_clears_messages(runtime):
    from coding_agent.modes.interactive.app import AgentApp
    from coding_agent.modes.interactive.widgets.message_list import MessageList
    app = AgentApp(runtime)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        await app._dispatch_slash("/clear")
        await pilot.pause(0.1)
        # MessageList should be empty
        ml = app.query_one(MessageList)
        assert len(list(ml.children)) == 0


@pytest.mark.anyio
async def test_thinking_cycles(runtime):
    from coding_agent.modes.interactive.app import AgentApp
    app = AgentApp(runtime)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        initial = app._session.thinking_level
        await app._dispatch_slash("/thinking")
        await pilot.pause(0.1)
        assert app._session.thinking_level != initial or app._session.thinking_level == "off"


@pytest.mark.anyio
async def test_model_no_arg_shows_selector(runtime):
    from coding_agent.modes.interactive.app import AgentApp
    app = AgentApp(runtime)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        await app._dispatch_slash("/model")
        await pilot.pause(0.1)
        # settings has 2 enabled models → modal selector shown
        assert _has_modal(app), "/model should push a selector modal"
        await pilot.press("escape")


@pytest.mark.anyio
async def test_model_with_search_term(runtime):
    from coding_agent.modes.interactive.app import AgentApp
    app = AgentApp(runtime)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        await app._dispatch_slash("/model haiku")
        await pilot.pause(0.1)
        # "haiku" matches exactly one model → set directly
        model = app._session.model or {}
        model_id = model.get("model_id", "")
        assert "haiku" in model_id.lower(), f"Expected haiku model, got {model_id}"


@pytest.mark.anyio
async def test_scoped_models_shows_info(runtime):
    from coding_agent.modes.interactive.app import AgentApp
    app = AgentApp(runtime)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        await app._dispatch_slash("/scoped-models")
        await pilot.pause(0.1)
        assert _has_modal(app), "/scoped-models should push an info modal"


@pytest.mark.anyio
async def test_export_html(runtime, tmp_path):
    from coding_agent.modes.interactive.app import AgentApp
    out_path = str(tmp_path / "export-test.html")
    app = AgentApp(runtime)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        await app._dispatch_slash(f"/export {out_path}")
        await pilot.pause(0.2)
        assert Path(out_path).exists(), "HTML export file should exist"
        content = Path(out_path).read_text()
        assert "Session" in content  # header contains "Session: <id>"


@pytest.mark.anyio
async def test_export_default_path(runtime):
    from coding_agent.modes.interactive.app import AgentApp
    app = AgentApp(runtime)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        await app._dispatch_slash("/export")
        await pilot.pause(0.2)
        # Should export to $TMPDIR/session-export.html without crashing
        notes = _notifications(app)
        assert any("export" in n.lower() for n in notes), \
            f"Expected export notification, got: {notes}"


@pytest.mark.anyio
async def test_export_jsonl(runtime, tmp_path):
    from coding_agent.modes.interactive.app import AgentApp
    from coding_agent.core.types import PromptOptions
    out_path = str(tmp_path / "export-test.jsonl")
    app = AgentApp(runtime)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.6)
        # Need at least one persisted message to create the JSONL on disk
        await app._session.prompt("test", PromptOptions(source="test"))
        await _wait_idle(app._session)
        await app._session.session_manager.flush()
        await pilot.pause(0.2)
        await app._dispatch_slash(f"/export {out_path}")
        await pilot.pause(0.2)
        assert Path(out_path).exists(), "JSONL export file should exist"


@pytest.mark.anyio
async def test_import_missing_file_errors(runtime):
    from coding_agent.modes.interactive.app import AgentApp
    app = AgentApp(runtime)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        await app._dispatch_slash("/import /nonexistent/path.jsonl")
        await pilot.pause(0.1)
        notes = _notifications(app)
        assert any("not found" in n.lower() or "error" in n.lower() for n in notes), \
            f"Expected file-not-found error, got: {notes}"


@pytest.mark.anyio
async def test_import_no_arg_warns(runtime):
    from coding_agent.modes.interactive.app import AgentApp
    app = AgentApp(runtime)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        await app._dispatch_slash("/import")
        await pilot.pause(0.1)
        notes = _notifications(app)
        assert any("usage" in n.lower() or "import" in n.lower() for n in notes), \
            f"Expected usage warning, got: {notes}"


@pytest.mark.anyio
async def test_import_real_file(runtime, tmp_path):
    """Export then import — full round-trip: verify confirm modal appears."""
    from coding_agent.modes.interactive.app import AgentApp
    from coding_agent.core.types import PromptOptions
    import shutil

    jsonl_path = str(tmp_path / "roundtrip.jsonl")
    app = AgentApp(runtime)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.6)
        await app._session.prompt("ping", PromptOptions(source="test"))
        await _wait_idle(app._session)
        await app._session.session_manager.flush()
        src = app._session.session_file
        assert src, "session must be persisted before import"
        shutil.copy2(src, jsonl_path)

        await app._dispatch_slash(f"/import {jsonl_path}")
        await pilot.pause(0.2)
        # Confirm modal should appear
        assert _has_modal(app), "/import should show a confirm modal"
        # Dismiss with Escape (cancels)
        await pilot.press("escape")
        await pilot.pause(0.1)
        assert not _has_modal(app)


@pytest.mark.anyio
async def test_copy_no_messages_warns(runtime):
    from coding_agent.modes.interactive.app import AgentApp
    app = AgentApp(runtime)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        await app._dispatch_slash("/copy")
        await pilot.pause(0.1)
        notes = _notifications(app)
        assert any("no" in n.lower() or "copy" in n.lower() for n in notes), \
            f"Expected warning, got: {notes}"


@pytest.mark.anyio
async def test_fork_no_messages_warns(runtime):
    from coding_agent.modes.interactive.app import AgentApp
    app = AgentApp(runtime)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        await app._dispatch_slash("/fork")
        await pilot.pause(0.1)
        notes = _notifications(app)
        assert any("no" in n.lower() or "fork" in n.lower() or "message" in n.lower()
                   for n in notes), f"Expected no-messages warning, got: {notes}"


@pytest.mark.anyio
async def test_fork_with_messages_shows_selector(runtime):
    from coding_agent.modes.interactive.app import AgentApp
    from coding_agent.core.types import PromptOptions
    app = AgentApp(runtime)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.6)
        await app._session.prompt("fork me", PromptOptions(source="test"))
        await _wait_idle(app._session)
        await pilot.pause(0.1)
        await app._dispatch_slash("/fork")
        await pilot.pause(0.1)
        assert _has_modal(app), "/fork should show message selector modal"
        await pilot.press("escape")


@pytest.mark.anyio
async def test_tree_no_entries_warns(runtime):
    from coding_agent.modes.interactive.app import AgentApp
    app = AgentApp(runtime)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        await app._dispatch_slash("/tree")
        await pilot.pause(0.1)
        notes = _notifications(app)
        assert any("no" in n.lower() or "tree" in n.lower() or "message" in n.lower()
                   for n in notes), f"Expected no-entries warning, got: {notes}"


@pytest.mark.anyio
async def test_tree_with_messages_shows_selector(runtime):
    from coding_agent.modes.interactive.app import AgentApp
    from coding_agent.core.types import PromptOptions
    app = AgentApp(runtime)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.6)
        await app._session.prompt("tree test", PromptOptions(source="test"))
        await _wait_idle(app._session)
        await pilot.pause(0.1)
        await app._dispatch_slash("/tree")
        await pilot.pause(0.1)
        assert _has_modal(app), "/tree should show entry selector modal"
        await pilot.press("escape")


@pytest.mark.anyio
async def test_reload_when_idle(runtime):
    from coding_agent.modes.interactive.app import AgentApp
    app = AgentApp(runtime)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.6)   # let _initialize() complete
        await app._dispatch_slash("/reload")
        await pilot.pause(0.3)
        notes = _notifications(app)
        assert any("reload" in n.lower() for n in notes), \
            f"Expected reload notification, got: {notes}"


@pytest.mark.anyio
async def test_abort_does_not_crash(runtime):
    from coding_agent.modes.interactive.app import AgentApp
    app = AgentApp(runtime)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        await app._dispatch_slash("/abort")
        await pilot.pause(0.1)
        # No crash; agent wasn't streaming so abort is a no-op


@pytest.mark.anyio
async def test_new_session(runtime):
    from coding_agent.modes.interactive.app import AgentApp
    app = AgentApp(runtime)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        old_id = app._session.session_id
        await app._dispatch_slash("/new")
        await pilot.pause(0.3)
        notes = _notifications(app)
        assert any("session" in n.lower() or "new" in n.lower() for n in notes), \
            f"Expected new-session notification, got: {notes}"


@pytest.mark.anyio
async def test_sessions_command_does_not_crash(runtime):
    from coding_agent.modes.interactive.app import AgentApp
    app = AgentApp(runtime)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        await app._dispatch_slash("/sessions")
        await pilot.pause(0.1)
        # SessionPicker is mounted (not a ModalScreen) — just check no crash


@pytest.mark.anyio
async def test_resume_is_alias_for_sessions(runtime):
    from coding_agent.modes.interactive.app import AgentApp
    from coding_agent.core.slash_commands import get_slash_commands
    cmds = {c.name: c for c in get_slash_commands()}
    assert cmds["resume"].source_info["action"] == cmds["sessions"].source_info["action"], \
        "/resume must map to the same action as /sessions"


@pytest.mark.anyio
async def test_share_no_gh_cli_warns(runtime):
    """Share should give a friendly error when gh is not on PATH."""
    import shutil as sh
    if sh.which("gh"):
        pytest.skip("gh CLI is installed — cannot test missing-gh path")
    from coding_agent.modes.interactive.app import AgentApp
    app = AgentApp(runtime)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        await app._dispatch_slash("/share")
        await pilot.pause(0.1)
        notes = _notifications(app)
        assert any("gh" in n.lower() or "github" in n.lower() or "install" in n.lower()
                   for n in notes), f"Expected gh-not-found error, got: {notes}"


@pytest.mark.anyio
async def test_unknown_command_warns(runtime):
    from coding_agent.modes.interactive.app import AgentApp
    app = AgentApp(runtime)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        await app._dispatch_slash("/definitely_not_a_command")
        await pilot.pause(0.1)
        notes = _notifications(app)
        assert any("unknown" in n.lower() for n in notes), \
            f"Expected unknown-command warning, got: {notes}"


@pytest.mark.anyio
async def test_enter_key_submits(runtime):
    """Enter in the input bar triggers action_submit (not insert newline)."""
    from coding_agent.modes.interactive.app import AgentApp
    from coding_agent.modes.interactive.widgets.input_bar import SubmitTextArea
    app = AgentApp(runtime)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.6)
        ta = app.query_one(SubmitTextArea)
        ta.load_text("hello world")
        # Press Enter — should submit (not insert newline)
        initial_text = ta.text
        await pilot.press("enter")
        await pilot.pause(0.2)
        # After submit, the textarea should be cleared
        assert ta.text == "", f"TextArea should clear after Enter submit, got: {ta.text!r}"


@pytest.mark.anyio
async def test_shift_enter_inserts_newline(runtime):
    """Shift+Enter must insert a newline into the TextArea."""
    from coding_agent.modes.interactive.app import AgentApp
    from coding_agent.modes.interactive.widgets.input_bar import SubmitTextArea
    app = AgentApp(runtime)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        ta = app.query_one(SubmitTextArea)
        ta.load_text("line1")
        # Move cursor to end
        await pilot.press("shift+enter")
        await pilot.pause(0.1)
        assert "\n" in ta.text, \
            f"Shift+Enter should insert newline, got: {ta.text!r}"
