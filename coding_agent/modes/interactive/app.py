from __future__ import annotations
import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical
    from textual.message import Message
    from textual.screen import ModalScreen
    from textual.widget import Widget
    from textual.widgets import Button, Input, Label, ListItem, ListView, Static
    from .widgets.message_list import MessageList
    from .widgets.input_bar import InputBar
    from .widgets.status_bar import StatusBar
    from .widgets.sidebar import Sidebar
    from .widgets.loader_bar import LoaderBar
    from .widgets.session_picker import SessionPicker
    from .theme import get_theme_vars
    from .keybindings import BINDINGS

    # ── Reusable modal screens ────────────────────────────────────────────────

    class _InfoScreen(ModalScreen):
        """Scrollable read-only info panel. Press Esc or Q to close."""
        CSS = """
        _InfoScreen { align: center middle; }
        #box {
            background: $surface; border: solid $primary;
            width: 90%; height: 80%; padding: 1 2;
        }
        #inf-title { color: $accent; text-style: bold;
                     border-bottom: solid $panel; padding-bottom: 1; }
        #inf-body  { height: 1fr; overflow-y: auto; color: $text; }
        #inf-foot  { height: 1; color: $text-muted; text-align: center; margin-top: 1; }
        """

        def __init__(self, title: str, content: str, **kwargs) -> None:
            super().__init__(**kwargs)
            self._title = title
            self._content = content

        def compose(self) -> ComposeResult:
            with Widget(id="box"):
                yield Static(self._title, id="inf-title")
                yield Static(self._content, id="inf-body")
                yield Static("Esc / Q  — close", id="inf-foot")

        def on_key(self, event) -> None:
            if event.key in ("escape", "q"):
                self.dismiss(None)

    class _SelectScreen(ModalScreen):
        """Pick one item from a list. Returns the selected string or None."""
        CSS = """
        _SelectScreen { align: center middle; }
        #box { background: $surface; border: solid $primary;
               width: 70%; height: auto; max-height: 22; padding: 1 2; }
        #sel-title { color: $accent; text-style: bold; margin-bottom: 1; }
        """

        def __init__(self, title: str, options: list[str], **kwargs) -> None:
            super().__init__(**kwargs)
            self._title = title
            self._options = options

        def compose(self) -> ComposeResult:
            with Widget(id="box"):
                yield Static(self._title, id="sel-title")
                # Store the option text in the ListItem's name (not id) so we
                # can retrieve it without touching Label.renderable (removed in
                # Textual 8) and without putting arbitrary strings in CSS ids.
                yield ListView(
                    *[ListItem(Label(o), name=o) for o in self._options],
                    id="choices",
                )

        def on_list_view_selected(self, event: ListView.Selected) -> None:
            # Retrieve via name (set above), fall back to Label text extraction
            value = getattr(event.item, "name", None)
            if value is None:
                try:
                    lbl = event.item.query_one(Label)
                    # Textual 8: Label stores content in _label (Text) or via render()
                    value = str(getattr(lbl, "_label", None) or lbl.render())
                except Exception:
                    value = ""
            self.dismiss(value or None)

        def on_key(self, event) -> None:
            if event.key == "escape":
                self.dismiss(None)

    class _ConfirmScreen(ModalScreen):
        """Yes/No confirmation dialog. Returns True/False."""
        CSS = """
        _ConfirmScreen { align: center middle; }
        #box { background: $surface; border: solid $primary;
               width: 55%; height: auto; padding: 1 2; }
        #conf-msg { margin-bottom: 1; }
        #conf-btns { layout: horizontal; height: auto; }
        """

        def __init__(self, title: str, message: str, **kwargs) -> None:
            super().__init__(**kwargs)
            self._title = title
            self._message = message

        def compose(self) -> ComposeResult:
            with Widget(id="box"):
                yield Static(f"[bold]{self._title}[/bold]\n{self._message}", id="conf-msg")
                with Widget(id="conf-btns"):
                    yield Button("Yes", id="yes", variant="success")
                    yield Button("No",  id="no",  variant="error")

        def on_button_pressed(self, event: Button.Pressed) -> None:
            self.dismiss(event.button.id == "yes")

        def on_key(self, event) -> None:
            if event.key == "escape":
                self.dismiss(False)

    class _InputScreen(ModalScreen):
        """Single-line text input dialog. Returns the entered string or None."""
        CSS = """
        _InputScreen { align: center middle; }
        #box { background: $surface; border: solid $primary;
               width: 60%; height: auto; padding: 1 2; }
        #inp-title { margin-bottom: 1; }
        """

        def __init__(self, title: str, placeholder: str = "", **kwargs) -> None:
            super().__init__(**kwargs)
            self._title = title
            self._placeholder = placeholder

        def compose(self) -> ComposeResult:
            with Widget(id="box"):
                yield Static(self._title, id="inp-title")
                yield Input(placeholder=self._placeholder, id="inp-field")

        def on_input_submitted(self, event: Input.Submitted) -> None:
            self.dismiss(event.value)

        def on_key(self, event) -> None:
            if event.key == "escape":
                self.dismiss(None)

    # ── Main app ──────────────────────────────────────────────────────────────

    class AgentApp(App):
        CSS = """
        Screen { layers: base overlay; background: #0d1117; }
        StatusBar { height: 1; dock: bottom; }
        #layout { height: 1fr; }
        #sidebar { width: 36; }
        #sidebar.hidden { display: none; }
        #chat-col { width: 1fr; }
        MessageList { height: 1fr; }
        LoaderBar { height: 1; }
        InputBar { height: auto; max-height: 12; }
        """
        BINDINGS = BINDINGS

        class _SessionEvent(Message):
            """Internal: agent event forwarded to the Textual event loop."""
            def __init__(self, event) -> None:
                super().__init__()
                self.session_event = event

        def __init__(self, runtime) -> None:
            super().__init__()
            self._runtime = runtime
            self._session = runtime.session
            self._unsub = None

        def compose(self) -> ComposeResult:
            yield StatusBar(id="status")
            with Horizontal(id="layout"):
                yield Sidebar(id="sidebar")
                with Vertical(id="chat-col"):
                    yield LoaderBar(id="loader")
                    yield MessageList(id="messages")
                    yield InputBar(id="input")

        def on_mount(self) -> None:
            self._unsub = self._session.subscribe(self._on_session_event)
            self.call_after_refresh(self._refresh_status)
            self.call_after_refresh(self._refresh_sidebar)

        def on_unmount(self) -> None:
            if self._unsub:
                self._unsub()

        # ── Session event routing ─────────────────────────────────────────────

        def _on_session_event(self, event) -> None:
            self.post_message(self._SessionEvent(event))

        def on_agent_app__session_event(self, message: "_SessionEvent") -> None:
            self._dispatch_event(message.session_event)

        def _dispatch_event(self, event) -> None:
            try:
                if not isinstance(event, dict):
                    return
                evt_type = event.get("type", "")

                # Forward message/tool events to the message list
                msg_list = self.query_one(MessageList)
                if evt_type in ("message_start", "message_update", "message_end"):
                    msg_list.handle_message_event(event)
                elif evt_type in (
                    "tool_execution_start", "tool_execution_update", "tool_execution_end",
                ):
                    msg_list.handle_tool_event(event)

                # Loader bar: show while streaming, hide when turn ends
                try:
                    loader = self.query_one(LoaderBar)
                    if evt_type in ("agent_start", "turn_start", "message_start"):
                        loader.set_active(True)
                    elif evt_type in ("agent_end", "turn_end", "message_end"):
                        # Only deactivate if agent is no longer streaming overall
                        if not self._session.is_streaming:
                            loader.set_active(False)
                except Exception:
                    pass

                # Status bar refresh on state-changing events
                if evt_type in (
                    "agent_start", "agent_end", "turn_start", "turn_end",
                    "message_end", "tool_execution_end",
                ):
                    self._refresh_status()

                # Sidebar refresh after initialization completes
                if evt_type == "session_initialized":
                    self._refresh_sidebar()
                    self._refresh_status()
            except Exception:
                pass

        def _refresh_status(self) -> None:
            try:
                self.query_one(StatusBar).refresh_state(self._session)
            except Exception:
                pass

        def _rewire_session(self) -> None:
            """Re-attach to the (possibly new) session after a fork/import/new."""
            if self._unsub:
                try:
                    self._unsub()
                except Exception:
                    pass
            self._session = self._runtime.session
            self._unsub = self._session.subscribe(self._on_session_event)
            try:
                msg_list = self.query_one(MessageList)
                msg_list.clear()
                # Replay historical messages so the conversation is visible
                try:
                    msg_list.load_history(self._session)
                    # Show brief feedback with entry count
                    sm = getattr(self._session, "_session_manager", None)
                    n = len(getattr(sm, "_entries", []))
                    if n > 0:
                        self.notify(
                            f"Resumed session \u2014 {n} entries loaded",
                            timeout=3,
                        )
                except Exception:
                    import logging
                    logging.getLogger(__name__).exception("_rewire_session: load_history failed")
            except Exception:
                pass
            # Stop the loader in case it was active
            try:
                self.query_one(LoaderBar).set_active(False)
            except Exception:
                pass
            self._refresh_status()
            self._refresh_sidebar()
            # Restore keyboard focus to the input bar
            try:
                self.query_one(InputBar).set_focus()
            except Exception:
                pass

        def _refresh_sidebar(self) -> None:
            try:
                self.query_one(Sidebar).refresh_sidebar(self._session)
            except Exception:
                pass

        def action_toggle_sidebar(self) -> None:
            try:
                self.query_one("#sidebar").toggle_class("hidden")
            except Exception:
                pass

        # ── Input bar submit (Enter key) ──────────────────────────────────────

        async def on_input_bar_submit(self, message) -> None:  # type: ignore[override]
            await self.action_submit()

        # ── Submission + slash dispatch ───────────────────────────────────────

        async def action_submit(self) -> None:
            input_bar = self.query_one(InputBar)
            text = input_bar.take_value()
            if not text.strip():
                return
            stripped = text.strip()
            if stripped.startswith("/"):
                await self._dispatch_slash(stripped)
                return

            # ── Steering vs. normal prompt ────────────────────────────────────
            # If the agent is currently streaming, send text as a steering
            # message so the agent can adjust its response mid-stream.
            # exclusive=True on _do_prompt would CANCEL the streaming worker,
            # so we branch here instead of letting exclusive resolve it.
            if self._session.is_streaming:
                self.run_worker(self._do_steer(stripped), exclusive=False)
            else:
                # Run prompt in a worker so the Textual event loop is free to
                # process _SessionEvent messages (streaming updates).
                self.run_worker(self._do_prompt(stripped), exclusive=True)

        async def _do_prompt(self, text: str) -> None:
            try:
                from coding_agent.core.types import PromptOptions
                await self._session.prompt(text, PromptOptions(source="interactive"))
            except Exception as e:
                self.notify(f"Error: {e}", severity="error")
            finally:
                # Always deactivate the loader when the prompt coroutine exits
                try:
                    self.query_one(LoaderBar).set_active(False)
                except Exception:
                    pass
                self._refresh_status()

        async def _do_steer(self, text: str) -> None:
            """Send a steering / follow-up message while the agent is running.

            Agent.steer() injects text into the current turn so the
            agent can adjust its response mid-stream.  fall_back to follow_up()
            if steer() is unavailable or raises.

            The MessageList detects the user message as a follow-up because
            _streaming_asst_keys is non-empty when message_start(user) fires.
            """
            try:
                await self._session.steer(text)
            except Exception:
                # steer() not available or failed — try follow_up()
                try:
                    await self._session.follow_up(text)
                except Exception as e:
                    self.notify(f"Steer/follow-up failed: {e}", severity="warning")

        async def _dispatch_slash(self, cmd_text: str) -> None:
            """Parse /name [arg] and route to the appropriate handler."""
            parts = cmd_text[1:].split(None, 1)
            name = parts[0].lower() if parts else ""
            arg  = parts[1].strip() if len(parts) > 1 else ""

            # Build command map from registry
            try:
                from coding_agent.core.slash_commands import get_slash_commands
                runner  = self._session._get_extension_runner() \
                          if hasattr(self._session, "_get_extension_runner") else None
                skills  = getattr(self._session, "_skills", [])
                prompts = getattr(self._session, "_prompt_templates", [])
                settings = self._session.settings_manager.get_settings()
                cmd_map  = {c.name: c for c in get_slash_commands(runner, prompts, skills, settings)}
            except Exception:
                cmd_map = {}

            cmd_obj = cmd_map.get(name)
            if cmd_obj is None:
                self.notify(
                    f"Unknown command: /{name}  — type /help for a list",
                    severity="warning",
                )
                return

            # Extension handler takes priority
            if cmd_obj.handler is not None:
                try:
                    result = cmd_obj.handler(arg)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    self.notify(f"Command error: {e}", severity="error")
                return

            source = getattr(cmd_obj, "source", "")
            if source == "builtin":
                action = (cmd_obj.source_info or {}).get("action", "")
                await self._run_builtin(action, arg)
            elif source == "prompt":
                await self._run_prompt_template(cmd_obj, arg)
            elif source == "skill":
                skill_name = (cmd_obj.source_info or {}).get("skill_name", name)
                await self._run_skill_command(skill_name, arg)
            else:
                self.notify(f"Unknown command: /{name}", severity="warning")

        # ── Built-in command dispatcher ───────────────────────────────────────

        async def _run_builtin(self, action: str, arg: str) -> None:
            handlers = {
                "show_help":           lambda: self._cmd_help(),
                "session_info":        lambda: self._cmd_session(),
                "session_name":        lambda: self._cmd_name(arg),
                "compact":             lambda: self._cmd_compact(arg),
                "new_session":         lambda: self.action_new_session(),
                "clear_messages":      lambda: self._clear_messages(),
                "select_model":        lambda: self._cmd_model(arg),
                "scoped_models":       lambda: self._cmd_scoped_models(),
                "cycle_thinking":      lambda: self.action_cycle_thinking(),
                "export":              lambda: self._cmd_export(arg),
                "import_session":      lambda: self._cmd_import(arg),
                "share":               lambda: self._cmd_share(),
                "copy":                lambda: self._cmd_copy(),
                "show_session_picker": lambda: self.action_show_session_picker(),
                "fork":                lambda: self._cmd_fork(),
                "tree":                lambda: self._cmd_tree(),
                "show_skills":         lambda: self._cmd_skills(),
                "settings":            lambda: self._cmd_settings(),
                "hotkeys":             lambda: self._cmd_hotkeys(),
                "changelog":           lambda: self._cmd_changelog(),
                "reload":              lambda: self._cmd_reload(),
                "abort_agent":         lambda: self.action_abort_agent(),
                "quit":                lambda: self._cmd_quit(),
            }
            fn = handlers.get(action)
            if fn is None:
                self.notify(f"Unhandled action: {action}", severity="warning")
                return
            result = fn()
            if asyncio.iscoroutine(result):
                await result

        # ── /help ─────────────────────────────────────────────────────────────

        def _cmd_help(self) -> None:
            try:
                from coding_agent.core.slash_commands import get_slash_commands
                runner  = self._session._get_extension_runner() \
                          if hasattr(self._session, "_get_extension_runner") else None
                skills  = getattr(self._session, "_skills", [])
                prompts = getattr(self._session, "_prompt_templates", [])
                settings = self._session.settings_manager.get_settings()
                cmds = get_slash_commands(runner, prompts, skills, settings)
                lines = []
                src_prev = None
                for c in cmds:
                    src = c.source
                    if src != src_prev:
                        lines.append(f"\n{'─' * 40}")
                        lines.append({"builtin": "Built-in", "extension": "Extensions",
                                      "prompt": "Prompt templates", "skill": "Skills"}.get(src, src.title()))
                        src_prev = src
                    desc = c.description or ""
                    lines.append(f"  /{c.name:<20} {desc}")
                lines.append(f"\n{'─' * 40}")
                lines.append("  Ctrl+J / Enter    Send message")
                lines.append("  Shift+Enter       New line in input")
                lines.append("  Ctrl+C            Abort streaming")
                lines.append("  PageUp/PageDown   Scroll messages")
                self._show_info("Help", "\n".join(lines))
            except Exception as e:
                self.notify(f"Help error: {e}", severity="error")

        # ── /session ──────────────────────────────────────────────────────────

        def _cmd_session(self) -> None:
            try:
                stats = self._session.get_session_stats()
                session_name = self._session.session_name

                # Accumulate token stats and tool counts from message history
                tool_calls = 0
                tool_results = 0
                inp = out = cr = cw = 0
                for m in self._session.messages:
                    role = getattr(m, "role", None)
                    if role == "assistant":
                        for c in getattr(m, "content", []):
                            if getattr(c, "type", None) == "toolCall":
                                tool_calls += 1
                        u = getattr(m, "usage", None)
                        if u:
                            inp += getattr(u, "input",       0) or 0
                            out += getattr(u, "output",      0) or 0
                            cr  += getattr(u, "cache_read",  0) or 0
                            cw  += getattr(u, "cache_write", 0) or 0
                    elif role == "toolResult":
                        tool_results += 1

                total_tok = inp + out + cr + cw
                lines = []
                if session_name:
                    lines.append(f"  Name         {session_name}")
                lines.append(f"  ID           {stats.session_id}")
                lines.append(f"  File         {stats.session_file or '(in-memory)'}")
                lines.append(f"\nMessages")
                lines.append(f"  User         {stats.user_messages}")
                lines.append(f"  Assistant    {stats.assistant_messages}")
                if tool_calls:
                    lines.append(f"  Tool calls   {tool_calls}")
                if tool_results:
                    lines.append(f"  Tool results {tool_results}")
                lines.append(f"  Total        {stats.total_messages}")
                if total_tok:
                    lines.append(f"\nTokens")
                    lines.append(f"  Input        {inp:,}")
                    lines.append(f"  Output       {out:,}")
                    if cr:  lines.append(f"  Cache read   {cr:,}")
                    if cw:  lines.append(f"  Cache write  {cw:,}")
                    lines.append(f"  Total        {total_tok:,}")
                self._show_info("Session", "\n".join(lines))
            except Exception as e:
                self.notify(f"Session info error: {e}", severity="error")

        # ── /name ─────────────────────────────────────────────────────────────

        def _cmd_name(self, arg: str) -> None:
            if not arg:
                current = self._session.session_name
                if current:
                    self.notify(f"Session name: {current}")
                else:
                    self.notify("No name set.  Usage: /name <name>", severity="warning")
                return
            self._session.set_session_name(arg)
            self.notify(f"Session name: {arg}")

        # ── /compact ──────────────────────────────────────────────────────────

        async def _cmd_compact(self, arg: str) -> None:
            msgs = self._session.messages
            if len(msgs) < 2:
                self.notify("Nothing to compact (need at least 2 messages).", severity="warning")
                return
            custom = arg.strip() or None
            self.notify("Compacting\u2026")
            try:
                await self._session.compact(custom)
                self.notify("Compacted \u2713")
            except Exception as e:
                self.notify(f"Compaction failed: {e}", severity="error")

        # ── /model ────────────────────────────────────────────────────────────

        async def _cmd_model(self, arg: str) -> None:
            settings = self._session.settings_manager.get_settings()
            models = list(settings.enabled_models or [])
            if not models:
                self.notify(
                    "No models in settings.enabled_models.\n"
                    "Add e.g. 'anthropic/claude-opus-4-5' to your settings.",
                    severity="warning",
                )
                return

            def _apply(chosen: str | None) -> None:
                if chosen:
                    self.run_worker(self._set_model_str(chosen), exclusive=False)

            if arg:
                matched = [m for m in models if arg.lower() in m.lower()]
                if not matched:
                    self.notify(f"No model matching '{arg}'", severity="warning")
                    return
                if len(matched) == 1:
                    await self._set_model_str(matched[0])
                    return
                self.push_screen(_SelectScreen(f"Models matching '{arg}'", matched), _apply)
            else:
                self.push_screen(_SelectScreen("Select model", models), _apply)

        async def _set_model_str(self, model_str: str) -> None:
            parts = model_str.split("/", 1)
            provider = parts[0] if len(parts) == 2 else "anthropic"
            model_id = parts[1] if len(parts) == 2 else parts[0]
            await self._session.set_model(provider, model_id)
            self._refresh_status()
            self.notify(f"Model: {model_str}")
            try:
                self.query_one(MessageList).add_audit_note(
                    f"Model \u2192 {model_str}"
                )
            except Exception:
                pass

        # ── /scoped-models ────────────────────────────────────────────────────

        def _cmd_scoped_models(self) -> None:
            settings = self._session.settings_manager.get_settings()
            models = list(settings.enabled_models or [])
            if not models:
                self.notify(
                    "settings.enabled_models is empty.\n"
                    "Add model strings like 'anthropic/claude-haiku-4-5' to cycle with Ctrl+M.",
                    severity="warning",
                    timeout=10,
                )
                return
            lines = ["Models available for Ctrl+M cycling:\n"]
            current = self._session.model or {}
            cur_str = f"{current.get('provider','')}/{current.get('model_id','')}".strip("/")
            for m in models:
                marker = " ◀ current" if m == cur_str else ""
                lines.append(f"  {m}{marker}")
            self._show_info("Scoped Models", "\n".join(lines))

        # ── /export ───────────────────────────────────────────────────────────

        async def _cmd_export(self, arg: str) -> None:
            try:
                if arg:
                    out_path = arg.strip()
                    if out_path.endswith(".jsonl"):
                        # JSONL export: flush then copy session file
                        await self._session.session_manager.flush()
                        src = self._session.session_file
                        if not src:
                            self.notify("Session not persisted yet.", severity="warning")
                            return
                        shutil.copy2(src, out_path)
                        self.notify(f"Session exported (JSONL): {out_path}", timeout=10)
                        return
                else:
                    out_path = os.path.join(tempfile.gettempdir(), "session-export.html")
                path = await self._session.export_to_html(out_path)
                self.notify(f"Exported (HTML): {path}", timeout=10)
            except Exception as e:
                self.notify(f"Export failed: {e}", severity="error")

        # ── /import ───────────────────────────────────────────────────────────

        async def _cmd_import(self, arg: str) -> None:
            if not arg:
                self.notify("Usage: /import <path.jsonl>", severity="warning")
                return
            path = arg.strip()
            if not os.path.exists(path):
                self.notify(f"File not found: {path}", severity="error")
                return

            def _on_confirmed(confirmed: bool) -> None:
                if not confirmed:
                    self.notify("Import cancelled.")
                    return
                self.run_worker(self._do_import(path), exclusive=False)

            self.push_screen(
                _ConfirmScreen("Import session", f"Replace current session with:\n  {path}"),
                _on_confirmed,
            )

        async def _do_import(self, path: str) -> None:
            try:
                result = await self._runtime.import_from_jsonl(path)
                if result.get("cancelled"):
                    self.notify("Import cancelled.")
                    return
                self._rewire_session()
                self.notify(f"Imported: {path} \u2713")
            except Exception as e:
                self.notify(f"Import failed: {e}", severity="error")

        # ── /share ────────────────────────────────────────────────────────────

        async def _cmd_share(self) -> None:
            # Verify gh CLI is available and logged in
            try:
                r = subprocess.run(["gh", "auth", "status"],
                                   capture_output=True, text=True)
                if r.returncode != 0:
                    self.notify(
                        "GitHub CLI is not logged in.\nRun:  gh auth login",
                        severity="error", timeout=10,
                    )
                    return
            except FileNotFoundError:
                self.notify(
                    "GitHub CLI (gh) is not installed.\nGet it from https://cli.github.com/",
                    severity="error", timeout=10,
                )
                return

            self.notify("Exporting and creating gist\u2026")
            tmp = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
            tmp.close()
            try:
                await self._session.export_to_html(tmp.name)
                r = subprocess.run(
                    ["gh", "gist", "create", "--public=false", tmp.name],
                    capture_output=True, text=True,
                )
                if r.returncode != 0:
                    self.notify(
                        f"Gist creation failed:\n{r.stderr.strip()[:200]}",
                        severity="error", timeout=10,
                    )
                    return
                gist_url = r.stdout.strip()
                self.notify(f"Shared \u2713\n{gist_url}", timeout=30)
            except Exception as e:
                self.notify(f"Share failed: {e}", severity="error")
            finally:
                try:
                    os.unlink(tmp.name)
                except Exception:
                    pass

        # ── /copy ─────────────────────────────────────────────────────────────

        def _cmd_copy(self) -> None:
            text = self._session.get_last_assistant_text()
            if not text:
                self.notify("No assistant messages to copy yet.", severity="warning")
                return
            try:
                _clipboard_write(text)
                self.notify("Copied last assistant message to clipboard \u2713")
            except Exception as e:
                self.notify(f"Copy failed: {e}", severity="error")

        # ── /fork ─────────────────────────────────────────────────────────────

        async def _cmd_fork(self) -> None:
            msgs = self._session.get_user_messages_for_forking()
            if not msgs:
                self.notify("No user messages to fork from.", severity="warning")
                return
            options = [f"[{m['entry_id'][:8]}\u2026]  {m['text'][:70]}" for m in msgs]

            def _on_chosen(chosen: str | None) -> None:
                if not chosen:
                    return
                try:
                    idx = options.index(chosen)
                    entry_id = msgs[idx]["entry_id"]
                    self.run_worker(self._do_fork(entry_id), exclusive=False)
                except (ValueError, IndexError):
                    pass

            self.push_screen(_SelectScreen("Fork from message", options), _on_chosen)

        async def _do_fork(self, entry_id: str) -> None:
            try:
                result = await self._runtime.fork(entry_id)
            except Exception as e:
                self.notify(f"Fork failed: {e}", severity="error")
                return
            if result.get("cancelled"):
                err = result.get("error", "")
                self.notify(
                    f"Fork cancelled.{' ' + err if err else ''}",
                    severity="warning",
                )
                return
            self._rewire_session()
            # Schedule an extra sidebar refresh once skills finish initializing
            self.set_timer(1.0, self._refresh_sidebar)
            self.notify("Forked session \u2713")

        # ── /tree ─────────────────────────────────────────────────────────────

        async def _cmd_tree(self) -> None:
            sm = self._session.session_manager
            msg_entries = [
                e for e in sm._entries
                if getattr(e, "type", "") == "message" and getattr(e, "message", None)
            ]
            if not msg_entries:
                self.notify("No messages in tree.", severity="warning")
                return
            recent = msg_entries[-30:]
            options, entry_ids = [], []
            for e in recent:
                msg = e.message
                role = getattr(msg, "role", "?")
                snippet = ""
                for c in getattr(msg, "content", []):
                    t = getattr(c, "text", None) or getattr(c, "thinking", None)
                    if t:
                        snippet = t[:60].replace("\n", " ")
                        break
                options.append(f"[{e.id[:8]}\u2026] {role}: {snippet}")
                entry_ids.append(e.id)

            def _on_chosen(chosen: str | None) -> None:
                if not chosen:
                    return
                try:
                    idx = options.index(chosen)
                    entry_id = entry_ids[idx]
                    self.run_worker(self._do_navigate(entry_id), exclusive=False)
                except (ValueError, IndexError):
                    pass

            self.push_screen(_SelectScreen("Navigate to entry", options), _on_chosen)

        async def _do_navigate(self, entry_id: str) -> None:
            try:
                await self._session.navigate_tree(entry_id)
                self.notify(f"Navigated to {entry_id[:8]}\u2026")
            except Exception as e:
                self.notify(f"Navigation failed: {e}", severity="error")

        # ── /skills ───────────────────────────────────────────────────────────

        def _cmd_skills(self) -> None:
            skills = list(getattr(self._session, "_skills", []) or [])
            prompts = list(getattr(self._session, "_prompt_templates", []) or [])
            lines = []
            settings = self._session.settings_manager.get_settings()
            skill_paths = list(settings.skills or [])

            if skill_paths:
                lines.append(f"Skill search paths:")
                for p in skill_paths:
                    lines.append(f"  {p}")
                lines.append("")
            else:
                lines.append("No skill paths configured in settings.skills")
                lines.append("Add paths to ~/.coding-agent/settings.json:")
                lines.append('  { "skills": ["~/.coding-agent/skills"] }')
                lines.append("")

            if skills:
                lines.append(f"Loaded skills ({len(skills)}):")
                for s in skills:
                    cmds = s.commands or [s.name]
                    cmd_str = ", ".join(f"/{c}" for c in cmds)
                    desc = s.description or "(no description)"
                    lines.append(f"  {cmd_str:<28} {desc}")
                    lines.append(f"    Path: {s.path}")
            else:
                lines.append("No skills loaded.")
                if not skill_paths:
                    lines.append("(configure settings.skills to load skills)")

            if prompts:
                lines.append(f"\nLoaded prompt templates ({len(prompts)}):")
                for p in prompts:
                    lines.append(f"  /{p.name:<26} {p.description or ''}")

            self._show_info("Skills & Prompts", "\n".join(lines))

        # ── /settings ─────────────────────────────────────────────────────────

        def _cmd_settings(self) -> None:
            import dataclasses
            settings = self._session.settings_manager.get_settings()
            lines = []
            for f in dataclasses.fields(settings):
                val = getattr(settings, f.name)
                # Show all fields (skip deeply nested ones for brevity)
                if dataclasses.is_dataclass(val):
                    lines.append(f"\n[{f.name}]")
                    for sf in dataclasses.fields(val):
                        lines.append(f"  {sf.name:<30} {getattr(val, sf.name)!r}")
                else:
                    lines.append(f"  {f.name:<32} {val!r}")
            self._show_info("Settings", "\n".join(lines))

        # ── /hotkeys ──────────────────────────────────────────────────────────

        def _cmd_hotkeys(self) -> None:
            lines = [
                "Keyboard Shortcuts",
                "",
                "  Input",
                "  ─────────────────────────────────────",
                "  Enter                 Send message",
                "  Shift+Enter           Insert new line",
                "  /command              Run slash command",
                "",
                "  App bindings",
                "  ─────────────────────────────────────",
            ]
            for b in BINDINGS:
                key  = b.key.replace("ctrl+", "Ctrl+").replace("shift+", "Shift+")
                desc = b.description or b.action
                lines.append(f"  {key:<22} {desc}")
            lines += [
                "",
                "  Navigation",
                "  ─────────────────────────────────────",
                "  PageUp / PageDown     Scroll messages",
                "  Esc                   Close modals",
            ]
            # Extension shortcuts
            try:
                runner = self._session._get_extension_runner()
                if runner and hasattr(runner, "get_shortcuts"):
                    shortcuts = runner.get_shortcuts({})
                    if shortcuts:
                        lines.append("\n  Extensions\n  " + "─" * 37)
                        for key, shortcut in shortcuts.items():
                            desc = getattr(shortcut, "description", str(shortcut))
                            lines.append(f"  {key:<22} {desc}")
            except Exception:
                pass
            self._show_info("Keyboard Shortcuts", "\n".join(lines))

        # ── /changelog ────────────────────────────────────────────────────────

        def _cmd_changelog(self) -> None:
            pkg_dir = Path(__file__).parent.parent.parent
            candidates = [
                pkg_dir / "CHANGELOG.md",
                pkg_dir / "CHANGELOG",
                pkg_dir.parent / "CHANGELOG.md",
                pkg_dir.parent / "CHANGELOG",
            ]
            for p in candidates:
                if p.exists():
                    try:
                        content = p.read_text(encoding="utf-8")
                        excerpt = content[:4000]
                        if len(content) > 4000:
                            excerpt += "\n\n… (truncated — open the file for full contents)"
                        self._show_info("Changelog", excerpt)
                        return
                    except Exception:
                        pass
            self.notify("No CHANGELOG.md found.", severity="warning")

        # ── /reload ───────────────────────────────────────────────────────────

        async def _cmd_reload(self) -> None:
            if self._session.is_streaming:
                self.notify("Wait for streaming to finish before reloading.", severity="warning")
                return
            if self._session.is_compacting:
                self.notify("Wait for compaction to finish before reloading.", severity="warning")
                return
            self.notify("Reloading skills, extensions, prompts\u2026")
            try:
                await self._session.reload()
                self.notify("Reloaded \u2713")
            except Exception as e:
                self.notify(f"Reload failed: {e}", severity="error")

        # ── /quit ─────────────────────────────────────────────────────────────

        def _cmd_quit(self) -> None:
            self.exit(0)

        # ── Existing actions ──────────────────────────────────────────────────

        async def action_abort_agent(self) -> None:
            try:
                await self._session.abort()
            except Exception:
                pass

        def action_escape_or_abort(self) -> None:
            """Escape key: abort if streaming, close any overlay widget otherwise."""
            # Try to close a mounted overlay (SessionPicker, etc.) first
            try:
                from .widgets.session_picker import SessionPicker
                pickers = self.query(SessionPicker)
                if pickers:
                    for p in pickers:
                        p.remove()
                    return
            except Exception:
                pass
            # Abort streaming if active
            if self._session.is_streaming:
                self.run_worker(self._session.abort(), exclusive=False)

        def action_new_session(self) -> None:
            self.run_worker(self._do_new_session(), exclusive=True)

        async def _do_new_session(self) -> None:
            try:
                result = await self._runtime.new_session()
            except Exception as e:
                self.notify(f"New session failed: {e}", severity="error")
                return
            if not result.get("cancelled"):
                self._rewire_session()
                self.set_timer(1.0, self._refresh_sidebar)
                self.notify("New session started")

        def action_compact(self) -> None:
            self.run_worker(self._cmd_compact(""), exclusive=True)

        def action_cycle_model(self) -> None:
            self.run_worker(self._session.cycle_model(), exclusive=True)
            self._refresh_status()

        def action_cycle_thinking(self) -> None:
            level = self._session.cycle_thinking_level()
            if level:
                self.notify(f"Thinking: {level}")
                try:
                    self.query_one(MessageList).add_audit_note(
                        f"Thinking \u2192 {level}"
                    )
                except Exception:
                    pass
            self._refresh_status()

        def action_show_session_picker(self) -> None:
            try:
                picker = SessionPicker(self._runtime)
                self.mount(picker)
            except Exception as e:
                self.notify(f"Session picker error: {e}", severity="error")

        async def action_export(self) -> None:
            await self._cmd_export("")

        def action_clear_messages(self) -> None:
            self._clear_messages()

        def action_toggle_thinking(self) -> None:
            try:
                self.query_one(MessageList).toggle_thinking_visibility()
            except Exception:
                pass

        def action_scroll_up(self) -> None:
            try:
                self.query_one(MessageList).scroll_up(10)
            except Exception:
                pass

        def action_scroll_down(self) -> None:
            try:
                self.query_one(MessageList).scroll_down(10)
            except Exception:
                pass

        # ── Helpers ───────────────────────────────────────────────────────────

        def _clear_messages(self) -> None:
            try:
                self.query_one(MessageList).clear()
                self.notify("Messages cleared")
            except Exception:
                pass

        def _show_info(self, title: str, content: str) -> None:
            """Push a scrollable info modal (non-blocking)."""
            self.push_screen(_InfoScreen(title, content))

        # ── Extension UI helpers (used by extension API) ──────────────────────
        # These use asyncio.Future so extension code can `await` them even though
        # push_screen itself is non-blocking.

        async def extension_select(self, title: str, options: list[str],
                                   timeout: int | None = None) -> str | None:
            fut: asyncio.Future[str | None] = asyncio.get_event_loop().create_future()
            self.push_screen(_SelectScreen(title, options),
                             lambda v: fut.set_result(v) if not fut.done() else None)
            return await fut

        async def extension_confirm(self, title: str, message: str,
                                    timeout: int | None = None) -> bool:
            fut: asyncio.Future[bool] = asyncio.get_event_loop().create_future()
            self.push_screen(_ConfirmScreen(title, message),
                             lambda v: fut.set_result(bool(v)) if not fut.done() else None)
            return await fut

        async def extension_input(self, title: str, placeholder: str = "",
                                  timeout: int | None = None) -> str | None:
            fut: asyncio.Future[str | None] = asyncio.get_event_loop().create_future()
            self.push_screen(_InputScreen(title, placeholder),
                             lambda v: fut.set_result(v) if not fut.done() else None)
            return await fut

        async def _run_prompt_template(self, cmd_obj, arg: str) -> None:
            try:
                for pt in getattr(self._session, "_prompt_templates", []):
                    if pt.name == cmd_obj.name:
                        content = pt.content
                        if arg:
                            content = content + "\n\n" + arg
                        from coding_agent.core.types import PromptOptions
                        await self._session.prompt(content, PromptOptions(source="interactive"))
                        return
                self.notify(f"Template not found: {cmd_obj.name}", severity="warning")
            except Exception as e:
                self.notify(f"Template error: {e}", severity="error")

        async def _run_skill_command(self, skill_name: str, arg: str) -> None:
            try:
                text = f"/{skill_name}"
                if arg:
                    text += f" {arg}"
                from coding_agent.core.types import PromptOptions
                await self._session.prompt(text, PromptOptions(source="interactive"))
            except Exception as e:
                self.notify(f"Skill error: {e}", severity="error")

    # ── Clipboard helper (module-level, no Textual dependency) ────────────────

    def _clipboard_write(text: str) -> None:
        """Write text to the system clipboard using the best available method."""
        if sys.platform == "win32":
            subprocess.run(["clip"], input=text.encode("utf-16"), check=True)
        elif sys.platform == "darwin":
            subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)
        else:
            # Linux: try wl-copy (Wayland), then xclip, then xsel
            for cmd in [
                ["wl-copy"],
                ["xclip", "-selection", "clipboard"],
                ["xsel", "--clipboard", "--input"],
            ]:
                try:
                    subprocess.run(cmd, input=text.encode("utf-8"), check=True)
                    return
                except (FileNotFoundError, subprocess.CalledProcessError):
                    continue
            raise RuntimeError(
                "No clipboard tool found. Install xclip, xsel, or wl-copy."
            )

    # ── Entry point ───────────────────────────────────────────────────────────

    async def run_interactive_mode(runtime) -> int:
        """Launch the interactive Textual TUI. Returns exit code."""
        try:
            from textual.app import App  # confirm installed
        except ImportError:
            print(
                "Interactive mode requires textual.\n"
                "Install: pip install coding-agent[tui]",
                file=sys.stderr,
            )
            return 1
        app = AgentApp(runtime)
        await app.run_async()
        return 0

except ImportError:
    async def run_interactive_mode(runtime) -> int:  # type: ignore[misc]
        print(
            "Interactive mode requires textual.\n"
            "Install: pip install coding-agent[tui]",
            file=__import__("sys").stderr,
        )
        return 1
