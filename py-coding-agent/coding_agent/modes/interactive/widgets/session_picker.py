from __future__ import annotations
import datetime
import json
from pathlib import Path

try:
    from textual.app import ComposeResult
    from textual.widget import Widget
    from textual.widgets import ListView, ListItem, Label, Static, Input

    class SessionPicker(Widget):
        DEFAULT_CSS = """
        SessionPicker {
            background: $surface; border: solid $primary;
            padding: 1; layer: overlay;
            offset: 8% 4%; width: 84%; height: 88%;
        }
        SessionPicker .sp-title {
            color: $primary; text-style: bold; margin-bottom: 1;
        }
        SessionPicker #sp-search {
            background: $surface; border: solid $panel;
            height: 3; margin-bottom: 1;
        }
        SessionPicker ListView { height: 1fr; }
        SessionPicker .sp-main   { color: $text; text-style: bold; }
        SessionPicker .sp-sub    { color: $text-muted; padding-left: 2; }
        SessionPicker .sp-empty  { color: $text-muted; text-style: italic; }
        """

        def __init__(self, runtime, **kwargs):
            super().__init__(**kwargs)
            self._runtime = runtime
            self._all_items: list[tuple[str, str, dict]] = []  # (path, title, meta)

        def compose(self) -> ComposeResult:
            yield Static(
                "Resume Session  [\u2191\u2193 navigate \u00b7 Enter select \u00b7 Esc close]",
                classes="sp-title",
            )
            yield Input(placeholder="Search sessions...", id="sp-search")
            yield ListView(id="session-list")

        def on_mount(self) -> None:
            self._load_sessions()

        def _load_sessions(self, filter_text: str = "") -> None:
            settings = self._runtime.services.settings_manager.get_settings()
            session_dir = settings.session_dir
            if not session_dir:
                session_dir = str(Path.home() / ".coding-agent" / "sessions")
            lv = self.query_one(ListView)
            lv.clear()
            try:
                files = sorted(
                    Path(session_dir).glob("*.jsonl"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )[:100]
                if not files:
                    lv.append(ListItem(Label(
                        f"(no sessions in {session_dir})", classes="sp-empty"
                    )))
                    return
                self._all_items = []
                for f in files:
                    meta = _read_session_meta(str(f))
                    title = meta.get("name") or meta.get("first_user_msg") or f.stem[:30]
                    title = title.strip() or f.stem[:30]
                    self._all_items.append((str(f), title, meta))
                self._render_list(filter_text)
            except Exception as exc:
                lv.append(ListItem(Label(f"(error loading sessions: {exc})", classes="sp-empty")))

        def _render_list(self, filter_text: str = "") -> None:
            lv = self.query_one(ListView)
            lv.clear()
            q = filter_text.lower().strip()
            shown = 0
            for path, title, meta in self._all_items:
                if q and q not in title.lower() and q not in Path(path).stem.lower():
                    continue
                age = meta.get("age", "")
                count = meta.get("msg_count", 0)
                sub = f"{age} \u00b7 {count} messages" if age else f"{count} messages"
                lv.append(ListItem(
                    Label(title[:70], classes="sp-main"),
                    Label(sub, classes="sp-sub"),
                    name=path,
                ))
                shown += 1
                if shown >= 50:
                    break
            if shown == 0:
                lv.append(ListItem(Label("(no matches)", classes="sp-empty")))

        def on_input_changed(self, event: Input.Changed) -> None:
            self._render_list(event.value)

        def on_list_view_selected(self, event: ListView.Selected) -> None:
            path = getattr(event.item, "name", None)
            app = self.app
            self.remove()
            if path:
                app.run_worker(_switch(app, self._runtime, path))

        def on_key(self, event) -> None:
            if event.key == "escape":
                self.remove()

    # ── Switch helper (module-level so it survives widget removal) ─────────────

    async def _switch(app, runtime, path: str) -> None:
        try:
            await runtime.switch_session(path)
        except Exception as e:
            try:
                app.notify(f"Switch failed: {e}", severity="error")
            except Exception:
                pass
            return

        # Rewire app to the new session (includes load_history)
        try:
            app._rewire_session()
        except Exception:
            try:
                from .message_list import MessageList
                app.query_one(MessageList).clear()
            except Exception:
                pass

        # Schedule a delayed sidebar refresh so skills have time to initialize
        try:
            app.set_timer(0.8, app._refresh_sidebar)
        except Exception:
            pass

        # Return focus to the input bar
        try:
            from .input_bar import InputBar
            app.query_one(InputBar).set_focus()
        except Exception:
            pass

    # ── Session metadata helpers ───────────────────────────────────────────────

    def _read_session_meta(path: str) -> dict:
        """Read session metadata: name, first user message, message count, age."""
        try:
            lines = open(path, encoding="utf-8").readlines()
            if not lines:
                return {}
            raw = json.loads(lines[0])
            session_id = raw.get("id", Path(path).stem)
            timestamp = raw.get("timestamp", "")

            # Human-readable age
            age_str = ""
            try:
                ts = datetime.datetime.fromisoformat(
                    timestamp.replace("Z", "+00:00")
                )
                now = datetime.datetime.now(datetime.timezone.utc)
                secs = int((now - ts).total_seconds())
                if secs < 120:
                    age_str = "just now"
                elif secs < 3600:
                    age_str = f"{secs // 60}m ago"
                elif secs < 86400:
                    age_str = f"{secs // 3600}h ago"
                else:
                    age_str = f"{secs // 86400}d ago"
            except Exception:
                pass

            first_user_msg = ""
            msg_count = 0
            session_name = None

            for line in lines[1:]:
                try:
                    entry = json.loads(line)
                except Exception:
                    continue
                etype = entry.get("type", "")
                if etype == "session_info" and entry.get("name"):
                    session_name = entry["name"]
                elif etype == "message":
                    msg = entry.get("message") or {}
                    if isinstance(msg, dict):
                        if msg.get("role") in ("user", "assistant"):
                            msg_count += 1
                        if msg.get("role") == "user" and not first_user_msg:
                            content = msg.get("content", [])
                            if isinstance(content, str):
                                first_user_msg = content[:100]
                            elif isinstance(content, list):
                                for block in content:
                                    if isinstance(block, dict) and block.get("type") == "text":
                                        first_user_msg = block.get("text", "")[:100]
                                        break

            return {
                "id": session_id,
                "name": session_name,
                "first_user_msg": first_user_msg.replace("\n", " ").strip(),
                "msg_count": msg_count,
                "age": age_str,
            }
        except Exception:
            return {"id": Path(path).stem, "name": None, "first_user_msg": "",
                    "msg_count": 0, "age": ""}

except ImportError:
    pass
