from __future__ import annotations
try:
    from textual.app import ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal
    from textual.message import Message
    from textual.widget import Widget
    from textual.widgets import TextArea, ListView, ListItem, Label, Static

    class SubmitTextArea(TextArea):
        """TextArea that submits on Enter and inserts newline on Shift+Enter."""

        BINDINGS = [
            Binding("enter",       "submit_text",  "Send",    show=False, priority=True),
            Binding("shift+enter", "insert_line",  "Newline", show=False, priority=True),
        ]

        class Submit(Message):
            """Posted when the user presses Enter to send."""

        def action_submit_text(self) -> None:
            self.post_message(self.Submit())

        def action_insert_line(self) -> None:
            self.insert("\n")

    class InputBar(Widget):
        DEFAULT_CSS = """
        InputBar {
            height: auto; padding: 0 0;
            background: $surface; border-top: solid $panel;
        }
        InputBar #input-row {
            height: auto; padding: 0 1;
        }
        InputBar #prompt-char {
            width: 2; color: $accent; text-style: bold;
            padding: 0 0; height: auto;
        }
        InputBar SubmitTextArea {
            height: auto; min-height: 1; max-height: 10;
            background: $surface; border: none; width: 1fr;
        }
        InputBar #autocomplete {
            background: $surface; border: solid $primary;
            max-height: 20; width: 72; display: none;
        }
        InputBar #autocomplete .ac-name {
            color: $accent; text-style: bold; padding: 0 1;
        }
        InputBar #autocomplete .ac-desc {
            color: $text-muted; padding: 0 2 0 3;
        }
        """

        class Submit(Message):
            """Bubbles up to the App when user requests submission."""

        def compose(self) -> ComposeResult:
            with Horizontal(id="input-row"):
                yield Static("> ", id="prompt-char")
                yield SubmitTextArea(id="editor")
            yield ListView(id="autocomplete")

        def on_mount(self) -> None:
            self.query_one(SubmitTextArea).focus()

        def on_submit_text_area_submit(self, event: SubmitTextArea.Submit) -> None:
            event.stop()
            self.post_message(self.Submit())

        def on_text_area_changed(self, event: TextArea.Changed) -> None:
            text = event.text_area.text
            if text.startswith("/") and "\n" not in text and " " not in text.lstrip("/"):
                self._show_autocomplete(text[1:])
            else:
                self._hide_autocomplete()

        def _show_autocomplete(self, prefix: str) -> None:
            cmds = self._get_commands(prefix)
            lv = self.query_one(ListView)
            lv.clear()
            for cmd in cmds[:10]:
                desc = cmd.description or ""
                lv.append(ListItem(
                    Label(f"/{cmd.name}", classes="ac-name"),
                    Label(desc, classes="ac-desc"),
                ))
            lv.display = bool(cmds)

        def _hide_autocomplete(self) -> None:
            try:
                self.query_one(ListView).display = False
            except Exception:
                pass

        def _get_commands(self, prefix: str):
            try:
                from coding_agent.core.slash_commands import get_slash_commands
                session = getattr(self.app, "_session", None)
                if not session:
                    return []
                runner = session._get_extension_runner() if hasattr(session, "_get_extension_runner") else None
                skills = getattr(session, "_skills", [])
                prompts = getattr(session, "_prompt_templates", [])
                settings = session.settings_manager.get_settings()
                cmds = get_slash_commands(runner, prompts, skills, settings)
                return [c for c in cmds if c.name.startswith(prefix)]
            except Exception:
                return []

        def on_list_view_selected(self, event: ListView.Selected) -> None:
            # The first Label has class "ac-name" and contains "/{name}"
            try:
                name_label = event.item.query_one(".ac-name", Label)
                text = str(name_label.renderable)
            except Exception:
                # Fallback: parse from any label
                labels = list(event.item.query(Label))
                text = str(labels[0].renderable).split("  ")[0] if labels else ""
            self.query_one(SubmitTextArea).load_text(text + " ")
            self._hide_autocomplete()

        def on_key(self, event) -> None:
            if event.key == "escape":
                self._hide_autocomplete()

        def take_value(self) -> str:
            ta = self.query_one(SubmitTextArea)
            text = ta.text
            ta.load_text("")
            return text

        def set_focus(self) -> None:
            self.query_one(SubmitTextArea).focus()

except ImportError:
    pass
