"""
Two-phase thinking block:
  • Streaming  → animated "💭 Thinking…" in a purple panel
  • Finalized  → collapsible "💭 Thinking: preview…" block (click to expand)
"""
from __future__ import annotations

try:
    from textual.app import ComposeResult
    from textual.widget import Widget
    from textual.widgets import Collapsible, Static

    _DOTS = ["   ", ".  ", ".. ", "..."]

    class ThinkingBlock(Widget):
        DEFAULT_CSS = """
        ThinkingBlock {
            margin: 0 0 1 0;
            height: auto;
            border-left: thick $accent-darken-2;
            background: $panel;
            padding: 0;
        }

        /* ── Streaming phase ── */
        ThinkingBlock .th-streaming {
            color: $accent;
            padding: 0 1;
            text-style: italic;
        }

        /* ── Finalized phase (Collapsible) ── */
        ThinkingBlock .th-done {
            background: transparent;
        }
        ThinkingBlock .th-content {
            color: $text-muted;
            padding: 0 1 0 2;
            text-style: italic;
        }
        """

        def __init__(self, thinking_content, is_history: bool = False, **kwargs):
            super().__init__(**kwargs)
            self._thinking = thinking_content
            self._is_history = is_history
            self._dot_frame = 0
            self._stream_timer = None

        def _get_thinking_text(self) -> str:
            """Extract thinking text from either a dict (history) or typed object."""
            if isinstance(self._thinking, dict):
                return self._thinking.get("thinking", "") or ""
            return getattr(self._thinking, "thinking", "") or ""

        def compose(self) -> ComposeResult:
            thinking_text = self._get_thinking_text()
            yield Static("💭 Thinking   ", classes="th-streaming", id="th-stream")
            with Collapsible(
                title=self._make_title(thinking_text),
                collapsed=True,
                classes="th-done",
            ):
                yield Static(thinking_text, classes="th-content")

        def on_mount(self) -> None:
            if self._is_history:
                # History mode: skip animation, immediately finalize
                self.finalize()
            else:
                # Start in streaming mode: hide collapsible, animate dots
                try:
                    self.query_one(".th-done").display = False
                except Exception:
                    pass
                self._stream_timer = self.set_interval(0.35, self._animate_dots)

        def _animate_dots(self) -> None:
            dots = _DOTS[self._dot_frame % len(_DOTS)]
            try:
                self.query_one("#th-stream", Static).update(
                    f"[bold]💭 Thinking{dots}[/bold]"
                )
            except Exception:
                pass
            self._dot_frame += 1

        # ── Updates during streaming ─────────────────────────────────────────

        def update_thinking(self, thinking_content) -> None:
            """Called while streaming — keep internal state current."""
            self._thinking = thinking_content
            thinking_text = self._get_thinking_text()
            try:
                self.query_one(".th-content", Static).update(thinking_text)
            except Exception:
                pass

        # ── Finalization ─────────────────────────────────────────────────────

        def finalize(self) -> None:
            """Switch from animated streaming indicator to the collapsed block."""
            if self._stream_timer is not None:
                self._stream_timer.stop()
                self._stream_timer = None

            thinking_text = self._get_thinking_text()
            try:
                self.query_one("#th-stream", Static).display = False
                collapsible = self.query_one(Collapsible)
                collapsible.title = self._make_title(thinking_text)
                self.query_one(".th-content", Static).update(thinking_text)
                collapsible.display = True
            except Exception:
                pass

        @staticmethod
        def _make_title(thinking_text: str) -> str:
            preview = (thinking_text or "")[:70].replace("\n", " ")
            if preview:
                char_count = len(thinking_text)
                return f"💭 Thinking: {preview}…  [{char_count:,} chars]"
            return "💭 Thinking…"

except ImportError:
    pass
