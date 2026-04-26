"""
Compact tool call block with distinct visual:

  ▶ bash  $ git status                      ← header (yellow, click to expand)
    On branch main                          ← output (truncated, PREVIEW_LINES)
    ... (3 more lines, click to expand)     ← hint when collapsed

States:
  • Running  → spinner in header, no output yet
  • Done     → ▶ header, truncated output
  • Error    → ✖ header (red), output shown
"""
from __future__ import annotations
from pathlib import Path

try:
    from textual.app import ComposeResult
    from textual.widget import Widget
    from textual.widgets import Static

    _FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    class ToolBlock(Widget):
        DEFAULT_CSS = """
        ToolBlock {
            margin: 0 0 1 0;
            height: auto;
            border-left: thick $warning-darken-2;
            background: $panel;
            padding: 0;
        }
        ToolBlock .th {
            color: $warning;
            padding: 0 1;
            text-style: bold;
        }
        ToolBlock .th.err {
            color: $error;
        }
        ToolBlock .tout {
            color: $text;
            padding: 0 1 0 2;
            text-style: none;
        }
        ToolBlock .tmore {
            color: $text-muted;
            padding: 0 1 0 2;
            text-style: italic;
        }
        """

        PREVIEW_LINES = 8

        def __init__(
            self,
            tool_call,
            preloaded_result: str | None = None,
            preloaded_is_error: bool = False,
            **kwargs,
        ):
            super().__init__(**kwargs)
            self._tool_call = tool_call
            self._output = ""
            self._expanded = False
            self._running = True
            self._is_error = False
            self._spin_frame = 0
            self._spin_timer = None
            self._preloaded_result = preloaded_result
            self._preloaded_is_error = preloaded_is_error

        # ── Summary ──────────────────────────────────────────────────────────

        def _call_summary(self) -> str:
            name = getattr(self._tool_call, "name", "?")
            args = getattr(self._tool_call, "arguments", {}) or {}
            home = str(Path.home())

            def s(p: str) -> str:
                return str(p).replace(home, "~")

            if name == "bash":
                cmd = args.get("command", "")
                if len(cmd) > 80:
                    cmd = cmd[:77] + "..."
                return f"bash  $ {cmd}"
            elif name in ("read", "write"):
                path = args.get("path", args.get("file_path", ""))
                return f"{name}  {s(path)}"
            elif name == "edit":
                path = args.get("path", args.get("file_path", ""))
                return f"edit  {s(path)}"
            elif name == "grep":
                pat = args.get("pattern", "")
                path = args.get("path", "")
                return f"grep  {pat!r} {s(path)}"
            elif name == "find":
                pat = args.get("pattern", args.get("glob", ""))
                return f"find  {pat}"
            elif name == "ls":
                return f"ls  {s(args.get('path', '.'))}"
            else:
                first = next(
                    (str(v)[:60] for v in args.values() if v is not None), ""
                )
                return f"{name}  {first}" if first else name

        def _header_text(self) -> str:
            summary = self._call_summary()
            if self._running:
                frame = _FRAMES[self._spin_frame % len(_FRAMES)]
                return f"{frame} {summary}"
            elif self._is_error:
                return f"✖ {summary}"
            else:
                indicator = "▼" if self._expanded else "▶"
                return f"{indicator} {summary}"

        # ── Compose ──────────────────────────────────────────────────────────

        def compose(self) -> ComposeResult:
            yield Static(self._header_text(), classes="th", id="th-label")
            yield Static("", classes="tout", id="tout")
            yield Static("", classes="tmore", id="tmore")

        def on_mount(self) -> None:
            if self._preloaded_result is not None:
                # History mode: immediately show finalized state, no spinner
                self._running = False
                self._is_error = self._preloaded_is_error
                self._output = self._preloaded_result
                try:
                    lbl = self.query_one("#th-label", Static)
                    lbl.update(self._header_text())
                    if self._is_error:
                        lbl.add_class("err")
                except Exception:
                    pass
                self._redraw()
            else:
                self._spin_timer = self.set_interval(0.1, self._tick_spinner)

        def _tick_spinner(self) -> None:
            if self._running:
                self._spin_frame += 1
                try:
                    self.query_one("#th-label", Static).update(self._header_text())
                except Exception:
                    pass

        # ── Updates ──────────────────────────────────────────────────────────

        def update_output(self, partial: str) -> None:
            self._output = partial
            self._redraw()

        def _redraw(self) -> None:
            lines = self._output.splitlines()
            n_total = len(lines)
            if self._expanded:
                shown = lines
                more = ""
            else:
                shown = lines[: self.PREVIEW_LINES]
                extra = n_total - self.PREVIEW_LINES
                more = (
                    f"... ({extra} more lines, click to expand)"
                    if extra > 0
                    else ""
                )
            try:
                self.query_one("#tout", Static).update("\n".join(shown))
                self.query_one("#tmore", Static).update(more)
            except Exception:
                pass

        # ── Finalize ─────────────────────────────────────────────────────────

        def finalize(self, result, is_error: bool) -> None:
            # Stop spinner
            if self._spin_timer is not None:
                self._spin_timer.stop()
                self._spin_timer = None
            self._running = False
            self._is_error = is_error

            # Extract output text from the result object
            if result is not None:
                parts: list[str] = []
                for c in getattr(result, "content", []):
                    if hasattr(c, "text"):
                        parts.append(c.text)
                if not parts:
                    if isinstance(result, str):
                        parts = [result]
                    elif isinstance(result, dict):
                        parts = [result.get("text", "")]
                self._output = "\n".join(parts)

            # Update header styling
            try:
                lbl = self.query_one("#th-label", Static)
                lbl.update(self._header_text())
                if is_error:
                    lbl.add_class("err")
            except Exception:
                pass

            self._redraw()

        # ── Interaction ──────────────────────────────────────────────────────

        def on_click(self) -> None:
            if self._running:
                return
            self._expanded = not self._expanded
            try:
                self.query_one("#th-label", Static).update(self._header_text())
            except Exception:
                pass
            self._redraw()

except ImportError:
    pass
