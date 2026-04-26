"""Animated braille spinner shown while the agent is working."""
from __future__ import annotations

try:
    from textual.app import ComposeResult
    from textual.widget import Widget
    from textual.widgets import Static

    _FRAMES = ["\u280b", "\u2819", "\u2839", "\u2838", "\u283c", "\u2834",
               "\u2826", "\u2827", "\u2807", "\u280f"]

    class LoaderBar(Widget):
        DEFAULT_CSS = """
        LoaderBar {
            height: 1;
            padding: 0 1;
            background: $surface;
            border-bottom: dashed $panel;
            display: none;
        }
        LoaderBar.active {
            display: block;
        }
        LoaderBar #lb-text {
            color: $text;
        }
        """

        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            self._frame = 0
            self._timer = None

        def compose(self) -> ComposeResult:
            yield Static("", id="lb-text")

        def set_active(self, active: bool) -> None:
            if active:
                self.add_class("active")
                if self._timer is None:
                    self._timer = self.set_interval(0.1, self._tick)
            else:
                self.remove_class("active")
                if self._timer is not None:
                    self._timer.stop()
                    self._timer = None
                try:
                    self.query_one("#lb-text", Static).update("")
                except Exception:
                    pass

        def _tick(self) -> None:
            frame = _FRAMES[self._frame % len(_FRAMES)]
            try:
                self.query_one("#lb-text", Static).update(
                    f"{frame} Working...  [dim](esc to interrupt)[/dim]"
                )
            except Exception:
                pass
            self._frame += 1

except ImportError:
    pass
