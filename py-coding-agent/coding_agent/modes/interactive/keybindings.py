from __future__ import annotations
try:
    from textual.binding import Binding
    BINDINGS = [
        Binding("ctrl+j",    "submit",               "Send",            show=True),
        Binding("ctrl+c",    "abort_agent",          "Abort",           show=True),
        Binding("escape",    "escape_or_abort",      "Abort/close",     show=False, priority=False),
        Binding("ctrl+n",    "new_session",          "New session",     show=False),
        Binding("ctrl+e",    "export",               "Export HTML",     show=False, priority=True),
        Binding("ctrl+r",    "show_session_picker",  "Sessions",        show=True),
        Binding("ctrl+k",    "compact",              "Compact",         show=False),
        Binding("ctrl+m",    "cycle_model",          "Cycle model",     show=False),
        Binding("ctrl+t",    "cycle_thinking",       "Cycle thinking",  show=False),
        Binding("ctrl+h",    "toggle_thinking",      "Hide thinking",   show=False),
        Binding("ctrl+l",    "clear_messages",       "Clear",           show=False),
        Binding("ctrl+b",    "toggle_sidebar",       "Sidebar",         show=False),
        Binding("pageup",    "scroll_up",            "Scroll up",       show=False),
        Binding("pagedown",  "scroll_down",          "Scroll down",     show=False),
    ]
except ImportError:
    BINDINGS = []
