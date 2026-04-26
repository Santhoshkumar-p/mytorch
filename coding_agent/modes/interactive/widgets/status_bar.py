from __future__ import annotations
import threading
from pathlib import Path

try:
    from textual.app import ComposeResult
    from textual.widget import Widget
    from textual.widgets import Static

    class StatusBar(Widget):
        DEFAULT_CSS = """
        StatusBar {
            background: $surface;
            border-top: solid $panel;
            height: 1;
            layout: horizontal;
            padding: 0 1;
        }
        StatusBar .sl { color: $text-muted; width: 1fr; }
        StatusBar .sc { color: $text-muted; width: 1fr; text-align: center; }
        StatusBar .sr { color: $text-muted; width: 1fr; text-align: right; }
        """

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._git_branch = ""
            self._last_cwd   = ""

        def compose(self) -> ComposeResult:
            yield Static("", classes="sl", id="s-left")
            yield Static("", classes="sc", id="s-center")
            yield Static("", classes="sr", id="s-right")

        # ── Git branch (background thread) ───────────────────────────────────

        def _fetch_git(self, cwd: str) -> None:
            try:
                import subprocess
                r = subprocess.run(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                if r.returncode == 0:
                    self._git_branch = r.stdout.strip()
                else:
                    self._git_branch = ""
            except Exception:
                self._git_branch = ""

        # ── Refresh ──────────────────────────────────────────────────────────

        def refresh_state(self, session) -> None:
            cwd = getattr(session, "cwd", None) or str(Path.home())
            if cwd != self._last_cwd:
                self._last_cwd = cwd
                threading.Thread(
                    target=self._fetch_git, args=(cwd,), daemon=True
                ).start()

            # ── Left: cwd name + git branch ──────────────────────────────
            cwd_name = Path(cwd).name or cwd
            left = cwd_name
            if self._git_branch:
                left += f" ({self._git_branch})"

            streaming  = getattr(session, "is_streaming", False)
            compacting = getattr(session, "is_compacting", False)
            if compacting:
                left += " [compacting]"

            # ── Center: token stats ───────────────────────────────────────
            center = ""
            try:
                msgs = getattr(session, "messages", [])
                for msg in reversed(msgs):
                    if getattr(msg, "role", None) == "assistant":
                        usage = getattr(msg, "usage", None)
                        if usage:
                            tok_in  = getattr(usage, "input_tokens",      0) or 0
                            tok_out = getattr(usage, "output_tokens",     0) or 0
                            tok_r   = getattr(usage, "cache_read_tokens", 0) or 0
                            if tok_in or tok_out:
                                center = (
                                    f"\u2191{_fmt(tok_in)} \u2193{_fmt(tok_out)}"
                                )
                                if tok_r:
                                    center += f" R{_fmt(tok_r)}"
                        break
            except Exception:
                pass
            if streaming:
                center = ("\u27f3 " + center).strip()

            # ── Right: provider + model + thinking ────────────────────────
            model    = getattr(session, "model", None) or {}
            provider = model.get("provider", "")
            model_id = model.get("model_id", "")
            thinking = getattr(session, "thinking_level", "off")

            if provider and model_id:
                right = f"({provider}) {model_id}"
            elif model_id:
                right = model_id
            else:
                right = "\u2014"

            if thinking and thinking != "off":
                right += f" [{thinking}]"

            try:
                self.query_one("#s-left",   Static).update(left)
                self.query_one("#s-center", Static).update(center)
                self.query_one("#s-right",  Static).update(right)
            except Exception:
                pass


    def _fmt(n: int) -> str:
        if n < 1000:
            return str(n)
        if n < 10_000:
            return f"{n / 1000:.1f}k"
        return f"{n // 1000}k"

except ImportError:
    pass
