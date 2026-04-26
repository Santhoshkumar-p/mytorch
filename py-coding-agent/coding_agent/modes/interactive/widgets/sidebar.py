from __future__ import annotations
from pathlib import Path

try:
    from textual.app import ComposeResult
    from textual.containers import ScrollableContainer
    from textual.widgets import Static

    class Sidebar(ScrollableContainer):
        """Left panel showing loaded Skills, Prompts, and Extensions."""

        DEFAULT_CSS = """
        Sidebar {
            width: 36;
            border-right: solid $panel;
            background: $background;
            overflow-y: auto;
            overflow-x: hidden;
            padding: 0;
        }
        Sidebar Static {
            padding: 0 1;
            color: $text-muted;
        }
        """

        def compose(self) -> ComposeResult:
            yield Static("", id="sb-body")

        def refresh_sidebar(self, session) -> None:
            """Re-build the sidebar content from the current session resources."""
            content = self._build(session)
            try:
                self.query_one("#sb-body", Static).update(content)
            except Exception:
                pass

        # ── Content builder ──────────────────────────────────────────────────

        def _build(self, session) -> str:
            home = str(Path.home())
            parts: list[str] = []

            def shorten(p: str) -> str:
                return p.replace(home, "~")

            def scope_of(path: str) -> str:
                """Guess scope from path: user-global config → 'user', else 'project'."""
                norm = path.replace("\\", "/")
                home_norm = home.replace("\\", "/")
                if norm.startswith(home_norm + "/."):
                    return "user"
                return "project"

            # ── Skills ──────────────────────────────────────────────────────
            skills = list(getattr(session, "_skills", []) or [])
            if skills:
                parts.append("[bold][Skills][/bold]")
                by_scope: dict[str, list] = {}
                for s in skills:
                    sc = (
                        (s.source_info or {}).get("scope")
                        or scope_of(s.path)
                    )
                    by_scope.setdefault(sc, []).append(s)
                for sc in ("user", "project") + tuple(
                    k for k in by_scope if k not in ("user", "project")
                ):
                    if sc not in by_scope:
                        continue
                    parts.append(f"  [dim]{sc}[/dim]")
                    for s in by_scope[sc]:
                        parts.append(f"    [dim]{shorten(s.path)}[/dim]")

            # ── Prompts ─────────────────────────────────────────────────────
            prompts = list(getattr(session, "_prompt_templates", []) or [])
            if prompts:
                if parts:
                    parts.append("")
                parts.append("[bold][Prompts][/bold]")
                by_scope2: dict[str, list] = {}
                for p in prompts:
                    sc = (
                        (p.source_info or {}).get("scope")
                        or scope_of(p.file_path)
                    )
                    by_scope2.setdefault(sc, []).append(p)
                for sc in ("user", "project") + tuple(
                    k for k in by_scope2 if k not in ("user", "project")
                ):
                    if sc not in by_scope2:
                        continue
                    parts.append(f"  [dim]{sc}[/dim]")
                    for p in by_scope2[sc]:
                        parts.append(f"    [dim]/{p.name}[/dim]")

            # ── Extensions ──────────────────────────────────────────────────
            try:
                runner = None
                if hasattr(session, "_get_extension_runner"):
                    runner = session._get_extension_runner()
                if runner:
                    cmds = runner.get_registered_commands() or []
                    if cmds:
                        if parts:
                            parts.append("")
                        parts.append("[bold][Extensions][/bold]")
                        # Group by source package if info is available
                        pkg_prev = None
                        for cmd in cmds:
                            pkg = cmd.get("package") or cmd.get("source", "")
                            if pkg and pkg != pkg_prev:
                                parts.append(f"  [dim]{pkg}[/dim]")
                                pkg_prev = pkg
                            parts.append(f"    [dim]/{cmd.get('name', '?')}[/dim]")
            except Exception:
                pass

            if not parts:
                return "[dim](no resources loaded)[/dim]"

            return "\n".join(parts)

except ImportError:
    pass
