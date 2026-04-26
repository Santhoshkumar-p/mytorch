from __future__ import annotations
import html
import json


class ToolHtmlRenderer:
    """Renders tool call/result to HTML for the export view."""

    def render_call(self, tool_call_id: str, tool_name: str, args: dict) -> str | None:
        """Return HTML string for a tool call header, or None for default."""
        renderers = {
            "bash": self._render_bash_call,
            "read": self._render_read_call,
            "write": self._render_write_call,
            "edit": self._render_edit_call,
            "grep": self._render_grep_call,
            "find": self._render_find_call,
            "ls": self._render_ls_call,
        }
        fn = renderers.get(tool_name)
        return fn(tool_call_id, args) if fn else None

    def render_result(
        self,
        tool_call_id,
        tool_name: str,
        result,
        details,
        is_error: bool,
    ) -> dict | None:
        """Return {"collapsed": html, "expanded": html} or None for default."""
        error_class = " error" if is_error else ""
        if isinstance(result, (dict, list)):
            try:
                text = json.dumps(result, indent=2)
            except (TypeError, ValueError):
                text = str(result)
        else:
            text = str(result) if result is not None else ""

        collapsed_text = text[:200] + ("..." if len(text) > 200 else "")
        return {
            "collapsed": f'<code class="tool-result{error_class}">{html.escape(collapsed_text)}</code>',
            "expanded": f'<pre class="tool-result{error_class}">{html.escape(text)}</pre>',
        }

    def _render_bash_call(self, id: str, args: dict) -> str:
        cmd = args.get("command", "")
        return f'<code class="tool-call bash">$ {html.escape(cmd)}</code>'

    def _render_read_call(self, id: str, args: dict) -> str:
        path = args.get("path", "")
        return f'<code class="tool-call read">read {html.escape(path)}</code>'

    def _render_write_call(self, id: str, args: dict) -> str:
        path = args.get("path", "")
        return f'<code class="tool-call write">write {html.escape(path)}</code>'

    def _render_edit_call(self, id: str, args: dict) -> str:
        path = args.get("path", "")
        return f'<code class="tool-call edit">edit {html.escape(path)}</code>'

    def _render_grep_call(self, id: str, args: dict) -> str:
        pattern = args.get("pattern", "")
        path = args.get("path", "")
        return f'<code class="tool-call grep">grep {html.escape(pattern)} {html.escape(path)}</code>'

    def _render_find_call(self, id: str, args: dict) -> str:
        pattern = args.get("pattern", args.get("glob", ""))
        return f'<code class="tool-call find">find {html.escape(pattern)}</code>'

    def _render_ls_call(self, id: str, args: dict) -> str:
        path = args.get("path", ".")
        return f'<code class="tool-call ls">ls {html.escape(path)}</code>'
