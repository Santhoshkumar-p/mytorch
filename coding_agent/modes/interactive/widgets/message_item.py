from __future__ import annotations

try:
    from textual.app import ComposeResult
    from textual.widget import Widget
    from textual.widgets import Static, Markdown

    class MessageItem(Widget):
        DEFAULT_CSS = """
        MessageItem { padding: 0 1; margin-bottom: 1; height: auto; }

        /* Normal user message */
        MessageItem.user      { background: $surface; border-left: thick $primary; }

        /* Follow-up / steering message sent while agent was running */
        MessageItem.follow-up { background: $surface; border-left: thick $accent-darken-1; }

        /* Assistant message */
        MessageItem.assistant { background: $background; }

        .role     { color: $text-muted; text-style: bold; }
        .role-followup { color: $accent; text-style: bold; }

        MessageItem Markdown  { height: auto; }
        """

        def __init__(
            self,
            message,
            is_follow_up: bool = False,
            tool_results: "dict | None" = None,
            **kwargs,
        ):
            super().__init__(**kwargs)
            self._message = message
            self._is_follow_up = is_follow_up
            # tool_results: {tool_use_id -> {"text": str, "is_error": bool}}
            # Provided for history-mode rendering so ToolBlocks show pre-loaded output.
            self._tool_results = tool_results or {}
            role = (
                message.get("role", "user")
                if isinstance(message, dict)
                else getattr(message, "role", "user")
            )
            if is_follow_up:
                self.add_class("follow-up")
            else:
                self.add_class(role)

        def compose(self) -> ComposeResult:
            msg = self._message
            role = (
                msg.get("role", "user")
                if isinstance(msg, dict)
                else getattr(msg, "role", "user")
            )
            if self._is_follow_up:
                label = "↩ You (steering)"
                label_css = "role role-followup"
            elif role == "user":
                label = "You"
                label_css = "role"
            else:
                label = "Assistant"
                label_css = "role"
            yield Static(label, classes=label_css)
            yield from self._render_blocks(msg)

        def _render_blocks(self, message) -> ComposeResult:
            # Normalise content to a list (handles str / list / typed objects)
            if isinstance(message, dict):
                raw = message.get("content")
            else:
                raw = getattr(message, "content", None)
            if isinstance(raw, str):
                content = [{"type": "text", "text": raw}] if raw else []
            else:
                content = list(raw or [])

            for block in content:
                if isinstance(block, dict):
                    # History mode: plain-dict blocks from JSONL round-trip
                    yield from self._render_dict_block(block)
                else:
                    # Streaming mode: typed agent objects
                    try:
                        from pi_agent import TextContent, ThinkingContent
                        from .thinking_block import ThinkingBlock
                        if isinstance(block, TextContent):
                            yield Markdown(block.text or "")
                        elif isinstance(block, ThinkingContent):
                            yield ThinkingBlock(block)
                    except ImportError:
                        pass

        def _render_dict_block(self, block: dict):
            """Render a plain-dict content block (history / JSONL replay)."""
            from .thinking_block import ThinkingBlock
            from .tool_block import ToolBlock

            block_type = block.get("type", "")

            if block_type == "text":
                text = block.get("text", "")
                if text:
                    yield Markdown(text)

            elif block_type == "thinking":
                yield ThinkingBlock(block, is_history=True)

            elif block_type in ("tool_use", "tool_call", "toolCall"):
                # Build a lightweight proxy that ToolBlock can consume
                class _Proxy:
                    pass
                proxy = _Proxy()
                proxy.id = block.get("id", "")
                proxy.name = block.get("name", "?")
                # Agent serialises as "arguments"; Anthropic API uses "input"
                proxy.arguments = (
                    block.get("arguments") or block.get("input") or {}
                )
                # Look up pre-recorded result from the tool_results map.
                # For history blocks we always use preloaded mode (no spinner) — even
                # if no matching result is found the tool already ran.
                result_info = self._tool_results.get(proxy.id)
                preloaded_result = result_info["text"] if result_info is not None else ""
                preloaded_is_error = (
                    result_info.get("is_error", False) if result_info is not None else False
                )
                yield ToolBlock(
                    proxy,
                    preloaded_result=preloaded_result,
                    preloaded_is_error=preloaded_is_error,
                )

            elif block_type == "tool_result":
                # Already reflected inside the ToolBlock above; skip standalone rendering
                return
            # Unknown block types are silently ignored

        def update_message(self, partial) -> None:
            self._message = partial
            try:
                from pi_agent import TextContent, ThinkingContent
                from .thinking_block import ThinkingBlock

                content = getattr(partial, "content", []) or []
                thinking_blocks_content = [
                    b for b in content if isinstance(b, ThinkingContent)
                ]
                text = "".join(
                    b.text for b in content if isinstance(b, TextContent)
                )

                # Update or create ThinkingBlock widgets
                existing_thinking = list(self.query(ThinkingBlock))
                for i, tb_content in enumerate(thinking_blocks_content):
                    if i < len(existing_thinking):
                        existing_thinking[i].update_thinking(tb_content)
                    else:
                        new_block = ThinkingBlock(tb_content)
                        mds = list(self.query(Markdown))
                        if mds:
                            self.mount(new_block, before=mds[0])
                        else:
                            self.mount(new_block)

                # Update Markdown text
                mds = list(self.query(Markdown))
                if mds:
                    mds[-1].update(text)
                elif text:
                    self.mount(Markdown(text))
            except Exception:
                pass

        def finalize(self, message) -> None:
            try:
                from .thinking_block import ThinkingBlock
                for tb in self.query(ThinkingBlock):
                    tb.finalize()
            except Exception:
                pass
            self.update_message(message)

except ImportError:
    pass
