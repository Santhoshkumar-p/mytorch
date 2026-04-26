from __future__ import annotations


def _normalise_content(msg) -> list:
    """Return a list of content blocks regardless of message representation."""
    if isinstance(msg, dict):
        content = msg.get("content") or []
    else:
        content = getattr(msg, "content", None) or []
    # Some backends send content as a bare string
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    return list(content)


try:
    from textual.app import ComposeResult
    from textual.containers import ScrollableContainer
    from textual.widgets import Static
    from .message_item import MessageItem
    from .tool_block import ToolBlock
    from .thinking_block import ThinkingBlock

    class AuditNote(Static):
        """Inline audit record (model/thinking changes, session events)."""
        DEFAULT_CSS = """
        AuditNote {
            height: auto;
            padding: 0 1;
            color: $text-muted;
            text-style: italic;
            margin-bottom: 0;
        }
        """

    class MessageList(ScrollableContainer):
        DEFAULT_CSS = """
        MessageList { height: 1fr; }
        """

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._widget_map: dict = {}
            # Keys of assistant messages that are currently streaming.
            # Used to detect "follow-up" user messages sent while agent is busy.
            self._streaming_asst_keys: set = set()

        # ── Message events ────────────────────────────────────────────────────

        def handle_message_event(self, event) -> None:
            try:
                if not isinstance(event, dict):
                    return
                evt_type = event.get("type", "")
                msg = event.get("message")
                if msg is None:
                    return

                role = getattr(msg, "role", "user")
                key = id(msg)

                if evt_type == "message_start":
                    if role == "assistant":
                        self._streaming_asst_keys.add(key)
                        w = MessageItem(msg, is_follow_up=False)
                    else:
                        # User message — follow-up if agent was streaming
                        is_follow_up = bool(self._streaming_asst_keys)
                        w = MessageItem(msg, is_follow_up=is_follow_up)
                    self._widget_map[key] = w
                    self.mount(w)
                    self.scroll_end(animate=False)

                elif evt_type == "message_update":
                    w = self._widget_map.get(key)
                    if w:
                        w.update_message(msg)
                        self.scroll_end(animate=False)

                elif evt_type == "message_end":
                    self._streaming_asst_keys.discard(key)
                    w = self._widget_map.get(key)
                    if w:
                        w.finalize(msg)
                    else:
                        # Non-streaming: message arrived directly as end event
                        w = MessageItem(msg, is_follow_up=False)
                        self._widget_map[key] = w
                        self.mount(w)
                        self.scroll_end(animate=False)
            except Exception:
                pass

        # ── Tool events ───────────────────────────────────────────────────────

        def handle_tool_event(self, event) -> None:
            try:
                if not isinstance(event, dict):
                    return
                evt_type = event.get("type", "")
                tc_id = event.get("tool_call_id", "")

                if evt_type == "tool_execution_start":
                    class _Proxy:
                        pass
                    proxy = _Proxy()
                    proxy.id = tc_id
                    proxy.name = event.get("tool_name", "?")
                    proxy.arguments = event.get("args", {})
                    w = ToolBlock(proxy)
                    self._widget_map[tc_id] = w
                    self.mount(w)
                    self.scroll_end(animate=False)

                elif evt_type == "tool_execution_update":
                    w = self._widget_map.get(tc_id)
                    if w:
                        w.update_output(event.get("partial_output", ""))

                elif evt_type == "tool_execution_end":
                    w = self._widget_map.get(tc_id)
                    if w:
                        w.finalize(event.get("result"), event.get("is_error", False))
            except Exception:
                pass

        # ── Audit notes ───────────────────────────────────────────────────────

        def add_audit_note(self, text: str) -> None:
            """Add an inline audit note (model/thinking changes, session events)."""
            try:
                w = AuditNote(f"\u2014\u2014 {text} \u2014\u2014")
                self.mount(w)
                self.scroll_end(animate=False)
            except Exception:
                pass

        # ── History replay ────────────────────────────────────────────────────

        def load_history(self, session) -> None:
            """Replay historical messages from a session into the widget list.

            Called after clear() when switching / resuming / forking a session.
            Handles both typed agent objects (fresh sessions) and plain-dict
            messages that have been round-tripped through the JSONL store.
            Also shows inline audit notes for model/thinking-level changes.
            """
            import logging
            log = logging.getLogger(__name__)
            try:
                from coding_agent.core.types import (
                    SessionMessageEntry,
                    ModelChangeEntry,
                    ThinkingLevelChangeEntry,
                )

                sm = getattr(session, "_session_manager", None)
                if sm is None:
                    log.debug("load_history: session has no _session_manager")
                    return
                entries = list(getattr(sm, "_entries", []) or [])
                log.debug("load_history: %d entries found", len(entries))

                # ── Pass 1: collect all tool result messages ───────────────────
                # Tool results arrive as separate messages with
                # role="toolResult" (not embedded in user messages).  We map
                # tool_call_id → {"text": str, "is_error": bool} so that
                # ToolBlock widgets can be pre-finalized during rendering.
                tool_results: dict = {}
                for entry in entries:
                    if not isinstance(entry, SessionMessageEntry):
                        continue
                    msg = entry.message
                    if not msg:
                        continue
                    role = (
                        msg.get("role") if isinstance(msg, dict)
                        else getattr(msg, "role", None)
                    )
                    # Agent uses role="toolResult" for tool result messages
                    if role not in ("toolResult", "tool_result"):
                        # Also handle Anthropic-style: user message with tool_result blocks
                        if role == "user":
                            for block in _normalise_content(msg):
                                b_type = (
                                    block.get("type", "") if isinstance(block, dict)
                                    else getattr(block, "type", "")
                                )
                                if b_type != "tool_result":
                                    continue
                                if isinstance(block, dict):
                                    tc_id = (
                                        block.get("tool_use_id")
                                        or block.get("tool_call_id")
                                        or block.get("toolUseId")
                                        or ""
                                    )
                                    nested = block.get("content") or []
                                    parts: list[str] = []
                                    for c in nested:
                                        if isinstance(c, dict) and c.get("type") == "text":
                                            parts.append(c.get("text", ""))
                                        elif isinstance(c, str):
                                            parts.append(c)
                                    if not parts and isinstance(block.get("content"), str):
                                        parts = [block["content"]]
                                    if tc_id:
                                        tool_results[tc_id] = {
                                            "text": "\n".join(parts),
                                            "is_error": bool(block.get("is_error", False)),
                                        }
                        continue
                    if isinstance(msg, dict):
                        tc_id = (
                            msg.get("tool_call_id")
                            or msg.get("toolCallId")
                            or msg.get("tool_use_id")
                            or ""
                        )
                        is_err = bool(msg.get("is_error") or msg.get("isError"))
                        raw_content = msg.get("content") or []
                        parts: list[str] = []
                        if isinstance(raw_content, list):
                            for c in raw_content:
                                if isinstance(c, dict):
                                    t = c.get("text", "")
                                else:
                                    t = str(c)
                                if t:
                                    parts.append(t)
                        elif isinstance(raw_content, str):
                            parts = [raw_content]
                        if tc_id:
                            tool_results[tc_id] = {
                                "text": "\n".join(parts),
                                "is_error": is_err,
                            }

                # ── Pass 2: render entries in order ───────────────────────────
                mounted = 0
                for entry in entries:
                    # Model / thinking changes → audit note
                    if isinstance(entry, ModelChangeEntry):
                        provider = getattr(entry, "provider", "")
                        model_id = getattr(entry, "model_id", "")
                        label = f"{provider}/{model_id}".strip("/")
                        self.add_audit_note(f"Model \u2192 {label}")
                        continue

                    if isinstance(entry, ThinkingLevelChangeEntry):
                        level = getattr(entry, "thinking_level", "")
                        self.add_audit_note(f"Thinking \u2192 {level}")
                        continue

                    if not isinstance(entry, SessionMessageEntry):
                        continue
                    msg = entry.message
                    if not msg:
                        continue
                    role = (
                        msg.get("role") if isinstance(msg, dict)
                        else getattr(msg, "role", None)
                    )
                    # Skip tool result messages (shown inside ToolBlocks)
                    if role in ("toolResult", "tool_result"):
                        continue
                    if role not in ("user", "assistant"):
                        continue

                    # Also skip user messages whose content is ONLY tool_result blocks
                    content = _normalise_content(msg)
                    if role == "user" and content:
                        b_types = [
                            (b.get("type", "") if isinstance(b, dict)
                             else getattr(b, "type", ""))
                            for b in content
                        ]
                        non_empty = [t for t in b_types if t]
                        if non_empty and all(
                            t in ("tool_result", "tool_result_content")
                            for t in non_empty
                        ):
                            continue

                    w = MessageItem(msg, is_follow_up=False, tool_results=tool_results)
                    self._widget_map[entry.id] = w
                    self.mount(w)
                    mounted += 1

                log.debug("load_history: mounted %d message widgets", mounted)
                if mounted > 0:
                    self.scroll_end(animate=False)
            except Exception:
                log.exception("load_history failed")

        # ── Helpers ───────────────────────────────────────────────────────────

        def toggle_thinking_visibility(self) -> None:
            try:
                for w in self.query(ThinkingBlock):
                    w.display = not w.display
            except Exception:
                pass

        def clear(self) -> None:
            self._widget_map.clear()
            self._streaming_asst_keys.clear()
            self.remove_children()

except ImportError:
    pass
