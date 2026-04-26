from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from typing import Callable, Any
from ..types import PromptOptions

EXTENSION_EVENTS = [
    "session_start", "message_start", "message_end",
    "agent_start", "agent_end",
    "tool_call", "tool_result",
    "turn_start", "turn_end",
    "compaction_start", "compaction_end",
    "session_before_switch",   # handlers can return {"cancel": True}
    "session_before_fork",
    "session_shutdown",
]


@dataclass
class ExtensionContext:
    session: Any
    session_manager: Any
    model: dict | None
    cwd: str
    flags: dict = field(default_factory=dict)
    signal: Any = None
    entry: Any = None

    def is_idle(self) -> bool:
        return getattr(self.session, "is_idle", lambda: False)()

    def has_pending_messages(self) -> bool:
        return getattr(self.session, "has_pending_messages", lambda: False)()

    def get_context_usage(self):
        return getattr(self.session, "context_usage", None)

    def get_system_prompt(self) -> str:
        return getattr(self.session, "system_prompt", "")

    async def compact(self, options: dict | None = None) -> None:
        if hasattr(self.session, "compact"):
            await self.session.compact(options or {})

    async def abort(self) -> None:
        if hasattr(self.session, "abort"):
            await self.session.abort()

    async def shutdown(self) -> None:
        if hasattr(self.session, "shutdown"):
            await self.session.shutdown()


class ExtensionAPI:
    """Passed to extension factory. Collects registrations, then bound to a session."""

    def __init__(self):
        self._handlers: dict[str, list[Callable]] = {}
        self._tools: list = []
        self._commands: list[dict] = []
        self._flags: list[dict] = []
        self._session = None

    # Registration phase (before bind)

    def on(self, event: str, handler: Callable) -> None:
        self._handlers.setdefault(event, []).append(handler)

    def register_tool(self, tool) -> None:
        self._tools.append(tool)

    def register_command(
        self,
        name: str,
        description: str | None = None,
        handler: Callable | None = None,
    ) -> None:
        self._commands.append({"name": name, "description": description, "handler": handler})

    def register_flag(
        self,
        name: str,
        type: str = "boolean",
        default=None,
        description: str | None = None,
    ) -> None:
        self._flags.append({"name": name, "type": type, "default": default, "description": description})

    # Action phase (after bind_to_session)

    def _require_session(self):
        if self._session is None:
            raise RuntimeError("ExtensionAPI not bound to a session yet")

    def send_message(self, message: str, source: str | None = None) -> None:
        self._require_session()
        if hasattr(self._session, "send_message"):
            self._session.send_message(message, source=source)

    def append_entry(self, custom_type: str, data=None) -> None:
        self._require_session()
        if hasattr(self._session, "session_manager"):
            self._session.session_manager.append_custom_entry(custom_type, data)

    def set_session_name(self, name: str) -> None:
        self._require_session()
        if hasattr(self._session, "session_manager"):
            sm = self._session.session_manager
            if hasattr(sm, "append_session_info"):
                sm.append_session_info(name)

    def get_session_name(self) -> str | None:
        self._require_session()
        if hasattr(self._session, "session_manager"):
            return self._session.session_manager.get_session_name()
        return None

    def set_label(self, entry_id: str, label: str | None = None) -> None:
        self._require_session()
        if hasattr(self._session, "session_manager"):
            sm = self._session.session_manager
            if hasattr(sm, "append_label"):
                sm.append_label(entry_id, label)

    def get_active_tools(self) -> list[str]:
        self._require_session()
        return getattr(self._session, "active_tools", [])

    def set_active_tools(self, tool_names: list[str]) -> None:
        self._require_session()
        if hasattr(self._session, "set_active_tools"):
            self._session.set_active_tools(tool_names)

    def get_commands(self) -> list[dict]:
        return list(self._commands)

    def set_thinking_level(self, level: str) -> None:
        self._require_session()
        if hasattr(self._session, "set_thinking_level"):
            self._session.set_thinking_level(level)

    def get_thinking_level(self) -> str:
        self._require_session()
        return getattr(self._session, "thinking_level", "off")
