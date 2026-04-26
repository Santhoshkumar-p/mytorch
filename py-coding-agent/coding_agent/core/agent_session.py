from __future__ import annotations
import asyncio
import dataclasses
from dataclasses import dataclass, field
from typing import Callable, Any

from .types import (
    AgentState, SessionStats, PromptOptions, BashResult, Settings,
    SessionMessageEntry, ModelChangeEntry, ThinkingLevelChangeEntry, SessionInfoEntry,
)
from .session_manager import SessionManager
from .settings_manager import SettingsManager
from .system_prompt import build_system_prompt, BuildSystemPromptOptions
from .tools.registry import build_tools
from .compaction.compaction import should_compact, run_compaction


@dataclass
class AgentSessionConfig:
    agent: Any                              # Agent instance
    session_manager: SessionManager
    settings_manager: SettingsManager
    cwd: str
    resource_loader: Any                    # ResourceLoader
    model_config: Any = None               # ModelConfig from agent library
    custom_tools: list | None = None
    initial_active_tool_names: list[str] = field(
        default_factory=lambda: ["read", "bash", "edit", "write"]
    )
    extension_runner_ref: list | None = None   # [ExtensionRunner | None]
    custom_system_prompt: str | None = None
    append_system_prompts: list[str] = field(default_factory=list)
    no_context_files: bool = False


class AgentSession:
    def __init__(self, config: AgentSessionConfig) -> None:
        self._agent = config.agent
        self._session_manager = config.session_manager
        self._settings_manager = config.settings_manager
        self._cwd = config.cwd
        self._resource_loader = config.resource_loader
        self._model_config = config.model_config
        self._custom_tools = config.custom_tools
        self._initial_active_tool_names = config.initial_active_tool_names
        self._extension_runner_ref = config.extension_runner_ref
        self._custom_system_prompt = config.custom_system_prompt
        self._append_system_prompts = list(config.append_system_prompts)
        self._no_context_files = config.no_context_files

        self._thinking_level: str = "off"
        self._steering_mode: str = "all"
        self._follow_up_mode: str = "all"
        self._auto_compaction: bool = True
        self._auto_retry: bool = True
        self._is_compacting: bool = False

        self._active_tool_names: list[str] = list(config.initial_active_tool_names)
        self._listeners: list[Callable] = []

        self._skills: list = []
        self._prompt_templates: list = []
        self._context_files: list = []
        self._system_prompt: str = ""

        asyncio.ensure_future(self._initialize())

    # ── Initialization ────────────────────────────────────────────────────────

    async def _initialize(self) -> None:
        try:
            self._skills = await self._resource_loader.load_skills()
            if not self._no_context_files:
                self._context_files = await self._resource_loader.load_context_files()
            self._prompt_templates = await self._resource_loader.load_prompt_templates()

            append_combined = "\n\n".join(
                s for s in self._append_system_prompts if s
            ) or None

            self._system_prompt = build_system_prompt(BuildSystemPromptOptions(
                custom_prompt=self._custom_system_prompt,
                selected_tools=self._active_tool_names,
                cwd=self._cwd,
                context_files=self._context_files,
                skills=self._skills,
                append_system_prompt=append_combined,
            ))

            ctx = self._session_manager.build_session_context()
            if ctx.thinking_level:
                self._thinking_level = ctx.thinking_level

            settings = self._settings_manager.get_settings()
            tools = build_tools(
                self._cwd,
                settings,
                active_names=self._active_tool_names,
                custom_tools=self._custom_tools,
            )

            # Apply system prompt and tools using the real agent API
            if hasattr(self._agent, "set_system_prompt"):
                self._agent.set_system_prompt(self._system_prompt)
            elif hasattr(self._agent, "system_prompt"):
                self._agent.system_prompt = self._system_prompt

            if hasattr(self._agent, "set_tools"):
                self._agent.set_tools(tools)
            elif hasattr(self._agent, "tools"):
                self._agent.tools = tools

            # Load persisted messages into agent history
            if ctx.messages:
                if hasattr(self._agent, "replace_messages"):
                    self._agent.replace_messages(ctx.messages)
                elif hasattr(self._agent, "_state") and hasattr(self._agent._state, "messages"):
                    self._agent._state.messages = list(ctx.messages)

            # Set thinking level on agent
            if hasattr(self._agent, "set_thinking_level"):
                self._agent.set_thinking_level(self._thinking_level)

            # Subscribe to agent events
            if hasattr(self._agent, "subscribe"):
                self._agent.subscribe(self._on_agent_event)

            # Bind extensions
            runner = self._get_extension_runner()
            if runner:
                await runner.bind_to_session(self)
                from .extensions.types import ExtensionContext
                await runner.emit("session_start", ExtensionContext(
                    session=self,
                    session_manager=self._session_manager,
                    model=self.model,
                    cwd=self._cwd,
                ))
        except Exception as exc:
            import logging
            logging.error("AgentSession._initialize error: %s", exc)
        finally:
            # Notify all listeners that initialization is complete so the TUI
            # can refresh skills/sidebar regardless of success/failure.
            for listener in list(self._listeners):
                try:
                    listener({"type": "session_initialized"})
                except Exception:
                    pass

    def _get_extension_runner(self):
        if self._extension_runner_ref and self._extension_runner_ref[0]:
            return self._extension_runner_ref[0]
        return None

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def agent(self):
        return self._agent

    @property
    def session_manager(self) -> SessionManager:
        return self._session_manager

    @property
    def settings_manager(self) -> SettingsManager:
        return self._settings_manager

    @property
    def model(self) -> dict | None:
        ctx = self._session_manager.build_session_context()
        if ctx.model:
            return ctx.model
        settings = self._settings_manager.get_settings()
        if settings.default_provider or settings.default_model:
            return {
                "provider": settings.default_provider or "anthropic",
                "model_id": settings.default_model or "",
            }
        return None

    @property
    def thinking_level(self) -> str:
        return self._thinking_level

    @property
    def is_streaming(self) -> bool:
        state = getattr(self._agent, "state", None)
        if state:
            # Agent uses is_streaming bool + stream_message for the partial message
            if hasattr(state, "is_streaming"):
                return bool(state.is_streaming)
            # fallback for stub agents
            return bool(getattr(state, "streaming_message", None) or getattr(state, "stream_message", None))
        return False

    @property
    def is_compacting(self) -> bool:
        return self._is_compacting

    @property
    def session_file(self) -> str | None:
        return self._session_manager.get_session_file()

    @property
    def session_id(self) -> str:
        return self._session_manager.get_session_id()

    @property
    def session_name(self) -> str | None:
        return self._session_manager.get_session_name()

    @property
    def auto_compaction_enabled(self) -> bool:
        return self._auto_compaction

    @property
    def pending_message_count(self) -> int:
        state = getattr(self._agent, "state", None)
        if state:
            return len(getattr(state, "pending_tool_calls", frozenset()))
        return 0

    @property
    def steering_mode(self) -> str:
        return self._steering_mode

    @property
    def follow_up_mode(self) -> str:
        return self._follow_up_mode

    @property
    def messages(self) -> list:
        state = getattr(self._agent, "state", None)
        if state:
            return list(getattr(state, "messages", []))
        return []

    # ── Agent actions ─────────────────────────────────────────────────────────

    async def prompt(self, text: str, options: PromptOptions | None = None) -> None:
        options = options or PromptOptions()
        try:
            from pi_agent import UserMessage, TextContent
            msg = UserMessage(content=[TextContent(text=text)])
            self._session_manager.append_message_entry(msg)
            await self._session_manager.flush()
        except ImportError:
            pass

        if hasattr(self._agent, "prompt"):
            await self._agent.prompt(text, images=options.images or [])

    async def steer(self, text: str, images: list | None = None) -> None:
        if not hasattr(self._agent, "steer"):
            return
        try:
            from pi_agent import UserMessage, TextContent, ImageContent as IC
            content = [TextContent(text=text)]
            if images:
                content.extend(images)
            # Agent.steer() is synchronous
            self._agent.steer(UserMessage(content=content))
        except ImportError:
            # Stub agent — try async or direct call
            if asyncio.iscoroutinefunction(self._agent.steer):
                await self._agent.steer(text, images=images or [])
            else:
                self._agent.steer(text)

    async def follow_up(self, text: str, images: list | None = None) -> None:
        if not hasattr(self._agent, "follow_up"):
            return
        try:
            from pi_agent import UserMessage, TextContent
            content = [TextContent(text=text)]
            if images:
                content.extend(images)
            # Agent.follow_up() is synchronous
            self._agent.follow_up(UserMessage(content=content))
        except ImportError:
            if asyncio.iscoroutinefunction(self._agent.follow_up):
                await self._agent.follow_up(text, images=images or [])
            else:
                self._agent.follow_up(text)

    async def abort(self) -> None:
        if not hasattr(self._agent, "abort"):
            return
        # Agent.abort() is synchronous
        if asyncio.iscoroutinefunction(self._agent.abort):
            await self._agent.abort()
        else:
            self._agent.abort()

    async def wait_for_idle(self) -> None:
        if hasattr(self._agent, "wait_for_idle"):
            await self._agent.wait_for_idle()

    async def set_model(self, provider: str, model_id: str) -> None:
        self._session_manager.append_model_change(provider, model_id)
        await self._session_manager.flush()
        # Update agent's active model
        if hasattr(self._agent, "set_model"):
            try:
                from pi_agent import Model
                api = _provider_to_api(provider)
                self._agent.set_model(Model(id=model_id, provider=provider, api=api))
            except ImportError:
                pass

    async def cycle_model(self) -> dict | None:
        settings = self._settings_manager.get_settings()
        models = settings.enabled_models
        if not models:
            return None
        current = self.model or {}
        current_key = f"{current.get('provider', '')}/{current.get('model_id', '')}"
        try:
            idx = models.index(current_key)
            next_model = models[(idx + 1) % len(models)]
        except ValueError:
            next_model = models[0]
        parts = next_model.split("/", 1)
        provider = parts[0] if len(parts) == 2 else "anthropic"
        model_id = parts[1] if len(parts) == 2 else parts[0]
        await self.set_model(provider, model_id)
        return {"provider": provider, "model_id": model_id}

    def set_thinking_level(self, level: str) -> None:
        self._thinking_level = level
        self._session_manager.append_thinking_level_change(level)
        if hasattr(self._agent, "set_thinking_level"):
            self._agent.set_thinking_level(level)

    def cycle_thinking_level(self) -> str:
        levels = ["off", "minimal", "low", "medium", "high", "xhigh"]
        try:
            idx = levels.index(self._thinking_level)
            next_level = levels[(idx + 1) % len(levels)]
        except ValueError:
            next_level = "off"
        self.set_thinking_level(next_level)
        return next_level

    def set_steering_mode(self, mode: str) -> None:
        self._steering_mode = mode

    def set_follow_up_mode(self, mode: str) -> None:
        self._follow_up_mode = mode

    async def compact(self, custom_instructions: str | None = None) -> None:
        if self._is_compacting:
            return
        self._is_compacting = True
        try:
            settings = self._settings_manager.get_settings()
            await run_compaction(self, settings.compaction, custom_instructions)
        finally:
            self._is_compacting = False

    def set_auto_compaction_enabled(self, enabled: bool) -> None:
        self._auto_compaction = enabled

    def set_auto_retry_enabled(self, enabled: bool) -> None:
        self._auto_retry = enabled

    async def abort_retry(self) -> None:
        if hasattr(self._agent, "abort"):
            await self._agent.abort()

    def get_session_stats(self) -> SessionStats:
        msgs = self.messages
        user_msgs = sum(1 for m in msgs if getattr(m, "role", None) == "user")
        asst_msgs = sum(1 for m in msgs if getattr(m, "role", None) == "assistant")
        state = getattr(self._agent, "state", None)
        usage = getattr(state, "context_usage", None) if state else None
        return SessionStats(
            session_file=self.session_file,
            session_id=self.session_id,
            user_messages=user_msgs,
            assistant_messages=asst_msgs,
            total_messages=len(msgs),
            context_usage=usage,
        )

    def get_system_prompt(self) -> str:
        return self._system_prompt

    async def execute_bash(self, command: str, timeout: int | None = None) -> BashResult:
        settings = self._settings_manager.get_settings()
        try:
            from .tools.bash import execute_bash as _exec_bash
            args = {"command": command}
            if timeout:
                args["timeout"] = timeout
            result = await _exec_bash(args, self._cwd, settings)
            # result is list[{"type": "text", "text": ..., "exit_code"?: ...}]
            text = "".join(
                c.get("text", "") if isinstance(c, dict) else getattr(c, "text", "")
                for c in result
            )
            exit_code = next(
                (c.get("exit_code", 0) for c in result if isinstance(c, dict) and "exit_code" in c),
                0,
            )
            return BashResult(output=text, exit_code=exit_code)
        except Exception as exc:
            return BashResult(output=str(exc), exit_code=1)

    async def export_to_html(self, output_path: str | None = None) -> str:
        try:
            from .export_html.export import export_session_to_html
            return await export_session_to_html(
                self._session_manager,
                output_path,
                system_prompt=self._system_prompt or None,
            )
        except Exception as exc:
            raise RuntimeError(f"Export failed: {exc}") from exc

    async def bind_extensions(self, runner) -> None:
        await runner.bind_to_session(self)

    def set_session_name(self, name: str) -> None:
        self._session_manager.append_session_info(name)

    def get_last_assistant_text(self) -> str | None:
        for m in reversed(self.messages):
            if getattr(m, "role", None) == "assistant":
                parts = [getattr(c, "text", "") for c in getattr(m, "content", []) if hasattr(c, "text")]
                return "".join(parts) or None
        return None

    def get_user_messages_for_forking(self) -> list[dict]:
        result = []
        for entry in self._session_manager._entries:
            if isinstance(entry, SessionMessageEntry):
                msg = entry.message
                if msg and getattr(msg, "role", None) == "user":
                    parts = [getattr(c, "text", "") for c in getattr(msg, "content", []) if hasattr(c, "text")]
                    result.append({"entry_id": entry.id, "text": "".join(parts)})
        return result

    async def navigate_tree(self, target_id: str, options: dict | None = None) -> dict:
        entry = self._session_manager.get_entry(target_id)
        if not entry:
            return {"error": f"Entry {target_id!r} not found"}
        self._session_manager._leaf_id = target_id
        return {"navigated": True, "target_id": target_id}

    async def reload(self) -> None:
        await self._resource_loader.reload()
        await self._initialize()

    def _set_active_tools(self, names: list[str]) -> None:
        self._active_tool_names = list(names)
        settings = self._settings_manager.get_settings()
        tools = build_tools(
            self._cwd, settings,
            active_names=self._active_tool_names,
            custom_tools=self._custom_tools,
        )
        if hasattr(self._agent, "set_tools"):
            self._agent.set_tools(tools)
        elif hasattr(self._agent, "tools"):
            self._agent.tools = tools

    def _get_commands(self) -> list[dict]:
        runner = self._get_extension_runner()
        if runner:
            return runner.get_registered_commands()
        return []

    def subscribe(self, listener: Callable) -> Callable:
        self._listeners.append(listener)
        def unsub():
            try:
                self._listeners.remove(listener)
            except ValueError:
                pass
        return unsub

    def dispose(self) -> None:
        self._listeners.clear()

    # ── Agent event handler ───────────────────────────────────────────────────

    def _on_agent_event(self, event) -> None:
        # Events arrive as plain dicts: {"type": "...", ...}
        # Older stub agents may emit objects — handle both gracefully.
        event_type = event.get("type") if isinstance(event, dict) else type(event).__name__

        if event_type in ("message_end", "MessageEndEvent"):
            msg = event.get("message") if isinstance(event, dict) else getattr(event, "message", None)
            if msg and getattr(msg, "role", "assistant") == "assistant":
                self._session_manager.append_message_entry(msg)
                asyncio.ensure_future(self._session_manager.flush())

                # Auto-compaction check using usage from message
                usage = getattr(msg, "usage", None)
                if usage and self._auto_compaction:
                    settings = self._settings_manager.get_settings()
                    cs = settings.compaction
                    ctx_window = 200_000
                    total = getattr(usage, "total_tokens", 0) or 0
                    if should_compact(total, ctx_window, cs):
                        asyncio.ensure_future(self.compact())

        elif event_type in ("tool_execution_end", "ToolExecutionEndEvent"):
            # tool_execution_end carries the raw AgentToolResult, not a message.
            # The ToolResultMessage is appended separately by agent_loop.
            # We don't need to persist anything here — it's handled at turn_end.
            pass

        elif event_type in ("turn_end", "TurnEndEvent"):
            # Persist tool result messages included in the turn
            tool_results = event.get("tool_results", []) if isinstance(event, dict) else []
            for result_msg in tool_results:
                self._session_manager.append_message_entry(result_msg)
            if tool_results:
                asyncio.ensure_future(self._session_manager.flush())

        for listener in list(self._listeners):
            try:
                listener(event)
            except Exception as exc:
                import logging
                logging.error("AgentSession listener error: %s", exc)


def _provider_to_api(provider: str) -> str:
    """Map a provider name to the api string expected by the agent Model."""
    mapping = {
        "anthropic": "anthropic",
        "openai": "openai-completions",
        "vertex": "vertex",
    }
    return mapping.get(provider, provider)
