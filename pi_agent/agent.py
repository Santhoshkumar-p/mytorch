"""
Stateful Agent — owns the transcript, emits events, manages queues.
Maps directly to agent.ts.
"""

from __future__ import annotations
import copy
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from .types import (
    AgentContext, AgentEvent, AgentLoopConfig, AgentMessage, AgentTool,
    AssistantMessage, ModelConfig, TextContent, Usage,
    ContextUsage, ContextWindowBreakdown, ThinkingLevel, ToolExecutionMode,
    ThinkingBudgets, Transport,
    AbortController,
    AgentStartEvent, AgentEndEvent, TurnEndEvent,
    MessageStartEvent, MessageUpdateEvent, MessageEndEvent,
    ToolExecutionStartEvent, ToolExecutionEndEvent,
    UserMessage, ImageContent,
)
from .agent_loop import run_agent_loop, run_agent_loop_continue

QueueMode = Literal["all", "one-at-a-time"]


class _PendingQueue:
    def __init__(self, mode: QueueMode):
        self.mode = mode
        self._items: list[AgentMessage] = []

    def enqueue(self, msg: AgentMessage) -> None:
        self._items.append(msg)

    def has_items(self) -> bool:
        return bool(self._items)

    def drain(self) -> list[AgentMessage]:
        if self.mode == "all":
            out, self._items = self._items[:], []
            return out
        if not self._items:
            return []
        first, self._items = self._items[0], self._items[1:]
        return [first]

    def clear(self) -> None:
        self._items = []


@dataclass
class _State:
    system_prompt: str = ""
    model_config: ModelConfig = field(default_factory=lambda: ModelConfig(model="unknown"))
    thinking_level: ThinkingLevel = "off"
    _tools: list[AgentTool] = field(default_factory=list, repr=False)
    _messages: list[AgentMessage] = field(default_factory=list, repr=False)
    is_streaming: bool = False
    streaming_message: AgentMessage | None = None
    pending_tool_calls: set[str] = field(default_factory=set)
    error_message: str | None = None
    context_usage: ContextUsage | None = None

    @property
    def tools(self) -> list[AgentTool]:
        return self._tools

    @tools.setter
    def tools(self, v: list[AgentTool]) -> None:
        self._tools = list(v)

    @property
    def messages(self) -> list[AgentMessage]:
        return self._messages

    @messages.setter
    def messages(self, v: list[AgentMessage]) -> None:
        self._messages = list(v)


@dataclass
class AgentOptions:
    model_config: ModelConfig | None = None
    initial_system_prompt: str = ""
    initial_tools: list[AgentTool] = field(default_factory=list)
    initial_messages: list[AgentMessage] = field(default_factory=list)
    thinking_level: ThinkingLevel = "off"
    convert_to_llm: Callable | None = None
    transform_context: Callable | None = None
    stream_fn: Callable | None = None
    get_api_key: Callable | None = None
    on_payload: Callable | None = None   # async (payload, model_config) -> dict | None
    on_response: Callable | None = None  # async (status, headers, model_config) -> None
    before_tool_call: Callable | None = None
    after_tool_call: Callable | None = None
    steering_mode: QueueMode = "one-at-a-time"
    follow_up_mode: QueueMode = "one-at-a-time"
    tool_execution: ToolExecutionMode = "parallel"
    session_id: str | None = None
    thinking_budgets: ThinkingBudgets | None = None
    transport: Transport = "sse"
    max_retry_delay_ms: int | None = None


def _default_convert_to_llm(messages: list[AgentMessage]) -> list[AgentMessage]:
    return [m for m in messages if getattr(m, "role", None) in ("user", "assistant", "toolResult")]


class Agent:
    """
    Stateful wrapper around the agent loop.
    Owns the transcript, emits lifecycle events, manages steering/follow-up queues.
    """

    def __init__(self, options: AgentOptions = None):
        opts = options or AgentOptions()
        self._state = _State(
            system_prompt=opts.initial_system_prompt,
            model_config=opts.model_config or ModelConfig(model="unknown"),
            thinking_level=opts.thinking_level,
        )
        self._state.tools = opts.initial_tools
        self._state.messages = opts.initial_messages

        self.convert_to_llm = opts.convert_to_llm or _default_convert_to_llm
        self.transform_context = opts.transform_context
        self.stream_fn = opts.stream_fn
        self.get_api_key = opts.get_api_key
        self.on_payload = opts.on_payload
        self.on_response = opts.on_response
        self.before_tool_call = opts.before_tool_call
        self.after_tool_call = opts.after_tool_call
        self.tool_execution = opts.tool_execution
        self.session_id = opts.session_id
        self.thinking_budgets = opts.thinking_budgets
        self.transport: Transport = opts.transport
        self.max_retry_delay_ms = opts.max_retry_delay_ms

        self._steering_queue = _PendingQueue(opts.steering_mode)
        self._follow_up_queue = _PendingQueue(opts.follow_up_mode)
        self._listeners: list[Callable] = []
        self._abort_controller: AbortController | None = None
        self._run_promise = None  # asyncio.Future for waitForIdle

    # ------------------------------------------------------------------
    # Public state
    # ------------------------------------------------------------------

    @property
    def state(self) -> _State:
        return self._state

    @property
    def signal(self) -> AbortController | None:
        return self._abort_controller

    @property
    def steering_mode(self) -> QueueMode:
        return self._steering_queue.mode

    @steering_mode.setter
    def steering_mode(self, v: QueueMode) -> None:
        self._steering_queue.mode = v

    @property
    def follow_up_mode(self) -> QueueMode:
        return self._follow_up_queue.mode

    @follow_up_mode.setter
    def follow_up_mode(self, v: QueueMode) -> None:
        self._follow_up_queue.mode = v

    # ------------------------------------------------------------------
    # Queue API
    # ------------------------------------------------------------------

    def steer(self, message: AgentMessage) -> None:
        self._steering_queue.enqueue(message)

    def follow_up(self, message: AgentMessage) -> None:
        self._follow_up_queue.enqueue(message)

    def clear_steering_queue(self) -> None:
        self._steering_queue.clear()

    def clear_follow_up_queue(self) -> None:
        self._follow_up_queue.clear()

    def clear_all_queues(self) -> None:
        self._steering_queue.clear()
        self._follow_up_queue.clear()

    def has_queued_messages(self) -> bool:
        return self._steering_queue.has_items() or self._follow_up_queue.has_items()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def subscribe(self, listener: Callable[[AgentEvent, AbortController], Any]) -> Callable:
        self._listeners.append(listener)
        return lambda: self._listeners.remove(listener)

    def abort(self) -> None:
        if self._abort_controller:
            self._abort_controller.abort()

    async def wait_for_idle(self) -> None:
        if self._run_promise:
            await self._run_promise

    def reset(self) -> None:
        self._state.messages = []
        self._state.is_streaming = False
        self._state.streaming_message = None
        self._state.pending_tool_calls = set()
        self._state.error_message = None
        self._state.context_usage = None
        self.clear_all_queues()

    # ------------------------------------------------------------------
    # Prompt / continue
    # ------------------------------------------------------------------

    async def prompt(
        self,
        input: str | AgentMessage | list[AgentMessage],
        images: list[ImageContent] | None = None,
    ) -> None:
        if self._abort_controller:
            raise RuntimeError("Agent is already running. Use steer() or follow_up() to queue messages.")

        messages = self._normalize_input(input, images)
        await self._run_prompt_messages(messages)

    async def continue_run(self) -> None:
        if self._abort_controller:
            raise RuntimeError("Agent is already running.")

        last = self._state.messages[-1] if self._state.messages else None
        if last is None:
            raise RuntimeError("No messages to continue from")

        if getattr(last, "role", None) == "assistant":
            queued = self._steering_queue.drain()
            if queued:
                await self._run_prompt_messages(queued, skip_initial_steering=True)
                return
            queued = self._follow_up_queue.drain()
            if queued:
                await self._run_prompt_messages(queued)
                return
            raise RuntimeError("Cannot continue from role: assistant")

        await self._run_continuation()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _normalize_input(
        self,
        input: str | AgentMessage | list[AgentMessage],
        images: list[ImageContent] | None,
    ) -> list[AgentMessage]:
        if isinstance(input, list):
            return input
        if isinstance(input, str):
            content: list = [TextContent(text=input)]
            if images:
                content.extend(images)
            return [UserMessage(content=content)]
        return [input]

    async def _run_prompt_messages(
        self,
        messages: list[AgentMessage],
        skip_initial_steering: bool = False,
    ) -> None:
        await self._run_with_lifecycle(lambda signal: run_agent_loop(
            messages,
            self._context_snapshot(),
            self._loop_config(skip_initial_steering=skip_initial_steering),
            self._emit,
            signal,
        ))

    async def _run_continuation(self) -> None:
        await self._run_with_lifecycle(lambda signal: run_agent_loop_continue(
            self._context_snapshot(),
            self._loop_config(),
            self._emit,
            signal,
        ))

    def _context_snapshot(self) -> AgentContext:
        return AgentContext(
            system_prompt=self._state.system_prompt,
            messages=list(self._state.messages),
            tools=list(self._state.tools),
        )

    def _loop_config(self, skip_initial_steering: bool = False) -> AgentLoopConfig:
        _skip = [skip_initial_steering]  # mutable cell

        async def get_steering():
            if _skip[0]:
                _skip[0] = False
                return []
            return self._steering_queue.drain()

        return AgentLoopConfig(
            model_config=self._state.model_config,
            convert_to_llm=self.convert_to_llm,
            stream_fn=self.stream_fn,
            transform_context=self.transform_context,
            get_api_key=self.get_api_key,
            on_payload=self.on_payload,
            on_response=self.on_response,
            get_steering_messages=get_steering,
            get_follow_up_messages=lambda: self._follow_up_queue.drain(),
            tool_execution=self.tool_execution,
            before_tool_call=self.before_tool_call,
            after_tool_call=self.after_tool_call,
            thinking_level=self._state.thinking_level,
            session_id=self.session_id,
            thinking_budgets=self.thinking_budgets,
            transport=self.transport,
            max_retry_delay_ms=self.max_retry_delay_ms,
        )

    async def _run_with_lifecycle(self, executor: Callable) -> None:
        import asyncio

        if self._abort_controller:
            raise RuntimeError("Agent is already running.")

        self._abort_controller = AbortController()
        self._state.is_streaming = True
        self._state.streaming_message = None
        self._state.error_message = None

        loop = asyncio.get_event_loop()
        self._run_promise = loop.create_future()

        try:
            await executor(self._abort_controller)
        except Exception as error:
            await self._handle_failure(error)
        finally:
            self._state.is_streaming = False
            self._state.streaming_message = None
            self._state.pending_tool_calls = set()
            if self._run_promise and not self._run_promise.done():
                self._run_promise.set_result(None)
            self._abort_controller = None
            self._run_promise = None

    async def _handle_failure(self, error: Exception) -> None:
        aborted = self._abort_controller and self._abort_controller.aborted
        msg = AssistantMessage(
            content=[TextContent(text="")],
            stop_reason="aborted" if aborted else "error",
            error_message=str(error),
            model=self._state.model_config.model,
        )
        self._state.messages.append(msg)
        self._state.error_message = msg.error_message
        await self._emit(AgentEndEvent(messages=[msg]))

    async def _emit(self, event: AgentEvent) -> None:
        # Reduce internal state
        t = event.type

        if t == "message_start":
            self._state.streaming_message = event.message

        elif t == "message_update":
            self._state.streaming_message = event.message
            # Update context usage from streaming message
            msg = event.message
            if hasattr(msg, "usage") and self._state.model_config:
                u = msg.usage
                self._state.context_usage = ContextUsage(
                    input_tokens=u.input,
                    output_tokens=u.output,
                    cache_read_tokens=u.cache_read,
                    cache_write_tokens=u.cache_write,
                    total_tokens=u.total_tokens,
                    context_window=self._state.model_config.context_window,
                )

        elif t == "message_end":
            self._state.streaming_message = None
            self._state.messages.append(event.message)
            # Finalize context usage from completed assistant message
            msg = event.message
            if hasattr(msg, "usage") and msg.usage and self._state.model_config:
                u = msg.usage
                self._state.context_usage = ContextUsage(
                    input_tokens=u.input,
                    output_tokens=u.output,
                    cache_read_tokens=u.cache_read,
                    cache_write_tokens=u.cache_write,
                    total_tokens=u.total_tokens,
                    context_window=self._state.model_config.context_window,
                )

        elif t == "tool_execution_start":
            s = set(self._state.pending_tool_calls)
            s.add(event.tool_call_id)
            self._state.pending_tool_calls = s

        elif t == "tool_execution_end":
            s = set(self._state.pending_tool_calls)
            s.discard(event.tool_call_id)
            self._state.pending_tool_calls = s

        elif t == "turn_end":
            msg = event.message
            if hasattr(msg, "error_message") and msg.error_message:
                self._state.error_message = msg.error_message

        elif t == "agent_end":
            self._state.streaming_message = None

        # Dispatch to all subscribers
        signal = self._abort_controller
        for listener in list(self._listeners):
            result = listener(event, signal)
            if result is not None:
                import asyncio
                if asyncio.iscoroutine(result):
                    await result
