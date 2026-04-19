"""
Core types for pi-agent. Dataclasses for loop objects, plain callables for hooks.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Union


# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    model: str
    provider: str = "anthropic"          # "anthropic" | "openai" | "vertex" | custom
    base_url: str = ""                   # defaults per provider if empty
    api_key: str = ""
    headers: dict[str, str] = field(default_factory=dict)
    max_tokens: int = 8096
    context_window: int = 200_000        # for context usage tracking
    supports_images: bool = True
    supports_thinking: bool = False
    # Vertex-specific
    vertex_project: str | None = None
    vertex_location: str | None = None


# ---------------------------------------------------------------------------
# Content blocks
# ---------------------------------------------------------------------------

@dataclass
class TextContent:
    text: str
    type: str = "text"
    text_signature: str | None = None   # for cache continuity (provider-specific)


@dataclass
class ThinkingContent:
    thinking: str
    type: str = "thinking"
    thinking_signature: str | None = None
    redacted: bool = False


@dataclass
class ImageContent:
    data: str        # base64 encoded
    mime_type: str   # "image/jpeg" | "image/png" | "image/gif" | "image/webp"
    type: str = "image"


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict
    type: str = "toolCall"
    thought_signature: str | None = None  # Google Vertex thought context
    _partial_json: str = field(default="", repr=False)


Content = Union[TextContent, ThinkingContent, ImageContent, ToolCall]


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

@dataclass
class Usage:
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0


@dataclass
class UserMessage:
    content: list[TextContent | ImageContent]
    role: str = "user"
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))


@dataclass
class AssistantMessage:
    content: list[TextContent | ThinkingContent | ToolCall]
    stop_reason: str = "stop"           # "stop" | "length" | "toolUse" | "error" | "aborted"
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    provider: str = ""
    response_id: str | None = None      # provider-specific response identifier
    error_message: str | None = None
    role: str = "assistant"
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))


@dataclass
class ToolResultMessage:
    tool_call_id: str
    tool_name: str
    content: list[TextContent | ImageContent]
    is_error: bool = False
    details: Any = None
    role: str = "toolResult"
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))


AgentMessage = Union[UserMessage, AssistantMessage, ToolResultMessage, Any]


# ---------------------------------------------------------------------------
# Streaming events emitted by stream_fn
# ---------------------------------------------------------------------------

@dataclass
class StreamStartEvent:
    partial: AssistantMessage
    type: str = "start"

@dataclass
class TextStartEvent:
    content_index: int
    partial: AssistantMessage
    type: str = "text_start"

@dataclass
class TextDeltaEvent:
    content_index: int
    delta: str
    partial: AssistantMessage
    type: str = "text_delta"

@dataclass
class TextEndEvent:
    content_index: int
    content: str
    partial: AssistantMessage
    type: str = "text_end"

@dataclass
class ThinkingStartEvent:
    content_index: int
    partial: AssistantMessage
    type: str = "thinking_start"

@dataclass
class ThinkingDeltaEvent:
    content_index: int
    delta: str
    partial: AssistantMessage
    type: str = "thinking_delta"

@dataclass
class ThinkingEndEvent:
    content_index: int
    content: str
    partial: AssistantMessage
    type: str = "thinking_end"

@dataclass
class ToolCallStartEvent:
    content_index: int
    partial: AssistantMessage
    type: str = "toolcall_start"

@dataclass
class ToolCallDeltaEvent:
    content_index: int
    delta: str
    partial: AssistantMessage
    type: str = "toolcall_delta"

@dataclass
class ToolCallEndEvent:
    content_index: int
    tool_call: ToolCall
    partial: AssistantMessage
    type: str = "toolcall_end"

@dataclass
class StreamDoneEvent:
    message: AssistantMessage
    type: str = "done"

@dataclass
class StreamErrorEvent:
    error: AssistantMessage
    type: str = "error"


AssistantMessageEvent = Union[
    StreamStartEvent, TextStartEvent, TextDeltaEvent, TextEndEvent,
    ThinkingStartEvent, ThinkingDeltaEvent, ThinkingEndEvent,
    ToolCallStartEvent, ToolCallDeltaEvent, ToolCallEndEvent,
    StreamDoneEvent, StreamErrorEvent,
]


# ---------------------------------------------------------------------------
# Agent events
# ---------------------------------------------------------------------------

@dataclass
class AgentStartEvent:
    type: str = "agent_start"

@dataclass
class AgentEndEvent:
    messages: list[AgentMessage]
    type: str = "agent_end"

@dataclass
class TurnStartEvent:
    type: str = "turn_start"

@dataclass
class TurnEndEvent:
    message: AgentMessage
    tool_results: list[ToolResultMessage]
    type: str = "turn_end"

@dataclass
class MessageStartEvent:
    message: AgentMessage
    type: str = "message_start"

@dataclass
class MessageUpdateEvent:
    message: AgentMessage
    assistant_message_event: AssistantMessageEvent
    type: str = "message_update"

@dataclass
class MessageEndEvent:
    message: AgentMessage
    type: str = "message_end"

@dataclass
class ToolExecutionStartEvent:
    tool_call_id: str
    tool_name: str
    args: Any
    type: str = "tool_execution_start"

@dataclass
class ToolExecutionUpdateEvent:
    tool_call_id: str
    tool_name: str
    args: Any
    partial_result: Any
    type: str = "tool_execution_update"

@dataclass
class ToolExecutionEndEvent:
    tool_call_id: str
    tool_name: str
    result: Any
    is_error: bool
    type: str = "tool_execution_end"


AgentEvent = Union[
    AgentStartEvent, AgentEndEvent,
    TurnStartEvent, TurnEndEvent,
    MessageStartEvent, MessageUpdateEvent, MessageEndEvent,
    ToolExecutionStartEvent, ToolExecutionUpdateEvent, ToolExecutionEndEvent,
]


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

@dataclass
class AgentToolResult:
    content: list[TextContent | ImageContent]
    details: Any = None


AgentToolUpdateCallback = Callable[[AgentToolResult], None]
ToolExecutionMode = Literal["sequential", "parallel"]
ThinkingLevel = Literal["off", "minimal", "low", "medium", "high", "xhigh"]

# Transport preference forwarded to providers that support multiple transports.
# Providers that don't support a given transport ignore it.
Transport = Literal["sse", "websocket", "auto"]

# Per-level thinking token budget overrides. Unset levels use provider defaults.
@dataclass
class ThinkingBudgets:
    minimal: int | None = None
    low: int | None = None
    medium: int | None = None
    high: int | None = None
    xhigh: int | None = None

    def get(self, level: str, default: int) -> int:
        return getattr(self, level, None) or default


@dataclass
class AgentTool:
    name: str
    label: str
    description: str
    parameters: dict   # JSON Schema
    execute: Callable  # async (tool_call_id, params, signal, on_update) -> AgentToolResult
    prepare_arguments: Callable | None = None
    execution_mode: ToolExecutionMode = "parallel"


# ---------------------------------------------------------------------------
# Context and loop config
# ---------------------------------------------------------------------------

@dataclass
class AgentContext:
    system_prompt: str
    messages: list[AgentMessage]
    tools: list[AgentTool] | None = None


@dataclass
class BeforeToolCallContext:
    assistant_message: AssistantMessage
    tool_call: ToolCall
    args: Any
    context: AgentContext


@dataclass
class AfterToolCallContext:
    assistant_message: AssistantMessage
    tool_call: ToolCall
    args: Any
    result: AgentToolResult
    is_error: bool
    context: AgentContext


@dataclass
class BeforeToolCallResult:
    block: bool = False
    reason: str | None = None


@dataclass
class AfterToolCallResult:
    content: list[TextContent | ImageContent] | None = None
    details: Any = None
    is_error: bool | None = None


@dataclass
class AgentLoopConfig:
    model_config: ModelConfig
    convert_to_llm: Callable          # (list[AgentMessage]) -> list[AgentMessage]
    stream_fn: Callable | None = None
    transform_context: Callable | None = None
    get_api_key: Callable | None = None
    get_steering_messages: Callable | None = None
    get_follow_up_messages: Callable | None = None
    tool_execution: ToolExecutionMode = "parallel"
    before_tool_call: Callable | None = None
    after_tool_call: Callable | None = None
    thinking_level: ThinkingLevel = "off"
    # Forwarded to providers
    session_id: str | None = None
    thinking_budgets: ThinkingBudgets | None = None
    transport: Transport = "sse"
    max_retry_delay_ms: int | None = None
    # Hooks into the HTTP layer — called before sending / after receiving
    on_payload: Callable | None = None  # async (payload: dict, model_config) -> dict | None
    on_response: Callable | None = None  # async (status: int, headers: dict, model_config) -> None


# ---------------------------------------------------------------------------
# Context window tracking — full breakdown like the UI shows
# ---------------------------------------------------------------------------

@dataclass
class ContextBucket:
    """A single named slice of the context window."""
    tokens: int = 0
    count: int = 0   # number of items (files, tools, etc.) — 0 means N/A


@dataclass
class ContextWindowBreakdown:
    """
    Mirrors the UI breakdown:
      Messages / System prompt / System tools / Skills /
      MCP tools / Memory files / deferred variants / Autocompact buffer / Free space
    """
    messages: ContextBucket = field(default_factory=ContextBucket)
    system_prompt: ContextBucket = field(default_factory=ContextBucket)
    system_tools: ContextBucket = field(default_factory=ContextBucket)
    system_tools_deferred: ContextBucket = field(default_factory=ContextBucket)
    skills: ContextBucket = field(default_factory=ContextBucket)
    mcp_tools: ContextBucket = field(default_factory=ContextBucket)
    mcp_tools_deferred: ContextBucket = field(default_factory=ContextBucket)
    memory_files: ContextBucket = field(default_factory=ContextBucket)
    autocompact_buffer: ContextBucket = field(default_factory=ContextBucket)

    def total_tracked(self) -> int:
        return (
            self.messages.tokens
            + self.system_prompt.tokens
            + self.system_tools.tokens
            + self.system_tools_deferred.tokens
            + self.skills.tokens
            + self.mcp_tools.tokens
            + self.mcp_tools_deferred.tokens
            + self.memory_files.tokens
            + self.autocompact_buffer.tokens
        )


@dataclass
class ContextUsage:
    # From the API response
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    total_tokens: int = 0
    context_window: int = 0

    # Optional fine-grained breakdown (set by higher-level code)
    breakdown: ContextWindowBreakdown | None = None

    @property
    def used_fraction(self) -> float:
        if self.context_window == 0:
            return 0.0
        return self.total_tokens / self.context_window

    @property
    def free_tokens(self) -> int:
        return max(0, self.context_window - self.total_tokens)

    def summary(self) -> str:
        total_k = self.total_tokens / 1000
        window_k = self.context_window / 1000
        pct = self.used_fraction * 100
        lines = [f"Context: {total_k:.1f}k / {window_k:.1f}k ({pct:.0f}%)"]
        if self.breakdown:
            b = self.breakdown
            cw = self.context_window or 1
            for name, bucket in [
                ("Messages", b.messages),
                ("System prompt", b.system_prompt),
                ("System tools", b.system_tools),
                ("System tools (deferred)", b.system_tools_deferred),
                ("Skills", b.skills),
                ("MCP tools", b.mcp_tools),
                ("MCP tools (deferred)", b.mcp_tools_deferred),
                ("Memory files", b.memory_files),
                ("Autocompact buffer", b.autocompact_buffer),
            ]:
                if bucket.tokens > 0:
                    pct_b = bucket.tokens / cw * 100
                    count_str = f"  ({bucket.count} items)" if bucket.count else ""
                    lines.append(f"  {name}: {bucket.tokens/1000:.1f}k  {pct_b:.1f}%{count_str}")
            free = self.free_tokens
            lines.append(f"  Free space: {free/1000:.1f}k  {free/cw*100:.1f}%")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Abort controller
# ---------------------------------------------------------------------------

class AbortController:
    def __init__(self):
        self._aborted = False

    def abort(self):
        self._aborted = True

    @property
    def aborted(self) -> bool:
        return self._aborted

    @property
    def signal(self) -> "AbortController":
        return self


StreamFn = Callable  # async (ModelConfig, AgentContext, AgentLoopConfig, AbortController|None) -> AsyncIterator[AssistantMessageEvent]
