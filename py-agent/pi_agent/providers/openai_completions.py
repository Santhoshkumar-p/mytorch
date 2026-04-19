"""
OpenAI chat completions streaming provider.
Works with any OpenAI-compatible endpoint (OpenAI, Azure, Groq, OpenRouter, etc.).
"""

from __future__ import annotations
import copy
import json
from typing import AsyncIterator

from ..types import (
    AgentContext, AgentLoopConfig, AssistantMessage, ModelConfig,
    TextContent, ThinkingContent, ImageContent, ToolCall, Usage,
    StreamStartEvent, TextStartEvent, TextDeltaEvent, TextEndEvent,
    ThinkingStartEvent, ThinkingDeltaEvent, ThinkingEndEvent,
    ToolCallStartEvent, ToolCallDeltaEvent, ToolCallEndEvent,
    StreamDoneEvent, StreamErrorEvent, AssistantMessageEvent,
    AbortController,
)
from ._shared import map_stop_reason, parse_usage_openai

_DEFAULT_BASE_URL = "https://api.openai.com/v1"

# Reasoning delta field names tried in order (different providers use different keys)
_REASONING_FIELDS = ["reasoning_content", "reasoning", "reasoning_text"]


def _build_messages(context: AgentContext, model_config: ModelConfig) -> list[dict]:
    out: list[dict] = []
    messages = context.messages

    if context.system_prompt:
        out.append({"role": "system", "content": context.system_prompt})

    i = 0
    while i < len(messages):
        msg = messages[i]
        role = getattr(msg, "role", None)

        if role == "user":
            content = msg.content
            if isinstance(content, str):
                out.append({"role": "user", "content": content})
            else:
                parts = []
                for c in content:
                    if isinstance(c, ImageContent) and model_config.supports_images:
                        parts.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{c.mime_type};base64,{c.data}"},
                        })
                    elif isinstance(c, TextContent):
                        parts.append({"type": "text", "text": c.text})
                if parts:
                    out.append({"role": "user", "content": parts})

        elif role == "assistant":
            amsg: dict = {"role": "assistant", "content": None}

            text_blocks = [c for c in msg.content if isinstance(c, TextContent) and c.text.strip()]
            if text_blocks:
                amsg["content"] = "".join(c.text for c in text_blocks)

            tool_calls = [c for c in msg.content if isinstance(c, ToolCall)]
            if tool_calls:
                amsg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in tool_calls
                ]

            # Skip empty assistant messages (aborted turns with no content)
            has_content = amsg["content"] and len(amsg["content"]) > 0
            if not has_content and not amsg.get("tool_calls"):
                i += 1
                continue

            out.append(amsg)

        elif role == "toolResult":
            # Batch all consecutive tool results into tool messages
            image_blocks: list[dict] = []
            while i < len(messages) and getattr(messages[i], "role", None) == "toolResult":
                tr = messages[i]
                text_parts = [c.text for c in tr.content if isinstance(c, TextContent)]
                text = "\n".join(text_parts) or "(see image)"

                out.append({
                    "role": "tool",
                    "tool_call_id": tr.tool_call_id,
                    "content": text,
                })

                if model_config.supports_images:
                    for c in tr.content:
                        if isinstance(c, ImageContent):
                            image_blocks.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:{c.mime_type};base64,{c.data}"},
                            })
                i += 1

            # Images from tool results go as a follow-up user message
            if image_blocks:
                out.append({
                    "role": "user",
                    "content": [{"type": "text", "text": "Images from tool results:"}] + image_blocks,
                })
            continue  # i already advanced

        i += 1

    return out


def _build_tools(tools: list) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
                "strict": False,
            },
        }
        for t in (tools or [])
    ]


async def stream_openai(
    model_config: ModelConfig,
    context: AgentContext,
    config: AgentLoopConfig,
    signal: AbortController | None = None,
) -> AsyncIterator[AssistantMessageEvent]:
    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise ImportError("pip install openai  — required for OpenAI provider")

    api_key = model_config.api_key
    if config.get_api_key:
        resolved = await config.get_api_key(model_config.provider)
        if resolved:
            api_key = resolved

    base_url = model_config.base_url or _DEFAULT_BASE_URL
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        default_headers=model_config.headers or {},
    )

    messages = _build_messages(context, model_config)
    tools = _build_tools(context.tools or [])

    params: dict = {
        "model": model_config.model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if tools:
        params["tools"] = tools
    if model_config.max_tokens:
        params["max_completion_tokens"] = model_config.max_tokens

    # Thinking / reasoning effort
    if config.thinking_level and config.thinking_level != "off" and model_config.supports_thinking:
        params["reasoning_effort"] = config.thinking_level

    partial = AssistantMessage(
        content=[],
        model=model_config.model,
        provider=model_config.provider,
        usage=Usage(),
    )

    # Track current streaming block
    current_text: TextContent | None = None
    current_thinking: ThinkingContent | None = None
    current_tool: ToolCall | None = None

    def block_idx() -> int:
        return len(partial.content) - 1

    def finish_block() -> None:
        nonlocal current_text, current_thinking, current_tool
        if current_text is not None:
            stream_events.append(TextEndEvent(
                content_index=block_idx(), content=current_text.text, partial=copy.copy(partial)
            ))
            current_text = None
        elif current_thinking is not None:
            stream_events.append(ThinkingEndEvent(
                content_index=block_idx(), content=current_thinking.thinking, partial=copy.copy(partial)
            ))
            current_thinking = None
        elif current_tool is not None:
            try:
                current_tool.arguments = json.loads(current_tool._partial_json or "{}")
            except json.JSONDecodeError:
                pass
            current_tool._partial_json = ""
            stream_events.append(ToolCallEndEvent(
                content_index=block_idx(), tool_call=current_tool, partial=copy.copy(partial)
            ))
            current_tool = None

    stream_events: list[AssistantMessageEvent] = []

    try:
        import asyncio
        abort_event = asyncio.Event()
        if signal and signal.aborted:
            raise RuntimeError("aborted")

        stream_events.append(StreamStartEvent(partial=copy.copy(partial)))
        yield StreamStartEvent(partial=copy.copy(partial))

        async with client.chat.completions.stream(**params) as stream:
            async for chunk in stream:
                if signal and signal.aborted:
                    raise RuntimeError("aborted")

                if chunk.usage:
                    u = parse_usage_openai({
                        "prompt_tokens": chunk.usage.prompt_tokens,
                        "completion_tokens": chunk.usage.completion_tokens,
                        "prompt_tokens_details": vars(chunk.usage.prompt_tokens_details) if chunk.usage.prompt_tokens_details else {},
                        "completion_tokens_details": vars(chunk.usage.completion_tokens_details) if chunk.usage.completion_tokens_details else {},
                    })
                    partial.usage.input = u["input"]
                    partial.usage.output = u["output"]
                    partial.usage.cache_read = u["cache_read"]
                    partial.usage.cache_write = u["cache_write"]
                    partial.usage.total_tokens = u["total_tokens"]

                choice = chunk.choices[0] if chunk.choices else None
                if not choice:
                    continue

                if choice.finish_reason:
                    partial.stop_reason = map_stop_reason(choice.finish_reason)

                delta = choice.delta
                if not delta:
                    continue

                # Thinking/reasoning fields (varies by provider)
                reasoning_delta: str | None = None
                for field in _REASONING_FIELDS:
                    val = getattr(delta, field, None)
                    if val:
                        reasoning_delta = val
                        break

                if reasoning_delta:
                    if current_text or current_tool:
                        finish_block()
                    if current_thinking is None:
                        current_thinking = ThinkingContent(thinking="")
                        partial.content.append(current_thinking)
                        ev = ThinkingStartEvent(content_index=block_idx(), partial=copy.copy(partial))
                        yield ev
                    current_thinking.thinking += reasoning_delta
                    ev = ThinkingDeltaEvent(
                        content_index=block_idx(), delta=reasoning_delta, partial=copy.copy(partial)
                    )
                    yield ev

                if delta.content:
                    if current_thinking or current_tool:
                        finish_block()
                    if current_text is None:
                        current_text = TextContent(text="")
                        partial.content.append(current_text)
                        ev = TextStartEvent(content_index=block_idx(), partial=copy.copy(partial))
                        yield ev
                    current_text.text += delta.content
                    ev = TextDeltaEvent(
                        content_index=block_idx(), delta=delta.content, partial=copy.copy(partial)
                    )
                    yield ev

                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        # New tool call starts when id appears
                        if tc_delta.id or (
                            current_tool is None
                        ):
                            if current_text or current_thinking or current_tool:
                                finish_block()
                            current_tool = ToolCall(
                                id=tc_delta.id or "",
                                name=(tc_delta.function.name if tc_delta.function else "") or "",
                                arguments={},
                            )
                            partial.content.append(current_tool)
                            ev = ToolCallStartEvent(content_index=block_idx(), partial=copy.copy(partial))
                            yield ev

                        if current_tool:
                            if tc_delta.id:
                                current_tool.id = tc_delta.id
                            if tc_delta.function:
                                if tc_delta.function.name:
                                    current_tool.name = tc_delta.function.name
                                if tc_delta.function.arguments:
                                    delta_str = tc_delta.function.arguments
                                    current_tool._partial_json += delta_str
                                    try:
                                        current_tool.arguments = json.loads(current_tool._partial_json)
                                    except json.JSONDecodeError:
                                        pass
                                    ev = ToolCallDeltaEvent(
                                        content_index=block_idx(),
                                        delta=delta_str,
                                        partial=copy.copy(partial),
                                    )
                                    yield ev

        # Finalize last block
        if current_text or current_thinking or current_tool:
            finish_block()

        if partial.stop_reason in ("error", "aborted"):
            raise RuntimeError(partial.error_message or "Stream ended with error")

        yield StreamDoneEvent(message=copy.copy(partial))

    except RuntimeError as e:
        msg = str(e)
        partial.stop_reason = "aborted" if "aborted" in msg else "error"
        partial.error_message = msg
        yield StreamErrorEvent(error=copy.copy(partial))
    except Exception as e:
        partial.stop_reason = "error"
        partial.error_message = str(e)
        yield StreamErrorEvent(error=copy.copy(partial))
