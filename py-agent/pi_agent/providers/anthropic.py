"""
Anthropic messages API — SSE streaming.
"""

from __future__ import annotations
import copy
import json
from typing import AsyncIterator

import httpx

from ..types import (
    AgentContext, AgentLoopConfig, AssistantMessage, ModelConfig,
    TextContent, ThinkingContent, ImageContent, ToolCall, Usage,
    StreamStartEvent, TextStartEvent, TextDeltaEvent, TextEndEvent,
    ThinkingStartEvent, ThinkingDeltaEvent, ThinkingEndEvent,
    ToolCallStartEvent, ToolCallDeltaEvent, ToolCallEndEvent,
    StreamDoneEvent, StreamErrorEvent, AssistantMessageEvent,
    ToolResultMessage, UserMessage, AbortController,
)
from ._shared import map_stop_reason

_THINKING_BUDGETS = {
    "minimal": 1024,
    "low": 2048,
    "medium": 4096,
    "high": 8192,
    "xhigh": 16384,
}

_DEFAULT_BASE_URL = "https://api.anthropic.com"


def _build_messages(messages: list) -> list[dict]:
    out = []
    for msg in messages:
        role = getattr(msg, "role", None)

        if role == "user":
            content = []
            for c in msg.content:
                if isinstance(c, ImageContent):
                    content.append({
                        "type": "image",
                        "source": {"type": "base64", "media_type": c.mime_type, "data": c.data},
                    })
                else:
                    content.append({"type": "text", "text": c.text})
            out.append({"role": "user", "content": content})

        elif role == "assistant":
            content = []
            for c in msg.content:
                if isinstance(c, TextContent):
                    content.append({"type": "text", "text": c.text})
                elif isinstance(c, ThinkingContent):
                    block: dict = {"type": "thinking", "thinking": c.thinking}
                    if c.thinking_signature:
                        block["signature"] = c.thinking_signature
                    content.append(block)
                elif isinstance(c, ToolCall):
                    content.append({
                        "type": "tool_use",
                        "id": c.id,
                        "name": c.name,
                        "input": c.arguments,
                    })
            out.append({"role": "assistant", "content": content})

        elif role == "toolResult":
            tool_content = []
            for c in msg.content:
                if isinstance(c, ImageContent):
                    tool_content.append({
                        "type": "image",
                        "source": {"type": "base64", "media_type": c.mime_type, "data": c.data},
                    })
                else:
                    tool_content.append({"type": "text", "text": c.text})
            out.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id,
                    "content": tool_content,
                    "is_error": msg.is_error,
                }],
            })

    return out


def _build_tools(tools: list) -> list[dict]:
    return [
        {"name": t.name, "description": t.description, "input_schema": t.parameters}
        for t in (tools or [])
    ]


async def stream_anthropic(
    model_config: ModelConfig,
    context: AgentContext,
    config: AgentLoopConfig,
    signal: AbortController | None = None,
) -> AsyncIterator[AssistantMessageEvent]:
    api_key = model_config.api_key
    if config.get_api_key:
        resolved = await config.get_api_key("anthropic")
        if resolved:
            api_key = resolved

    base_url = model_config.base_url or _DEFAULT_BASE_URL
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
        **model_config.headers,
    }

    llm_messages = _build_messages(context.messages)
    tools = _build_tools(context.tools or [])

    body: dict = {
        "model": model_config.model,
        "max_tokens": model_config.max_tokens,
        "messages": llm_messages,
        "stream": True,
    }
    if context.system_prompt:
        body["system"] = context.system_prompt
    if tools:
        body["tools"] = tools

    budget = _THINKING_BUDGETS.get(config.thinking_level)
    if budget and model_config.supports_thinking:
        body["thinking"] = {"type": "enabled", "budget_tokens": budget}

    partial = AssistantMessage(
        content=[],
        model=model_config.model,
        provider=model_config.provider,
        usage=Usage(),
    )

    try:
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{base_url}/v1/messages",
                headers=headers,
                json=body,
                timeout=600,
            ) as resp:
                if resp.status_code != 200:
                    text = await resp.aread()
                    raise RuntimeError(f"Anthropic {resp.status_code}: {text.decode()}")

                async for line in resp.aiter_lines():
                    if signal and signal.aborted:
                        raise RuntimeError("aborted")
                    if not line.startswith("data: "):
                        continue
                    data = line[6:].strip()
                    if not data or data == "[DONE]":
                        continue
                    event = json.loads(data)
                    result = _process(event, partial)
                    if result is not None:
                        yield result

    except RuntimeError as e:
        msg = str(e)
        partial.stop_reason = "aborted" if "aborted" in msg else "error"
        partial.error_message = msg
        yield StreamErrorEvent(error=copy.copy(partial))


def _process(event: dict, partial: AssistantMessage) -> AssistantMessageEvent | None:
    t = event.get("type")

    if t == "message_start":
        u = event.get("message", {}).get("usage", {})
        partial.usage.input = u.get("input_tokens", 0)
        partial.usage.cache_read = u.get("cache_read_input_tokens", 0)
        partial.usage.cache_write = u.get("cache_creation_input_tokens", 0)
        return StreamStartEvent(partial=copy.copy(partial))

    elif t == "content_block_start":
        idx = event["index"]
        block = event["content_block"]
        _grow(partial.content, idx)

        btype = block["type"]
        if btype == "text":
            partial.content[idx] = TextContent(text="")
            return TextStartEvent(content_index=idx, partial=copy.copy(partial))
        elif btype == "thinking":
            partial.content[idx] = ThinkingContent(thinking="")
            return ThinkingStartEvent(content_index=idx, partial=copy.copy(partial))
        elif btype == "tool_use":
            partial.content[idx] = ToolCall(id=block["id"], name=block["name"], arguments={})
            return ToolCallStartEvent(content_index=idx, partial=copy.copy(partial))

    elif t == "content_block_delta":
        idx = event["index"]
        delta = event["delta"]
        dtype = delta["type"]
        blk = _get(partial.content, idx)

        if dtype == "text_delta" and isinstance(blk, TextContent):
            blk.text += delta["text"]
            return TextDeltaEvent(content_index=idx, delta=delta["text"], partial=copy.copy(partial))

        elif dtype == "thinking_delta" and isinstance(blk, ThinkingContent):
            blk.thinking += delta["thinking"]
            return ThinkingDeltaEvent(content_index=idx, delta=delta["thinking"], partial=copy.copy(partial))

        elif dtype == "input_json_delta" and isinstance(blk, ToolCall):
            blk._partial_json += delta["partial_json"]
            try:
                blk.arguments = json.loads(blk._partial_json)
            except json.JSONDecodeError:
                pass
            return ToolCallDeltaEvent(content_index=idx, delta=delta["partial_json"], partial=copy.copy(partial))

    elif t == "content_block_stop":
        idx = event["index"]
        blk = _get(partial.content, idx)
        if isinstance(blk, TextContent):
            return TextEndEvent(content_index=idx, content=blk.text, partial=copy.copy(partial))
        elif isinstance(blk, ThinkingContent):
            return ThinkingEndEvent(content_index=idx, content=blk.thinking, partial=copy.copy(partial))
        elif isinstance(blk, ToolCall):
            blk._partial_json = ""
            return ToolCallEndEvent(content_index=idx, tool_call=blk, partial=copy.copy(partial))

    elif t == "message_delta":
        delta = event.get("delta", {})
        usage = event.get("usage", {})
        partial.stop_reason = map_stop_reason(delta.get("stop_reason"))
        partial.usage.output = usage.get("output_tokens", partial.usage.output)
        partial.usage.total_tokens = (
            partial.usage.input + partial.usage.output
            + partial.usage.cache_read + partial.usage.cache_write
        )

    elif t == "message_stop":
        return StreamDoneEvent(message=copy.copy(partial))

    elif t == "error":
        err = event.get("error", {})
        partial.stop_reason = "error"
        partial.error_message = err.get("message", "Unknown error")
        return StreamErrorEvent(error=copy.copy(partial))

    return None


def _grow(lst: list, idx: int) -> None:
    while len(lst) <= idx:
        lst.append(None)  # type: ignore


def _get(lst: list, idx: int):
    return lst[idx] if idx < len(lst) else None
