"""
Proxy stream function — for apps that route LLM calls through their own server.
Server sends stripped delta events (no partial field); we reconstruct client-side.
Maps directly to proxy.ts.
"""

from __future__ import annotations
import copy
import json
from dataclasses import dataclass, field
from typing import AsyncIterator

import httpx

from .types import (
    AgentContext, AgentLoopConfig, AssistantMessage, ModelConfig,
    TextContent, ThinkingContent, ToolCall, Usage,
    StreamStartEvent, TextStartEvent, TextDeltaEvent, TextEndEvent,
    ThinkingStartEvent, ThinkingDeltaEvent, ThinkingEndEvent,
    ToolCallStartEvent, ToolCallDeltaEvent, ToolCallEndEvent,
    StreamDoneEvent, StreamErrorEvent, AssistantMessageEvent,
    AbortController,
)


@dataclass
class ProxyStreamOptions:
    auth_token: str
    proxy_url: str  # e.g. "https://genai.example.com"
    temperature: float | None = None
    max_tokens: int | None = None


async def stream_proxy(
    model_config: ModelConfig,
    context: AgentContext,
    config: AgentLoopConfig,
    signal: AbortController | None = None,
    proxy_options: ProxyStreamOptions | None = None,
) -> AsyncIterator[AssistantMessageEvent]:
    if proxy_options is None:
        raise ValueError("proxy_options required for stream_proxy")

    partial = AssistantMessage(
        content=[],
        model=model_config.model,
        usage=Usage(),
    )

    try:
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{proxy_options.proxy_url}/api/stream",
                headers={
                    "Authorization": f"Bearer {proxy_options.auth_token}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model_config.model,
                    "context": {
                        "system_prompt": context.system_prompt,
                        "messages": _serialize_messages(context.messages),
                        "tools": _serialize_tools(context.tools or []),
                    },
                    "options": {
                        "temperature": proxy_options.temperature,
                        "max_tokens": proxy_options.max_tokens,
                        "thinking_level": config.thinking_level,
                    },
                },
                timeout=600,
            ) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    raise RuntimeError(f"Proxy error {response.status_code}: {body.decode()}")

                async for line in response.aiter_lines():
                    if signal and signal.aborted:
                        raise RuntimeError("aborted")
                    if not line.startswith("data: "):
                        continue
                    data = line[6:].strip()
                    if not data:
                        continue
                    proxy_event = json.loads(data)
                    result = _process_proxy_event(proxy_event, partial)
                    if result is not None:
                        yield result

    except RuntimeError as e:
        msg = str(e)
        reason = "aborted" if "aborted" in msg else "error"
        partial.stop_reason = reason
        partial.error_message = msg
        yield StreamErrorEvent(error=copy.copy(partial))


def _process_proxy_event(event: dict, partial: AssistantMessage) -> AssistantMessageEvent | None:
    t = event.get("type")

    if t == "start":
        return StreamStartEvent(partial=copy.copy(partial))

    elif t == "text_start":
        idx = event["content_index"]
        _grow(partial.content, idx)
        partial.content[idx] = TextContent(text="")
        return TextStartEvent(content_index=idx, partial=copy.copy(partial))

    elif t == "text_delta":
        idx = event["content_index"]
        block = _get(partial.content, idx)
        if isinstance(block, TextContent):
            block.text += event["delta"]
            return TextDeltaEvent(content_index=idx, delta=event["delta"], partial=copy.copy(partial))

    elif t == "text_end":
        idx = event["content_index"]
        block = _get(partial.content, idx)
        if isinstance(block, TextContent):
            block.text_signature = event.get("content_signature")
            return TextEndEvent(content_index=idx, content=block.text, partial=copy.copy(partial))

    elif t == "thinking_start":
        idx = event["content_index"]
        _grow(partial.content, idx)
        partial.content[idx] = ThinkingContent(thinking="")
        return ThinkingStartEvent(content_index=idx, partial=copy.copy(partial))

    elif t == "thinking_delta":
        idx = event["content_index"]
        block = _get(partial.content, idx)
        if isinstance(block, ThinkingContent):
            block.thinking += event["delta"]
            return ThinkingDeltaEvent(content_index=idx, delta=event["delta"], partial=copy.copy(partial))

    elif t == "thinking_end":
        idx = event["content_index"]
        block = _get(partial.content, idx)
        if isinstance(block, ThinkingContent):
            block.thinking_signature = event.get("content_signature")
            return ThinkingEndEvent(content_index=idx, content=block.thinking, partial=copy.copy(partial))

    elif t == "toolcall_start":
        idx = event["content_index"]
        _grow(partial.content, idx)
        partial.content[idx] = ToolCall(id=event["id"], name=event["tool_name"], arguments={})
        return ToolCallStartEvent(content_index=idx, partial=copy.copy(partial))

    elif t == "toolcall_delta":
        idx = event["content_index"]
        block = _get(partial.content, idx)
        if isinstance(block, ToolCall):
            block._partial_json += event["delta"]
            try:
                block.arguments = json.loads(block._partial_json)
            except json.JSONDecodeError:
                pass
            return ToolCallDeltaEvent(content_index=idx, delta=event["delta"], partial=copy.copy(partial))

    elif t == "toolcall_end":
        idx = event["content_index"]
        block = _get(partial.content, idx)
        if isinstance(block, ToolCall):
            block._partial_json = ""
            return ToolCallEndEvent(content_index=idx, tool_call=block, partial=copy.copy(partial))

    elif t == "done":
        partial.stop_reason = event["reason"]
        usage = event.get("usage", {})
        partial.usage.input = usage.get("input", 0)
        partial.usage.output = usage.get("output", 0)
        partial.usage.cache_read = usage.get("cache_read", 0)
        partial.usage.cache_write = usage.get("cache_write", 0)
        partial.usage.total_tokens = usage.get("total_tokens", 0)
        return StreamDoneEvent(message=copy.copy(partial))

    elif t == "error":
        partial.stop_reason = event.get("reason", "error")
        partial.error_message = event.get("error_message")
        usage = event.get("usage", {})
        partial.usage.input = usage.get("input", 0)
        partial.usage.output = usage.get("output", 0)
        partial.usage.total_tokens = usage.get("total_tokens", 0)
        return StreamErrorEvent(error=copy.copy(partial))

    return None


def _grow(lst: list, idx: int) -> None:
    while len(lst) <= idx:
        lst.append(None)


def _get(lst: list, idx: int):
    return lst[idx] if idx < len(lst) else None


def _serialize_messages(messages: list) -> list[dict]:
    out = []
    for m in messages:
        role = getattr(m, "role", None)
        if role == "user":
            out.append({"role": "user", "content": [{"type": c.type, **_content_dict(c)} for c in m.content]})
        elif role == "assistant":
            out.append({"role": "assistant", "content": [{"type": c.type, **_content_dict(c)} for c in m.content]})
        elif role == "toolResult":
            out.append({
                "role": "toolResult",
                "tool_call_id": m.tool_call_id,
                "tool_name": m.tool_name,
                "content": [{"type": c.type, **_content_dict(c)} for c in m.content],
                "is_error": m.is_error,
            })
    return out


def _content_dict(c) -> dict:
    if isinstance(c, TextContent):
        return {"text": c.text}
    if isinstance(c, ThinkingContent):
        return {"thinking": c.thinking}
    if isinstance(c, ToolCall):
        return {"id": c.id, "name": c.name, "arguments": c.arguments}
    return {}


def _serialize_tools(tools: list) -> list[dict]:
    return [{"name": t.name, "description": t.description, "parameters": t.parameters} for t in tools]
