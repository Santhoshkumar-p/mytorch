"""
Anthropic streaming provider for the coding agent.
Implements the agent Provider protocol via the Anthropic Messages API (SSE).
"""
from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

import httpx

from pi_agent import (
    AssistantMessageEventStream,
    TextContent,
    ToolCall,
    Usage,
    UsageCost,
)
from pi_agent import AssistantMessage as PiMsg

if TYPE_CHECKING:
    from pi_agent.pi_ai.types import PiAIRequest

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"


# ── Stop-reason mapping ────────────────────────────────────────────────────────

def _map_stop(raw: str | None) -> str:
    return {
        "end_turn":      "stop",
        "stop_sequence": "stop",
        "tool_use":      "toolUse",
        "max_tokens":    "length",
    }.get(raw or "", "stop")


# ── Message format conversion ─────────────────────────────────────────────────

def _to_anthropic_messages(messages) -> list[dict]:
    """Convert LlmContext messages to Anthropic wire format, grouping tool results."""
    out: list[dict] = []
    tool_buf: list[dict] = []

    def flush():
        if tool_buf:
            out.append({"role": "user", "content": list(tool_buf)})
            tool_buf.clear()

    for msg in messages:
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)

        if role == "user":
            flush()
            content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
            if isinstance(content, str):
                out.append({"role": "user", "content": content})
            else:
                blocks = []
                for b in (content or []):
                    if hasattr(b, "text"):
                        blocks.append({"type": "text", "text": b.text})
                    elif hasattr(b, "data"):
                        blocks.append({"type": "image", "source": {
                            "type": "base64",
                            "media_type": b.mime_type,
                            "data": b.data,
                        }})
                    elif isinstance(b, dict):
                        blocks.append(b)
                out.append({"role": "user", "content": blocks})

        elif role == "assistant":
            flush()
            content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", [])
            blocks = []
            for b in (content if isinstance(content, list) else []):
                if hasattr(b, "thinking"):
                    pass  # skip ThinkingContent
                elif isinstance(b, TextContent):
                    blocks.append({"type": "text", "text": b.text})
                elif isinstance(b, ToolCall):
                    blocks.append({
                        "type": "tool_use",
                        "id": b.id,
                        "name": b.name,
                        "input": b.arguments or {},
                    })
                elif isinstance(b, dict):
                    blocks.append(b)
            if blocks:
                out.append({"role": "assistant", "content": blocks})

        elif role == "toolResult":
            tool_call_id = (
                msg.get("tool_call_id") if isinstance(msg, dict)
                else getattr(msg, "tool_call_id", "")
            )
            is_error = (
                msg.get("is_error", False) if isinstance(msg, dict)
                else getattr(msg, "is_error", False)
            )
            raw_content = (
                msg.get("content", []) if isinstance(msg, dict)
                else getattr(msg, "content", [])
            )
            cb = []
            for c in (raw_content if isinstance(raw_content, list) else []):
                if hasattr(c, "text"):
                    cb.append({"type": "text", "text": c.text})
                elif isinstance(c, dict) and c.get("type") == "text":
                    cb.append(c)
            tool_buf.append({
                "type": "tool_result",
                "tool_use_id": tool_call_id,
                "content": cb,
                "is_error": is_error,
            })

    flush()
    return out


# ── Provider ───────────────────────────────────────────────────────────────────

class AnthropicProvider:
    """
    Anthropic Messages API provider (SSE streaming).
    Implements the agent Provider protocol.
    """

    async def stream(
        self,
        request: "PiAIRequest",
        abort_event: asyncio.Event | None = None,
    ) -> AssistantMessageEventStream:
        es = AssistantMessageEventStream()
        asyncio.ensure_future(self._do_stream(request, es, abort_event))
        return es

    # ── Internal streaming ─────────────────────────────────────────────────────

    async def _do_stream(
        self,
        request: "PiAIRequest",
        es: AssistantMessageEventStream,
        abort_event: asyncio.Event | None,
    ) -> None:
        try:
            payload = self._build_payload(request)
            headers = {
                "x-api-key": request.api_key or "",
                "anthropic-version": API_VERSION,
                "content-type": "application/json",
            }
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream(
                    "POST", API_URL, json=payload, headers=headers
                ) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        raise RuntimeError(
                            f"Anthropic API error {resp.status_code}: {body.decode()[:400]}"
                        )
                    await self._parse_sse(resp, es, request.model)
        except Exception as exc:
            err = PiMsg(
                content=[TextContent(text=str(exc))],
                api="anthropic",
                provider="anthropic",
                model=request.model.id,
                usage=Usage(
                    input=0, output=0, cache_read=0, cache_write=0,
                    total_tokens=0, cost=UsageCost(),
                ),
                stop_reason="error",
                error_message=str(exc),
            )
            es.push({"type": "error", "reason": "error", "error": err})

    def _build_payload(self, request: "PiAIRequest") -> dict:
        ctx = request.context
        payload: dict = {
            "model": request.model.id,
            "messages": _to_anthropic_messages(ctx.messages),
            "max_tokens": 8096,
            "stream": True,
        }
        if ctx.system_prompt:
            payload["system"] = ctx.system_prompt
        if ctx.tools:
            payload["tools"] = [
                {
                    "name": t.name,
                    "description": t.description or "",
                    "input_schema": (
                        dict(t.parameters) if t.parameters
                        else {"type": "object", "properties": {}}
                    ),
                }
                for t in ctx.tools
            ]
        if request.reasoning and request.reasoning != "off":
            budget = (request.thinking_budgets or {}).get(request.reasoning, 8000)
            payload["thinking"] = {"type": "enabled", "budget_tokens": budget}
        return payload

    async def _parse_sse(self, resp, es: AssistantMessageEventStream, model) -> None:
        partial = PiMsg(
            content=[],
            api="anthropic",
            provider="anthropic",
            model=model.id,
            usage=Usage(
                input=0, output=0, cache_read=0, cache_write=0,
                total_tokens=0, cost=UsageCost(),
            ),
            stop_reason="stop",
        )
        es.push({"type": "start", "partial": partial})

        sse_to_ci: dict[int, int] = {}
        json_bufs: dict[int, str] = {}
        usage = partial.usage
        stop_reason = "stop"

        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                ev = json.loads(data)
            except json.JSONDecodeError:
                continue
            etype = ev.get("type")

            if etype == "message_start":
                u = ev.get("message", {}).get("usage", {})
                usage.input = u.get("input_tokens", 0)
                usage.output = u.get("output_tokens", 0)
                usage.total_tokens = usage.input + usage.output

            elif etype == "content_block_start":
                si = ev["index"]
                blk = ev["content_block"]
                bt = blk.get("type")
                ci = len(partial.content)
                sse_to_ci[si] = ci
                if bt == "text":
                    partial.content.append(TextContent(text=""))
                    es.push({"type": "text_start", "content_index": si, "partial": partial})
                elif bt == "thinking":
                    from pi_agent import ThinkingContent
                    partial.content.append(ThinkingContent(thinking=""))
                    es.push({"type": "thinking_start", "content_index": si, "partial": partial})
                elif bt == "tool_use":
                    json_bufs[si] = ""
                    partial.content.append(
                        ToolCall(id=blk["id"], name=blk["name"], arguments={})
                    )
                    es.push({"type": "toolcall_start", "content_index": si, "partial": partial})

            elif etype == "content_block_delta":
                si = ev["index"]
                delta = ev["delta"]
                dtype = delta.get("type")
                ci = sse_to_ci.get(si)

                if dtype == "text_delta" and ci is not None:
                    text = delta.get("text", "")
                    b = partial.content[ci]
                    if isinstance(b, TextContent):
                        b.text = b.text + text
                    es.push({
                        "type": "text_delta",
                        "content_index": si,
                        "delta": text,
                        "partial": partial,
                    })

                elif dtype == "thinking_delta" and ci is not None:
                    from pi_agent import ThinkingContent
                    text = delta.get("thinking", "")
                    b = partial.content[ci]
                    if isinstance(b, ThinkingContent):
                        b.thinking = b.thinking + text
                    es.push({
                        "type": "thinking_delta",
                        "content_index": si,
                        "delta": text,
                        "partial": partial,
                    })

                elif dtype == "input_json_delta" and si in json_bufs:
                    chunk = delta.get("partial_json", "")
                    json_bufs[si] += chunk
                    es.push({
                        "type": "toolcall_delta",
                        "content_index": si,
                        "delta": chunk,
                        "partial": partial,
                    })

            elif etype == "content_block_stop":
                si = ev["index"]
                ci = sse_to_ci.get(si)
                if ci is not None and ci < len(partial.content):
                    b = partial.content[ci]
                    if isinstance(b, TextContent):
                        es.push({
                            "type": "text_end",
                            "content_index": si,
                            "content": b.text,
                            "partial": partial,
                        })
                    else:
                        from pi_agent import ThinkingContent
                        if isinstance(b, ThinkingContent):
                            es.push({
                                "type": "thinking_end",
                                "content_index": si,
                                "content": b.thinking,
                                "partial": partial,
                            })
                        elif isinstance(b, ToolCall) and si in json_bufs:
                            try:
                                b.arguments = json.loads(json_bufs[si]) if json_bufs[si] else {}
                            except json.JSONDecodeError:
                                b.arguments = {}
                            es.push({
                                "type": "toolcall_end",
                                "content_index": si,
                                "tool_call": b,
                                "partial": partial,
                            })

            elif etype == "message_delta":
                delta = ev.get("delta", {})
                stop_reason = _map_stop(delta.get("stop_reason"))
                u = ev.get("usage", {})
                usage.output       = u.get("output_tokens", usage.output)
                usage.cache_read   = u.get("cache_read_input_tokens", 0)
                usage.cache_write  = u.get("cache_creation_input_tokens", 0)
                usage.total_tokens = (
                    usage.input + usage.output + usage.cache_read + usage.cache_write
                )

            elif etype == "message_stop":
                break

        partial.usage = usage
        partial.stop_reason = stop_reason  # type: ignore[assignment]

        if stop_reason in ("stop", "length", "toolUse"):
            es.push({"type": "done", "reason": stop_reason, "message": partial})
        else:
            es.push({"type": "error", "reason": "error", "error": partial})
