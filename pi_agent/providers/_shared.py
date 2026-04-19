"""
Shared message/tool conversion utilities used by all providers.
"""

from __future__ import annotations
from typing import Any
from ..types import (
    AgentMessage, AgentTool,
    UserMessage, AssistantMessage, ToolResultMessage,
    TextContent, ThinkingContent, ImageContent, ToolCall,
)


def build_tools_json(tools: list[AgentTool]) -> list[dict]:
    """Convert AgentTool list to JSON Schema tool array (provider-neutral)."""
    return [
        {"name": t.name, "description": t.description, "parameters": t.parameters}
        for t in tools
    ]


def map_stop_reason(raw: str | None) -> str:
    """Map provider finish_reason strings to our internal stop reasons."""
    if raw is None:
        return "stop"
    mapping = {
        "stop": "stop",
        "end": "stop",
        "length": "length",
        "max_tokens": "length",
        "tool_calls": "toolUse",
        "tool_use": "toolUse",
        "function_call": "toolUse",
        "content_filter": "error",
        "network_error": "error",
        "STOP": "stop",
        "MAX_TOKENS": "length",
        "SAFETY": "error",
    }
    return mapping.get(raw, "error")


def parse_usage_openai(raw: dict) -> dict:
    """
    Parse OpenAI usage dict into our Usage fields.
    Handles cache_read / cache_write from prompt_tokens_details.
    """
    prompt = raw.get("prompt_tokens", 0)
    details = raw.get("prompt_tokens_details", {}) or {}
    cached = details.get("cached_tokens", 0)
    cache_write = details.get("cache_write_tokens", 0)
    # Some providers report cached = previous hits + current writes; subtract to get reads only
    cache_read = max(0, cached - cache_write) if cache_write else cached

    completion_details = raw.get("completion_tokens_details", {}) or {}
    reasoning = completion_details.get("reasoning_tokens", 0)
    output = (raw.get("completion_tokens", 0) or 0) + reasoning

    net_input = max(0, prompt - cache_read - cache_write)
    total = net_input + output + cache_read + cache_write
    return {
        "input": net_input,
        "output": output,
        "cache_read": cache_read,
        "cache_write": cache_write,
        "total_tokens": total,
    }
