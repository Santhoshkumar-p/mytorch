"""
Default stream function — routes to the right provider based on ModelConfig.provider.
You can also pass any custom async stream_fn directly via AgentLoopConfig.stream_fn.
"""

from __future__ import annotations
from typing import AsyncIterator

from .types import AgentContext, AgentLoopConfig, AssistantMessageEvent, ModelConfig, AbortController
from .providers.anthropic import stream_anthropic
from .providers.openai_completions import stream_openai
from .providers.vertex import stream_vertex


async def stream_llm(
    model_config: ModelConfig,
    context: AgentContext,
    config: AgentLoopConfig,
    signal: AbortController | None = None,
) -> AsyncIterator[AssistantMessageEvent]:
    """Route to the correct provider based on model_config.provider."""
    provider = model_config.provider.lower()

    if provider == "anthropic":
        async for event in stream_anthropic(model_config, context, config, signal):
            yield event

    elif provider in ("openai", "azure", "groq", "openrouter", "xai", "cerebras", "mistral"):
        async for event in stream_openai(model_config, context, config, signal):
            yield event

    elif provider in ("vertex", "google-vertex"):
        async for event in stream_vertex(model_config, context, config, signal):
            yield event

    else:
        raise ValueError(
            f"Unknown provider '{model_config.provider}'. "
            "Set stream_fn in AgentLoopConfig to use a custom provider."
        )
