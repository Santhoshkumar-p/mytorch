"""
Google Vertex AI streaming provider via google-genai SDK.
Supports Gemini models on Vertex with thinking (budget-based and level-based).
"""

from __future__ import annotations
import copy
import time
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
from ._shared import map_stop_reason

_THINKING_BUDGETS = {
    "minimal": 128,
    "low": 2048,
    "medium": 8192,
    "high": 32768,
}

_tool_call_counter = 0


def _build_contents(messages: list) -> list:
    """Convert AgentMessage list to google-genai Content objects."""
    from google.genai import types as gtypes  # type: ignore

    out = []
    for msg in messages:
        role = getattr(msg, "role", None)

        if role == "user":
            parts = []
            for c in msg.content:
                if isinstance(c, ImageContent):
                    parts.append(gtypes.Part.from_bytes(
                        data=__import__("base64").b64decode(c.data),
                        mime_type=c.mime_type,
                    ))
                else:
                    parts.append(gtypes.Part.from_text(text=c.text))
            out.append(gtypes.Content(role="user", parts=parts))

        elif role == "assistant":
            parts = []
            for c in msg.content:
                if isinstance(c, TextContent):
                    parts.append(gtypes.Part.from_text(text=c.text))
                elif isinstance(c, ThinkingContent):
                    # Pass back thinking with thought signature for multi-turn continuity
                    part = gtypes.Part.from_text(text=c.thinking)
                    if c.thinking_signature:
                        part._thought = True
                    parts.append(part)
                elif isinstance(c, ToolCall):
                    parts.append(gtypes.Part.from_function_call(
                        name=c.name,
                        args=c.arguments,
                    ))
            out.append(gtypes.Content(role="model", parts=parts))

        elif role == "toolResult":
            out.append(gtypes.Content(
                role="user",
                parts=[gtypes.Part.from_function_response(
                    name=msg.tool_name,
                    response={"result": "\n".join(
                        c.text for c in msg.content if isinstance(c, TextContent)
                    )},
                )],
            ))

    return out


def _build_tools(tools: list):
    """Convert AgentTool list to google-genai tool declarations."""
    from google.genai import types as gtypes  # type: ignore

    declarations = []
    for t in (tools or []):
        declarations.append(gtypes.FunctionDeclaration(
            name=t.name,
            description=t.description,
            parameters=t.parameters,
        ))
    return [gtypes.Tool(function_declarations=declarations)] if declarations else None


async def stream_vertex(
    model_config: ModelConfig,
    context: AgentContext,
    config: AgentLoopConfig,
    signal: AbortController | None = None,
) -> AsyncIterator[AssistantMessageEvent]:
    try:
        from google import genai  # type: ignore
        from google.genai import types as gtypes  # type: ignore
    except ImportError:
        raise ImportError("pip install google-genai  — required for Vertex provider")

    global _tool_call_counter

    # Build client
    api_key = model_config.api_key
    if config.get_api_key:
        resolved = await config.get_api_key("vertex")
        if resolved:
            api_key = resolved

    project = model_config.vertex_project or __import__("os").environ.get("GOOGLE_CLOUD_PROJECT")
    location = model_config.vertex_location or __import__("os").environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

    if api_key:
        client = genai.Client(vertexai=True, api_key=api_key, http_options={"headers": model_config.headers})
    elif project and location:
        client = genai.Client(vertexai=True, project=project, location=location,
                              http_options={"headers": model_config.headers} if model_config.headers else None)
    else:
        raise RuntimeError("Vertex requires api_key or (vertex_project + vertex_location) in ModelConfig")

    contents = _build_contents(context.messages)
    tools = _build_tools(context.tools or [])

    gen_config: dict = {}
    if model_config.max_tokens:
        gen_config["max_output_tokens"] = model_config.max_tokens

    # Thinking config — custom budgets take priority over defaults
    if config.thinking_level and config.thinking_level != "off" and model_config.supports_thinking:
        if config.thinking_budgets:
            budget = config.thinking_budgets.get(config.thinking_level, _THINKING_BUDGETS.get(config.thinking_level, 4096))
        else:
            budget = _THINKING_BUDGETS.get(config.thinking_level, 4096)
        gen_config["thinking_config"] = gtypes.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=budget,
        )
    elif model_config.supports_thinking:
        gen_config["thinking_config"] = gtypes.ThinkingConfig(thinking_budget=0)

    if context.system_prompt:
        gen_config["system_instruction"] = context.system_prompt

    partial = AssistantMessage(
        content=[],
        model=model_config.model,
        provider=model_config.provider,
        usage=Usage(),
    )

    current_text: TextContent | None = None
    current_thinking: ThinkingContent | None = None

    def block_idx() -> int:
        return len(partial.content) - 1

    def finish_text_or_thinking() -> None:
        nonlocal current_text, current_thinking
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

    stream_events: list[AssistantMessageEvent] = []

    try:
        yield StreamStartEvent(partial=copy.copy(partial))

        # on_payload: Vertex uses SDK params, not raw JSON. Pass gen_config dict as payload.
        if config.on_payload:
            override = await config.on_payload(gen_config, model_config)
            if override is not None:
                gen_config = override

        response_stream = await client.aio.models.generate_content_stream(
            model=model_config.model,
            contents=contents,
            config=gtypes.GenerateContentConfig(**gen_config),
            **({"tools": tools} if tools else {}),
        )

        async for chunk in response_stream:
            if signal and signal.aborted:
                raise RuntimeError("aborted")

            candidate = chunk.candidates[0] if chunk.candidates else None
            if not candidate:
                continue

            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    is_thinking = getattr(part, "_thought", False) or getattr(part, "thought", False)

                    if part.text is not None:
                        if is_thinking:
                            if current_text:
                                finish_text_or_thinking()
                            if current_thinking is None:
                                current_thinking = ThinkingContent(thinking="")
                                partial.content.append(current_thinking)
                                ev = ThinkingStartEvent(content_index=block_idx(), partial=copy.copy(partial))
                                yield ev
                            current_thinking.thinking += part.text
                            if hasattr(part, "thought_signature") and part.thought_signature:
                                current_thinking.thinking_signature = part.thought_signature
                            ev = ThinkingDeltaEvent(
                                content_index=block_idx(), delta=part.text, partial=copy.copy(partial)
                            )
                            yield ev
                        else:
                            if current_thinking:
                                finish_text_or_thinking()
                            if current_text is None:
                                current_text = TextContent(text="")
                                partial.content.append(current_text)
                                ev = TextStartEvent(content_index=block_idx(), partial=copy.copy(partial))
                                yield ev
                            current_text.text += part.text
                            ev = TextDeltaEvent(
                                content_index=block_idx(), delta=part.text, partial=copy.copy(partial)
                            )
                            yield ev

                    if part.function_call:
                        if current_text or current_thinking:
                            finish_text_or_thinking()

                        fc = part.function_call
                        provided_id = getattr(fc, "id", None)
                        # Vertex sometimes gives non-unique IDs — deduplicate
                        existing_ids = {c.id for c in partial.content if isinstance(c, ToolCall)}
                        if not provided_id or provided_id in existing_ids:
                            _tool_call_counter += 1
                            tc_id = f"{fc.name}_{int(time.time())}_{_tool_call_counter}"
                        else:
                            tc_id = provided_id

                        tc = ToolCall(id=tc_id, name=fc.name or "", arguments=dict(fc.args or {}))
                        if hasattr(part, "thought_signature") and part.thought_signature:
                            tc.thought_signature = part.thought_signature

                        partial.content.append(tc)
                        idx = block_idx()
                        ev = ToolCallStartEvent(content_index=idx, partial=copy.copy(partial))
                        yield ev
                        ev = ToolCallDeltaEvent(
                            content_index=idx, delta=__import__("json").dumps(tc.arguments),
                            partial=copy.copy(partial),
                        )
                        yield ev
                        ev = ToolCallEndEvent(content_index=idx, tool_call=tc, partial=copy.copy(partial))
                        yield ev

            if candidate.finish_reason:
                partial.stop_reason = map_stop_reason(str(candidate.finish_reason))
                if any(isinstance(c, ToolCall) for c in partial.content):
                    partial.stop_reason = "toolUse"

            if chunk.usage_metadata:
                um = chunk.usage_metadata
                input_t = (um.prompt_token_count or 0) - (um.cached_content_token_count or 0)
                output_t = (um.candidates_token_count or 0) + (um.thoughts_token_count or 0)
                cache_read = um.cached_content_token_count or 0
                partial.usage = Usage(
                    input=input_t,
                    output=output_t,
                    cache_read=cache_read,
                    total_tokens=um.total_token_count or (input_t + output_t + cache_read),
                )

        # Close any open block
        if current_text or current_thinking:
            finish_text_or_thinking()

        if partial.stop_reason in ("error", "aborted"):
            raise RuntimeError(partial.error_message or "Stream error")

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
