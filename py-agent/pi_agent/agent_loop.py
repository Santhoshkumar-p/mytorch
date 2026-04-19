"""
Pure agent loop — no state, no side effects beyond calling emit.
Maps directly to agent-loop.ts.
"""

from __future__ import annotations
import copy
import json
import time
from typing import AsyncIterator, Callable

import jsonschema

from .types import (
    AgentContext, AgentEvent, AgentLoopConfig, AgentMessage, AgentTool,
    AgentToolResult, AssistantMessage, ToolCall, ToolResultMessage,
    BeforeToolCallContext, AfterToolCallContext,
    BeforeToolCallResult, AfterToolCallResult,
    TextContent, ImageContent, Usage,
    AgentStartEvent, AgentEndEvent, TurnStartEvent, TurnEndEvent,
    MessageStartEvent, MessageUpdateEvent, MessageEndEvent,
    ToolExecutionStartEvent, ToolExecutionUpdateEvent, ToolExecutionEndEvent,
    StreamDoneEvent, StreamErrorEvent,
    AbortController,
)
from .llm import stream_llm

AgentEventSink = Callable[[AgentEvent], None]  # sync or async, handled by agent


async def run_agent_loop(
    prompts: list[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: AbortController | None = None,
) -> list[AgentMessage]:
    new_messages = list(prompts)
    current = AgentContext(
        system_prompt=context.system_prompt,
        messages=list(context.messages) + list(prompts),
        tools=list(context.tools or []),
    )

    await emit(AgentStartEvent())
    await emit(TurnStartEvent())
    for prompt in prompts:
        await emit(MessageStartEvent(message=prompt))
        await emit(MessageEndEvent(message=prompt))

    await _run_loop(current, new_messages, config, signal, emit)
    return new_messages


async def run_agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: AbortController | None = None,
) -> list[AgentMessage]:
    if not context.messages:
        raise ValueError("Cannot continue: no messages in context")
    last = context.messages[-1]
    if getattr(last, "role", None) == "assistant":
        raise ValueError("Cannot continue from message role: assistant")

    new_messages: list[AgentMessage] = []
    current = AgentContext(
        system_prompt=context.system_prompt,
        messages=list(context.messages),
        tools=list(context.tools or []),
    )

    await emit(AgentStartEvent())
    await emit(TurnStartEvent())
    await _run_loop(current, new_messages, config, signal, emit)
    return new_messages


async def _run_loop(
    context: AgentContext,
    new_messages: list[AgentMessage],
    config: AgentLoopConfig,
    signal: AbortController | None,
    emit: AgentEventSink,
) -> None:
    first_turn = True
    pending: list[AgentMessage] = []
    if config.get_steering_messages:
        pending = await config.get_steering_messages()

    while True:
        has_tool_calls = True

        while has_tool_calls or pending:
            if not first_turn:
                await emit(TurnStartEvent())
            else:
                first_turn = False

            # Inject pending steering messages
            if pending:
                for msg in pending:
                    await emit(MessageStartEvent(message=msg))
                    await emit(MessageEndEvent(message=msg))
                    context.messages.append(msg)
                    new_messages.append(msg)
                pending = []

            message = await _stream_assistant_response(context, config, signal, emit)
            new_messages.append(message)

            if message.stop_reason in ("error", "aborted"):
                await emit(TurnEndEvent(message=message, tool_results=[]))
                await emit(AgentEndEvent(messages=new_messages))
                return

            tool_calls = [c for c in message.content if c.type == "toolCall"]
            has_tool_calls = len(tool_calls) > 0

            tool_results: list[ToolResultMessage] = []
            if has_tool_calls:
                tool_results = await _execute_tool_calls(context, message, config, signal, emit)
                for r in tool_results:
                    context.messages.append(r)
                    new_messages.append(r)

            await emit(TurnEndEvent(message=message, tool_results=tool_results))

            if config.get_steering_messages:
                pending = await config.get_steering_messages()

        # Check for follow-up messages
        follow_ups: list[AgentMessage] = []
        if config.get_follow_up_messages:
            follow_ups = await config.get_follow_up_messages()

        if follow_ups:
            pending = follow_ups
            continue

        break

    await emit(AgentEndEvent(messages=new_messages))


async def _stream_assistant_response(
    context: AgentContext,
    config: AgentLoopConfig,
    signal: AbortController | None,
    emit: AgentEventSink,
) -> AssistantMessage:
    messages = context.messages
    if config.transform_context:
        messages = await config.transform_context(messages, signal)

    llm_messages = await config.convert_to_llm(messages)

    llm_context = AgentContext(
        system_prompt=context.system_prompt,
        messages=llm_messages,
        tools=context.tools,
    )

    stream_fn = config.stream_fn or stream_llm
    stream = stream_fn(config.model_config, llm_context, config, signal)

    partial: AssistantMessage | None = None
    added_partial = False

    async for event in stream:
        if event.type == "start":
            partial = event.partial
            context.messages.append(partial)
            added_partial = True
            await emit(MessageStartEvent(message=copy.copy(partial)))

        elif event.type in (
            "text_start", "text_delta", "text_end",
            "thinking_start", "thinking_delta", "thinking_end",
            "toolcall_start", "toolcall_delta", "toolcall_end",
        ):
            if partial is not None:
                partial = event.partial
                context.messages[-1] = partial
                await emit(MessageUpdateEvent(
                    message=copy.copy(partial),
                    assistant_message_event=event,
                ))

        elif event.type in ("done", "error"):
            final = event.message if event.type == "done" else event.error
            if added_partial:
                context.messages[-1] = final
            else:
                context.messages.append(final)
                await emit(MessageStartEvent(message=copy.copy(final)))
            await emit(MessageEndEvent(message=final))
            return final

    # Fallback: shouldn't normally reach here
    final = partial or AssistantMessage(
        content=[TextContent(text="")],
        stop_reason="error",
        error_message="Stream ended without a done event",
    )
    if added_partial:
        context.messages[-1] = final
    else:
        context.messages.append(final)
        await emit(MessageStartEvent(message=copy.copy(final)))
    await emit(MessageEndEvent(message=final))
    return final


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

async def _execute_tool_calls(
    context: AgentContext,
    assistant_message: AssistantMessage,
    config: AgentLoopConfig,
    signal: AbortController | None,
    emit: AgentEventSink,
) -> list[ToolResultMessage]:
    tool_calls = [c for c in assistant_message.content if c.type == "toolCall"]

    # Force sequential if any tool has sequential mode or config says sequential
    has_sequential = any(
        _find_tool(context.tools, tc.name) and
        _find_tool(context.tools, tc.name).execution_mode == "sequential"
        for tc in tool_calls
    )

    if config.tool_execution == "sequential" or has_sequential:
        return await _sequential(context, assistant_message, tool_calls, config, signal, emit)
    return await _parallel(context, assistant_message, tool_calls, config, signal, emit)


async def _sequential(
    context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: list[ToolCall],
    config: AgentLoopConfig,
    signal: AbortController | None,
    emit: AgentEventSink,
) -> list[ToolResultMessage]:
    results = []
    for tc in tool_calls:
        await emit(ToolExecutionStartEvent(tool_call_id=tc.id, tool_name=tc.name, args=tc.arguments))
        prep = await _prepare(context, assistant_message, tc, config, signal)

        if prep["kind"] == "immediate":
            results.append(await _emit_outcome(tc, prep["result"], prep["is_error"], emit))
        else:
            executed = await _execute_prepared(prep, signal, emit)
            results.append(await _finalize(context, assistant_message, prep, executed, config, signal, emit))

    return results


async def _parallel(
    context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: list[ToolCall],
    config: AgentLoopConfig,
    signal: AbortController | None,
    emit: AgentEventSink,
) -> list[ToolResultMessage]:
    import asyncio

    results = []
    runnable = []

    for tc in tool_calls:
        await emit(ToolExecutionStartEvent(tool_call_id=tc.id, tool_name=tc.name, args=tc.arguments))
        prep = await _prepare(context, assistant_message, tc, config, signal)

        if prep["kind"] == "immediate":
            results.append(await _emit_outcome(tc, prep["result"], prep["is_error"], emit))
        else:
            runnable.append(prep)

    # Launch all runnable in parallel, but finalize in order
    executions = [asyncio.ensure_future(_execute_prepared(p, signal, emit)) for p in runnable]
    for prep, execution in zip(runnable, executions):
        executed = await execution
        results.append(await _finalize(context, assistant_message, prep, executed, config, signal, emit))

    return results


def _find_tool(tools: list[AgentTool] | None, name: str) -> AgentTool | None:
    if not tools:
        return None
    return next((t for t in tools if t.name == name), None)


def _validate_args(tool: AgentTool, args: dict) -> dict:
    try:
        jsonschema.validate(instance=args, schema=tool.parameters)
    except jsonschema.ValidationError as e:
        raise ValueError(f"Invalid args for {tool.name}: {e.message}")
    return args


async def _prepare(
    context: AgentContext,
    assistant_message: AssistantMessage,
    tool_call: ToolCall,
    config: AgentLoopConfig,
    signal: AbortController | None,
) -> dict:
    tool = _find_tool(context.tools, tool_call.name)
    if not tool:
        return {
            "kind": "immediate",
            "result": _error_result(f"Tool '{tool_call.name}' not found"),
            "is_error": True,
        }

    try:
        args = tool_call.arguments
        if tool.prepare_arguments:
            args = tool.prepare_arguments(args)
        _validate_args(tool, args)

        if config.before_tool_call:
            before = await config.before_tool_call(
                BeforeToolCallContext(
                    assistant_message=assistant_message,
                    tool_call=tool_call,
                    args=args,
                    context=context,
                ),
                signal,
            )
            if before and before.block:
                return {
                    "kind": "immediate",
                    "result": _error_result(before.reason or "Tool execution was blocked"),
                    "is_error": True,
                }

        return {"kind": "prepared", "tool_call": tool_call, "tool": tool, "args": args}

    except Exception as e:
        return {"kind": "immediate", "result": _error_result(str(e)), "is_error": True}


async def _execute_prepared(
    prep: dict,
    signal: AbortController | None,
    emit: AgentEventSink,
) -> dict:
    tool: AgentTool = prep["tool"]
    tool_call: ToolCall = prep["tool_call"]

    def on_update(partial_result: AgentToolResult) -> None:
        # Fire and forget — emit is async but we can't await in a sync callback
        import asyncio
        asyncio.ensure_future(emit(ToolExecutionUpdateEvent(
            tool_call_id=tool_call.id,
            tool_name=tool_call.name,
            args=tool_call.arguments,
            partial_result=partial_result,
        )))

    try:
        result = await tool.execute(tool_call.id, prep["args"], signal, on_update)
        return {"result": result, "is_error": False}
    except Exception as e:
        return {"result": _error_result(str(e)), "is_error": True}


async def _finalize(
    context: AgentContext,
    assistant_message: AssistantMessage,
    prep: dict,
    executed: dict,
    config: AgentLoopConfig,
    signal: AbortController | None,
    emit: AgentEventSink,
) -> ToolResultMessage:
    result: AgentToolResult = executed["result"]
    is_error: bool = executed["is_error"]

    if config.after_tool_call:
        try:
            after = await config.after_tool_call(
                AfterToolCallContext(
                    assistant_message=assistant_message,
                    tool_call=prep["tool_call"],
                    args=prep["args"],
                    result=result,
                    is_error=is_error,
                    context=context,
                ),
                signal,
            )
            if after:
                result = AgentToolResult(
                    content=after.content if after.content is not None else result.content,
                    details=after.details if after.details is not None else result.details,
                )
                if after.is_error is not None:
                    is_error = after.is_error
        except Exception as e:
            result = _error_result(str(e))
            is_error = True

    return await _emit_outcome(prep["tool_call"], result, is_error, emit)


async def _emit_outcome(
    tool_call: ToolCall,
    result: AgentToolResult,
    is_error: bool,
    emit: AgentEventSink,
) -> ToolResultMessage:
    await emit(ToolExecutionEndEvent(
        tool_call_id=tool_call.id,
        tool_name=tool_call.name,
        result=result,
        is_error=is_error,
    ))

    msg = ToolResultMessage(
        tool_call_id=tool_call.id,
        tool_name=tool_call.name,
        content=result.content,
        details=result.details,
        is_error=is_error,
    )
    await emit(MessageStartEvent(message=msg))
    await emit(MessageEndEvent(message=msg))
    return msg


def _error_result(message: str) -> AgentToolResult:
    return AgentToolResult(content=[TextContent(text=message)], details={})
