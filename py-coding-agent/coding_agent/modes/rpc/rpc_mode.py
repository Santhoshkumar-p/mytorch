from __future__ import annotations
import asyncio
import dataclasses
import json
import signal
import sys

from .rpc_types import (
    CMD_TYPES, RpcSuccess, RpcError, RpcEvent, RpcSessionState,
    PromptCmd, SteerCmd, FollowUpCmd, AbortCmd, AbortRetryCmd, NewSessionCmd,
    GetStateCmd, SetModelCmd, CycleModelCmd, GetAvailableModelsCmd,
    SetThinkingLevelCmd, CycleThinkingLevelCmd, SetSteeringModeCmd, SetFollowUpModeCmd,
    CompactCmd, SetAutoCompactionCmd, SetAutoRetryCmd, BashCmd, AbortBashCmd,
    GetSessionStatsCmd, ExportHtmlCmd, SwitchSessionCmd, ForkCmd, NavigateTreeCmd,
    ReloadCmd, GetForkMessagesCmd, GetLastAssistantTextCmd, SetSessionNameCmd,
    GetMessagesCmd, GetCommandsCmd, ImportCmd,
)
from ...core.types import PromptOptions


async def run_rpc_mode(runtime) -> int:
    session = runtime.session
    _pending_ui: dict[str, asyncio.Future] = {}
    loop = asyncio.get_event_loop()

    def write(obj):
        sys.stdout.write(json.dumps(_safe_asdict(obj)) + "\n")
        sys.stdout.flush()

    def on_event(event):
        write(RpcEvent(event=type(event).__name__, data=_serialize_event(event)))

    unsub = session.subscribe(on_event)

    def _handle_signal():
        asyncio.ensure_future(runtime.dispose())

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _handle_signal)
        except (NotImplementedError, OSError):
            pass  # Windows

    reader = asyncio.StreamReader()
    proto = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: proto, sys.stdin)

    try:
        async for line in _iter_lines(reader):
            line = line.strip()
            if not line:
                continue

            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                write(RpcError(error=f"JSON parse error: {exc}"))
                continue

            cmd_type = raw.get("type", "")
            cmd_id = raw.get("id")

            # Extension UI response — resolves a pending future
            if cmd_type == "extension_ui_response":
                req_id = raw.get("request_id", "")
                if req_id in _pending_ui:
                    _pending_ui[req_id].set_result(raw)
                continue

            cls = CMD_TYPES.get(cmd_type)
            if cls is None:
                write(RpcError(id=cmd_id, error=f"Unknown command: {cmd_type}"))
                continue

            try:
                valid_fields = {f.name for f in dataclasses.fields(cls)}
                kwargs = {k: v for k, v in raw.items() if k in valid_fields and k != "type"}
                cmd = cls(**kwargs)
            except Exception as exc:
                write(RpcError(id=cmd_id, error=f"Bad command: {exc}"))
                continue

            asyncio.ensure_future(_dispatch(runtime, session, cmd, write, _pending_ui))
    finally:
        unsub()

    return 0


async def _dispatch(runtime, session, cmd, write, pending_ui: dict) -> None:
    cmd_id = getattr(cmd, "id", None)
    try:
        result = await _handle(runtime, session, cmd)
        write(RpcSuccess(id=cmd_id, data=result))
    except Exception as exc:
        write(RpcError(id=cmd_id, error=str(exc)))


async def _handle(runtime, session, cmd):  # noqa: C901
    match cmd:
        case PromptCmd():
            await session.prompt(
                cmd.message,
                PromptOptions(
                    images=cmd.images or [],
                    streaming_behavior=cmd.streaming_behavior,
                    source="rpc",
                ),
            )
        case SteerCmd():
            await session.steer(cmd.message, cmd.images)
        case FollowUpCmd():
            await session.follow_up(cmd.message, cmd.images)
        case AbortCmd():
            await session.abort()
        case AbortRetryCmd():
            await session.abort_retry()
        case NewSessionCmd():
            return await runtime.new_session(cmd.parent_session)
        case GetStateCmd():
            return _build_state(session)
        case SetModelCmd():
            await session.set_model(cmd.provider, cmd.model_id)
        case CycleModelCmd():
            return await session.cycle_model()
        case GetAvailableModelsCmd():
            return {"models": []}
        case SetThinkingLevelCmd():
            session.set_thinking_level(cmd.level)
        case CycleThinkingLevelCmd():
            return {"thinking_level": session.cycle_thinking_level()}
        case SetSteeringModeCmd():
            session.set_steering_mode(cmd.mode)
        case SetFollowUpModeCmd():
            session.set_follow_up_mode(cmd.mode)
        case CompactCmd():
            await session.compact(cmd.custom_instructions)
        case SetAutoCompactionCmd():
            session.set_auto_compaction_enabled(cmd.enabled)
        case SetAutoRetryCmd():
            session.set_auto_retry_enabled(cmd.enabled)
        case BashCmd():
            r = await session.execute_bash(cmd.command)
            return dataclasses.asdict(r)
        case AbortBashCmd():
            await session.abort()
        case GetSessionStatsCmd():
            return dataclasses.asdict(session.get_session_stats())
        case ExportHtmlCmd():
            return {"path": await session.export_to_html(cmd.output_path)}
        case SwitchSessionCmd():
            return await runtime.switch_session(cmd.session_path)
        case ForkCmd():
            return await runtime.fork(cmd.entry_id)
        case NavigateTreeCmd():
            return await session.navigate_tree(cmd.target_id)
        case ReloadCmd():
            await session.reload()
        case GetForkMessagesCmd():
            return {"messages": session.get_user_messages_for_forking()}
        case GetLastAssistantTextCmd():
            return {"text": session.get_last_assistant_text()}
        case SetSessionNameCmd():
            session.set_session_name(cmd.name)
        case GetMessagesCmd():
            return {"messages": [_serialize_msg(m) for m in session.messages]}
        case GetCommandsCmd():
            runner = getattr(session, "_extension_runner_ref", None)
            if runner and runner[0]:
                cmds = runner[0].get_registered_commands()
            else:
                cmds = []
            return {"commands": cmds}
        case ImportCmd():
            return await runtime.import_from_jsonl(cmd.path)
        case _:
            raise ValueError(f"Unhandled command: {type(cmd).__name__}")
    return None


def _build_state(session) -> dict:
    state = RpcSessionState(
        model=session.model,
        thinking_level=session.thinking_level,
        is_streaming=session.is_streaming,
        is_compacting=session.is_compacting,
        steering_mode=session.steering_mode,
        follow_up_mode=session.follow_up_mode,
        session_file=session.session_file,
        session_id=session.session_id,
        session_name=session.session_name,
        auto_compaction_enabled=session.auto_compaction_enabled,
        message_count=len(session.messages),
        pending_message_count=session.pending_message_count,
    )
    return dataclasses.asdict(state)


async def _iter_lines(reader: asyncio.StreamReader):
    while True:
        try:
            line = await reader.readline()
            if not line:
                break
            yield line.decode(errors="replace")
        except Exception:
            break


def _serialize_event(event) -> dict:
    return _safe_asdict(event)


def _serialize_msg(m) -> dict:
    return _safe_asdict(m)


def _safe_asdict(obj) -> dict:
    try:
        return dataclasses.asdict(obj)
    except Exception:
        try:
            return vars(obj)
        except Exception:
            return {"type": type(obj).__name__}
