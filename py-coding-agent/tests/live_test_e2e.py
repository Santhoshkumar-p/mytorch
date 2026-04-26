"""
Comprehensive end-to-end live test suite for py-coding-agent.
Tests: basic text, tool use (bash/write/read/grep), skills, extensions,
       session persistence, compaction, HTML export, context usage,
       model switching, abort, multi-turn.
"""
from __future__ import annotations

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import asyncio
import json
import os
import shutil
import tempfile
import traceback
from pathlib import Path
from typing import Any

import httpx

# ── API Config ─────────────────────────────────────────────────────────────────

API_KEY  = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL_ID = "claude-haiku-4-5-20251001"
API_URL  = "https://api.anthropic.com/v1/messages"
API_VER  = "2023-06-01"

# ── agent imports ─────────────────────────────────────────────────────────────

from pi_agent import (
    Agent,
    AssistantMessageEventStream,
    Model,
    ProviderRegistry,
    create_agent_stream_fn,
    TextContent,
    ToolCall,
    Usage,
    UsageCost,
)
from pi_agent import AssistantMessage as PiMsg
from pi_agent import UserMessage as PiUser
from pi_agent.pi_ai.types import PiAIRequest

# ── coding_agent imports ───────────────────────────────────────────────────────

from coding_agent.core.agent_session import AgentSession, AgentSessionConfig
from coding_agent.core.session_manager import SessionManager
from coding_agent.core.settings_manager import SettingsManager
from coding_agent.core.resource_loader import ResourceLoader
from coding_agent.core.types import CompactionEntry


# ══════════════════════════════════════════════════════════════════════════════
# Anthropic provider
# ══════════════════════════════════════════════════════════════════════════════

def _map_stop(raw: str | None) -> str:
    return {
        "end_turn":       "stop",
        "stop_sequence":  "stop",
        "tool_use":       "toolUse",
        "max_tokens":     "length",
    }.get(raw or "", "stop")


def _to_anthropic_messages(messages) -> list[dict]:
    """Convert LlmContext messages to Anthropic wire format."""
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
                            "type": "base64", "media_type": b.mime_type, "data": b.data,
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


class AnthropicProvider:
    """Streams Anthropic SSE events into an AssistantMessageEventStream."""

    async def stream(self, request: PiAIRequest, abort_event=None) -> AssistantMessageEventStream:
        es = AssistantMessageEventStream()
        asyncio.ensure_future(self._do_stream(request, es, abort_event))
        return es

    async def _do_stream(self, request: PiAIRequest, es: AssistantMessageEventStream, abort_event):
        try:
            payload = self._build_payload(request)
            headers = {
                "x-api-key": request.api_key or API_KEY,
                "anthropic-version": API_VER,
                "content-type": "application/json",
            }
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream("POST", API_URL, json=payload, headers=headers) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        raise RuntimeError(f"HTTP {resp.status_code}: {body.decode()[:300]}")
                    await self._parse(resp, es, request.model)
        except Exception as exc:
            err = PiMsg(
                content=[TextContent(text=str(exc))],
                api="anthropic", provider="anthropic", model=request.model.id,
                usage=Usage(input=0, output=0, cache_read=0, cache_write=0,
                            total_tokens=0, cost=UsageCost()),
                stop_reason="error", error_message=str(exc),
            )
            es.push({"type": "error", "reason": "error", "error": err})

    def _build_payload(self, request: PiAIRequest) -> dict:
        ctx = request.context
        payload: dict = {
            "model": request.model.id,
            "messages": _to_anthropic_messages(ctx.messages),
            "max_tokens": 4096,
            "stream": True,
        }
        if ctx.system_prompt:
            payload["system"] = ctx.system_prompt
        if ctx.tools:
            payload["tools"] = [
                {
                    "name": t.name,
                    "description": t.description or "",
                    "input_schema": dict(t.parameters) if t.parameters
                                   else {"type": "object", "properties": {}},
                }
                for t in ctx.tools
            ]
        return payload

    async def _parse(self, resp, es: AssistantMessageEventStream, model):
        partial = PiMsg(
            content=[], api="anthropic", provider="anthropic", model=model.id,
            usage=Usage(input=0, output=0, cache_read=0, cache_write=0,
                        total_tokens=0, cost=UsageCost()),
            stop_reason="stop",
        )
        es.push({"type": "start", "partial": partial})

        sse_to_ci: dict[int, int] = {}   # sse block index -> content list index
        json_bufs: dict[int, str] = {}   # sse block index -> accumulated JSON
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
                elif bt == "tool_use":
                    json_bufs[si] = ""
                    partial.content.append(ToolCall(id=blk["id"], name=blk["name"], arguments={}))
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
                    es.push({"type": "text_delta", "content_index": si,
                             "delta": text, "partial": partial})
                elif dtype == "input_json_delta" and si in json_bufs:
                    chunk = delta.get("partial_json", "")
                    json_bufs[si] += chunk
                    es.push({"type": "toolcall_delta", "content_index": si,
                             "delta": chunk, "partial": partial})

            elif etype == "content_block_stop":
                si = ev["index"]
                ci = sse_to_ci.get(si)
                if ci is not None and ci < len(partial.content):
                    b = partial.content[ci]
                    if isinstance(b, TextContent):
                        es.push({"type": "text_end", "content_index": si,
                                 "content": b.text, "partial": partial})
                    elif isinstance(b, ToolCall) and si in json_bufs:
                        try:
                            b.arguments = json.loads(json_bufs[si]) if json_bufs[si] else {}
                        except json.JSONDecodeError:
                            b.arguments = {}
                        es.push({"type": "toolcall_end", "content_index": si,
                                 "tool_call": b, "partial": partial})

            elif etype == "message_delta":
                delta = ev.get("delta", {})
                stop_reason = _map_stop(delta.get("stop_reason"))
                u = ev.get("usage", {})
                usage.output        = u.get("output_tokens", usage.output)
                usage.cache_read    = u.get("cache_read_input_tokens", 0)
                usage.cache_write   = u.get("cache_creation_input_tokens", 0)
                usage.total_tokens  = usage.input + usage.output + usage.cache_read + usage.cache_write

            elif etype == "message_stop":
                break

        partial.usage       = usage
        partial.stop_reason = stop_reason  # type: ignore[assignment]

        if stop_reason in ("stop", "length", "toolUse"):
            es.push({"type": "done", "reason": stop_reason, "message": partial})
        else:
            es.push({"type": "error", "reason": "error", "error": partial})


# ══════════════════════════════════════════════════════════════════════════════
# Session factory helpers
# ══════════════════════════════════════════════════════════════════════════════

def _make_agent() -> Agent:
    registry = ProviderRegistry()
    registry.register("anthropic", AnthropicProvider())
    stream_fn = create_agent_stream_fn(registry)
    agent = Agent(stream_fn=stream_fn)
    agent.set_model(Model(id=MODEL_ID, provider="anthropic", api="anthropic"))
    agent.get_api_key = lambda _: API_KEY
    return agent


async def make_session(
    agent_dir: str,
    cwd: str | None = None,
    *,
    custom_tools=None,
    custom_system_prompt: str | None = None,
    extension_runner_ref: list | None = None,
    settings_overrides: dict | None = None,
) -> AgentSession:
    cwd = cwd or agent_dir
    sm = SettingsManager.create(cwd, agent_dir)
    if settings_overrides:
        sm.apply_overrides(settings_overrides)
    rl = ResourceLoader(cwd, agent_dir, sm)

    settings    = sm.get_settings()
    session_dir = settings.session_dir or str(Path(agent_dir) / "sessions")
    session_mgr = SessionManager.create(cwd, session_dir=session_dir)

    config = AgentSessionConfig(
        agent=_make_agent(),
        session_manager=session_mgr,
        settings_manager=sm,
        cwd=cwd,
        resource_loader=rl,
        custom_tools=custom_tools,
        custom_system_prompt=custom_system_prompt,
        extension_runner_ref=extension_runner_ref,
    )
    session = AgentSession(config)
    await asyncio.sleep(0.6)   # let _initialize() complete
    return session


async def run_prompt(session: AgentSession, text: str) -> list[dict]:
    """Run a prompt and collect all events."""
    events: list[dict] = []
    unsub = session.subscribe(lambda e: events.append(e))
    await session.prompt(text)
    unsub()
    return events


def get_last_text(session: AgentSession) -> str:
    return session.get_last_assistant_text() or ""


# ══════════════════════════════════════════════════════════════════════════════
# Test harness
# ══════════════════════════════════════════════════════════════════════════════

passed = 0
failed = 0
_section = ""

def section(name: str) -> None:
    global _section
    _section = name
    print(f"\n--- {name} ---")

def ok(name: str) -> None:
    global passed
    passed += 1
    print(f"  [PASS] {name}")

def fail(name: str, reason: str = "") -> None:
    global failed
    failed += 1
    snippet = (": " + reason[:120]) if reason else ""
    print(f"  [FAIL] {name}{snippet}")

def check(name: str, condition: bool, reason: str = "") -> None:
    if condition:
        ok(name)
    else:
        fail(name, reason or "condition is False")


# ══════════════════════════════════════════════════════════════════════════════
# Tests
# ══════════════════════════════════════════════════════════════════════════════

async def test_basic_text():
    section("Basic text response")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        try:
            session = await make_session(td)
            events = await run_prompt(session, 'Reply with exactly: HELLO_WORLD')
            text = get_last_text(session)
            msgs = session.messages

            check("received assistant message",   any(getattr(m, "role", None) == "assistant" for m in msgs))
            check("text contains HELLO_WORLD",    "HELLO_WORLD" in text, text[:80])
            check("message_end event fired",      any(e.get("type") == "message_end" for e in events))
            check("agent_end event fired",        any(e.get("type") == "agent_end"   for e in events))
        except Exception as exc:
            fail("test_basic_text", traceback.format_exc()[-200:])


async def test_tool_bash():
    section("Tool use - bash")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        try:
            session = await make_session(td, cwd=td)
            events = await run_prompt(
                session,
                "Use the bash tool to run: echo BASH_WORKS_FINE && exit 0. "
                "Then confirm you see the output."
            )
            text = get_last_text(session)
            tool_starts = [e for e in events if e.get("type") == "tool_execution_start"]
            tool_ends   = [e for e in events if e.get("type") == "tool_execution_end"]

            check("bash tool was invoked",        len(tool_starts) >= 1,
                  f"tool_starts={len(tool_starts)}")
            check("tool completed",               len(tool_ends) >= 1)
            check("bash tool name",
                  any(e.get("tool_name") == "bash" for e in tool_starts),
                  str([e.get("tool_name") for e in tool_starts]))
            check("response mentions output",     len(text) > 0, text[:80])
        except Exception as exc:
            fail("test_tool_bash", traceback.format_exc()[-200:])


async def test_tool_write():
    section("Tool use - write file")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        try:
            target = os.path.join(td, "hello.txt").replace("\\", "/")
            session = await make_session(td, cwd=td)
            events = await run_prompt(
                session,
                f"Use the write tool to write 'WRITTEN_CONTENT' to the file {target}. "
                "Just write the file and confirm."
            )
            tool_starts = [e for e in events if e.get("type") == "tool_execution_start"]
            check("write tool was invoked",      len(tool_starts) >= 1)
            check("file exists on disk",          os.path.exists(target),
                  f"path={target}")
            if os.path.exists(target):
                content = Path(target).read_text()
                check("file has correct content", "WRITTEN_CONTENT" in content, content[:60])
        except Exception as exc:
            fail("test_tool_write", traceback.format_exc()[-200:])


async def test_tool_read():
    section("Tool use - read file")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        try:
            target = os.path.join(td, "readme.txt")
            Path(target).write_text("SECRET_TOKEN_42")
            session = await make_session(td, cwd=td)
            events = await run_prompt(
                session,
                f"Use the read tool to read the file {target.replace(chr(92), '/')} "
                "and tell me what it contains."
            )
            text = get_last_text(session)
            tool_starts = [e for e in events if e.get("type") == "tool_execution_start"]
            check("read tool invoked",             len(tool_starts) >= 1)
            check("response mentions file content", "SECRET_TOKEN_42" in text, text[:120])
        except Exception as exc:
            fail("test_tool_read", traceback.format_exc()[-200:])


async def test_tool_grep():
    section("Tool use - grep")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        try:
            Path(os.path.join(td, "data.txt")).write_text(
                "line one\nUNIQUE_GREP_MARKER\nline three"
            )
            session = await make_session(td, cwd=td)
            events = await run_prompt(
                session,
                "Use the grep tool to search for 'UNIQUE_GREP_MARKER' in the current directory "
                "and tell me the result."
            )
            text = get_last_text(session)
            tool_starts = [e for e in events if e.get("type") == "tool_execution_start"]
            check("grep tool invoked",            len(tool_starts) >= 1)
            check("response mentions marker",      "UNIQUE_GREP_MARKER" in text, text[:120])
        except Exception as exc:
            fail("test_tool_grep", traceback.format_exc()[-200:])


async def test_multi_tool_task():
    section("Multi-tool task (write then read)")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        try:
            target = os.path.join(td, "combo.txt").replace("\\", "/")
            session = await make_session(td, cwd=td)
            events = await run_prompt(
                session,
                f"First, use the write tool to write 'MULTI_TOOL_PASS' to {target}. "
                f"Then use the read tool to read {target} back. "
                "Confirm what you read."
            )
            text = get_last_text(session)
            tool_starts = [e for e in events if e.get("type") == "tool_execution_start"]
            check("multiple tools used",          len(tool_starts) >= 2,
                  f"tool_starts={len(tool_starts)}")
            check("file created",                 os.path.exists(target.replace("/", os.sep)))
            check("response confirms content",    "MULTI_TOOL_PASS" in text, text[:120])
        except Exception as exc:
            fail("test_multi_tool_task", traceback.format_exc()[-200:])


async def test_multi_turn():
    section("Multi-turn conversation")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        try:
            session = await make_session(td)
            await run_prompt(session, "Remember the number TURNTEST_7734.")
            events2 = await run_prompt(session, "What number did I ask you to remember?")
            text2 = get_last_text(session)
            msgs = session.messages
            user_count = sum(1 for m in msgs if getattr(m, "role", None) == "user")
            asst_count = sum(1 for m in msgs if getattr(m, "role", None) == "assistant")
            check("at least 2 user turns",       user_count >= 2, f"user={user_count}")
            check("at least 2 asst turns",       asst_count >= 2, f"asst={asst_count}")
            check("remembers number",             "7734" in text2, text2[:120])
        except Exception as exc:
            fail("test_multi_turn", traceback.format_exc()[-200:])


async def test_skills():
    section("Skills (markdown injected into system prompt)")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        try:
            # Name must match ^[a-z0-9][a-z0-9-]{0,62}$ (no underscores)
            skill_path = os.path.join(td, "myskill.md")
            Path(skill_path).write_text(
                "When the user says SKILL_TRIGGER, "
                "respond with exactly: SKILL_ACTIVATED"
            )
            session = await make_session(
                td,
                settings_overrides={"skills": [skill_path]},
            )
            sys_prompt = session.get_system_prompt()
            check("skill content in system prompt",
                  "myskill" in sys_prompt or "SKILL_TRIGGER" in sys_prompt,
                  sys_prompt[:200] if sys_prompt else "(empty)")

            events = await run_prompt(session, "SKILL_TRIGGER")
            text = get_last_text(session)
            check("skill influenced response",    "SKILL_ACTIVATED" in text, text[:120])
        except Exception as exc:
            fail("test_skills", traceback.format_exc()[-200:])


async def test_extension_event():
    section("Extension - session_start event fires")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        try:
            ext_path = os.path.join(td, "my_ext.py")
            Path(ext_path).write_text(
                "SESSION_STARTED = False\n\n"
                "def setup(api):\n"
                "    def on_start(ctx):\n"
                "        import my_ext as _m\n"
                "        _m.SESSION_STARTED = True\n"
                "    api.on('session_start', on_start)\n"
            )

            from coding_agent.core.extensions.runner import ExtensionRunner
            runner = ExtensionRunner()
            await runner.load([ext_path])

            session = await make_session(td, extension_runner_ref=[runner])

            # The session_start event is emitted by _initialize()
            # Check that at least one api was loaded
            check("extension runner loaded",     len(runner._apis) == 1,
                  f"apis={len(runner._apis)}")
            check("extension has session_start handler",
                  "session_start" in runner._apis[0]._handlers,
                  str(list(runner._apis[0]._handlers.keys())))
        except Exception as exc:
            fail("test_extension_event", traceback.format_exc()[-200:])


async def test_custom_tool():
    section("Custom tool registered via AgentSessionConfig")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        try:
            from pi_agent import AgentTool, AgentToolResult

            call_log: list[dict] = []

            async def execute_echo(tool_call_id, params, abort_event=None, on_update=None):
                call_log.append({"id": tool_call_id, "params": dict(params)})
                msg = params.get("message", "")
                return AgentToolResult(
                    content=[TextContent(text=f"ECHO: {msg}")],
                    details={"message": msg},
                )

            custom = AgentTool(
                name="echo_tool",
                label="Echo",
                description="Echoes back the message parameter.",
                execute=execute_echo,
                parameters={
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "description": "text to echo"},
                    },
                    "required": ["message"],
                },
            )

            session = await make_session(td, custom_tools=[custom])
            events = await run_prompt(
                session,
                "Use the echo_tool with message='CUSTOM_TOOL_TEST'. Then confirm what it returned."
            )
            text = get_last_text(session)
            tool_starts = [e for e in events if e.get("type") == "tool_execution_start"
                           and e.get("tool_name") == "echo_tool"]

            check("echo_tool was invoked",        len(tool_starts) >= 1,
                  f"all tools: {[e.get('tool_name') for e in events if e.get('type')=='tool_execution_start']}")
            check("execute was called",           len(call_log) >= 1, str(call_log))
            check("response contains echo output","ECHO" in text or "CUSTOM_TOOL_TEST" in text,
                  text[:120])
        except Exception as exc:
            fail("test_custom_tool", traceback.format_exc()[-200:])


async def test_session_persistence():
    section("Session persistence (JSONL save + reload)")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        try:
            # --- save session ---
            session1 = await make_session(td)
            await run_prompt(session1, "Say exactly: PERSIST_MARKER_OK")
            msgs_before = len(session1.messages)
            await session1.session_manager.flush()
            file_path = session1.session_file
            check("session file created",         file_path is not None and os.path.exists(file_path),
                  str(file_path))

            if file_path:
                # Verify JSONL content
                lines = [l for l in Path(file_path).read_text(encoding="utf-8").splitlines() if l.strip()]
                check("JSONL has multiple lines",  len(lines) >= 2, f"lines={len(lines)}")

                # --- reload session ---
                sm2 = SessionManager.open(file_path)
                ctx = sm2.build_session_context()
                check("context has messages after reload",
                      len(ctx.messages) >= 1, f"msgs={len(ctx.messages)}")
        except Exception as exc:
            fail("test_session_persistence", traceback.format_exc()[-200:])


async def test_compaction():
    section("Compaction")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        try:
            session = await make_session(td)
            await run_prompt(session, "Say: BEFORE_COMPACT")
            await session.compact()
            entries = session.session_manager._entries
            compaction_entries = [e for e in entries if isinstance(e, CompactionEntry)]
            check("compaction entry created",     len(compaction_entries) >= 1,
                  f"entries={[type(e).__name__ for e in entries]}")
            check("not still compacting",        not session.is_compacting)
        except Exception as exc:
            fail("test_compaction", traceback.format_exc()[-200:])


async def test_html_export():
    section("HTML export")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        try:
            session = await make_session(td)
            await run_prompt(session, "Reply with: EXPORT_READY")
            await session.session_manager.flush()

            out_path = os.path.join(td, "export.html")
            result_path = await session.export_to_html(out_path)
            check("HTML file created",            os.path.exists(out_path), out_path)
            if os.path.exists(out_path):
                html = Path(out_path).read_text(encoding="utf-8", errors="replace")
                check("HTML has DOCTYPE",         "<!DOCTYPE" in html or "<html" in html,
                      html[:80])
        except Exception as exc:
            fail("test_html_export", traceback.format_exc()[-200:])


async def test_context_usage():
    section("Context usage tracking")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        try:
            session = await make_session(td)
            await run_prompt(session, "Reply with: USAGE_CHECK")

            state = session.agent.state
            msgs  = session.messages
            asst  = [m for m in msgs if getattr(m, "role", None) == "assistant"]

            check("at least one assistant message", len(asst) >= 1)
            if asst:
                usage = getattr(asst[-1], "usage", None)
                check("usage present",            usage is not None)
                if usage:
                    check("input_tokens > 0",     getattr(usage, "input", 0) > 0,
                          str(usage))
                    check("output_tokens > 0",    getattr(usage, "output", 0) > 0,
                          str(usage))
                    check("total_tokens > 0",     getattr(usage, "total_tokens", 0) > 0,
                          str(usage))
        except Exception as exc:
            fail("test_context_usage", traceback.format_exc()[-200:])


async def test_model_switch():
    section("Model switching")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        try:
            session = await make_session(td)
            original_model = session.agent.state.model.id

            await session.set_model("anthropic", "claude-sonnet-4-5-20250929")
            new_model = session.agent.state.model.id

            check("model changed",               new_model != original_model,
                  f"orig={original_model} new={new_model}")
            check("new model id correct",        "sonnet" in new_model.lower(), new_model)

            # Model change entry in session
            from coding_agent.core.types import ModelChangeEntry
            mc_entries = [e for e in session.session_manager._entries
                          if isinstance(e, ModelChangeEntry)]
            check("model change entry saved",    len(mc_entries) >= 1,
                  str([type(e).__name__ for e in session.session_manager._entries]))
        except Exception as exc:
            fail("test_model_switch", traceback.format_exc()[-200:])


async def test_thinking_level():
    section("Thinking level")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        try:
            session = await make_session(td)
            check("default thinking level",      session.thinking_level == "off",
                  session.thinking_level)

            session.set_thinking_level("low")
            check("thinking level set to low",   session.thinking_level == "low",
                  session.thinking_level)

            next_level = session.cycle_thinking_level()
            check("cycle advances level",        next_level != "low", next_level)

            from coding_agent.core.types import ThinkingLevelChangeEntry
            tl_entries = [e for e in session.session_manager._entries
                          if isinstance(e, ThinkingLevelChangeEntry)]
            check("thinking level entries saved", len(tl_entries) >= 1,
                  str(len(tl_entries)))
        except Exception as exc:
            fail("test_thinking_level", traceback.format_exc()[-200:])


async def test_abort():
    section("Abort mid-stream")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        try:
            session = await make_session(td)

            # Start a long prompt in the background
            task = asyncio.create_task(
                session.prompt(
                    "Count from 1 to 500 very slowly with a brief pause between each number."
                )
            )
            await asyncio.sleep(1.5)
            await session.abort()

            # Wait up to 8 s for task to finish
            try:
                await asyncio.wait_for(task, timeout=8)
            except asyncio.TimeoutError:
                task.cancel()

            msgs = session.messages
            asst = [m for m in msgs if getattr(m, "role", None) == "assistant"]
            check("agent is idle after abort",   not session.is_streaming)
            check("at least one message added",  len(asst) >= 1, f"asst_count={len(asst)}")
            if asst:
                stop = getattr(asst[-1], "stop_reason", "")
                # Either aborted or completed early
                check("stop_reason is aborted or stop",
                      stop in ("aborted", "stop", "error", "toolUse"),
                      f"stop_reason={stop}")
        except Exception as exc:
            fail("test_abort", traceback.format_exc()[-200:])


async def test_session_stats():
    section("Session stats")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        try:
            session = await make_session(td)
            await run_prompt(session, "Say: STATS_OK")
            stats = session.get_session_stats()
            check("stats.user_messages >= 1",    stats.user_messages >= 1,
                  str(stats.user_messages))
            check("stats.assistant_messages >= 1",stats.assistant_messages >= 1,
                  str(stats.assistant_messages))
            check("stats.total_messages >= 2",   stats.total_messages >= 2,
                  str(stats.total_messages))
        except Exception as exc:
            fail("test_session_stats", traceback.format_exc()[-200:])


async def test_event_types():
    section("Event type coverage")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        try:
            session = await make_session(td, cwd=td)
            events = await run_prompt(
                session,
                "Use the bash tool to run `echo EVENT_TEST` and report back."
            )
            types_seen = {e.get("type") for e in events}
            for expected in ("agent_start", "turn_start", "message_start",
                             "message_end", "agent_end"):
                check(f"event '{expected}' fired",
                      expected in types_seen,
                      f"seen: {sorted(types_seen)}")
            check("tool events fired",
                  "tool_execution_start" in types_seen and "tool_execution_end" in types_seen,
                  f"seen: {sorted(types_seen)}")
        except Exception as exc:
            fail("test_event_types", traceback.format_exc()[-200:])


async def test_custom_system_prompt():
    section("Custom system prompt")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        try:
            session = await make_session(
                td,
                custom_system_prompt=(
                    "You are a test bot. When asked for your name, "
                    "always reply with: MY_NAME_IS_TESTBOT"
                ),
            )
            sp = session.get_system_prompt()
            check("custom prompt in system prompt",
                  "MY_NAME_IS_TESTBOT" in sp, sp[:200])

            events = await run_prompt(session, "What is your name?")
            text = get_last_text(session)
            check("agent follows custom prompt",  "TESTBOT" in text, text[:120])
        except Exception as exc:
            fail("test_custom_system_prompt", traceback.format_exc()[-200:])


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

async def main():
    print("=" * 60)
    print("py-coding-agent end-to-end live test suite")
    print(f"Model: {MODEL_ID}")
    print("=" * 60)

    await test_basic_text()
    await test_tool_bash()
    await test_tool_write()
    await test_tool_read()
    await test_tool_grep()
    await test_multi_tool_task()
    await test_multi_turn()
    await test_skills()
    await test_extension_event()
    await test_custom_tool()
    await test_session_persistence()
    await test_compaction()
    await test_html_export()
    await test_context_usage()
    await test_model_switch()
    await test_thinking_level()
    await test_abort()
    await test_session_stats()
    await test_event_types()
    await test_custom_system_prompt()

    print("\n" + "=" * 60)
    total = passed + failed
    print(f"Results: {passed}/{total} passed, {failed} failed")
    print("=" * 60)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
