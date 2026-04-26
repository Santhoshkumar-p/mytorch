from __future__ import annotations
import asyncio
import os
import sys
from pathlib import Path

from .cli.args import parse_args, parse_model_shorthand, validate_fork_flags
from .cli.file_processor import process_file_arguments, read_stdin_if_piped
from .core.agent_session import AgentSession, AgentSessionConfig
from .core.agent_session_runtime import AgentSessionRuntime
from .core.agent_session_services import create_agent_session_services
from .core.extensions.runner import ExtensionRunner
from .core.session_manager import SessionManager


async def _run(argv=None) -> int:
    args = parse_args(argv)

    if args.version:
        print("coding-agent 0.1.0")
        return 0

    if args.list_models:
        from .cli.list_models import list_models
        pattern = args.list_models if isinstance(args.list_models, str) else None
        await list_models(pattern)
        return 0

    errors = validate_fork_flags(args)
    if errors:
        for e in errors:
            print(f"Error: {e}", file=sys.stderr)
        return 1

    cwd = os.getcwd()
    agent_dir = str(Path.home() / ".coding-agent")

    services = await create_agent_session_services(
        cwd=cwd,
        agent_dir=agent_dir,
        settings_overrides=_build_overrides(args),
    )
    settings = services.settings_manager.get_settings()

    # Determine mode
    mode = args.mode
    if not mode and args.print:
        mode = "text"
    if not mode:
        # Default to interactive TUI if textual is installed, otherwise error
        try:
            import textual  # noqa: F401
            mode = "interactive"
        except ImportError:
            print(
                "No mode specified. Install textual for TUI (pip install coding-agent[tui])\n"
                "or use: --mode text | --mode json | --mode rpc",
                file=sys.stderr,
            )
            return 1

    # Build session manager
    session_manager = _build_session_manager(args, services, cwd)

    # Model config
    model_str = args.model or ""
    thinking_override = None
    if args.model:
        model_str, thinking_override = parse_model_shorthand(args.model)

    provider = args.provider or settings.default_provider or "anthropic"
    model_name = model_str or settings.default_model or "claude-opus-4-5"
    thinking_level = thinking_override or args.thinking or settings.default_thinking_level

    try:
        from pi_agent import (
            Agent, Model,
            create_default_registry, create_agent_stream_fn,
        )
        from .providers.anthropic import AnthropicProvider
        registry = create_default_registry()
        registry.register("anthropic", AnthropicProvider())
        stream_fn = create_agent_stream_fn(registry)

        api_key = args.api_key
        def _get_api_key(prov: str) -> str | None:
            if api_key:
                return api_key
            import os
            env_map = {
                "anthropic": "ANTHROPIC_API_KEY",
                "openai": "OPENAI_API_KEY",
                "vertex": "GOOGLE_API_KEY",
            }
            return os.environ.get(env_map.get(prov, ""), None)

        agent = Agent(
            stream_fn=stream_fn,
            get_api_key=_get_api_key,
            thinking_budgets=_make_thinking_budgets(settings),
            max_retry_delay_ms=settings.retry.max_delay_ms,
        )
        # Set model on agent state
        api = {"anthropic": "anthropic", "openai": "openai-completions", "vertex": "vertex"}.get(provider, provider)
        agent.set_model(Model(id=model_name, provider=provider, api=api))
        agent.set_thinking_level(thinking_level)
        model_config = None  # no separate model_config needed
    except ImportError:
        # agent SDK not installed — use a stub for syntax-check / test purposes
        model_config = None
        agent = _StubAgent()

    # Extensions
    runner = ExtensionRunner()
    if not args.no_extensions:
        ext_paths = list(args.extensions or []) + list(settings.extensions)
        if ext_paths:
            await runner.load(ext_paths)

    # Active tools
    if args.no_tools:
        active_tools: list[str] = []
    elif args.tools:
        active_tools = [t.strip() for t in args.tools.split(",")]
    else:
        active_tools = ["read", "bash", "edit", "write"]

    # Extra skills/prompts
    extra_skills = list(args.skills) if not args.no_skills else []
    extra_prompts = list(args.prompt_templates) if not args.no_prompt_templates else []

    resource_loader = services.resource_loader
    resource_loader.extend_resources(skill_paths=extra_skills, prompt_paths=extra_prompts)

    # Build session
    session = AgentSession(AgentSessionConfig(
        agent=agent,
        session_manager=session_manager,
        settings_manager=services.settings_manager,
        cwd=cwd,
        resource_loader=resource_loader,
        model_config=model_config,
        initial_active_tool_names=active_tools,
        extension_runner_ref=[runner],
        custom_system_prompt=args.system_prompt,
        append_system_prompts=args.append_system_prompt or [],
        no_context_files=args.no_context_files,
    ))

    runtime = AgentSessionRuntime(session, services)

    # Export-only mode
    if args.export:
        path = await session.export_to_html(args.export)
        print(f"Exported to {path}")
        return 0

    # Fork at startup
    if args.fork:
        r = await runtime.fork(args.fork)
        if r.get("cancelled"):
            return 1

    # Initial message from positional args
    raw_messages = args.messages or []
    # Treat bare strings as text; @-prefixed as file refs
    file_refs = [
        (f"@{m}" if not m.startswith("@") and os.path.exists(m) else m)
        for m in raw_messages
    ]
    pf = await process_file_arguments(file_refs, cwd, auto_resize=settings.image_auto_resize)
    stdin_text = await read_stdin_if_piped()
    initial_text = "\n\n".join(t for t in [stdin_text, pf.text] if t) or None

    if mode == "rpc":
        from .modes.rpc.rpc_mode import run_rpc_mode
        return await run_rpc_mode(runtime)

    if mode == "interactive":
        from .modes.interactive.app import run_interactive_mode
        return await run_interactive_mode(runtime)

    from .modes.print_mode import run_print_mode, PrintModeOptions
    return await run_print_mode(
        runtime,
        PrintModeOptions(
            mode=mode,
            initial_message=initial_text,
            initial_images=pf.images,
        ),
    )


def _build_session_manager(args, services, cwd: str) -> SessionManager:
    settings = services.settings_manager.get_settings()
    session_dir = args.session_dir or settings.session_dir or _default_session_dir(cwd)

    if args.no_session:
        return SessionManager.create(cwd, session_dir=None)

    if args.session:
        return _open_by_id_or_path(args.session, session_dir, cwd)

    if args.continue_ or args.resume:
        latest = _find_latest(session_dir)
        if latest:
            return SessionManager.open(latest, cwd_override=cwd)

    return SessionManager.create(cwd, session_dir=session_dir)


def _open_by_id_or_path(identifier: str, session_dir: str, cwd: str) -> SessionManager:
    if os.path.exists(identifier):
        return SessionManager.open(identifier, cwd_override=cwd)
    # ID prefix search
    try:
        import json
        for p in Path(session_dir).glob("*.jsonl"):
            try:
                raw = json.loads(open(str(p)).readline())
                if raw.get("id", "").startswith(identifier):
                    return SessionManager.open(str(p), cwd_override=cwd)
            except Exception:
                continue
    except Exception:
        pass
    return SessionManager.create(cwd, session_dir=session_dir)


def _find_latest(session_dir: str) -> str | None:
    try:
        files = list(Path(session_dir).glob("*.jsonl"))
        if not files:
            return None
        return str(max(files, key=lambda p: p.stat().st_mtime))
    except Exception:
        return None


def _default_session_dir(cwd: str) -> str:
    return str(Path.home() / ".coding-agent" / "sessions")


def _build_overrides(args) -> dict:
    overrides: dict = {}
    if getattr(args, "verbose", False):
        overrides["verbose"] = True
    if getattr(args, "offline", False):
        overrides["offline"] = True
    return overrides


def _make_thinking_budgets(settings) -> dict | None:
    """Convert settings thinking_budgets dict to the format the agent expects."""
    b = settings.thinking_budgets
    if not b:
        return None
    valid_keys = {"minimal", "low", "medium", "high", "xhigh"}
    return {k: v for k, v in b.items() if k in valid_keys} or None


class _StubAgent:
    """Minimal stub used when the agent SDK is not installed."""
    class state:
        messages: list = []
        streaming_message = None
        pending_tool_calls: frozenset = frozenset()
        context_usage = None

    system_prompt: str = ""
    tools: list = []

    def subscribe(self, _handler):
        pass

    async def prompt(self, _text, **_kw):
        pass

    async def steer(self, _text, **_kw):
        pass

    async def follow_up(self, _text, **_kw):
        pass

    async def abort(self):
        pass

    async def wait_for_idle(self):
        pass


def main():
    sys.exit(asyncio.run(_run()))
