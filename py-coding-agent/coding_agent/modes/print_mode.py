from __future__ import annotations
import asyncio
import dataclasses
import json
import signal
import sys
from dataclasses import dataclass, field

from ..core.types import PromptOptions


@dataclass
class PrintModeOptions:
    mode: str = "text"    # "text" | "json"
    messages: list[str] = field(default_factory=list)
    initial_message: str | None = None
    initial_images: list = field(default_factory=list)


async def run_print_mode(runtime, options: PrintModeOptions) -> int:
    session = runtime.session
    loop = asyncio.get_event_loop()

    def _handle_signal():
        asyncio.ensure_future(runtime.dispose())

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _handle_signal)
        except (NotImplementedError, OSError):
            pass

    if options.mode == "json":
        def on_event(event):
            if isinstance(event, dict):
                evt_type = event.get("type", "unknown")
                data = {k: v for k, v in event.items() if k != "message" or True}
                # Serialize message objects inside the event
                if "message" in data:
                    try:
                        import dataclasses as _dc
                        msg = data["message"]
                        data = {**data, "message": _dc.asdict(msg) if _dc.is_dataclass(msg) else str(msg)}
                    except Exception:
                        data = {**data, "message": str(data["message"])}
            else:
                evt_type = type(event).__name__
                data = _safe_asdict(event)
            print(json.dumps({"type": evt_type, "data": data}), flush=True)
        unsub = session.subscribe(on_event)
    else:
        unsub = lambda: None

    try:
        if options.initial_message:
            await session.prompt(
                options.initial_message,
                PromptOptions(images=options.initial_images or [], source="rpc"),
            )
            await session.wait_for_idle()

        for msg in options.messages:
            await session.prompt(msg, PromptOptions(source="rpc"))
            await session.wait_for_idle()

        if options.mode == "text":
            for m in reversed(session.messages):
                if getattr(m, "role", None) == "assistant":
                    text = "".join(
                        getattr(c, "text", "")
                        for c in getattr(m, "content", [])
                        if hasattr(c, "text")
                    )
                    print(text)
                    break

        # Error check
        for m in reversed(session.messages):
            if getattr(m, "stop_reason", None) in ("error", "aborted"):
                return 1
        return 0

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        unsub()
        await runtime.dispose()


def _safe_asdict(event) -> dict:
    try:
        return dataclasses.asdict(event)
    except Exception:
        return {}
