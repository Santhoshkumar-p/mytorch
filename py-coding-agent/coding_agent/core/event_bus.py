from __future__ import annotations
import asyncio
import logging
from typing import Callable


class EventBus:
    def __init__(self):
        self._handlers: dict[str, list[Callable]] = {}

    def on(self, event: str, handler: Callable) -> Callable:
        """Register handler, returns unsubscribe function."""
        self._handlers.setdefault(event, []).append(handler)

        def unsub():
            try:
                self._handlers[event].remove(handler)
            except (KeyError, ValueError):
                pass

        return unsub

    def emit(self, event: str, *args, **kwargs) -> None:
        """Call all handlers for event. Async handlers are scheduled as tasks."""
        for h in list(self._handlers.get(event, [])):
            try:
                result = h(*args, **kwargs)
                if asyncio.iscoroutine(result):
                    asyncio.ensure_future(result)
            except Exception as e:
                logging.error("EventBus [%s] handler error: %s", event, e)

    def clear(self) -> None:
        self._handlers.clear()
