from __future__ import annotations
import logging
from .types import ExtensionAPI, ExtensionContext
from .loader import load_extension

logger = logging.getLogger(__name__)


class ExtensionRunner:
    def __init__(self):
        self._apis: list[ExtensionAPI] = []

    async def load(self, paths: list[str]) -> None:
        """Load each extension file, log errors without crashing."""
        for path in paths:
            api = ExtensionAPI()
            try:
                await load_extension(path, api)
                self._apis.append(api)
            except Exception as exc:
                logger.error("Failed to load extension %r: %s", path, exc)

    async def bind_to_session(self, session) -> None:
        """Wire each API's _session reference."""
        for api in self._apis:
            api._session = session

    async def emit(self, event: str, ctx: ExtensionContext) -> dict:
        """Call all handlers for event. Return merged result dicts."""
        merged: dict = {}
        for api in self._apis:
            handlers = api._handlers.get(event, [])
            for handler in handlers:
                try:
                    import asyncio
                    import inspect
                    if inspect.iscoroutinefunction(handler):
                        result = await handler(ctx)
                    else:
                        result = handler(ctx)
                    if isinstance(result, dict):
                        merged.update(result)
                except Exception as exc:
                    logger.error("Extension handler error [%s]: %s", event, exc)
        return merged

    def get_registered_tools(self) -> list:
        tools = []
        for api in self._apis:
            tools.extend(api._tools)
        return tools

    def get_registered_commands(self) -> list[dict]:
        commands = []
        for api in self._apis:
            commands.extend(api._commands)
        return commands

    def get_flag_values(self) -> dict:
        """Return a dict of flag_name -> default_value for all registered flags."""
        values: dict = {}
        for api in self._apis:
            for flag in api._flags:
                values[flag["name"]] = flag.get("default")
        return values
