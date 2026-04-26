from __future__ import annotations
import logging

from .agent_session import AgentSession, AgentSessionConfig
from .agent_session_services import AgentSessionServices
from .session_manager import SessionManager


class AgentSessionRuntime:
    def __init__(self, session: AgentSession, services: AgentSessionServices) -> None:
        self._session = session
        self._services = services

    @property
    def session(self) -> AgentSession:
        return self._session

    @property
    def services(self) -> AgentSessionServices:
        return self._services

    @property
    def cwd(self) -> str:
        return self._services.cwd

    async def switch_session(
        self, session_path: str, cwd_override: str | None = None
    ) -> dict:
        """Emit session_before_switch event (cancellable), then switch."""
        try:
            runner = self._session._get_extension_runner()
            if runner:
                from .extensions.types import ExtensionContext
                ctx = ExtensionContext(
                    session=self._session,
                    session_manager=self._session.session_manager,
                    model=self._session.model,
                    cwd=self._services.cwd,
                )
                result = await runner.emit("session_before_switch", ctx)
                if result.get("cancel"):
                    return {"cancelled": True}
        except Exception as exc:
            logging.warning("switch_session pre-hook error: %s", exc)

        try:
            await self._session.session_manager.flush()
        except Exception as exc:
            logging.warning("switch_session flush error: %s", exc)

        cwd = cwd_override or self._services.cwd
        new_sm = SessionManager.open(session_path, cwd_override=cwd)
        self._rebuild_session(new_sm)
        return {"cancelled": False}

    async def new_session(self, parent_session: str | None = None) -> dict:
        """Start a fresh session."""
        try:
            await self._session.session_manager.flush()
        except Exception:
            pass
        settings = self._services.settings_manager.get_settings()
        session_dir = settings.session_dir
        new_sm = SessionManager.create(self._services.cwd, session_dir=session_dir)
        if parent_session:
            new_sm._header.parent_session = parent_session
        self._rebuild_session(new_sm)
        return {"session_id": new_sm.get_session_id()}

    async def fork(self, entry_id: str) -> dict:
        """Fork from entry_id into a new session file."""
        # Flush the current session first so the fork file is up to date
        try:
            await self._session.session_manager.flush()
        except Exception:
            pass
        new_path = self._session.session_manager.create_branched_session(entry_id)
        if not new_path:
            return {"cancelled": True, "error": "Could not create branch — entry not found"}
        return await self.switch_session(new_path)

    async def import_from_jsonl(
        self, input_path: str, cwd_override: str | None = None
    ) -> dict:
        """Import/resume a session from a JSONL file."""
        return await self.switch_session(input_path, cwd_override)

    async def dispose(self) -> None:
        """Flush session and cleanup."""
        try:
            await self._session.session_manager.flush()
        except Exception:
            pass
        try:
            self._session.dispose()
        except Exception:
            pass

    def _rebuild_session(self, new_session_manager: SessionManager) -> None:
        """Replace the current session with a new one using the given session_manager."""
        old = self._session

        try:
            old.dispose()
        except Exception as exc:
            logging.warning("_rebuild_session dispose error: %s", exc)

        # Try to create a fresh agent; fall back to reusing the old one on any error
        new_agent = None
        try:
            from pi_agent import Agent, AgentOptions
            new_agent = Agent(AgentOptions())
        except ImportError:
            new_agent = old.agent
        except Exception as exc:
            logging.warning("_rebuild_session: Agent() failed (%s), reusing old agent", exc)
            new_agent = old.agent

        config = AgentSessionConfig(
            agent=new_agent,
            session_manager=new_session_manager,
            settings_manager=self._services.settings_manager,
            cwd=self._services.cwd,
            resource_loader=self._services.resource_loader,
            model_config=old._model_config,
            custom_tools=old._custom_tools,
            initial_active_tool_names=list(old._active_tool_names),
            extension_runner_ref=old._extension_runner_ref,
            custom_system_prompt=old._custom_system_prompt,
            append_system_prompts=list(old._append_system_prompts),
            no_context_files=old._no_context_files,
        )
        self._session = AgentSession(config)
