import pytest
import asyncio
import tempfile
import os
import json
from pathlib import Path


@pytest.fixture
def tmp_dir(tmp_path):
    return str(tmp_path)


@pytest.fixture
def sample_session_manager(tmp_path):
    """A fresh in-memory SessionManager."""
    from coding_agent.core.session_manager import SessionManager
    return SessionManager.create(str(tmp_path))


@pytest.fixture
def sample_settings():
    from coding_agent.core.settings_manager import SettingsManager
    return SettingsManager.in_memory()


@pytest.fixture
def mock_agent():
    """A fake agent that returns a fixed text response."""
    import asyncio
    from coding_agent.core.types import AgentState

    class MockAgent:
        def __init__(self):
            self.state = AgentState()
            self._listeners = []
            self.system_prompt = ""
            self.tools = []

        def subscribe(self, listener):
            self._listeners.append(listener)
            return lambda: self._listeners.remove(listener) if listener in self._listeners else None

        async def prompt(self, text, options=None):
            pass

        async def steer(self, text, images=None):
            pass

        async def follow_up(self, text, images=None):
            pass

        async def abort(self):
            pass

        async def wait_for_idle(self):
            pass

        def reset(self):
            self.state = AgentState()

    return MockAgent()
