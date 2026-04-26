from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path

from .settings_manager import SettingsManager
from .resource_loader import ResourceLoader


@dataclass
class AgentSessionServices:
    cwd: str
    agent_dir: str
    settings_manager: SettingsManager
    resource_loader: ResourceLoader


async def create_agent_session_services(
    cwd: str | None = None,
    agent_dir: str | None = None,
    settings_overrides: dict | None = None,
) -> AgentSessionServices:
    cwd = cwd or os.getcwd()
    agent_dir = agent_dir or str(Path.home() / ".coding-agent")
    sm = SettingsManager.create(cwd, agent_dir)
    if settings_overrides:
        sm.apply_overrides(settings_overrides)
    rl = ResourceLoader(cwd, agent_dir, sm)
    return AgentSessionServices(
        cwd=cwd,
        agent_dir=agent_dir,
        settings_manager=sm,
        resource_loader=rl,
    )
