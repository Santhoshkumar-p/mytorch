from __future__ import annotations
import asyncio
import os
from pathlib import Path
from .types import ContextFile, Skill, PromptTemplate, Settings
from .skills import load_skills_from_dir
from .prompt_templates import load_prompt_templates

_CONTEXT_FILE_NAMES = ("AGENTS.md", "CLAUDE.md")


class ResourceLoader:
    def __init__(self, cwd: str, agent_dir: str, settings_manager):
        self._cwd = cwd
        self._agent_dir = agent_dir
        self._settings = settings_manager
        self._extra_skill_paths: list[str] = []
        self._extra_prompt_paths: list[str] = []

    async def load_skills(self, extra: list[str] | None = None) -> list[Skill]:
        """Load skills from settings.skills paths + extra paths."""
        settings: Settings = self._settings.get_settings()
        all_paths = list(settings.skills) + self._extra_skill_paths + (extra or [])
        skills: list[Skill] = []
        for path in all_paths:
            expanded = os.path.expanduser(path)
            p = Path(expanded)
            if p.is_dir():
                skills.extend(load_skills_from_dir(str(p), source_info={"path": str(p)}))
            elif p.is_file() and p.suffix == ".md":
                from .skills import load_skill
                skill = load_skill(str(p), source_info={"path": str(p)})
                if skill:
                    skills.append(skill)
        return skills

    async def load_context_files(self) -> list[ContextFile]:
        """Find AGENTS.md or CLAUDE.md from global agent_dir + walking up from cwd.

        Order: global agent_dir file first, then root→cwd order.
        """
        found: list[ContextFile] = []

        # 1. Check agent_dir
        agent_dir = Path(os.path.expanduser(self._agent_dir))
        for name in _CONTEXT_FILE_NAMES:
            candidate = agent_dir / name
            if candidate.is_file():
                try:
                    content = candidate.read_text(encoding="utf-8")
                    found.append(ContextFile(path=str(candidate), content=content))
                except OSError:
                    pass
                break  # only the first match per dir

        # 2. Walk up from cwd to fs root, collect in root→cwd order
        cwd = Path(self._cwd).resolve()
        ancestors: list[Path] = []
        current = cwd
        while True:
            ancestors.append(current)
            parent = current.parent
            if parent == current:
                break
            current = parent
        ancestors.reverse()  # root first

        for directory in ancestors:
            for name in _CONTEXT_FILE_NAMES:
                candidate = directory / name
                if candidate.is_file():
                    # Skip if already loaded as the global agent_dir file
                    already = any(cf.path == str(candidate) for cf in found)
                    if not already:
                        try:
                            content = candidate.read_text(encoding="utf-8")
                            found.append(ContextFile(path=str(candidate), content=content))
                        except OSError:
                            pass
                    break  # only first match per directory

        return found

    async def load_prompt_templates(self, extra: list[str] | None = None) -> list[PromptTemplate]:
        """Load prompt templates from settings.prompts + extra paths."""
        settings: Settings = self._settings.get_settings()
        all_paths = list(settings.prompts) + self._extra_prompt_paths + (extra or [])
        # Expand user paths
        expanded = [os.path.expanduser(p) for p in all_paths]
        return load_prompt_templates(expanded)

    async def reload(self) -> None:
        """Reload settings from disk (re-reads config files)."""
        if hasattr(self._settings, "reload"):
            if asyncio.iscoroutinefunction(self._settings.reload):
                await self._settings.reload()
            else:
                self._settings.reload()
        elif hasattr(self._settings, "_load"):
            self._settings._load()

    def extend_resources(
        self,
        skill_paths: list[str] | None = None,
        prompt_paths: list[str] | None = None,
    ) -> None:
        if skill_paths:
            self._extra_skill_paths.extend(skill_paths)
        if prompt_paths:
            self._extra_prompt_paths.extend(prompt_paths)
