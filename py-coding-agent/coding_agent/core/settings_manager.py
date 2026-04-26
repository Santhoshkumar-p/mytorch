from __future__ import annotations
import json
import os
import re
from pathlib import Path

import portalocker

from coding_agent.core.types import (
    BranchSummarySettings,
    CompactionSettings,
    RetrySettings,
    Settings,
)


def _to_snake(s: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()


def _to_camel(s: str) -> str:
    parts = s.split("_")
    return parts[0] + "".join(p.title() for p in parts[1:])


def _keys_to_snake(d: dict) -> dict:
    if not isinstance(d, dict):
        return d
    return {_to_snake(k): _keys_to_snake(v) for k, v in d.items()}


def _keys_to_camel(d: dict) -> dict:
    if not isinstance(d, dict):
        return d
    return {_to_camel(k): _keys_to_camel(v) for k, v in d.items()}


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _run_migrations(raw: dict) -> dict:
    """v1→v3 key renames."""
    if "queueMode" in raw:
        raw["steeringMode"] = raw.pop("queueMode")
    if "websockets" in raw:
        raw["transport"] = "websocket" if raw.pop("websockets") else "sse"
    return raw


def _load_json(path: str) -> tuple[dict, Exception | None]:
    try:
        with open(path) as f:
            return _run_migrations(json.load(f)), None
    except FileNotFoundError:
        return {}, None
    except Exception as e:
        return {}, e


def _save_json(path: str, data: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = path + ".tmp"
    with portalocker.Lock(path + ".lock", timeout=5):
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)


def _dict_to_settings(d: dict) -> Settings:
    snake = _keys_to_snake(d)
    s = Settings()
    for k, v in snake.items():
        if not hasattr(s, k):
            continue
        if k == "compaction" and isinstance(v, dict):
            defaults = CompactionSettings()
            kwargs = {_to_snake(kk): vv for kk, vv in v.items() if hasattr(defaults, _to_snake(kk))}
            v = CompactionSettings(**kwargs)
        elif k == "retry" and isinstance(v, dict):
            defaults = RetrySettings()
            kwargs = {_to_snake(kk): vv for kk, vv in v.items() if hasattr(defaults, _to_snake(kk))}
            v = RetrySettings(**kwargs)
        elif k == "branch_summary" and isinstance(v, dict):
            defaults = BranchSummarySettings()
            kwargs = {_to_snake(kk): vv for kk, vv in v.items() if hasattr(defaults, _to_snake(kk))}
            v = BranchSummarySettings(**kwargs)
        setattr(s, k, v)
    return s


class SettingsManager:
    @classmethod
    def create(cls, cwd: str, agent_dir: str) -> "SettingsManager":
        sm = cls.__new__(cls)
        sm._global_path = str(Path(agent_dir) / "settings.json")
        sm._project_path = str(Path(cwd) / ".coding-agent" / "settings.json")
        sm._overrides: dict = {}
        sm._global_error: Exception | None = None
        sm._project_error: Exception | None = None
        sm._load()
        return sm

    @classmethod
    def in_memory(cls, settings: dict | None = None) -> "SettingsManager":
        sm = cls.__new__(cls)
        sm._global_path = None
        sm._project_path = None
        sm._global_raw: dict = settings or {}
        sm._project_raw: dict = {}
        sm._overrides: dict = {}
        sm._global_error = None
        sm._project_error = None
        return sm

    def _load(self) -> None:
        self._global_raw, self._global_error = _load_json(self._global_path)
        self._project_raw, self._project_error = _load_json(self._project_path)

    async def reload(self) -> None:
        self._load()

    def get_global_settings(self) -> Settings:
        return _dict_to_settings(self._global_raw)

    def get_project_settings(self) -> Settings:
        return _dict_to_settings(self._project_raw)

    def get_settings(self) -> Settings:
        merged = _deep_merge(_deep_merge(self._global_raw, self._project_raw), self._overrides)
        return _dict_to_settings(merged)

    def apply_overrides(self, overrides: dict) -> None:
        self._overrides = _deep_merge(self._overrides, _keys_to_camel(overrides))

    def drain_errors(self) -> list[Exception]:
        errs = [e for e in (self._global_error, self._project_error) if e]
        self._global_error = self._project_error = None
        return errs

    def get_retry_settings(self) -> RetrySettings:
        return self.get_settings().retry

    def get_compaction_settings(self) -> CompactionSettings:
        return self.get_settings().compaction

    def get_branch_summary_settings(self) -> BranchSummarySettings:
        return self.get_settings().branch_summary
