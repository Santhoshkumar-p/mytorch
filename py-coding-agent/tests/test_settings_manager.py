"""Tests for coding_agent.core.settings_manager.SettingsManager."""
import json
import pytest
from pathlib import Path


def test_in_memory_defaults():
    from coding_agent.core.settings_manager import SettingsManager
    sm = SettingsManager.in_memory()
    s = sm.get_settings()
    assert s.default_thinking_level == "off"
    assert s.compaction.reserve_tokens == 16384


def test_in_memory_with_values():
    from coding_agent.core.settings_manager import SettingsManager
    sm = SettingsManager.in_memory({"defaultThinkingLevel": "medium", "steeringMode": "one-at-a-time"})
    s = sm.get_settings()
    assert s.default_thinking_level == "medium"
    assert s.steering_mode == "one-at-a-time"


def test_apply_overrides():
    from coding_agent.core.settings_manager import SettingsManager
    sm = SettingsManager.in_memory({"defaultThinkingLevel": "low"})
    sm.apply_overrides({"default_thinking_level": "high"})
    assert sm.get_settings().default_thinking_level == "high"


def test_apply_overrides_multiple_times():
    from coding_agent.core.settings_manager import SettingsManager
    sm = SettingsManager.in_memory()
    sm.apply_overrides({"default_thinking_level": "low"})
    sm.apply_overrides({"default_thinking_level": "high"})
    assert sm.get_settings().default_thinking_level == "high"


def test_create_reads_global(tmp_path):
    from coding_agent.core.settings_manager import SettingsManager
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "settings.json").write_text(json.dumps({"defaultThinkingLevel": "medium"}))
    sm = SettingsManager.create(str(tmp_path), str(agent_dir))
    assert sm.get_settings().default_thinking_level == "medium"


def test_create_project_overrides_global(tmp_path):
    from coding_agent.core.settings_manager import SettingsManager
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "settings.json").write_text(json.dumps({"defaultThinkingLevel": "low"}))
    pi_dir = tmp_path / ".coding-agent"
    pi_dir.mkdir()
    (pi_dir / "settings.json").write_text(json.dumps({"defaultThinkingLevel": "high"}))
    sm = SettingsManager.create(str(tmp_path), str(agent_dir))
    assert sm.get_settings().default_thinking_level == "high"


def test_migration_queue_mode():
    from coding_agent.core.settings_manager import _run_migrations
    raw = _run_migrations({"queueMode": "one-at-a-time"})
    assert raw.get("steeringMode") == "one-at-a-time"
    assert "queueMode" not in raw


def test_migration_websockets_true():
    from coding_agent.core.settings_manager import _run_migrations
    raw = _run_migrations({"websockets": True})
    assert raw.get("transport") == "websocket"
    assert "websockets" not in raw


def test_migration_websockets_false():
    from coding_agent.core.settings_manager import _run_migrations
    raw = _run_migrations({"websockets": False})
    assert raw.get("transport") == "sse"
    assert "websockets" not in raw


def test_drain_errors_ok():
    from coding_agent.core.settings_manager import SettingsManager
    sm = SettingsManager.in_memory()
    errs = sm.drain_errors()
    assert errs == []


def test_drain_errors_clears_them(tmp_path):
    from coding_agent.core.settings_manager import SettingsManager
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    # Write invalid JSON to cause a parse error
    (agent_dir / "settings.json").write_text("NOT VALID JSON {{{{")
    sm = SettingsManager.create(str(tmp_path), str(agent_dir))
    errs = sm.drain_errors()
    assert len(errs) > 0
    # Second drain should be empty
    errs2 = sm.drain_errors()
    assert errs2 == []


def test_deep_merge_nested():
    from coding_agent.core.settings_manager import _deep_merge
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    override = {"a": {"y": 99, "z": 100}, "c": 4}
    result = _deep_merge(base, override)
    assert result == {"a": {"x": 1, "y": 99, "z": 100}, "b": 3, "c": 4}


def test_deep_merge_non_dict_override():
    from coding_agent.core.settings_manager import _deep_merge
    base = {"a": {"x": 1}}
    override = {"a": "string"}
    result = _deep_merge(base, override)
    assert result["a"] == "string"


def test_deep_merge_empty_override():
    from coding_agent.core.settings_manager import _deep_merge
    base = {"a": 1, "b": 2}
    result = _deep_merge(base, {})
    assert result == {"a": 1, "b": 2}


def test_get_global_settings(tmp_path):
    from coding_agent.core.settings_manager import SettingsManager
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "settings.json").write_text(json.dumps({"defaultThinkingLevel": "low"}))
    pi_dir = tmp_path / ".coding-agent"
    pi_dir.mkdir()
    (pi_dir / "settings.json").write_text(json.dumps({"defaultThinkingLevel": "high"}))
    sm = SettingsManager.create(str(tmp_path), str(agent_dir))
    # get_global_settings returns only global (agent_dir) settings
    global_s = sm.get_global_settings()
    assert global_s.default_thinking_level == "low"


def test_get_project_settings(tmp_path):
    from coding_agent.core.settings_manager import SettingsManager
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    pi_dir = tmp_path / ".coding-agent"
    pi_dir.mkdir()
    (pi_dir / "settings.json").write_text(json.dumps({"steeringMode": "one-at-a-time"}))
    sm = SettingsManager.create(str(tmp_path), str(agent_dir))
    project_s = sm.get_project_settings()
    assert project_s.steering_mode == "one-at-a-time"


def test_compaction_settings_nested(tmp_path):
    from coding_agent.core.settings_manager import SettingsManager
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "settings.json").write_text(json.dumps({
        "compaction": {"reserveTokens": 8192, "enabled": False}
    }))
    sm = SettingsManager.create(str(tmp_path), str(agent_dir))
    s = sm.get_settings()
    assert s.compaction.reserve_tokens == 8192
    assert s.compaction.enabled is False


def test_retry_settings_nested(tmp_path):
    from coding_agent.core.settings_manager import SettingsManager
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "settings.json").write_text(json.dumps({
        "retry": {"enabled": False, "maxRetries": 5}
    }))
    sm = SettingsManager.create(str(tmp_path), str(agent_dir))
    s = sm.get_settings()
    assert s.retry.enabled is False
    assert s.retry.max_retries == 5
