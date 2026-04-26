"""Tests for coding_agent.core.export_html.export."""
import asyncio
import json
from pathlib import Path
import pytest


async def test_export_creates_file(tmp_path):
    from coding_agent.core.session_manager import SessionManager
    from coding_agent.core.export_html.export import export_session_to_html
    sm = SessionManager.create(str(tmp_path))
    sm.append_model_change("anthropic", "claude-opus-4-5")
    sm.append_session_info("Test Export")
    await sm.flush()
    out = await export_session_to_html(sm, str(tmp_path / "out.html"))
    assert Path(out).exists()
    content = Path(out).read_text()
    assert "<html" in content.lower()


async def test_export_contains_session_id(tmp_path):
    from coding_agent.core.session_manager import SessionManager
    from coding_agent.core.export_html.export import export_session_to_html
    sm = SessionManager.create(str(tmp_path))
    sm.append_session_info("Test")
    await sm.flush()
    out = await export_session_to_html(sm, str(tmp_path / "out.html"))
    content = Path(out).read_text()
    # Session ID (first 8 chars at minimum) should be in HTML
    assert sm.get_session_id()[:8] in content


async def test_export_contains_session_name(tmp_path):
    from coding_agent.core.session_manager import SessionManager
    from coding_agent.core.export_html.export import export_session_to_html
    sm = SessionManager.create(str(tmp_path))
    sm.append_session_info("My Named Session")
    await sm.flush()
    out = await export_session_to_html(sm, str(tmp_path / "out.html"))
    content = Path(out).read_text()
    assert "My Named Session" in content


async def test_export_from_file(tmp_path):
    from coding_agent.core.session_manager import SessionManager
    from coding_agent.core.export_html.export import export_from_file
    sm = SessionManager.create(str(tmp_path))
    sm.append_session_info("From File")
    await sm.flush()
    session_file = sm.get_session_file()
    out = await export_from_file(session_file, str(tmp_path / "exported.html"))
    assert Path(out).exists()
    content = Path(out).read_text()
    assert "<html" in content.lower()


async def test_export_html_structure(tmp_path):
    from coding_agent.core.session_manager import SessionManager
    from coding_agent.core.export_html.export import export_session_to_html
    sm = SessionManager.create(str(tmp_path))
    await sm.flush()
    out = await export_session_to_html(sm, str(tmp_path / "out.html"))
    content = Path(out).read_text()
    # Basic HTML structure
    assert "<head>" in content
    assert "<body>" in content
    assert "</html>" in content


async def test_export_embeds_session_data(tmp_path):
    from coding_agent.core.session_manager import SessionManager
    from coding_agent.core.export_html.export import export_session_to_html
    import base64
    sm = SessionManager.create(str(tmp_path))
    sm.append_model_change("anthropic", "test-model")
    await sm.flush()
    out = await export_session_to_html(sm, str(tmp_path / "out.html"))
    content = Path(out).read_text()
    # Should embed JSONL data for download (base64 in the hidden textarea)
    assert 'id="jdata"' in content
    assert "dlJsonl" in content


async def test_export_compaction_entry_rendered(tmp_path):
    from coding_agent.core.session_manager import SessionManager
    from coding_agent.core.export_html.export import export_session_to_html
    sm = SessionManager.create(str(tmp_path))
    e1 = sm.append_model_change("anthropic", "claude-opus-4-5")
    sm.append_compaction("Compacted the history", first_kept_entry_id=e1.id, tokens_before=1000)
    await sm.flush()
    out = await export_session_to_html(sm, str(tmp_path / "out.html"))
    content = Path(out).read_text()
    assert "compaction" in content.lower() or "Compacted" in content


async def test_export_empty_session(tmp_path):
    from coding_agent.core.session_manager import SessionManager
    from coding_agent.core.export_html.export import export_session_to_html
    sm = SessionManager.create(str(tmp_path))
    out = await export_session_to_html(sm, str(tmp_path / "empty.html"))
    assert Path(out).exists()
    content = Path(out).read_text()
    assert "<html" in content.lower()


async def test_export_default_output_path(tmp_path):
    """If no output_path provided, file should be created in CWD."""
    import os
    from coding_agent.core.session_manager import SessionManager
    from coding_agent.core.export_html.export import export_session_to_html
    sm = SessionManager.create(str(tmp_path))
    # Change CWD to tmp_path so the default output path goes there
    old_cwd = os.getcwd()
    try:
        os.chdir(str(tmp_path))
        out = await export_session_to_html(sm)  # no output_path
        assert Path(out).exists()
    finally:
        os.chdir(old_cwd)
        # Cleanup
        try:
            Path(out).unlink()
        except Exception:
            pass
