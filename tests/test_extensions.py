"""Tests for coding_agent.core.extensions."""
import asyncio
import pytest
from pathlib import Path


async def test_extension_loads(tmp_path):
    from coding_agent.core.extensions.loader import load_extension
    from coding_agent.core.extensions.types import ExtensionAPI
    ext_file = tmp_path / "my_ext.py"
    ext_file.write_text("""
async def setup(api):
    api.register_command("test-cmd", description="A test command")
""")
    api = ExtensionAPI()
    await load_extension(str(ext_file), api)
    assert any(c["name"] == "test-cmd" for c in api._commands)


async def test_extension_sync_setup(tmp_path):
    from coding_agent.core.extensions.loader import load_extension
    from coding_agent.core.extensions.types import ExtensionAPI
    ext_file = tmp_path / "sync_ext.py"
    ext_file.write_text("""
def setup(api):
    api.register_command("sync-cmd", description="Sync command")
""")
    api = ExtensionAPI()
    await load_extension(str(ext_file), api)
    assert any(c["name"] == "sync-cmd" for c in api._commands)


async def test_extension_no_setup_raises(tmp_path):
    from coding_agent.core.extensions.loader import load_extension
    from coding_agent.core.extensions.types import ExtensionAPI
    ext_file = tmp_path / "no_setup.py"
    ext_file.write_text("""
# No setup function
x = 42
""")
    api = ExtensionAPI()
    with pytest.raises(AttributeError):
        await load_extension(str(ext_file), api)


async def test_extension_registers_tool(tmp_path):
    from coding_agent.core.extensions.loader import load_extension
    from coding_agent.core.extensions.types import ExtensionAPI
    ext_file = tmp_path / "tool_ext.py"
    ext_file.write_text("""
def setup(api):
    api.register_tool({"name": "my_tool", "description": "A custom tool"})
""")
    api = ExtensionAPI()
    await load_extension(str(ext_file), api)
    assert len(api._tools) == 1
    assert api._tools[0]["name"] == "my_tool"


async def test_extension_runner_emit(tmp_path):
    from coding_agent.core.extensions.runner import ExtensionRunner
    from coding_agent.core.extensions.types import ExtensionContext
    runner = ExtensionRunner()
    ext_file = tmp_path / "evt_ext.py"
    ext_file.write_text("""
def setup(api):
    def on_start(ctx):
        pass
    api.on("session_start", on_start)
""")
    await runner.load([str(ext_file)])
    ctx = ExtensionContext(session=None, session_manager=None, model=None, cwd="/")
    result = await runner.emit("session_start", ctx)
    assert isinstance(result, dict)


async def test_extension_runner_emit_with_return(tmp_path):
    from coding_agent.core.extensions.runner import ExtensionRunner
    from coding_agent.core.extensions.types import ExtensionContext
    runner = ExtensionRunner()
    ext_file = tmp_path / "cancel_ext.py"
    ext_file.write_text("""
def setup(api):
    def on_before_switch(ctx):
        return {"cancel": True}
    api.on("session_before_switch", on_before_switch)
""")
    await runner.load([str(ext_file)])
    ctx = ExtensionContext(session=None, session_manager=None, model=None, cwd="/")
    result = await runner.emit("session_before_switch", ctx)
    assert result.get("cancel") is True


async def test_extension_flag_defaults(tmp_path):
    from coding_agent.core.extensions.runner import ExtensionRunner
    ext_file = tmp_path / "flag_ext.py"
    ext_file.write_text("""
def setup(api):
    api.register_flag("verbose", type="boolean", default=False)
    api.register_flag("count", type="string", default="10")
""")
    runner = ExtensionRunner()
    await runner.load([str(ext_file)])
    flags = runner.get_flag_values()
    assert flags.get("verbose") is False
    assert flags.get("count") == "10"


async def test_extension_runner_load_bad_file_does_not_crash(tmp_path):
    from coding_agent.core.extensions.runner import ExtensionRunner
    runner = ExtensionRunner()
    # Nonexistent file — should log error but not raise
    await runner.load([str(tmp_path / "nonexistent.py")])
    assert len(runner._apis) == 0


async def test_extension_runner_load_multiple(tmp_path):
    from coding_agent.core.extensions.runner import ExtensionRunner
    ext1 = tmp_path / "ext1.py"
    ext2 = tmp_path / "ext2.py"
    ext1.write_text("""
def setup(api):
    api.register_command("cmd1", description="First")
""")
    ext2.write_text("""
def setup(api):
    api.register_command("cmd2", description="Second")
""")
    runner = ExtensionRunner()
    await runner.load([str(ext1), str(ext2)])
    commands = runner.get_registered_commands()
    names = [c["name"] for c in commands]
    assert "cmd1" in names
    assert "cmd2" in names


async def test_extension_runner_get_registered_tools(tmp_path):
    from coding_agent.core.extensions.runner import ExtensionRunner
    ext_file = tmp_path / "tool_ext.py"
    ext_file.write_text("""
def setup(api):
    api.register_tool({"name": "custom_tool"})
""")
    runner = ExtensionRunner()
    await runner.load([str(ext_file)])
    tools = runner.get_registered_tools()
    assert len(tools) == 1


async def test_extension_api_on_registers_handler():
    from coding_agent.core.extensions.types import ExtensionAPI
    api = ExtensionAPI()
    called = []
    api.on("test_event", lambda ctx: called.append(ctx))
    assert "test_event" in api._handlers
    assert len(api._handlers["test_event"]) == 1


async def test_extension_api_register_command():
    from coding_agent.core.extensions.types import ExtensionAPI
    api = ExtensionAPI()
    api.register_command("my-cmd", description="Does stuff", handler=None)
    assert len(api._commands) == 1
    assert api._commands[0]["name"] == "my-cmd"
    assert api._commands[0]["description"] == "Does stuff"


async def test_extension_api_get_commands():
    from coding_agent.core.extensions.types import ExtensionAPI
    api = ExtensionAPI()
    api.register_command("cmd-a")
    api.register_command("cmd-b")
    cmds = api.get_commands()
    names = [c["name"] for c in cmds]
    assert "cmd-a" in names
    assert "cmd-b" in names


async def test_extension_api_send_message_without_session():
    from coding_agent.core.extensions.types import ExtensionAPI
    api = ExtensionAPI()
    with pytest.raises(RuntimeError):
        api.send_message("hello")


async def test_extension_context_is_idle():
    from coding_agent.core.extensions.types import ExtensionContext
    ctx = ExtensionContext(session=None, session_manager=None, model=None, cwd="/")
    assert ctx.is_idle() is False  # no session → default False


async def test_extension_context_get_system_prompt():
    from coding_agent.core.extensions.types import ExtensionContext
    ctx = ExtensionContext(session=None, session_manager=None, model=None, cwd="/")
    assert ctx.get_system_prompt() == ""  # no session → empty string
