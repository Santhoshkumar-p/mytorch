"""Tests for coding_agent.cli.args."""
import pytest


def test_parse_basic():
    from coding_agent.cli.args import parse_args
    args = parse_args(["--provider", "anthropic", "--model", "claude-opus-4-5"])
    assert args.provider == "anthropic"
    assert args.model == "claude-opus-4-5"


def test_parse_mode():
    from coding_agent.cli.args import parse_args
    args = parse_args(["--mode", "rpc"])
    assert args.mode == "rpc"


def test_parse_mode_text():
    from coding_agent.cli.args import parse_args
    args = parse_args(["--mode", "text"])
    assert args.mode == "text"


def test_parse_mode_json():
    from coding_agent.cli.args import parse_args
    args = parse_args(["--mode", "json"])
    assert args.mode == "json"


def test_parse_print_flag():
    from coding_agent.cli.args import parse_args
    args = parse_args(["-p"])
    assert args.print is True


def test_parse_print_long_flag():
    from coding_agent.cli.args import parse_args
    args = parse_args(["--print"])
    assert args.print is True


def test_parse_extension():
    from coding_agent.cli.args import parse_args
    args = parse_args(["-e", "ext1.py", "-e", "ext2.py"])
    assert args.extensions == ["ext1.py", "ext2.py"]


def test_parse_no_tools():
    from coding_agent.cli.args import parse_args
    args = parse_args(["--no-tools"])
    assert args.no_tools is True


def test_parse_thinking():
    from coding_agent.cli.args import parse_args
    for level in ["off", "minimal", "low", "medium", "high", "xhigh"]:
        args = parse_args(["--thinking", level])
        assert args.thinking == level


def test_parse_api_key():
    from coding_agent.cli.args import parse_args
    args = parse_args(["--api-key", "sk-test-123"])
    assert args.api_key == "sk-test-123"


def test_parse_system_prompt():
    from coding_agent.cli.args import parse_args
    args = parse_args(["--system-prompt", "You are helpful"])
    assert args.system_prompt == "You are helpful"


def test_parse_append_system_prompt():
    from coding_agent.cli.args import parse_args
    args = parse_args(["--append-system-prompt", "Extra text", "--append-system-prompt", "More text"])
    assert "Extra text" in args.append_system_prompt
    assert "More text" in args.append_system_prompt


def test_parse_continue():
    from coding_agent.cli.args import parse_args
    args = parse_args(["-c"])
    assert args.continue_ is True


def test_parse_resume():
    from coding_agent.cli.args import parse_args
    args = parse_args(["-r"])
    assert args.resume is True


def test_parse_session():
    from coding_agent.cli.args import parse_args
    args = parse_args(["--session", "abc123"])
    assert args.session == "abc123"


def test_parse_fork():
    from coding_agent.cli.args import parse_args
    args = parse_args(["--fork", "entry123"])
    assert args.fork == "entry123"


def test_parse_no_session():
    from coding_agent.cli.args import parse_args
    args = parse_args(["--no-session"])
    assert args.no_session is True


def test_parse_verbose():
    from coding_agent.cli.args import parse_args
    args = parse_args(["--verbose"])
    assert args.verbose is True


def test_parse_version():
    from coding_agent.cli.args import parse_args
    args = parse_args(["-v"])
    assert args.version is True


def test_parse_model_shorthand():
    from coding_agent.cli.args import parse_model_shorthand
    model, level = parse_model_shorthand("anthropic/claude-opus-4-5:high")
    assert model == "anthropic/claude-opus-4-5"
    assert level == "high"


def test_parse_model_shorthand_no_level():
    from coding_agent.cli.args import parse_model_shorthand
    model, level = parse_model_shorthand("anthropic/claude-opus-4-5")
    assert model == "anthropic/claude-opus-4-5"
    assert level is None


def test_parse_model_shorthand_minimal():
    from coding_agent.cli.args import parse_model_shorthand
    model, level = parse_model_shorthand("gpt-4o:low")
    assert model == "gpt-4o"
    assert level == "low"


def test_validate_fork_exclusive():
    from coding_agent.cli.args import parse_args, validate_fork_flags
    args = parse_args(["--fork", "abc", "--continue"])
    errors = validate_fork_flags(args)
    assert len(errors) > 0


def test_validate_fork_ok():
    from coding_agent.cli.args import parse_args, validate_fork_flags
    args = parse_args(["--fork", "abc"])
    errors = validate_fork_flags(args)
    assert errors == []


def test_validate_fork_with_resume_exclusive():
    from coding_agent.cli.args import parse_args, validate_fork_flags
    args = parse_args(["--fork", "abc", "--resume"])
    errors = validate_fork_flags(args)
    assert len(errors) > 0


def test_validate_fork_with_session_exclusive():
    from coding_agent.cli.args import parse_args, validate_fork_flags
    args = parse_args(["--fork", "abc", "--session", "xyz"])
    errors = validate_fork_flags(args)
    assert len(errors) > 0


def test_parse_messages():
    from coding_agent.cli.args import parse_args
    args = parse_args(["hello world", "second message"])
    assert args.messages == ["hello world", "second message"]


def test_parse_messages_empty():
    from coding_agent.cli.args import parse_args
    args = parse_args([])
    assert args.messages == []


def test_parse_tools():
    from coding_agent.cli.args import parse_args
    args = parse_args(["--tools", "bash,read,edit"])
    assert args.tools == "bash,read,edit"


def test_parse_no_skills():
    from coding_agent.cli.args import parse_args
    args = parse_args(["-ns"])
    assert args.no_skills is True


def test_parse_session_dir():
    from coding_agent.cli.args import parse_args
    args = parse_args(["--session-dir", "/tmp/sessions"])
    assert args.session_dir == "/tmp/sessions"
