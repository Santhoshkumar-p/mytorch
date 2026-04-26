from __future__ import annotations
import argparse


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="agent", description="AI coding agent")

    # Model
    p.add_argument("--provider", help="Model provider (anthropic, openai, vertex)")
    p.add_argument("--model", help="Model name or provider/model[:thinking]")
    p.add_argument("--api-key", dest="api_key", help="API key for the provider")

    # System prompt
    p.add_argument("--system-prompt", dest="system_prompt", help="Override system prompt")
    p.add_argument(
        "--append-system-prompt",
        action="append",
        dest="append_system_prompt",
        default=[],
        help="Append text to system prompt (can be repeated)",
    )

    # Thinking
    p.add_argument(
        "--thinking",
        choices=["off", "minimal", "low", "medium", "high", "xhigh"],
        help="Thinking level",
    )

    # Session
    p.add_argument("-c", "--continue", action="store_true", dest="continue_",
                   help="Continue most recent session")
    p.add_argument("-r", "--resume", action="store_true",
                   help="Resume (alias for --continue)")
    p.add_argument("--no-session", action="store_true", dest="no_session",
                   help="Do not persist this session")
    p.add_argument("--session", metavar="ID_OR_PATH",
                   help="Open a specific session by ID prefix or file path")
    p.add_argument("--fork", metavar="ENTRY_ID",
                   help="Fork from a specific entry in the current session")
    p.add_argument("--session-dir", dest="session_dir", metavar="DIR",
                   help="Directory to store session files")

    # Model cycling
    p.add_argument("--models", metavar="PATTERNS",
                   help="Comma-separated model patterns to cycle through (e.g. 'anthropic/*,openai/gpt-4o')")

    # Tools
    p.add_argument("--tools", metavar="NAMES",
                   help="Comma-separated list of tools to enable")
    p.add_argument("--no-tools", action="store_true", dest="no_tools",
                   help="Disable all tools")

    # Output mode
    p.add_argument("--mode", choices=["text", "json", "rpc", "interactive"],
                   help="Output mode: text, json (events), rpc (stdin/stdout JSON), or interactive (TUI)")
    p.add_argument("-p", "--print", action="store_true",
                   help="Print mode (shorthand for --mode text)")
    p.add_argument("--export", metavar="PATH",
                   help="Export session to HTML and exit")

    # Resources
    p.add_argument("-e", "--extension", action="append", dest="extensions", default=[],
                   help="Path to extension module (can be repeated)")
    p.add_argument("-ne", "--no-extensions", action="store_true", dest="no_extensions",
                   help="Disable all extensions")
    p.add_argument("--skill", action="append", dest="skills", default=[],
                   help="Extra skill path (can be repeated)")
    p.add_argument("-ns", "--no-skills", action="store_true", dest="no_skills",
                   help="Disable all skills")
    p.add_argument("--prompt-template", action="append", dest="prompt_templates", default=[],
                   help="Extra prompt template path (can be repeated)")
    p.add_argument("-np", "--no-prompt-templates", action="store_true",
                   dest="no_prompt_templates", help="Disable all prompt templates")
    p.add_argument("--theme", action="append", dest="themes", default=[],
                   help="Theme paths (can be repeated)")
    p.add_argument("--no-themes", action="store_true", dest="no_themes",
                   help="Disable themes")

    # Context
    p.add_argument("-nc", "--no-context-files", action="store_true", dest="no_context_files",
                   help="Do not load AGENTS.md / CLAUDE.md context files")

    # Utility
    p.add_argument("-v", "--version", action="store_true", help="Print version and exit")
    p.add_argument("--list-models", nargs="?", const=True, metavar="PATTERN",
                   help="List available models (optionally filtered by pattern)")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    p.add_argument("--offline", action="store_true", help="Offline mode")

    # Positional — message text and @file references
    p.add_argument("messages", nargs="*", help="Initial message text or @file references")

    return p.parse_args(argv)


def parse_model_shorthand(s: str) -> tuple[str, str | None]:
    """'provider/model:thinking' -> ('provider/model', 'thinking')"""
    # Only split on colon after the last slash segment
    last_segment = s.rsplit("/", 1)[-1]
    if ":" in last_segment:
        model, _, level = s.rpartition(":")
        return model, level
    return s, None


def validate_fork_flags(args) -> list[str]:
    """Return error messages for mutually exclusive flag combinations."""
    errors: list[str] = []
    if getattr(args, "fork", None):
        if getattr(args, "continue_", False):
            errors.append("--fork is mutually exclusive with --continue")
        if getattr(args, "resume", False):
            errors.append("--fork is mutually exclusive with --resume")
        if getattr(args, "session", None):
            errors.append("--fork is mutually exclusive with --session")
    return errors
