"""
Slash command registry for the interactive TUI.

get_slash_commands() returns a list of SlashCommand objects used for:
- Autocomplete in InputBar
- Dispatch in app.py _dispatch_slash()
"""
from __future__ import annotations
from .types import SlashCommand, Skill, PromptTemplate, Settings

# Built-in TUI commands — ordered for autocomplete display.
# Tuple: (name, description, action_key)
_BUILTIN: list[tuple[str, str, str]] = [
    ("help",          "Show all available slash commands",               "show_help"),
    ("session",       "Show session info and token statistics",          "session_info"),
    ("name",          "Set or display session name  /name <name>",       "session_name"),
    ("compact",       "Compact history  /compact [custom instructions]", "compact"),
    ("new",           "Start a fresh session",                           "new_session"),
    ("clear",         "Clear the message display",                       "clear_messages"),
    ("model",         "Select/search model  /model [search term]",       "select_model"),
    ("scoped-models", "Show models configured for Ctrl+M cycling",       "scoped_models"),
    ("thinking",      "Cycle thinking level  off/minimal/low/medium/high/xhigh", "cycle_thinking"),
    ("export",        "Export session  /export [path.html|path.jsonl]",  "export"),
    ("import",        "Import session  /import <path.jsonl>",            "import_session"),
    ("share",         "Share session as a secret GitHub Gist",           "share"),
    ("copy",          "Copy last assistant message to clipboard",        "copy"),
    ("sessions",      "Open session picker",                             "show_session_picker"),
    ("resume",        "Resume a different session  (alias: /sessions)",  "show_session_picker"),
    ("fork",          "Fork conversation from a previous message",       "fork"),
    ("tree",          "Navigate the session tree / switch branches",     "tree"),
    ("skills",        "List loaded skills and their commands",           "show_skills"),
    ("settings",      "Show current settings",                           "settings"),
    ("hotkeys",       "Show all keyboard shortcuts",                     "hotkeys"),
    ("changelog",     "Show changelog",                                  "changelog"),
    ("reload",        "Reload skills, extensions, and prompts",          "reload"),
    ("abort",         "Abort the current agent operation",               "abort_agent"),
    ("quit",          "Quit the application",                            "quit"),
]


def get_slash_commands(
    extension_runner=None,
    prompt_templates: list[PromptTemplate] | None = None,
    skills: list[Skill] | None = None,
    settings: Settings | None = None,
) -> list[SlashCommand]:
    """Return all slash commands: builtins + extensions + prompt templates + skills."""
    commands: list[SlashCommand] = []

    # Built-in TUI commands first
    for name, description, action in _BUILTIN:
        commands.append(SlashCommand(
            name=name,
            description=description,
            source="builtin",
            source_info={"action": action},
            handler=None,
        ))

    # Extension-registered commands
    if extension_runner is not None:
        try:
            for cmd in extension_runner.get_registered_commands():
                commands.append(SlashCommand(
                    name=cmd.get("name", ""),
                    description=cmd.get("description"),
                    source="extension",
                    source_info=cmd,
                    handler=cmd.get("handler"),
                ))
        except Exception:
            pass

    # Prompt templates
    for template in (prompt_templates or []):
        commands.append(SlashCommand(
            name=template.name,
            description=template.description or None,
            source="prompt",
            source_info={"path": template.file_path},
            handler=None,
        ))

    # Skills — each skill command becomes a slash command.
    # If a skill has no explicit commands: in frontmatter, use the skill name itself.
    enable_skill_commands = True
    if settings is not None:
        enable_skill_commands = getattr(settings, "enable_skill_commands", True)

    if enable_skill_commands:
        for skill in (skills or []):
            cmd_names = list(skill.commands)
            if not cmd_names:
                # Auto-derive a slash command from the skill name
                cmd_names = [skill.name]
            for cmd_name in cmd_names:
                if not cmd_name:
                    continue
                commands.append(SlashCommand(
                    name=cmd_name,
                    description=skill.description or f"Run skill: {skill.name}",
                    source="skill",
                    source_info={"skill_name": skill.name, "path": skill.path},
                    handler=None,
                ))

    return commands
