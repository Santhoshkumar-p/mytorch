from __future__ import annotations
import html
from dataclasses import dataclass, field
from datetime import date
from .types import ContextFile, Skill

_TOOL_SNIPPETS = {
    "bash": "Execute shell commands in the project directory",
    "read": "Read files and images from the filesystem",
    "edit": "Make targeted text replacements in existing files",
    "write": "Create or overwrite files",
    "grep": "Search file contents with regex or literal patterns",
    "find": "Find files by glob pattern",
    "ls":   "List directory contents",
}

_BASE_INTRO = (
    "You are an AI coding assistant. You help users write, understand, and improve code.\n"
    "You have access to tools that let you read and modify files, run shell commands, and search codebases.\n"
    "Always think carefully before making changes, and prefer targeted edits over full rewrites."
)


@dataclass
class BuildSystemPromptOptions:
    custom_prompt: str | None = None
    selected_tools: list[str] = field(default_factory=lambda: ["read", "bash", "edit", "write"])
    tool_snippets: dict[str, str] = field(default_factory=dict)
    prompt_guidelines: list[str] = field(default_factory=list)
    append_system_prompt: str | None = None
    cwd: str | None = None
    context_files: list[ContextFile] = field(default_factory=list)
    skills: list[Skill] = field(default_factory=list)


def build_system_prompt(options: BuildSystemPromptOptions) -> str:
    """Build system prompt with conditional sections."""
    sections: list[str] = []

    if options.custom_prompt:
        sections.append(options.custom_prompt)
    else:
        sections.append(_BASE_INTRO)

        # Available tools section
        snippets = {**_TOOL_SNIPPETS, **options.tool_snippets}
        tool_lines = []
        for tool in options.selected_tools:
            desc = snippets.get(tool, tool)
            tool_lines.append(f"- **{tool}**: {desc}")
        if tool_lines:
            sections.append("## Available Tools\n" + "\n".join(tool_lines))

        # Guidelines
        guidelines = _build_guidelines(options.selected_tools, options.prompt_guidelines)
        if guidelines:
            sections.append("## Guidelines\n" + guidelines)

    # Context files
    for cf in options.context_files:
        sections.append(f"## {cf.path}\n{cf.content}")

    # Skills (only if "read" in selected_tools)
    if options.skills and "read" in options.selected_tools:
        sections.append(_format_skills(options.skills))

    # Append system prompt
    if options.append_system_prompt:
        sections.append(options.append_system_prompt)

    # Current date + cwd (always last)
    footer_parts = [f"Today's date: {date.today().isoformat()}"]
    if options.cwd:
        footer_parts.append(f"Working directory: {options.cwd}")
    sections.append("\n".join(footer_parts))

    return "\n\n".join(s for s in sections if s)


def _build_guidelines(tools: list[str], extra: list[str]) -> str:
    """Return bullet-list guidelines based on which tools are available."""
    lines: list[str] = []

    has_bash = "bash" in tools
    has_search_tools = any(t in tools for t in ("grep", "find", "ls"))

    if has_bash and not has_search_tools:
        lines.append("- Use bash for file searching and listing")
    elif has_bash and has_search_tools:
        lines.append("- Prefer grep, find, ls over bash for file operations")

    for guideline in extra:
        lines.append(f"- {guideline}" if not guideline.startswith("-") else guideline)

    return "\n".join(lines)


def _format_skills(skills: list[Skill]) -> str:
    """Format skills as <available_skills> XML for progressive disclosure.

    Only name, description, and file location are included here.  The full
    skill content is loaded on-demand by the model via the ``read`` tool when
    it determines a skill is relevant to the current task.

    Skills with ``disable_model_invocation=true`` are excluded (they can only
    be invoked explicitly via a slash command).
    """
    inner_parts: list[str] = []
    for skill in skills:
        if skill.disable_model_invocation:
            continue
        name = html.escape(skill.name)
        desc = html.escape(skill.description) if skill.description else ""
        loc = html.escape(skill.path)
        inner_parts.append(
            f"  <skill>\n"
            f"    <name>{name}</name>\n"
            f"    <description>{desc}</description>\n"
            f"    <location>{loc}</location>\n"
            f"  </skill>"
        )
    if not inner_parts:
        return ""
    inner = "\n".join(inner_parts)
    return (
        "<available_skills>\n"
        f"{inner}\n"
        "</available_skills>"
    )
