from __future__ import annotations
import shlex
from pathlib import Path
from .types import PromptTemplate


def _parse_frontmatter(content: str) -> tuple[dict, str]:
    """Parse YAML-lite frontmatter between --- markers. Returns (fm_dict, body)."""
    if not content.startswith("---"):
        return {}, content
    end = content.find("\n---", 3)
    if end == -1:
        return {}, content
    fm_block = content[3:end].strip()
    body = content[end + 4:].lstrip("\n")
    fm: dict = {}
    for line in fm_block.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            fm[k.strip()] = v.strip()
    return fm, body


def load_prompt_templates(paths: list[str]) -> list[PromptTemplate]:
    """Load .md files from given paths as prompt templates.

    Each path may be:
      - A single .md file   → loaded directly
      - A directory         → all *.md files in it are loaded (non-recursive)
    """
    templates: list[PromptTemplate] = []

    # Expand directories to their .md children
    expanded_paths: list[Path] = []
    for path in paths:
        p = Path(path)
        if p.is_dir():
            expanded_paths.extend(sorted(p.glob("*.md")))
        else:
            expanded_paths.append(p)

    for p in expanded_paths:
        try:
            content = p.read_text(encoding="utf-8")
        except OSError:
            continue

        fm, body = _parse_frontmatter(content)

        name = fm.get("name", p.stem).strip()
        description = fm.get("description", "").strip()
        argument_hint = fm.get("argument_hint", None)
        if argument_hint:
            argument_hint = argument_hint.strip() or None

        # If no description in frontmatter, use first non-empty line of body
        if not description:
            for line in body.splitlines():
                line = line.strip().lstrip("#").strip()
                if line:
                    description = line
                    break

        templates.append(PromptTemplate(
            name=name,
            description=description,
            content=body,
            file_path=str(p),
            source_info={"path": str(p)},
            argument_hint=argument_hint,
        ))

    return templates


def parse_command_args(raw: str) -> list[str]:
    """Bash-style argument parsing (handles quoting). Falls back to split on error."""
    try:
        return shlex.split(raw)
    except ValueError:
        return raw.split()


def substitute_args(template: str, args: list[str]) -> str:
    """Substitute $1/$2/..., $@/$ARGUMENTS, ${@:N}, ${@:N:L} in template."""
    import re

    def _slice(args: list[str], n: int, length: int | None = None) -> str:
        subset = args[n - 1:] if length is None else args[n - 1: n - 1 + length]
        return " ".join(subset)

    # ${@:N:L} — slice with length
    def replace_slice_len(m: re.Match) -> str:
        n = int(m.group(1))
        l = int(m.group(2))
        return _slice(args, n, l)

    template = re.sub(r'\$\{@:(\d+):(\d+)\}', replace_slice_len, template)

    # ${@:N} — slice from N
    def replace_slice(m: re.Match) -> str:
        n = int(m.group(1))
        return _slice(args, n)

    template = re.sub(r'\$\{@:(\d+)\}', replace_slice, template)

    # $@ and $ARGUMENTS
    all_args = " ".join(args)
    template = template.replace("$@", all_args)
    template = template.replace("$ARGUMENTS", all_args)

    # $1, $2, ...
    def replace_positional(m: re.Match) -> str:
        idx = int(m.group(1))
        return args[idx - 1] if idx <= len(args) else ""

    template = re.sub(r'\$(\d+)', replace_positional, template)

    return template
