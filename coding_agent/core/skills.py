from __future__ import annotations
import re
from pathlib import Path
from .types import Skill

# Valid skill names: 1-64 chars, lowercase letters/digits/hyphens, no leading/trailing hyphens
_NAME_RE = re.compile(r'^[a-z0-9][a-z0-9-]{0,62}[a-z0-9]$|^[a-z0-9]$')


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


def _parse_list_field(value: str) -> list[str]:
    """Parse '[item1, item2]' or 'item1, item2' into a list of strings."""
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    return [item.strip().lstrip("/") for item in value.split(",") if item.strip()]


def load_skill(path: str, source_info: dict | None = None) -> Skill | None:
    """Load a single skill.

    Accepts:
    - A directory path: looks for SKILL.md inside it
    - A SKILL.md file path: loads it directly
    - Any other .md file: loads as a legacy skill

    The skill name defaults to:
    - Frontmatter ``name`` field if present
    - Parent directory name when loading from SKILL.md
    - File stem otherwise
    """
    p = Path(path)

    # If a directory is given, resolve to SKILL.md inside it
    if p.is_dir():
        skill_md = p / "SKILL.md"
        if skill_md.exists():
            p = skill_md
        else:
            return None

    try:
        content = p.read_text(encoding="utf-8")
    except OSError:
        return None

    fm, body = _parse_frontmatter(content)

    # Default name: parent dir for SKILL.md files, file stem for others
    if p.name == "SKILL.md":
        default_name = p.parent.name.lower()
        # Sanitise: replace underscores/spaces with hyphens
        default_name = re.sub(r'[^a-z0-9-]', '-', default_name).strip('-')
    else:
        default_name = p.stem.lower()

    name = fm.get("name", default_name).strip()

    if not _NAME_RE.match(name):
        return None

    description = fm.get("description", "").strip()
    commands_raw = fm.get("commands", "")
    tags_raw = fm.get("tags", "")
    disable_model_invocation = fm.get("disable_model_invocation", "false").lower() == "true"

    commands = _parse_list_field(commands_raw) if commands_raw else []
    tags = _parse_list_field(tags_raw) if tags_raw else []

    return Skill(
        name=name,
        path=str(p),
        content=body,
        description=description,
        base_dir=str(p.parent),
        source_info=source_info or {},
        commands=commands,
        tags=tags,
        disable_model_invocation=disable_model_invocation,
    )


def load_skills_from_dir(dir_path: str, source_info: dict | None = None) -> list[Skill]:
    """Load skills from a directory.

    Rules (derived from agent behaviour):

    1. Root has ``SKILL.md`` **with an explicit ``name:`` field** → the root
       directory *is* a single skill; return it immediately.

    2. Root has ``SKILL.md`` **without an explicit name** (index / marker file)
       → load every ``.md`` file that is a *direct child* of root (not nested).
       The marker itself is skipped.

    3. Root has **no** ``SKILL.md`` → walk the tree recursively.  When a
       sub-directory is encountered:
       - If it *has* a ``SKILL.md``, treat it as a skill boundary: load that
         ``SKILL.md`` as a skill and do **not** recurse further into it.
       - Otherwise recurse into it looking for ``.md`` files.
    """
    root = Path(dir_path)
    if not root.is_dir():
        return []

    skills: list[Skill] = []
    root_skill_md = root / "SKILL.md"

    if root_skill_md.exists():
        # Peek at frontmatter to decide whether this is a "real" skill dir
        # or just an index marker.
        try:
            fm, _ = _parse_frontmatter(root_skill_md.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            fm = {}

        if fm.get("name"):
            # Explicit name → the root directory is itself one skill.
            skill = load_skill(str(root_skill_md), source_info)
            if skill:
                return [skill]

        # No explicit name → index marker.  Load direct .md children only.
        for p in sorted(root.glob("*.md")):
            if p.name == "SKILL.md":
                continue
            skill = load_skill(str(p), source_info)
            if skill:
                skills.append(skill)
        return skills

    # No root SKILL.md → recursive walk with SKILL.md boundary detection.
    def _walk(directory: Path) -> None:
        try:
            entries = sorted(directory.iterdir(), key=lambda e: e.name)
        except OSError:
            return
        for entry in entries:
            if entry.is_file() and entry.suffix == ".md" and entry.name != "SKILL.md":
                skill = load_skill(str(entry), source_info)
                if skill:
                    skills.append(skill)
            elif entry.is_dir():
                subskill_md = entry / "SKILL.md"
                if subskill_md.exists():
                    # Boundary: this subdir is a skill, don't recurse into it.
                    skill = load_skill(str(subskill_md), source_info)
                    if skill:
                        skills.append(skill)
                else:
                    _walk(entry)

    _walk(root)
    return skills
