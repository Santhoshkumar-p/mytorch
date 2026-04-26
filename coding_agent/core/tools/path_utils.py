from __future__ import annotations

import os
import unicodedata
from pathlib import Path


def expand_path(path: str) -> str:
    """Expand ~ and strip leading @ prefix."""
    if path.startswith("@"):
        path = path[1:]
    return os.path.expanduser(path)


def resolve_to_cwd(path: str, cwd: str) -> str:
    """Expand then make absolute relative to cwd."""
    expanded = expand_path(path)
    if os.path.isabs(expanded):
        return expanded
    return str(Path(cwd) / expanded)


def resolve_read_path(path: str, cwd: str) -> str | None:
    """Try four Unicode variants to handle macOS screenshot filenames.

    Returns the first existing path or None.
    """
    base = resolve_to_cwd(path, cwd)

    candidates: list[str] = []

    # 1. Exact path
    candidates.append(base)

    # 2. U+202F (narrow no-break space) <-> regular space swap
    if "\u202f" in base:
        candidates.append(base.replace("\u202f", " "))
    elif " " in base:
        candidates.append(base.replace(" ", "\u202f"))

    # 3. NFD decomposed
    candidates.append(unicodedata.normalize("NFD", base))

    # 4. Curly apostrophe U+2019 <-> straight apostrophe
    if "\u2019" in base:
        candidates.append(base.replace("\u2019", "'"))
    elif "'" in base:
        candidates.append(base.replace("'", "\u2019"))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    return None
