from __future__ import annotations
import importlib.util
import inspect
import os
from pathlib import Path


async def load_extension(path: str, api: "ExtensionAPI") -> None:
    """Load a Python extension module via importlib and call its setup(api) function."""
    expanded = os.path.expanduser(path)
    p = Path(expanded)
    if not p.is_absolute():
        p = Path.cwd() / p
    p = p.resolve()

    spec = importlib.util.spec_from_file_location(p.stem, str(p))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load extension from {p}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]

    setup_fn = getattr(module, "setup", None)
    if setup_fn is None:
        raise AttributeError(f"Extension {p} has no setup() function")

    if inspect.iscoroutinefunction(setup_fn):
        await setup_fn(api)
    else:
        setup_fn(api)
