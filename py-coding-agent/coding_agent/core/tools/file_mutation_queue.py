from __future__ import annotations

import asyncio
import os
from typing import Any, Callable

_locks: dict[str, asyncio.Lock] = {}


async def with_file_mutation_queue(abs_path: str, fn: Callable) -> Any:
    """Serialize mutations on the same file, resolving symlinks for identity."""
    real = os.path.realpath(abs_path)
    if real not in _locks:
        _locks[real] = asyncio.Lock()
    async with _locks[real]:
        return await fn()
