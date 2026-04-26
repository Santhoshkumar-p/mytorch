from __future__ import annotations
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import aiofiles

from coding_agent.core.types import (
    BranchSummaryEntry,
    CompactionEntry,
    CustomEntry,
    CustomMessageEntry,
    LabelEntry,
    ModelChangeEntry,
    SessionContext,
    SessionEntryBase,
    SessionHeader,
    SessionInfoEntry,
    SessionMessageEntry,
    ThinkingLevelChangeEntry,
)

CURRENT_SESSION_VERSION = 3

ENTRY_CLASSES = {
    "message": SessionMessageEntry,
    "model_change": ModelChangeEntry,
    "thinking_level_change": ThinkingLevelChangeEntry,
    "compaction": CompactionEntry,
    "branch_summary": BranchSummaryEntry,
    "custom": CustomEntry,
    "custom_message": CustomMessageEntry,
    "label": LabelEntry,
    "session_info": SessionInfoEntry,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _snake(s: str) -> str:
    """Convert a single camelCase key to snake_case."""
    import re
    return re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()


def _header_to_dict(header: SessionHeader) -> dict:
    d = {
        "type": "header",
        "id": header.id,
        "timestamp": header.timestamp,
        "cwd": header.cwd,
        "version": header.version,
    }
    if header.parent_session is not None:
        d["parentSession"] = header.parent_session
    return d


def _entry_to_dict(entry: SessionEntryBase) -> dict:
    """Convert any entry dataclass to a camelCase dict for JSON serialization."""
    d: dict = {
        "type": entry.type,
        "id": entry.id,
        "parentId": entry.parent_id,
        "timestamp": entry.timestamp,
    }

    if isinstance(entry, SessionMessageEntry):
        msg = entry.message
        # Serialize message to a plain dict if it's not already JSON-serializable.
        # Agent types are dataclasses with slots=True — use dataclasses.asdict().
        # Anthropic SDK objects have model_dump(). Plain dicts pass through.
        if msg is not None and not isinstance(msg, (dict, str, int, float, bool, list)):
            if hasattr(msg, "model_dump"):
                msg = msg.model_dump()
            else:
                try:
                    import dataclasses
                    if dataclasses.is_dataclass(msg):
                        msg = dataclasses.asdict(msg)
                    else:
                        msg = {"role": getattr(msg, "role", "user"),
                               "content": getattr(msg, "content", [])}
                except Exception:
                    msg = {"role": getattr(msg, "role", "user"),
                           "content": getattr(msg, "content", [])}
        d["message"] = msg

    elif isinstance(entry, ModelChangeEntry):
        d["provider"] = entry.provider
        d["modelId"] = entry.model_id

    elif isinstance(entry, ThinkingLevelChangeEntry):
        d["thinkingLevel"] = entry.thinking_level

    elif isinstance(entry, CompactionEntry):
        d["summary"] = entry.summary
        d["firstKeptEntryId"] = entry.first_kept_entry_id
        d["tokensBefore"] = entry.tokens_before
        d["details"] = entry.details
        d["fromHook"] = entry.from_hook

    elif isinstance(entry, BranchSummaryEntry):
        d["fromId"] = entry.from_id
        d["summary"] = entry.summary
        d["details"] = entry.details
        d["fromHook"] = entry.from_hook

    elif isinstance(entry, CustomEntry):
        d["customType"] = entry.custom_type
        d["data"] = entry.data

    elif isinstance(entry, CustomMessageEntry):
        d["customType"] = entry.custom_type
        d["content"] = entry.content
        d["details"] = entry.details
        d["display"] = entry.display

    elif isinstance(entry, LabelEntry):
        d["targetId"] = entry.target_id
        d["label"] = entry.label

    elif isinstance(entry, SessionInfoEntry):
        d["name"] = entry.name

    return d


def _deserialize_entry(raw: dict) -> SessionEntryBase:
    """Dispatch raw dict to the correct entry dataclass."""
    entry_type = raw.get("type", "")
    cls = ENTRY_CLASSES.get(entry_type)

    base_kwargs = {
        "id": raw["id"],
        "parent_id": raw.get("parentId"),
        "timestamp": raw.get("timestamp", ""),
    }

    if cls is SessionMessageEntry:
        return SessionMessageEntry(**base_kwargs, message=raw.get("message"))

    elif cls is ModelChangeEntry:
        return ModelChangeEntry(
            **base_kwargs,
            provider=raw.get("provider", ""),
            model_id=raw.get("modelId", ""),
        )

    elif cls is ThinkingLevelChangeEntry:
        return ThinkingLevelChangeEntry(
            **base_kwargs,
            thinking_level=raw.get("thinkingLevel", "off"),
        )

    elif cls is CompactionEntry:
        return CompactionEntry(
            **base_kwargs,
            summary=raw.get("summary", ""),
            first_kept_entry_id=raw.get("firstKeptEntryId", ""),
            tokens_before=raw.get("tokensBefore", 0),
            details=raw.get("details"),
            from_hook=raw.get("fromHook", False),
        )

    elif cls is BranchSummaryEntry:
        return BranchSummaryEntry(
            **base_kwargs,
            from_id=raw.get("fromId", ""),
            summary=raw.get("summary", ""),
            details=raw.get("details"),
            from_hook=raw.get("fromHook", False),
        )

    elif cls is CustomEntry:
        return CustomEntry(
            **base_kwargs,
            custom_type=raw.get("customType", ""),
            data=raw.get("data"),
        )

    elif cls is CustomMessageEntry:
        return CustomMessageEntry(
            **base_kwargs,
            custom_type=raw.get("customType", ""),
            content=raw.get("content", ""),
            details=raw.get("details"),
            display=raw.get("display", True),
        )

    elif cls is LabelEntry:
        return LabelEntry(
            **base_kwargs,
            target_id=raw.get("targetId", ""),
            label=raw.get("label"),
        )

    elif cls is SessionInfoEntry:
        return SessionInfoEntry(**base_kwargs, name=raw.get("name"))

    # Unknown entry type — return a minimal base-like custom entry
    return CustomEntry(
        **base_kwargs,
        custom_type=entry_type,
        data=raw,
    )


def _make_session_path(session_dir: str, session_id: str) -> str:
    return str(Path(session_dir) / f"{session_id}.jsonl")


def _default_session_dir(cwd: str) -> str:
    return str(Path.home() / ".coding-agent" / "sessions")


class SessionManager:
    # ── Construction ──────────────────────────────────────────────────────────

    @classmethod
    def create(cls, cwd: str, session_dir: str | None = None) -> "SessionManager":
        sm = cls.__new__(cls)
        sm._cwd = cwd
        sm._session_dir = session_dir or _default_session_dir(cwd)
        sm._file_path = None
        sm._entries: list = []
        sm._entry_map: dict = {}
        sm._leaf_id: str | None = None
        sm._header = SessionHeader(
            id=str(uuid.uuid4()),
            timestamp=_now(),
            cwd=cwd,
        )
        sm._pending_writes: list[str] = [json.dumps(_header_to_dict(sm._header))]
        return sm

    @classmethod
    def open(
        cls,
        file_path: str,
        session_dir: str | None = None,
        cwd_override: str | None = None,
    ) -> "SessionManager":
        sm = cls.__new__(cls)
        sm._file_path = file_path
        with open(file_path) as f:
            lines = [line.strip() for line in f if line.strip()]
        raw_header = json.loads(lines[0])
        # v1→v3 migration: ensure version field
        raw_header.setdefault("version", 1)
        sm._header = SessionHeader(
            id=raw_header["id"],
            timestamp=raw_header.get("timestamp", ""),
            cwd=raw_header.get("cwd", ""),
            version=raw_header.get("version", 1),
            parent_session=raw_header.get("parentSession"),
        )
        sm._cwd = cwd_override or sm._header.cwd
        sm._session_dir = session_dir or str(Path(file_path).parent)
        sm._entries = []
        sm._entry_map = {}
        sm._leaf_id = None
        sm._pending_writes = []
        for line in lines[1:]:
            raw = json.loads(line)
            entry = _deserialize_entry(raw)
            sm._entries.append(entry)
            sm._entry_map[entry.id] = entry
            sm._leaf_id = entry.id
        return sm

    # ── Read access ───────────────────────────────────────────────────────────

    def get_entry(self, id: str):
        return self._entry_map.get(id)

    def get_leaf_id(self) -> str | None:
        return self._leaf_id

    def get_leaf_entry(self):
        return self._entry_map.get(self._leaf_id) if self._leaf_id else None

    def get_session_file(self) -> str | None:
        return self._file_path

    def get_session_id(self) -> str:
        return self._header.id

    def get_session_dir(self) -> str:
        return self._session_dir

    def get_cwd(self) -> str:
        return self._cwd

    def is_persisted(self) -> bool:
        return self._file_path is not None

    def get_session_name(self) -> str | None:
        for e in reversed(self._entries):
            if isinstance(e, SessionInfoEntry) and e.name is not None:
                return e.name
        return None

    def get_label(self, entry_id: str) -> str | None:
        for e in reversed(self._entries):
            if isinstance(e, LabelEntry) and e.target_id == entry_id:
                return e.label
        return None

    def get_children(self, parent_id: str | None) -> list:
        return [e for e in self._entries if e.parent_id == parent_id]

    def get_branch(self, from_id: str) -> list:
        """Walk from entry to root (inclusive), return in chronological order."""
        chain, eid = [], from_id
        while eid:
            e = self._entry_map.get(eid)
            if not e:
                break
            chain.append(e)
            eid = e.parent_id
        return list(reversed(chain))

    def get_tree(self) -> list:
        """Return tree as list of root nodes each with .children populated."""
        nodes = {e.id: {"entry": e, "children": []} for e in self._entries}
        roots = []
        for e in self._entries:
            node = nodes[e.id]
            if e.parent_id and e.parent_id in nodes:
                nodes[e.parent_id]["children"].append(node)
            else:
                roots.append(node)
        return roots

    def build_session_context(self, leaf_id: str | None = None) -> SessionContext:
        """Walk parentId chain from leaf, respecting compaction cutoffs."""
        tip = leaf_id if leaf_id is not None else self._leaf_id
        if not tip:
            return SessionContext(messages=[])

        chain = self.get_branch(tip)

        # Find compaction cutoff: last compaction entry's first_kept_entry_id
        cutoff_id = None
        for e in reversed(chain):
            if isinstance(e, CompactionEntry):
                cutoff_id = e.first_kept_entry_id
                break

        if cutoff_id:
            cutoff_idx = next((i for i, e in enumerate(chain) if e.id == cutoff_id), 0)
            chain = chain[cutoff_idx:]

        thinking_level = "off"
        model = None
        messages = []
        for e in chain:
            if isinstance(e, ThinkingLevelChangeEntry):
                thinking_level = e.thinking_level
            elif isinstance(e, ModelChangeEntry):
                model = {"provider": e.provider, "model_id": e.model_id}
            elif isinstance(e, SessionMessageEntry) and e.message:
                messages.append(e.message)
            elif isinstance(e, CustomMessageEntry) and e.display:
                messages.append(e.content)

        return SessionContext(messages=messages, thinking_level=thinking_level, model=model)

    # ── Mutations (append to JSONL) ───────────────────────────────────────────

    def _append(self, entry) -> None:
        self._entries.append(entry)
        self._entry_map[entry.id] = entry
        self._leaf_id = entry.id
        self._pending_writes.append(json.dumps(_entry_to_dict(entry)))

    def _new_id(self) -> str:
        return uuid.uuid4().hex[:8]

    def _new_entry_base(self) -> dict:
        return {"id": self._new_id(), "parent_id": self._leaf_id, "timestamp": _now()}

    def append_message_entry(self, message) -> SessionMessageEntry:
        e = SessionMessageEntry(**self._new_entry_base(), message=message)
        self._append(e)
        return e

    def append_model_change(self, provider: str, model_id: str) -> ModelChangeEntry:
        e = ModelChangeEntry(**self._new_entry_base(), provider=provider, model_id=model_id)
        self._append(e)
        return e

    def append_thinking_level_change(self, level: str) -> ThinkingLevelChangeEntry:
        e = ThinkingLevelChangeEntry(**self._new_entry_base(), thinking_level=level)
        self._append(e)
        return e

    def append_compaction(
        self,
        summary: str,
        first_kept_entry_id: str,
        tokens_before: int,
        details=None,
        from_hook: bool = False,
    ) -> CompactionEntry:
        e = CompactionEntry(
            **self._new_entry_base(),
            summary=summary,
            first_kept_entry_id=first_kept_entry_id,
            tokens_before=tokens_before,
            details=details,
            from_hook=from_hook,
        )
        self._append(e)
        return e

    def append_branch_summary(
        self,
        from_id: str,
        summary: str,
        details=None,
        from_hook: bool = False,
    ) -> BranchSummaryEntry:
        e = BranchSummaryEntry(
            **self._new_entry_base(),
            from_id=from_id,
            summary=summary,
            details=details,
            from_hook=from_hook,
        )
        self._append(e)
        return e

    def append_custom_entry(self, custom_type: str, data=None) -> CustomEntry:
        e = CustomEntry(**self._new_entry_base(), custom_type=custom_type, data=data)
        self._append(e)
        return e

    def append_custom_message_entry(
        self,
        custom_type: str,
        content,
        display: bool = True,
        details=None,
    ) -> CustomMessageEntry:
        e = CustomMessageEntry(
            **self._new_entry_base(),
            custom_type=custom_type,
            content=content,
            display=display,
            details=details,
        )
        self._append(e)
        return e

    def append_label_change(self, target_id: str, label: str | None = None) -> LabelEntry:
        e = LabelEntry(**self._new_entry_base(), target_id=target_id, label=label)
        self._append(e)
        return e

    def append_session_info(self, name: str | None) -> SessionInfoEntry:
        e = SessionInfoEntry(**self._new_entry_base(), name=name)
        self._append(e)
        return e

    def reset_leaf(self) -> None:
        """Move leaf pointer to None (navigating to root)."""
        self._leaf_id = None

    def update_entry(self, entry) -> None:
        """Replace an existing entry in memory."""
        self._entry_map[entry.id] = entry
        for i, e in enumerate(self._entries):
            if e.id == entry.id:
                self._entries[i] = entry
                break

    # ── Persistence ───────────────────────────────────────────────────────────

    async def flush(self) -> None:
        if not self._pending_writes and not self._file_path:
            return
        if not self._file_path:
            self._file_path = _make_session_path(self._session_dir, self._header.id)
            Path(self._session_dir).mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(self._file_path, "a") as f:
            for line in self._pending_writes:
                await f.write(line + "\n")
        self._pending_writes.clear()

    # ── Branching ─────────────────────────────────────────────────────────────

    def create_branched_session(self, parent_id: str) -> str | None:
        """Write a new JSONL with only the branch from root to parent_id."""
        branch = self.get_branch(parent_id)
        if not branch:
            return None
        new_id = str(uuid.uuid4())
        new_path = _make_session_path(self._session_dir, new_id)
        Path(self._session_dir).mkdir(parents=True, exist_ok=True)
        header = SessionHeader(
            id=new_id,
            timestamp=_now(),
            cwd=self._cwd,
            parent_session=self._header.id,
        )
        lines = [json.dumps(_header_to_dict(header))]
        for e in branch:
            lines.append(json.dumps(_entry_to_dict(e)))
        Path(new_path).write_text("\n".join(lines) + "\n")
        return new_path

    def new_session(self, parent_session: str | None = None) -> None:
        self._entries.clear()
        self._entry_map.clear()
        self._leaf_id = None
        self._file_path = None
        self._header = SessionHeader(
            id=str(uuid.uuid4()),
            timestamp=_now(),
            cwd=self._cwd,
            parent_session=parent_session,
        )
        self._pending_writes = [json.dumps(_header_to_dict(self._header))]

    @classmethod
    def fork_from(
        cls,
        source_path: str,
        target_cwd: str,
        session_dir: str | None = None,
    ) -> "SessionManager":
        """Open a session from another project and re-anchor to target_cwd."""
        return cls.open(source_path, session_dir, cwd_override=target_cwd)

    @staticmethod
    def list_all(session_dir: str, on_progress=None) -> list[str]:
        """Return paths of all .jsonl files in session_dir, newest first."""
        p = Path(session_dir)
        if not p.exists():
            return []
        files = sorted(p.glob("*.jsonl"), key=lambda f: f.stat().st_mtime, reverse=True)
        paths = [str(f) for f in files]
        if on_progress:
            for path in paths:
                on_progress(path)
        return paths
