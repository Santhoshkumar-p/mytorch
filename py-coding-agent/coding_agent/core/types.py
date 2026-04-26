from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable

# ── Session JSONL entry types ─────────────────────────────────────────────────

@dataclass
class SessionHeader:
    id: str
    timestamp: str
    cwd: str
    version: int = 3
    parent_session: str | None = None

@dataclass
class SessionEntryBase:
    id: str
    parent_id: str | None
    timestamp: str

@dataclass
class SessionMessageEntry(SessionEntryBase):
    type: str = "message"
    message: Any = None

@dataclass
class ModelChangeEntry(SessionEntryBase):
    type: str = "model_change"
    provider: str = ""
    model_id: str = ""

@dataclass
class ThinkingLevelChangeEntry(SessionEntryBase):
    type: str = "thinking_level_change"
    thinking_level: str = "off"

@dataclass
class CompactionEntry(SessionEntryBase):
    type: str = "compaction"
    summary: str = ""
    first_kept_entry_id: str = ""
    tokens_before: int = 0
    details: Any = None
    from_hook: bool = False

@dataclass
class BranchSummaryEntry(SessionEntryBase):
    type: str = "branch_summary"
    from_id: str = ""
    summary: str = ""
    details: Any = None
    from_hook: bool = False

@dataclass
class CustomEntry(SessionEntryBase):
    type: str = "custom"
    custom_type: str = ""
    data: Any = None

@dataclass
class CustomMessageEntry(SessionEntryBase):
    type: str = "custom_message"
    custom_type: str = ""
    content: str | list = ""
    details: Any = None
    display: bool = True

@dataclass
class LabelEntry(SessionEntryBase):
    type: str = "label"
    target_id: str = ""
    label: str | None = None

@dataclass
class SessionInfoEntry(SessionEntryBase):
    type: str = "session_info"
    name: str | None = None

# ── Agent state ───────────────────────────────────────────────────────────────

@dataclass
class AgentState:
    messages: list = field(default_factory=list)
    streaming_message: Any = None
    pending_tool_calls: frozenset = field(default_factory=frozenset)
    error_message: str | None = None
    context_usage: Any = None  # ContextUsage from agent library

# ── Settings ──────────────────────────────────────────────────────────────────

@dataclass
class RetrySettings:
    enabled: bool = True
    max_retries: int = 3
    base_delay_ms: int = 2000
    max_delay_ms: int = 60000

@dataclass
class CompactionSettings:
    enabled: bool = True
    reserve_tokens: int = 16384
    keep_recent_tokens: int = 20000

@dataclass
class BranchSummarySettings:
    reserve_tokens: int = 16384
    skip_prompt: bool = False

@dataclass
class Settings:
    # Model
    default_provider: str | None = None
    default_model: str | None = None
    default_thinking_level: str = "off"
    enabled_models: list[str] = field(default_factory=list)
    # Network
    transport: str | None = None  # "sse" | "websocket"
    # Queue behavior
    steering_mode: str = "all"
    follow_up_mode: str = "all"
    # Shell
    shell_path: str | None = None
    shell_command_prefix: str | None = None
    # Storage
    session_dir: str | None = None
    # Resources
    extensions: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    prompts: list[str] = field(default_factory=list)
    themes: list[str] = field(default_factory=list)
    # Compaction / retry / branch summary
    compaction: CompactionSettings = field(default_factory=CompactionSettings)
    branch_summary: BranchSummarySettings = field(default_factory=BranchSummarySettings)
    retry: RetrySettings = field(default_factory=RetrySettings)
    # Images
    image_auto_resize: bool = True
    block_images: bool = False
    show_images: bool = True
    # Thinking
    thinking_budgets: dict[str, int] = field(default_factory=dict)
    # UI prefs
    hide_thinking_block: bool = False
    editor_padding_x: int = 2
    autocomplete_max_visible: int = 8
    show_hardware_cursor: bool = False
    double_escape_action: str = "fork"  # "fork" | "tree" | "none"
    tree_filter_mode: str = "default"
    # Feature flags
    enable_skill_commands: bool = True
    quiet_startup: bool = False
    collapse_changelog: bool = False
    enable_install_telemetry: bool = True
    code_block_indent: str = "  "
    # Meta
    last_changelog_version: str | None = None

# ── Session context & stats ───────────────────────────────────────────────────

@dataclass
class SessionContext:
    messages: list
    thinking_level: str = "off"
    model: dict | None = None  # {"provider": str, "model_id": str}

@dataclass
class SessionStats:
    session_file: str | None
    session_id: str
    user_messages: int = 0
    assistant_messages: int = 0
    tool_calls: int = 0
    tool_results: int = 0
    total_messages: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    total_tokens: int = 0
    context_usage: Any = None

# ── Prompt options ────────────────────────────────────────────────────────────

@dataclass
class PromptOptions:
    images: list = field(default_factory=list)
    expand_prompt_templates: bool = True
    streaming_behavior: str | None = None  # "steer" | "followUp"
    source: str | None = None  # "rpc" | "sdk" | "cli"

# ── Tool results ──────────────────────────────────────────────────────────────

@dataclass
class TruncationResult:
    truncated_by: str | None  # "lines" | "bytes" | None
    total_lines: int = 0
    output_lines: int = 0
    output_bytes: int = 0
    first_line_exceeds_limit: bool = False
    last_line_partial: bool = False

@dataclass
class BashResult:
    output: str
    exit_code: int | None
    cancelled: bool = False
    truncated: bool = False
    full_output_path: str | None = None
    truncation: TruncationResult | None = None

# ── Skills / prompts / context files ──────────────────────────────────────────

@dataclass
class Skill:
    name: str
    path: str
    content: str
    description: str
    base_dir: str
    source_info: dict
    commands: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    disable_model_invocation: bool = False

@dataclass
class PromptTemplate:
    name: str
    description: str
    content: str
    file_path: str
    source_info: dict
    argument_hint: str | None = None

@dataclass
class ContextFile:
    path: str
    content: str

@dataclass
class SlashCommand:
    name: str
    description: str | None
    source: str  # "extension" | "prompt" | "skill"
    source_info: dict
    handler: Callable | None = None

# ── File operations (for compaction tracking) ─────────────────────────────────

@dataclass
class FileOperations:
    read: set[str] = field(default_factory=set)
    written: set[str] = field(default_factory=set)
    edited: set[str] = field(default_factory=set)
