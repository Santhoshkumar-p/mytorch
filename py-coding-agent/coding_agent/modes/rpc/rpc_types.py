from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


# ── Inbound command types ─────────────────────────────────────────────────────

@dataclass
class PromptCmd:
    type: str = "prompt"
    message: str = ""
    images: list = field(default_factory=list)
    streaming_behavior: str | None = None
    id: str | None = None


@dataclass
class SteerCmd:
    type: str = "steer"
    message: str = ""
    images: list = field(default_factory=list)
    id: str | None = None


@dataclass
class FollowUpCmd:
    type: str = "follow_up"
    message: str = ""
    images: list = field(default_factory=list)
    id: str | None = None


@dataclass
class AbortCmd:
    type: str = "abort"
    id: str | None = None


@dataclass
class AbortRetryCmd:
    type: str = "abort_retry"
    id: str | None = None


@dataclass
class NewSessionCmd:
    type: str = "new_session"
    parent_session: str | None = None
    id: str | None = None


@dataclass
class GetStateCmd:
    type: str = "get_state"
    id: str | None = None


@dataclass
class SetModelCmd:
    type: str = "set_model"
    provider: str = ""
    model_id: str = ""
    id: str | None = None


@dataclass
class CycleModelCmd:
    type: str = "cycle_model"
    id: str | None = None


@dataclass
class GetAvailableModelsCmd:
    type: str = "get_available_models"
    id: str | None = None


@dataclass
class SetThinkingLevelCmd:
    type: str = "set_thinking_level"
    level: str = "off"
    id: str | None = None


@dataclass
class CycleThinkingLevelCmd:
    type: str = "cycle_thinking_level"
    id: str | None = None


@dataclass
class SetSteeringModeCmd:
    type: str = "set_steering_mode"
    mode: str = "all"
    id: str | None = None


@dataclass
class SetFollowUpModeCmd:
    type: str = "set_follow_up_mode"
    mode: str = "all"
    id: str | None = None


@dataclass
class CompactCmd:
    type: str = "compact"
    custom_instructions: str | None = None
    id: str | None = None


@dataclass
class SetAutoCompactionCmd:
    type: str = "set_auto_compaction"
    enabled: bool = True
    id: str | None = None


@dataclass
class SetAutoRetryCmd:
    type: str = "set_auto_retry"
    enabled: bool = True
    id: str | None = None


@dataclass
class BashCmd:
    type: str = "bash"
    command: str = ""
    id: str | None = None


@dataclass
class AbortBashCmd:
    type: str = "abort_bash"
    id: str | None = None


@dataclass
class GetSessionStatsCmd:
    type: str = "get_session_stats"
    id: str | None = None


@dataclass
class ExportHtmlCmd:
    type: str = "export_html"
    output_path: str | None = None
    id: str | None = None


@dataclass
class SwitchSessionCmd:
    type: str = "switch_session"
    session_path: str = ""
    id: str | None = None


@dataclass
class ForkCmd:
    type: str = "fork"
    entry_id: str = ""
    id: str | None = None


@dataclass
class NavigateTreeCmd:
    type: str = "navigate_tree"
    target_id: str = ""
    id: str | None = None


@dataclass
class ReloadCmd:
    type: str = "reload"
    id: str | None = None


@dataclass
class GetForkMessagesCmd:
    type: str = "get_fork_messages"
    id: str | None = None


@dataclass
class GetLastAssistantTextCmd:
    type: str = "get_last_assistant_text"
    id: str | None = None


@dataclass
class SetSessionNameCmd:
    type: str = "set_session_name"
    name: str = ""
    id: str | None = None


@dataclass
class GetMessagesCmd:
    type: str = "get_messages"
    id: str | None = None


@dataclass
class GetCommandsCmd:
    type: str = "get_commands"
    id: str | None = None


@dataclass
class ImportCmd:
    type: str = "import"
    path: str = ""
    id: str | None = None


CMD_TYPES: dict[str, type] = {
    "prompt": PromptCmd,
    "steer": SteerCmd,
    "follow_up": FollowUpCmd,
    "abort": AbortCmd,
    "abort_retry": AbortRetryCmd,
    "new_session": NewSessionCmd,
    "get_state": GetStateCmd,
    "set_model": SetModelCmd,
    "cycle_model": CycleModelCmd,
    "get_available_models": GetAvailableModelsCmd,
    "set_thinking_level": SetThinkingLevelCmd,
    "cycle_thinking_level": CycleThinkingLevelCmd,
    "set_steering_mode": SetSteeringModeCmd,
    "set_follow_up_mode": SetFollowUpModeCmd,
    "compact": CompactCmd,
    "set_auto_compaction": SetAutoCompactionCmd,
    "set_auto_retry": SetAutoRetryCmd,
    "bash": BashCmd,
    "abort_bash": AbortBashCmd,
    "get_session_stats": GetSessionStatsCmd,
    "export_html": ExportHtmlCmd,
    "switch_session": SwitchSessionCmd,
    "fork": ForkCmd,
    "navigate_tree": NavigateTreeCmd,
    "reload": ReloadCmd,
    "get_fork_messages": GetForkMessagesCmd,
    "get_last_assistant_text": GetLastAssistantTextCmd,
    "set_session_name": SetSessionNameCmd,
    "get_messages": GetMessagesCmd,
    "get_commands": GetCommandsCmd,
    "import": ImportCmd,
}

# ── Outbound types ────────────────────────────────────────────────────────────

@dataclass
class RpcSuccess:
    type: str = "success"
    id: str | None = None
    data: Any = None


@dataclass
class RpcError:
    type: str = "error"
    id: str | None = None
    error: str = ""


@dataclass
class RpcEvent:
    type: str = "event"
    event: str = ""
    data: Any = None


@dataclass
class ExtensionUIRequest:
    type: str = "extension_ui_request"
    request_id: str = ""
    request_type: str = ""     # "select" | "confirm" | "input" | "notify" | "editor"
    title: str = ""
    options: list = field(default_factory=list)
    message: str = ""
    placeholder: str = ""
    timeout: int | None = None


@dataclass
class ExtensionError:
    type: str = "extension_error"
    error: str = ""
    extension: str = ""


@dataclass
class RpcSessionState:
    model: dict | None = None
    thinking_level: str = "off"
    is_streaming: bool = False
    is_compacting: bool = False
    steering_mode: str = "all"
    follow_up_mode: str = "all"
    session_file: str | None = None
    session_id: str = ""
    session_name: str | None = None
    auto_compaction_enabled: bool = True
    message_count: int = 0
    pending_message_count: int = 0
