from __future__ import annotations
import logging
from dataclasses import dataclass
from ..types import BranchSummarySettings, FileOperations, SessionMessageEntry
from .utils import extract_file_ops_from_message, compute_file_lists, serialize_conversation

logger = logging.getLogger(__name__)


@dataclass
class BranchSummaryDetails:
    read_files: list[str]
    modified_files: list[str]


async def generate_branch_summary(
    agent,
    entries: list,
    settings: BranchSummarySettings,
    custom_instructions: str | None = None,
) -> tuple[str, BranchSummaryDetails]:
    """Summarize an abandoned branch. Returns (summary_text, details)."""
    # Collect file operations from all assistant messages in the branch
    combined_ops = FileOperations()
    messages = []
    for entry in entries:
        if isinstance(entry, SessionMessageEntry) and entry.message is not None:
            msg = entry.message
            role = getattr(msg, "role", None)
            if role == "assistant":
                ops = extract_file_ops_from_message(msg)
                combined_ops.read.update(ops.read)
                combined_ops.written.update(ops.written)
                combined_ops.edited.update(ops.edited)
            messages.append(msg)

    file_lists = compute_file_lists(combined_ops)
    details = BranchSummaryDetails(
        read_files=file_lists["read"],
        modified_files=file_lists["modified"],
    )

    if settings.skip_prompt:
        return "[Branch summary skipped]", details

    serialized = serialize_conversation(messages)

    instructions = ""
    if custom_instructions:
        instructions = f"\n\nAdditional instructions: {custom_instructions}"

    summary_prompt = (
        "The following conversation represents an abandoned branch of work. "
        "Please provide a concise summary of what was attempted, what was accomplished, "
        "and why it may have been abandoned. Focus on key decisions and any important context."
        f"{instructions}\n\n"
        f"<conversation>\n{serialized}\n</conversation>"
    )

    try:
        stream_llm = getattr(agent, "stream_llm", None)
        if stream_llm is None:
            return "[Branch summary unavailable: no stream_llm]", details

        text_parts: list[str] = []
        async for event in stream_llm(messages=[], tools=[], system=summary_prompt):
            event_type = getattr(event, "type", None) or type(event).__name__
            if event_type in ("StreamDoneEvent", "stream_done"):
                final_text = getattr(event, "text", None)
                if final_text:
                    return final_text, details
            elif event_type in ("TextEvent", "text"):
                text_parts.append(getattr(event, "text", ""))

        summary = "".join(text_parts) or "[Empty branch summary]"
        return summary, details

    except Exception as exc:
        logger.warning("Branch summary generation failed: %s", exc)
        return f"[Branch summary unavailable: {exc}]", details


def collect_entries_for_branch_summary(session_manager, from_id: str, to_id: str) -> list:
    """Return entries present in from_id's branch but not in to_id's branch."""
    from_set = {e.id for e in session_manager.get_branch(from_id)}
    to_set = {e.id for e in session_manager.get_branch(to_id)}
    abandoned = from_set - to_set
    return [e for e in session_manager._entries if e.id in abandoned]
