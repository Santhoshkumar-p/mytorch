from __future__ import annotations
import logging
from ..types import CompactionSettings, SessionMessageEntry, CompactionEntry

logger = logging.getLogger(__name__)


def should_compact(context_tokens: int, context_window: int, settings: CompactionSettings) -> bool:
    return context_tokens > context_window - settings.reserve_tokens


def estimate_tokens(message) -> int:
    """Heuristic: len(repr(message)) // 4."""
    return len(repr(message)) // 4


def find_cut_point(entries: list, keep_recent_tokens: int) -> str | None:
    """Walk backward accumulating token estimates. Return id of the entry where we've
    accumulated >= keep_recent_tokens. Ensures we never cut between tool call and result.
    Always returns entries[0].id if nothing found (keep everything).
    """
    if not entries:
        return None

    accumulated = 0
    for entry in reversed(entries):
        if isinstance(entry, SessionMessageEntry) and entry.message is not None:
            accumulated += estimate_tokens(entry.message)
        if accumulated >= keep_recent_tokens:
            return entry.id

    # Default: keep everything from the first entry
    return entries[0].id


async def run_compaction(
    session,
    settings: CompactionSettings,
    custom_instructions: str | None = None,
) -> None:
    """Run full compaction cycle on a session.

    1. Find cut point in current entries
    2. Serialize messages before cut for summarization
    3. Call _one_shot_summarize(session, prompt)
    4. Append CompactionEntry to session_manager
    5. Rebuild agent context
    """
    session_manager = session.session_manager
    entries = list(session_manager._entries)

    if not entries:
        return

    # Estimate current token count
    context_usage = getattr(session, "context_usage", None)
    context_tokens = getattr(context_usage, "input_tokens", 0) if context_usage else 0

    cut_id = find_cut_point(entries, settings.keep_recent_tokens)
    if cut_id is None:
        return

    # Collect entries before the cut point for summarization
    cut_idx = next((i for i, e in enumerate(entries) if e.id == cut_id), 0)
    entries_to_summarize = entries[:cut_idx]

    from .utils import serialize_conversation
    messages_to_summarize = [
        e.message for e in entries_to_summarize
        if isinstance(e, SessionMessageEntry) and e.message is not None
    ]

    serialized = serialize_conversation(messages_to_summarize)

    instructions = ""
    if custom_instructions:
        instructions = f"\n\nAdditional instructions: {custom_instructions}"

    summary_prompt = (
        "Please provide a concise summary of the following conversation. "
        "Focus on what was accomplished, key decisions made, files modified, "
        "and any important context needed to continue the work."
        f"{instructions}\n\n"
        f"<conversation>\n{serialized}\n</conversation>"
    )

    summary = await _one_shot_summarize(session, summary_prompt)

    session_manager.append_compaction(
        summary=summary,
        first_kept_entry_id=cut_id,
        tokens_before=context_tokens,
        details=None,
        from_hook=False,
    )

    # Rebuild agent context if the session supports it
    if hasattr(session, "rebuild_context"):
        await session.rebuild_context()


async def _one_shot_summarize(session, prompt: str) -> str:
    """Call the session's agent for a single summarization turn (no tools).
    Returns text of the assistant response.
    """
    try:
        agent = getattr(session, "agent", None)
        if agent is None:
            return "[Summary unavailable: no agent attached to session]"

        # Try to use a stream_llm or similar method with no tools
        stream_llm = getattr(agent, "stream_llm", None)
        if stream_llm is None:
            return "[Summary unavailable: agent has no stream_llm method]"

        # Build minimal context
        try:
            from pi_agent.types import UserMessage, TextContent  # type: ignore
            user_msg = UserMessage(content=[TextContent(type="text", text=prompt)])
        except ImportError:
            # Fallback: use dicts
            user_msg = {"role": "user", "content": prompt}

        text_parts: list[str] = []
        async for event in stream_llm(messages=[user_msg], tools=[]):
            event_type = getattr(event, "type", None) or type(event).__name__
            if event_type in ("StreamDoneEvent", "stream_done"):
                final_text = getattr(event, "text", None)
                if final_text:
                    return final_text
            elif event_type in ("TextEvent", "text"):
                text_parts.append(getattr(event, "text", ""))

        return "".join(text_parts) or "[Empty summary]"

    except Exception as exc:
        logger.warning("Compaction summarization failed: %s", exc)
        return f"[Summary unavailable: {exc}]"
