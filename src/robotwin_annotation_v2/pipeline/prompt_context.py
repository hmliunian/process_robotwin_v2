"""Shared, mode-aware values exposed to configurable Qwen prompts.

Prompt rendering is deliberately the only semantic stage that knows the
concrete timeline event types.  The prompt files may therefore describe the
real task timeline without teaching the Qwen and mask-QC stages about each
other's business rules or inventing release events for close-and-hold tasks.
"""

from __future__ import annotations

from ..models import LoopContext, PickPlaceEvents, TargetOnlyEvents


def timeline_prompt_fields(context: LoopContext) -> dict[str, str]:
    """Return the prompt fields that exist for this episode's timeline.

    A template that references an event from the wrong mode remains invalid:
    callers compare its placeholders with this mapping and fail before making
    a model request.
    """

    events = context.events
    common = {
        "active_arm": events.active_arm,
        "episode_end": str(context.frame_count - 1),
    }
    if isinstance(events, TargetOnlyEvents):
        return {
            **common,
            "remove_start": str(events.t_remove_start),
            "close_start": str(events.t_close_start),
            "close_end": str(events.t_close_end),
        }
    if isinstance(events, PickPlaceEvents):
        return {
            **common,
            "move_start": str(events.t_move_start),
            "close_start": str(events.t_close_start),
            "close_done": str(events.t_close_done),
            "open_start": str(events.t_open_start),
            "open_done": str(events.t_open_done),
        }
    raise TypeError(f"unsupported timeline event type: {type(events).__name__}")


__all__ = ["timeline_prompt_fields"]
