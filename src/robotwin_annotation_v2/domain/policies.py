"""Domain policies for keyframe selection.

Each role (target, receiver, gripper) has different rules for selecting keyframes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

from .models import (
    AnchorKind,
    EpisodeRef,
    FrameWindow,
    InstanceSlot,
    KeyframeRequest,
)


@dataclass(frozen=True)
class InteractionTimeline:
    """Action boundaries detected from state/gripper signals."""
    episode: EpisodeRef

    # Movement phases
    move_start: int | None = None
    move_end: int | None = None

    # Gripper events
    close_start: int | None = None
    close_end: int | None = None
    open_start: int | None = None
    open_end: int | None = None

    # Hold intervals (from process_data confirmed_hold)
    hold_start: int | None = None
    hold_end: int | None = None


@dataclass(frozen=True)
class SemanticPlan:
    """Text queries and role assignments from task understanding."""
    episode: EpisodeRef
    target_query: str
    receiver_query: str | None = None
    has_static_receiver: bool = False


class KeyframePolicy(Protocol):
    """Protocol for role-specific keyframe selection strategy."""

    def create_requests(
        self,
        semantic: SemanticPlan,
        timeline: InteractionTimeline,
    ) -> list[KeyframeRequest]:
        """Generate keyframe requests for this role."""
        ...


class TargetSeedPolicy:
    """Target object: find pre-grasp visible frame."""

    def create_requests(
        self,
        semantic: SemanticPlan,
        timeline: InteractionTimeline,
    ) -> list[KeyframeRequest]:
        """Target seed window: [move_start, close_start)."""

        if timeline.move_start is None or timeline.close_start is None:
            return []

        # Target should be visible before gripper closes
        window = FrameWindow(
            first=timeline.move_start,
            last=timeline.close_start - 1,
        )

        slot = InstanceSlot(name="target_0", role="target")

        request = KeyframeRequest(
            request_id=f"{semantic.episode.episode_id}_{slot.name}_r001",
            episode=semantic.episode,
            slot=slot,
            anchor_kind=AnchorKind.PRE_GRASP_VISIBLE,
            allowed_window=window,
            visual_query=semantic.target_query,
            exclusions=("gripper_left", "gripper_right"),
            revision=1,
        )

        return [request]


class StaticReceiverSeedPolicy:
    """Static receiver: find unoccluded frame (can differ from target frame)."""

    def create_requests(
        self,
        semantic: SemanticPlan,
        timeline: InteractionTimeline,
    ) -> list[KeyframeRequest]:
        """Receiver seed window: prefer before action, when unoccluded."""

        if not semantic.has_static_receiver or semantic.receiver_query is None:
            return []

        if timeline.move_start is None:
            return []

        # Prefer early frames when receiver is unoccluded
        # Can be extended to entire episode if needed
        window = FrameWindow(
            first=0,
            last=max(timeline.move_start, 0),
        )

        slot = InstanceSlot(name="receiver_0", role="receiver")

        request = KeyframeRequest(
            request_id=f"{semantic.episode.episode_id}_{slot.name}_r001",
            episode=semantic.episode,
            slot=slot,
            anchor_kind=AnchorKind.STATIC_RECEIVER_VISIBLE,
            allowed_window=window,
            visual_query=semantic.receiver_query,
            exclusions=("target_0", "gripper_left", "gripper_right"),
            revision=1,
        )

        return [request]


class RolePolicyRegistry:
    """Registry mapping roles to their keyframe policies."""

    def __init__(self) -> None:
        self._policies: dict[str, KeyframePolicy] = {
            "target": TargetSeedPolicy(),
            "receiver": StaticReceiverSeedPolicy(),
        }

    def get_requests(
        self,
        semantic: SemanticPlan,
        timeline: InteractionTimeline,
    ) -> list[KeyframeRequest]:
        """Generate all keyframe requests for this episode."""
        requests: list[KeyframeRequest] = []

        for policy in self._policies.values():
            requests.extend(policy.create_requests(semantic, timeline))

        return requests
