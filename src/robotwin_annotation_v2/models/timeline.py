"""Typed timeline events and derived output windows.

Pick/place and target-only episodes share one grasp prefix, but they do not
share the same complete state machine.  In particular, a target-only
close-and-hold episode has no release event.  Keeping two explicit event
types prevents downstream code from inventing fake ``open`` boundaries.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


@dataclass(frozen=True)
class FrameWindow:
    """Inclusive frame window ``[start, end]``."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"invalid frame window [{self.start}, {self.end}]")

    def __contains__(self, frame_id: int) -> bool:
        return self.start <= frame_id <= self.end

    def __len__(self) -> int:
        return self.end - self.start + 1

    def to_json(self) -> list[int]:
        return [self.start, self.end]


@dataclass(frozen=True)
class PickPlaceEvents:
    """Five ordered state-derived boundaries for one pick/place loop."""

    active_arm: Literal["left", "right"]
    t_move_start: int
    t_close_start: int
    t_close_done: int
    t_open_start: int
    t_open_done: int

    def __post_init__(self) -> None:
        values = (
            self.t_move_start,
            self.t_close_start,
            self.t_close_done,
            self.t_open_start,
            self.t_open_done,
        )
        if min(values) < 0:
            raise ValueError("loop event frames must be non-negative")
        if not (
            self.t_move_start <= self.t_close_start
            < self.t_close_done
            < self.t_open_start
            < self.t_open_done
        ):
            raise ValueError(f"loop events are not ordered: {values}")

    @property
    def loop_window(self) -> FrameWindow:
        return FrameWindow(self.t_move_start, self.t_open_done)

    @property
    def target_window(self) -> FrameWindow:
        return FrameWindow(self.t_move_start, self.t_close_done)

    @property
    def receiver_window(self) -> FrameWindow:
        return FrameWindow(self.t_close_done, self.t_open_done)

    def to_json(self) -> dict[str, object]:
        return asdict(self)


# Compatibility name retained while callers migrate to the explicit type.
LoopEvents = PickPlaceEvents


@dataclass(frozen=True)
class TargetOnlyEvents:
    """Three boundaries for an approach, close, and hold operation.

    ``t_remove_start`` means the start of the robot's removal operation (the
    first stable active-arm motion), not the later instant at which the object
    physically leaves its support surface.
    """

    active_arm: Literal["left", "right"]
    t_remove_start: int
    t_close_start: int
    t_close_end: int

    def __post_init__(self) -> None:
        values = (self.t_remove_start, self.t_close_start, self.t_close_end)
        if min(values) < 0:
            raise ValueError("target-only event frames must be non-negative")
        if not self.t_remove_start <= self.t_close_start < self.t_close_end:
            raise ValueError(f"target-only events are not ordered: {values}")

    @property
    def target_window(self) -> FrameWindow:
        return FrameWindow(self.t_remove_start, self.t_close_end)

    def operation_window(self, frame_count: int) -> FrameWindow:
        if frame_count <= self.t_close_end:
            raise ValueError("target-only close event extends beyond the episode")
        return FrameWindow(self.t_remove_start, frame_count - 1)

    def to_json(self) -> dict[str, object]:
        return asdict(self)


type TimelineEvents = PickPlaceEvents | TargetOnlyEvents


@dataclass(frozen=True)
class EpisodeWindows:
    """All windows consumed by model, gripper, and publication stages."""

    operation: FrameWindow
    target: FrameWindow
    receiver: FrameWindow | None
    gripper: FrameWindow

    def to_json(self) -> dict[str, list[int] | None]:
        return {
            "operation": self.operation.to_json(),
            "target_0": self.target.to_json(),
            "receiver_0": None if self.receiver is None else self.receiver.to_json(),
            "gripper": self.gripper.to_json(),
        }


def derive_episode_windows(events: TimelineEvents, *, frame_count: int) -> EpisodeWindows:
    """Derive all downstream windows in the one timeline-aware boundary."""

    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    if isinstance(events, TargetOnlyEvents):
        operation = events.operation_window(frame_count)
        return EpisodeWindows(
            operation=operation,
            target=events.target_window,
            receiver=None,
            gripper=operation,
        )
    if events.t_open_done >= frame_count:
        raise ValueError("pick/place loop extends beyond the episode")
    return EpisodeWindows(
        operation=events.loop_window,
        target=events.target_window,
        receiver=events.receiver_window,
        gripper=events.loop_window,
    )


__all__ = [
    "EpisodeWindows",
    "FrameWindow",
    "LoopEvents",
    "PickPlaceEvents",
    "TargetOnlyEvents",
    "TimelineEvents",
    "derive_episode_windows",
]
