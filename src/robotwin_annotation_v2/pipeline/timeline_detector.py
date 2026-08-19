"""Pure state-signal detection for pick/place and close/hold timelines."""

from __future__ import annotations

from typing import Any, Protocol, cast

import numpy as np

from ..models.timeline import PickPlaceEvents, TargetOnlyEvents

NDArray = np.ndarray[Any, Any]

OPEN_THRESHOLD = 0.9
CLOSED_THRESHOLD = 0.15


class EpisodeStateSignals(Protocol):
    """Read-only episode state view required by the timeline detector."""

    @property
    def gripper_states(self) -> NDArray: ...

    @property
    def eef_states(self) -> NDArray: ...


class StateLoopError(RuntimeError):
    """State signals do not satisfy the configured timeline contract."""


def _median_filter(values: NDArray) -> NDArray:
    padded = np.pad(values, (1, 1), mode="edge")
    return cast(
        NDArray,
        np.median(
            np.stack((padded[:-2], padded[1:-1], padded[2:])),
            axis=0,
        ),
    )


def _first_run(condition: NDArray, *, start: int, length: int) -> int | None:
    for frame_id in range(max(0, start), len(condition) - length + 1):
        if bool(condition[frame_id : frame_id + length].all()):
            return frame_id
    return None


def _close_transition(
    gripper_values: NDArray,
    *,
    stable_frames: int,
) -> tuple[NDArray, int, int]:
    """Return filtered gripper state and one stable close boundary."""

    gripper = np.asarray(gripper_values, dtype=np.float64)
    if stable_frames < 1:
        raise ValueError("stable_frames must be positive")
    if gripper.ndim != 1 or gripper.size < stable_frames * 2:
        raise StateLoopError("gripper_values must be a sufficiently long 1-D array")

    filtered = _median_filter(gripper)
    close_candidates = (
        np.flatnonzero((filtered[1:] < OPEN_THRESHOLD) & (filtered[1:] < filtered[:-1])) + 1
    )
    if close_candidates.size == 0:
        raise StateLoopError("no close transition detected")
    close_start = int(close_candidates[0])
    close_run = _first_run(
        filtered <= CLOSED_THRESHOLD,
        start=close_start,
        length=stable_frames,
    )
    if close_run is None:
        raise StateLoopError("no stable closed transition detected")
    return filtered, close_start, close_run + stable_frames - 1


def _motion_start(
    eef_values: NDArray,
    *,
    close_start: int,
    stable_frames: int,
    motion_floor: float,
    rotation_scale: float,
) -> int:
    """Detect the first stable active-arm motion before gripper closure."""

    eef = np.asarray(eef_values, dtype=np.float64)
    if eef.ndim != 2 or eef.shape[1:] != (6,):
        raise StateLoopError("eef_values must have shape [T,6]")
    delta = np.diff(eef, axis=0)
    delta[:, 3:] = (delta[:, 3:] + np.pi) % (2 * np.pi) - np.pi
    score = np.linalg.norm(delta[:, :3], axis=1)
    score += rotation_scale * np.linalg.norm(delta[:, 3:], axis=1)
    baseline = score[: max(1, min(close_start, 3))]
    median = float(np.median(baseline))
    deviation = float(np.median(np.abs(baseline - median)))
    threshold = max(median + 4.0 * deviation, motion_floor)
    motion = np.concatenate(([False], score > threshold))
    start = _first_run(motion, start=1, length=stable_frames)
    return close_start if start is None or start >= close_start else start


def detect_loop_events(
    gripper_values: NDArray,
    eef_values: NDArray,
    *,
    arm: str,
    stable_frames: int = 3,
    motion_floor: float = 0.002,
    rotation_scale: float = 0.05,
) -> PickPlaceEvents:
    """Detect five ordered events for one arm from gripper and EEF state."""

    gripper = np.asarray(gripper_values, dtype=np.float64)
    eef = np.asarray(eef_values, dtype=np.float64)
    if eef.shape != (gripper.size, 6):
        raise StateLoopError(f"eef_values must have shape {(gripper.size, 6)}")
    filtered, close_start, close_done = _close_transition(
        gripper,
        stable_frames=stable_frames,
    )

    open_candidates = (
        np.flatnonzero((filtered[1:] > filtered[:-1]) & (np.arange(1, len(filtered)) > close_done))
        + 1
    )
    if open_candidates.size == 0:
        raise StateLoopError("no open transition detected")
    open_start = int(open_candidates[0])
    open_run = _first_run(
        filtered >= OPEN_THRESHOLD,
        start=open_start,
        length=stable_frames,
    )
    if open_run is None:
        raise StateLoopError("no stable reopen transition detected")
    open_done = open_run + stable_frames - 1

    move_start = _motion_start(
        eef,
        close_start=close_start,
        stable_frames=stable_frames,
        motion_floor=motion_floor,
        rotation_scale=rotation_scale,
    )

    try:
        return PickPlaceEvents(
            active_arm=arm,  # type: ignore[arg-type]
            t_move_start=move_start,
            t_close_start=close_start,
            t_close_done=close_done,
            t_open_start=open_start,
            t_open_done=open_done,
        )
    except ValueError as exc:
        raise StateLoopError(str(exc)) from exc


def detect_target_only_events(
    gripper_values: NDArray,
    eef_values: NDArray,
    *,
    arm: str,
    stable_frames: int = 3,
    motion_floor: float = 0.002,
    rotation_scale: float = 0.05,
) -> TargetOnlyEvents:
    """Detect one approach/close/hold operation without release events."""

    gripper = np.asarray(gripper_values, dtype=np.float64)
    eef = np.asarray(eef_values, dtype=np.float64)
    if eef.shape != (gripper.size, 6):
        raise StateLoopError(f"eef_values must have shape {(gripper.size, 6)}")
    filtered, close_start, close_end = _close_transition(
        gripper,
        stable_frames=stable_frames,
    )
    if not bool((filtered[:stable_frames] >= OPEN_THRESHOLD).all()):
        raise StateLoopError("target-only gripper must begin stably open")
    reopen_run = _first_run(
        filtered >= OPEN_THRESHOLD,
        start=close_end + 1,
        length=stable_frames,
    )
    if reopen_run is not None:
        raise StateLoopError("target-only gripper unexpectedly reopens")
    if not bool((filtered[-stable_frames:] <= CLOSED_THRESHOLD).all()):
        raise StateLoopError("target-only gripper must remain closed at episode end")

    remove_start = _motion_start(
        eef,
        close_start=close_start,
        stable_frames=stable_frames,
        motion_floor=motion_floor,
        rotation_scale=rotation_scale,
    )
    try:
        return TargetOnlyEvents(
            active_arm=arm,  # type: ignore[arg-type]
            t_remove_start=remove_start,
            t_close_start=close_start,
            t_close_end=close_end,
        )
    except ValueError as exc:
        raise StateLoopError(str(exc)) from exc


def detect_arm_loops(
    gripper_values: NDArray,
    eef_values: NDArray,
    *,
    arm: str,
) -> tuple[PickPlaceEvents, ...]:
    """Return every complete loop so later gestures are not silently ignored."""

    gripper = np.asarray(gripper_values)
    eef = np.asarray(eef_values)
    events: list[PickPlaceEvents] = []
    offset = 0
    while offset < len(gripper) - 5:
        try:
            event = detect_loop_events(
                gripper[offset:],
                eef[offset:],
                arm=arm,
            )
        except StateLoopError:
            break
        absolute = PickPlaceEvents(
            active_arm=event.active_arm,
            t_move_start=event.t_move_start + offset,
            t_close_start=event.t_close_start + offset,
            t_close_done=event.t_close_done + offset,
            t_open_start=event.t_open_start + offset,
            t_open_done=event.t_open_done + offset,
        )
        events.append(absolute)
        offset = absolute.t_open_done + 1
    return tuple(events)


def detect_episode_loop(state: EpisodeStateSignals) -> PickPlaceEvents:
    """Require exactly one complete arm loop in the episode."""

    candidates: list[PickPlaceEvents] = []
    counts: dict[str, int] = {}
    for arm_index, arm in enumerate(("left", "right")):
        arm_events = detect_arm_loops(
            state.gripper_states[:, arm_index],
            state.eef_states[:, arm_index],
            arm=arm,
        )
        candidates.extend(arm_events)
        counts[arm] = len(arm_events)
    if len(candidates) != 1:
        raise StateLoopError(
            f"expected exactly one active-arm loop, got {len(candidates)}; per_arm={counts}"
        )
    return candidates[0]


def detect_episode_target_only(state: EpisodeStateSignals) -> TargetOnlyEvents:
    """Require exactly one arm to close once and remain closed."""

    candidates: list[TargetOnlyEvents] = []
    errors: dict[str, str] = {}
    for arm_index, arm in enumerate(("left", "right")):
        try:
            event = detect_target_only_events(
                state.gripper_states[:, arm_index],
                state.eef_states[:, arm_index],
                arm=arm,
            )
        except StateLoopError as exc:
            errors[arm] = str(exc)
        else:
            candidates.append(event)
    if len(candidates) != 1:
        raise StateLoopError(
            "expected exactly one target-only close-and-hold arm, "
            f"got {len(candidates)}; per_arm={errors}"
        )
    return candidates[0]


__all__ = [
    "CLOSED_THRESHOLD",
    "OPEN_THRESHOLD",
    "EpisodeStateSignals",
    "StateLoopError",
    "detect_arm_loops",
    "detect_episode_loop",
    "detect_episode_target_only",
    "detect_loop_events",
    "detect_target_only_events",
]
