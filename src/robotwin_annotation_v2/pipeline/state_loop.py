"""Stage 1: extract one state loop and sparse semantic frames."""

from __future__ import annotations

import numpy as np

from ..adapters.robotwin_dataset import EpisodeState, RoboTwinDataset
from ..models import (
    EpisodeRef,
    FramePurpose,
    LoopContext,
    LoopEvents,
    SemanticFrame,
)


OPEN_THRESHOLD = 0.9
CLOSED_THRESHOLD = 0.15


class StateLoopError(RuntimeError):
    """State signals do not describe exactly one valid pick-and-place loop."""


def _median_filter(values: np.ndarray) -> np.ndarray:
    padded = np.pad(values, (1, 1), mode="edge")
    return np.median(
        np.stack((padded[:-2], padded[1:-1], padded[2:])),
        axis=0,
    )


def _first_run(condition: np.ndarray, *, start: int, length: int) -> int | None:
    for frame_id in range(max(0, start), len(condition) - length + 1):
        if bool(condition[frame_id : frame_id + length].all()):
            return frame_id
    return None


def detect_loop_events(
    gripper_values: np.ndarray,
    eef_values: np.ndarray,
    *,
    arm: str,
    stable_frames: int = 3,
    motion_floor: float = 0.002,
    rotation_scale: float = 0.05,
) -> LoopEvents:
    """Detect five ordered events for one arm from gripper and EEF state."""

    gripper = np.asarray(gripper_values, dtype=np.float64)
    eef = np.asarray(eef_values, dtype=np.float64)
    if gripper.ndim != 1 or gripper.size < stable_frames * 2:
        raise StateLoopError("gripper_values must be a sufficiently long 1-D array")
    if eef.shape != (gripper.size, 6):
        raise StateLoopError(f"eef_values must have shape {(gripper.size, 6)}")
    if stable_frames < 1:
        raise ValueError("stable_frames must be positive")

    filtered = _median_filter(gripper)
    close_candidates = np.flatnonzero(
        (filtered[1:] < OPEN_THRESHOLD) & (filtered[1:] < filtered[:-1])
    ) + 1
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
    close_done = close_run + stable_frames - 1

    open_candidates = np.flatnonzero(
        (filtered[1:] > filtered[:-1])
        & (np.arange(1, len(filtered)) > close_done)
    ) + 1
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

    delta = np.diff(eef, axis=0)
    delta[:, 3:] = (delta[:, 3:] + np.pi) % (2 * np.pi) - np.pi
    score = np.linalg.norm(delta[:, :3], axis=1)
    score += rotation_scale * np.linalg.norm(delta[:, 3:], axis=1)
    baseline = score[: max(1, min(close_start, 3))]
    median = float(np.median(baseline))
    deviation = float(np.median(np.abs(baseline - median)))
    threshold = max(median + 4.0 * deviation, motion_floor)
    motion = np.concatenate(([False], score > threshold))
    move_start = _first_run(motion, start=1, length=stable_frames)
    if move_start is None or move_start >= close_start:
        move_start = close_start

    try:
        return LoopEvents(
            active_arm=arm,  # type: ignore[arg-type]
            t_move_start=move_start,
            t_close_start=close_start,
            t_close_done=close_done,
            t_open_start=open_start,
            t_open_done=open_done,
        )
    except ValueError as exc:
        raise StateLoopError(str(exc)) from exc


def detect_arm_loops(
    gripper_values: np.ndarray,
    eef_values: np.ndarray,
    *,
    arm: str,
) -> tuple[LoopEvents, ...]:
    """Return every complete loop so later gestures are not silently ignored."""

    gripper = np.asarray(gripper_values)
    eef = np.asarray(eef_values)
    events: list[LoopEvents] = []
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
        absolute = LoopEvents(
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


def detect_episode_loop(state: EpisodeState) -> LoopEvents:
    """Require exactly one complete arm loop in the episode."""

    candidates: list[LoopEvents] = []
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


def _uniform_frames(start: int, end: int, count: int) -> tuple[int, ...]:
    if end < start:
        return ()
    if start == end or count <= 1:
        return (start,)
    return tuple(sorted(set(np.linspace(start, end, num=count).round().astype(int).tolist())))


def sample_semantic_frames(
    events: LoopEvents,
    *,
    frame_count: int,
    seed_count: int = 4,
    seed_safety_margin: int = 4,
) -> tuple[SemanticFrame, ...]:
    """Select sparse, purpose-labelled frames without inspecting RGB pixels."""

    seed_end = max(0, events.t_close_start - seed_safety_margin)
    selected: dict[int, SemanticFrame] = {}
    for frame_id in _uniform_frames(0, seed_end, seed_count):
        selected[frame_id] = SemanticFrame(
            frame_id,
            FramePurpose.PRE_GRASP_SEED_CANDIDATE,
            ("target", "receiver"),
        )

    contexts: tuple[tuple[int, FramePurpose, tuple[str, ...]], ...] = (
        (
            min(events.t_close_done + 1, events.t_open_start - 1),
            FramePurpose.POST_GRASP_CONTEXT,
            ("target",),
        ),
        (
            (events.t_close_done + events.t_open_start) // 2,
            FramePurpose.POST_GRASP_CONTEXT,
            ("target",),
        ),
        (
            min(events.t_open_start + 1, events.t_open_done),
            FramePurpose.PLACE_CONTEXT,
            ("receiver",),
        ),
    )
    for frame_id, purpose, eligible_roles in contexts:
        frame_id = min(max(frame_id, 0), frame_count - 1)
        if frame_id not in selected:
            selected[frame_id] = SemanticFrame(
                frame_id,
                purpose,
                eligible_roles,  # type: ignore[arg-type]
            )
    return tuple(selected[frame_id] for frame_id in sorted(selected))


def build_loop_context(dataset: RoboTwinDataset, ref: EpisodeRef) -> LoopContext:
    """Run Stage 1 for one episode."""

    state = dataset.load_state(ref)
    events = detect_episode_loop(state)
    semantic_frames = sample_semantic_frames(events, frame_count=state.frame_count)
    return LoopContext(
        episode=ref,
        task_text=state.task_text,
        frame_count=state.frame_count,
        events=events,
        semantic_frames=semantic_frames,
        state_source=str(state.paths.parquet),
        video_source=str(state.paths.video),
    )
