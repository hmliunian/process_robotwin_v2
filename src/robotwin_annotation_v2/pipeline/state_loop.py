"""Stage 1: extract one state loop and sparse semantic frames."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import numpy as np

from ..domain import AnnotationMode, annotation_spec
from ..models import (
    EpisodeRef,
    FramePurpose,
    LoopContext,
    PickPlaceEvents,
    SemanticFrame,
    TargetOnlyEvents,
    TimelineEvents,
    derive_episode_windows,
)
from . import timeline_detector as _timeline_detector

if TYPE_CHECKING:
    from ..adapters.robotwin_dataset import RoboTwinDataset

CLOSED_THRESHOLD = _timeline_detector.CLOSED_THRESHOLD
OPEN_THRESHOLD = _timeline_detector.OPEN_THRESHOLD
StateLoopError = _timeline_detector.StateLoopError
_close_transition = _timeline_detector._close_transition
_first_run = _timeline_detector._first_run
_median_filter = _timeline_detector._median_filter
_motion_start = _timeline_detector._motion_start
detect_arm_loops = _timeline_detector.detect_arm_loops
detect_episode_loop = _timeline_detector.detect_episode_loop
detect_episode_target_only = _timeline_detector.detect_episode_target_only
detect_loop_events = _timeline_detector.detect_loop_events
detect_target_only_events = _timeline_detector.detect_target_only_events


def _uniform_frames(start: int, end: int, count: int) -> tuple[int, ...]:
    if end < start:
        return ()
    if start == end or count <= 1:
        return (start,)
    return tuple(sorted(set(np.linspace(start, end, num=count).round().astype(int).tolist())))


def sample_semantic_frames(
    events: TimelineEvents,
    *,
    frame_count: int,
    annotation_mode: AnnotationMode = AnnotationMode.PICK_PLACE,
    seed_count: int = 4,
    seed_safety_margin: int = 4,
) -> tuple[SemanticFrame, ...]:
    """Select mode-specific semantic evidence without inspecting RGB pixels.

    Both modes share pre-grasp seed candidates.  Pick/place adds transport and
    placement context; close-and-hold adds only target identity evidence after
    closure.  Context frames never extend an object's output mask window.
    """

    mode = AnnotationMode(annotation_mode)
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    if seed_count < 1:
        raise ValueError("seed_count must be positive")
    if seed_safety_margin < 0:
        raise ValueError("seed_safety_margin must be non-negative")
    if mode is AnnotationMode.PICK_PLACE and not isinstance(events, PickPlaceEvents):
        raise ValueError("pick_place semantic sampling requires PickPlaceEvents")
    if mode is AnnotationMode.TARGET_ONLY and not isinstance(events, TargetOnlyEvents):
        raise ValueError("target_only semantic sampling requires TargetOnlyEvents")
    derive_episode_windows(events, frame_count=frame_count)

    spec = annotation_spec(mode)
    required_roles = cast(
        tuple[Literal["target", "receiver"], ...],
        spec.required_role_names,
    )
    seed_end = max(0, events.t_close_start - seed_safety_margin)
    selected: dict[int, SemanticFrame] = {}
    for frame_id in _uniform_frames(0, seed_end, seed_count):
        selected[frame_id] = SemanticFrame(
            frame_id,
            FramePurpose.PRE_GRASP_SEED_CANDIDATE,
            required_roles,
        )

    contexts: tuple[tuple[int, FramePurpose, tuple[Literal["target", "receiver"], ...]], ...]
    if isinstance(events, PickPlaceEvents):
        contexts = (
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
    else:
        last_frame = frame_count - 1
        first_closed_context = min(events.t_close_end + 1, last_frame)
        held_context = (first_closed_context + last_frame) // 2
        contexts = (
            (
                first_closed_context,
                FramePurpose.POST_GRASP_CONTEXT,
                ("target",),
            ),
            (
                held_context,
                FramePurpose.POST_GRASP_CONTEXT,
                ("target",),
            ),
        )
    for frame_id, purpose, eligible_roles in contexts:
        frame_id = min(max(frame_id, 0), frame_count - 1)
        if frame_id not in selected:
            selected[frame_id] = SemanticFrame(
                frame_id,
                purpose,
                eligible_roles,
            )
    return tuple(selected[frame_id] for frame_id in sorted(selected))


def build_loop_context(
    dataset: RoboTwinDataset,
    ref: EpisodeRef,
    *,
    annotation_mode: AnnotationMode = AnnotationMode.PICK_PLACE,
) -> LoopContext:
    """Run the one mode-dispatch boundary for Stage 1."""

    state = dataset.load_state(ref)
    mode = AnnotationMode(annotation_mode)
    if mode is AnnotationMode.PICK_PLACE:
        events: TimelineEvents = detect_episode_loop(state)
    else:
        events = detect_episode_target_only(state)
    semantic_frames = sample_semantic_frames(
        events,
        frame_count=state.frame_count,
        annotation_mode=mode,
    )
    return LoopContext(
        episode=ref,
        task_text=state.task_text,
        frame_count=state.frame_count,
        events=events,
        semantic_frames=semantic_frames,
        state_source=str(state.paths.parquet),
        video_source=str(state.paths.video),
        annotation_mode=mode,
    )
