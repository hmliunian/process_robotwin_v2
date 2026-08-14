from __future__ import annotations

from pathlib import Path

import numpy as np

from robotwin_annotation_v2.adapters.robotwin_dataset import EpisodePaths, EpisodeState
from robotwin_annotation_v2.domain import AnnotationMode
from robotwin_annotation_v2.models import FramePurpose, LoopEvents, TargetOnlyEvents
from robotwin_annotation_v2.pipeline.state_loop import (
    detect_arm_loops,
    detect_episode_target_only,
    detect_loop_events,
    detect_target_only_events,
    sample_semantic_frames,
)


def test_detect_five_ordered_events_and_motion_start() -> None:
    gripper = np.array(
        [1.0] * 5
        + [0.7, 0.4, 0.1, 0.1, 0.1]
        + [0.3, 0.6, 0.95, 0.95, 0.95]
    )
    eef = np.zeros((len(gripper), 6), dtype=np.float64)
    eef[2:6, 0] = np.arange(4) * 0.01

    events = detect_loop_events(gripper, eef, arm="right")

    assert events == LoopEvents("right", 3, 5, 9, 10, 14)


def test_detect_target_only_close_and_hold_events() -> None:
    gripper = np.array([1.0] * 5 + [0.7, 0.4, 0.1, 0.1, 0.1] + [0.1] * 6)
    eef = np.zeros((len(gripper), 6), dtype=np.float64)
    eef[2:6, 0] = np.arange(4) * 0.01

    events = detect_target_only_events(gripper, eef, arm="left")

    assert events == TargetOnlyEvents("left", 3, 5, 9)


def test_target_only_rejects_pick_place_reopen() -> None:
    gripper = np.array(
        [1.0] * 5
        + [0.7, 0.4, 0.1, 0.1, 0.1]
        + [0.4, 0.95, 0.95, 0.95]
    )
    eef = np.zeros((len(gripper), 6), dtype=np.float64)

    with np.testing.assert_raises_regex(RuntimeError, "unexpectedly reopens"):
        detect_target_only_events(gripper, eef, arm="right")


def test_episode_target_only_selects_the_only_close_and_hold_arm() -> None:
    frame_count = 16
    closed = np.array([1.0] * 5 + [0.7, 0.4, 0.1, 0.1, 0.1] + [0.1] * 6)
    grippers = np.stack((closed, np.ones(frame_count)), axis=1)
    eef = np.zeros((frame_count, 2, 6), dtype=np.float64)
    eef[2:6, 0, 0] = np.arange(4) * 0.01
    state = EpisodeState(
        frame_count=frame_count,
        task_text="lift the bottle",
        gripper_states=grippers,
        eef_states=eef,
        paths=EpisodePaths(Path("state.parquet"), Path("video.mp4"), Path("sidecar.hdf5")),
    )

    events = detect_episode_target_only(state)

    assert events.active_arm == "left"


def test_semantic_frames_are_sparse_and_purpose_labelled() -> None:
    events = LoopEvents("right", 4, 56, 68, 123, 136)

    frames = sample_semantic_frames(events, frame_count=138)

    frame_ids = [frame.frame_id for frame in frames]
    assert frame_ids == sorted(set(frame_ids))
    assert frame_ids[:4] == [0, 17, 35, 52]
    assert any(frame.purpose is FramePurpose.POST_GRASP_CONTEXT for frame in frames)
    assert any(frame.purpose is FramePurpose.PLACE_CONTEXT for frame in frames)
    seed_frames = [frame for frame in frames if frame.seed_eligible]
    assert all(frame.eligible_roles == ("target", "receiver") for frame in seed_frames)


def test_target_only_semantic_frames_never_request_receiver_context() -> None:
    events = LoopEvents("right", 4, 56, 68, 123, 136)

    frames = sample_semantic_frames(
        events,
        frame_count=138,
        annotation_mode=AnnotationMode.TARGET_ONLY,
    )

    assert all(frame.eligible_roles == ("target",) for frame in frames)
    assert all(frame.purpose is not FramePurpose.PLACE_CONTEXT for frame in frames)


def test_arm_detector_keeps_multiple_loops_visible() -> None:
    one_loop = np.array(
        [1.0] * 4 + [0.7, 0.1, 0.1, 0.1] + [0.4, 0.95, 0.95, 0.95]
    )
    gripper = np.concatenate((one_loop, np.ones(3), one_loop))
    eef = np.zeros((len(gripper), 6), dtype=np.float64)

    events = detect_arm_loops(gripper, eef, arm="left")

    assert len(events) == 2
    assert events[0].t_open_done < events[1].t_move_start
