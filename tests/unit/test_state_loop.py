from __future__ import annotations

import numpy as np

from robotwin_annotation_v2.models import FramePurpose, LoopEvents
from robotwin_annotation_v2.pipeline.state_loop import (
    detect_arm_loops,
    detect_loop_events,
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


def test_arm_detector_keeps_multiple_loops_visible() -> None:
    one_loop = np.array(
        [1.0] * 4 + [0.7, 0.1, 0.1, 0.1] + [0.4, 0.95, 0.95, 0.95]
    )
    gripper = np.concatenate((one_loop, np.ones(3), one_loop))
    eef = np.zeros((len(gripper), 6), dtype=np.float64)

    events = detect_arm_loops(gripper, eef, arm="left")

    assert len(events) == 2
    assert events[0].t_open_done < events[1].t_move_start
