from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import robotwin_annotation_v2.pipeline as public_pipeline
import robotwin_annotation_v2.urdf_gripper_data as urdf_data
from robotwin_annotation_v2.adapters.robotwin_dataset import EpisodePaths, EpisodeState
from robotwin_annotation_v2.pipeline.state_loop import (
    StateLoopError,
    detect_arm_loops,
    detect_episode_loop,
    detect_loop_events,
)


def _happy_signal() -> tuple[np.ndarray, np.ndarray]:
    gripper = np.array(
        [1.0] * 5
        + [0.7, 0.4, 0.1, 0.1, 0.1]
        + [0.3, 0.6, 0.95, 0.95, 0.95],
        dtype=np.float64,
    )
    eef = np.zeros((len(gripper), 6), dtype=np.float64)
    eef[2:6, 0] = np.arange(4) * 0.01
    return gripper, eef


def _expected_events(
    arm: str,
    move_start: int,
    close_start: int,
    close_done: int,
    open_start: int,
    open_done: int,
) -> dict[str, object]:
    return {
        "active_arm": arm,
        "t_move_start": move_start,
        "t_close_start": close_start,
        "t_close_done": close_done,
        "t_open_start": open_start,
        "t_open_done": open_done,
    }


def _assert_event_parity(
    gripper: np.ndarray,
    eef: np.ndarray,
    *,
    arm: str,
    expected: dict[str, object],
    stable_frames: int = 3,
    motion_floor: float = 0.002,
    rotation_scale: float = 0.05,
) -> None:
    pipeline_event = detect_loop_events(
        gripper,
        eef,
        arm=arm,
        stable_frames=stable_frames,
        motion_floor=motion_floor,
        rotation_scale=rotation_scale,
    )
    urdf_event = urdf_data._detect_loop_events(
        gripper,
        eef,
        arm=arm,  # type: ignore[arg-type]
        stable_frames=stable_frames,
        motion_floor=motion_floor,
        rotation_scale=rotation_scale,
    )

    assert pipeline_event.to_json() == urdf_event.to_json() == expected


def test_state_loop_and_public_exports_preserve_canonical_detector_identity() -> None:
    from robotwin_annotation_v2.pipeline import state_loop
    from robotwin_annotation_v2.pipeline import timeline_detector as canonical_detector

    public_names = (
        "StateLoopError",
        "detect_arm_loops",
        "detect_episode_loop",
        "detect_episode_target_only",
        "detect_loop_events",
        "detect_target_only_events",
    )
    private_names = (
        "_close_transition",
        "_first_run",
        "_median_filter",
        "_motion_start",
    )
    for name in public_names:
        canonical = getattr(canonical_detector, name)
        assert getattr(state_loop, name) is canonical
        assert getattr(public_pipeline, name) is canonical
    for name in private_names:
        assert getattr(state_loop, name) is getattr(canonical_detector, name)


@pytest.mark.parametrize("arm", ["left", "right"])
def test_detectors_match_for_left_and_right_happy_paths(arm: str) -> None:
    gripper, eef = _happy_signal()

    _assert_event_parity(
        gripper,
        eef,
        arm=arm,
        expected=_expected_events(arm, 3, 5, 9, 10, 14),
    )


@pytest.mark.parametrize(
    "case",
    ["no_motion", "wrapped_rpy", "median_filtered_glitch", "threshold_boundaries"],
)
def test_detectors_match_at_signal_boundaries(case: str) -> None:
    gripper, eef = _happy_signal()
    expected = _expected_events("left", 3, 5, 9, 10, 14)

    if case == "no_motion":
        eef.fill(0.0)
        expected["t_move_start"] = 5
    elif case == "wrapped_rpy":
        eef.fill(0.0)
        eef[:, 5] = np.pi - 0.3
        eef[3, 5] = np.pi - 0.2
        eef[4, 5] = np.pi - 0.1
        eef[5:, 5] = -np.pi
    elif case == "median_filtered_glitch":
        gripper[2] = 0.0
    else:
        gripper = np.array(
            [1.0] * 4 + [0.7, 0.15, 0.15, 0.15] + [0.4, 0.9, 0.9, 0.9],
            dtype=np.float64,
        )
        eef = np.zeros((len(gripper), 6), dtype=np.float64)
        expected = _expected_events("left", 4, 4, 7, 8, 11)

    _assert_event_parity(gripper, eef, arm="left", expected=expected)


def test_detectors_match_with_nondefault_detection_parameters() -> None:
    gripper = np.array(
        [1.0] * 7 + [0.7, 0.1, 0.1] + [0.4, 0.9, 0.9],
        dtype=np.float64,
    )
    eef = np.zeros((len(gripper), 6), dtype=np.float64)
    eef[4, 5] = 0.075
    eef[5:, 5] = 0.15

    _assert_event_parity(
        gripper,
        eef,
        arm="right",
        stable_frames=2,
        motion_floor=0.01,
        rotation_scale=0.2,
        expected=_expected_events("right", 4, 7, 9, 10, 12),
    )


def _one_loop_signal() -> np.ndarray:
    return np.array(
        [1.0] * 4 + [0.7, 0.1, 0.1, 0.1] + [0.4, 0.95, 0.95, 0.95],
        dtype=np.float64,
    )


@pytest.mark.parametrize("tail", ["complete_loop", "incomplete_loop"])
def test_arm_loop_scanning_matches_for_complete_and_incomplete_tails(tail: str) -> None:
    one_loop = _one_loop_signal()
    trailing_signal = (
        one_loop
        if tail == "complete_loop"
        else np.array([1.0] * 4 + [0.7, 0.1, 0.1, 0.1], dtype=np.float64)
    )
    gripper = np.concatenate((one_loop, np.ones(3), trailing_signal))
    eef = np.zeros((len(gripper), 6), dtype=np.float64)

    pipeline_events = detect_arm_loops(gripper, eef, arm="left")
    urdf_events = urdf_data._detect_arm_loops(gripper, eef, arm="left")
    pipeline_json = tuple(event.to_json() for event in pipeline_events)
    urdf_json = tuple(event.to_json() for event in urdf_events)
    expected = (_expected_events("left", 4, 4, 7, 8, 11),)
    if tail == "complete_loop":
        expected += (_expected_events("left", 19, 19, 22, 23, 26),)

    assert pipeline_json == urdf_json == expected


def _observation_state(active_arms: tuple[str, ...]) -> np.ndarray:
    gripper, eef = _happy_signal()
    state = np.zeros((len(gripper), 14), dtype=np.float64)
    state[:, (6, 13)] = 1.0
    if "left" in active_arms:
        state[:, 6] = gripper
        state[:, 0:6] = eef
    if "right" in active_arms:
        state[:, 13] = gripper
        state[:, 7:13] = eef
    return state


def _episode_state(state: np.ndarray) -> EpisodeState:
    return EpisodeState(
        frame_count=len(state),
        task_text="test",
        gripper_states=state[:, (6, 13)],
        eef_states=np.stack((state[:, 0:6], state[:, 7:13]), axis=1),
        paths=EpisodePaths(Path("state.parquet"), Path("rgb.mp4"), Path("sidecar.h5")),
    )


@pytest.mark.parametrize(
    ("active_arms", "count", "per_arm"),
    [
        ((), 0, "{'left': 0, 'right': 0}"),
        (("left", "right"), 2, "{'left': 1, 'right': 1}"),
    ],
)
def test_episode_detectors_reject_zero_or_two_active_arms(
    active_arms: tuple[str, ...],
    count: int,
    per_arm: str,
) -> None:
    state = _observation_state(active_arms)
    expected_message = f"expected exactly one active-arm loop, got {count}; per_arm={per_arm}"

    with pytest.raises(StateLoopError) as pipeline_error:
        detect_episode_loop(_episode_state(state))
    with pytest.raises(urdf_data.UrdfGripperDataError) as urdf_error:
        urdf_data.infer_active_loop(state)

    assert str(pipeline_error.value) == expected_message
    assert str(urdf_error.value) == expected_message


def _error_signal(case: str) -> tuple[np.ndarray, np.ndarray]:
    if case == "no_close":
        gripper = np.ones(15, dtype=np.float64)
    elif case == "no_stable_close":
        gripper = np.array(
            [1.0] * 5 + [0.7, 0.4, 0.2, 0.2, 0.2] + [0.4, 0.95, 0.95, 0.95],
            dtype=np.float64,
        )
    elif case == "no_open":
        gripper = np.array(
            [1.0] * 5 + [0.7, 0.4, 0.1, 0.1, 0.1] + [0.1] * 3,
            dtype=np.float64,
        )
    else:
        gripper = np.array(
            [1.0] * 5 + [0.7, 0.4, 0.1, 0.1, 0.1] + [0.3, 0.6, 0.8, 0.8, 0.8],
            dtype=np.float64,
        )
    return gripper, np.zeros((len(gripper), 6), dtype=np.float64)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("no_close", "no close transition detected"),
        ("no_stable_close", "no stable closed transition detected"),
        ("no_open", "no open transition detected"),
        ("no_stable_reopen", "no stable reopen transition detected"),
    ],
)
def test_detector_transition_errors_remain_in_parity(case: str, message: str) -> None:
    gripper, eef = _error_signal(case)

    with pytest.raises(StateLoopError) as pipeline_error:
        detect_loop_events(gripper, eef, arm="left")
    with pytest.raises(urdf_data.UrdfGripperDataError) as urdf_error:
        urdf_data._detect_loop_events(gripper, eef, arm="left")

    assert str(pipeline_error.value) == message
    assert str(urdf_error.value) == message


@pytest.mark.parametrize("case", ["gripper_shape", "eef_shape"])
def test_detector_shape_errors_freeze_each_implementation_message(case: str) -> None:
    if case == "gripper_shape":
        gripper = np.ones((3, 4), dtype=np.float64)
        eef = np.zeros((gripper.size, 6), dtype=np.float64)
        pipeline_message = "gripper_values must be a sufficiently long 1-D array"
        urdf_message = "gripper values must be a sufficiently long one-dimensional array"
    else:
        gripper, _ = _happy_signal()
        eef = np.zeros((len(gripper), 5), dtype=np.float64)
        pipeline_message = "eef_values must have shape (15, 6)"
        urdf_message = "eef values must have shape (15, 6)"

    with pytest.raises(StateLoopError) as pipeline_error:
        detect_loop_events(gripper, eef, arm="right")
    with pytest.raises(urdf_data.UrdfGripperDataError) as urdf_error:
        urdf_data._detect_loop_events(gripper, eef, arm="right")

    assert str(pipeline_error.value) == pipeline_message
    assert str(urdf_error.value) == urdf_message
