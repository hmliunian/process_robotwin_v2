from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import robotwin_annotation_v2.urdf_gripper_data as data_module
from robotwin_annotation_v2.adapters.robotwin_dataset import EpisodePaths, EpisodeState
from robotwin_annotation_v2.models.timeline import TargetOnlyEvents
from robotwin_annotation_v2.pipeline.state_loop import detect_episode_loop
from robotwin_annotation_v2.urdf_gripper_data import (
    ActiveGripperLoop,
    UrdfGripperDataError,
    format_episode_id,
    infer_active_loop,
    load_authoritative_loop_context,
    load_camera_calibration,
    load_episode_arrays,
    load_urdf_gripper_episode,
    resolve_episode_paths,
)


def _one_right_arm_loop_state() -> np.ndarray:
    gripper = np.array(
        [1.0] * 5
        + [0.7, 0.4, 0.1, 0.1, 0.1]
        + [0.3, 0.6, 0.95, 0.95, 0.95],
        dtype=np.float64,
    )
    state = np.zeros((len(gripper), 14), dtype=np.float64)
    state[:, (6, 13)] = 1.0
    state[:, 13] = gripper
    state[2:6, 7] = np.arange(4) * 0.01
    return state


def _write_parquet(path: Path, *, episode_index: int = 7152) -> tuple[np.ndarray, np.ndarray]:
    observation_state = _one_right_arm_loop_state()
    joint_absolute = np.arange(observation_state.size, dtype=np.float64).reshape(
        observation_state.shape
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "frame_index": np.arange(len(observation_state), dtype=np.int64),
            "episode_index": np.full(len(observation_state), episode_index, dtype=np.int64),
            "observation.state": list(observation_state),
            "observation.state.joint_absolute": list(joint_absolute),
        }
    )
    frame.to_parquet(path, index=False)
    return observation_state, joint_absolute


def _write_loop_context(path: Path, events: ActiveGripperLoop) -> None:
    payload = {
        "format_version": "robotwin_loop_context_v1",
        "episode": {
            "task": "move_pillbottle_pad",
            "episode_index": 7152,
            "episode_id": "007152",
            "camera": "cam_high",
        },
        "frame_count": 15,
        "events": events.to_json(),
        "windows": {
            "loop": list(events.inclusive_window),
            "target_0": [events.t_move_start, events.t_close_done],
            "receiver_0": [events.t_close_done, events.t_open_done],
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_target_only_v2_normalizes_close_hold_without_open_events(
    tmp_path: Path,
) -> None:
    events = TargetOnlyEvents("right", 3, 5, 9)
    path = tmp_path / "loop.json"
    payload = {
        "format_version": "robotwin_loop_context_v2",
        "annotation_mode": "target_only",
        "timeline_kind": "close_hold",
        "required_object_roles": ["target"],
        "episode": {
            "task": "move_pillbottle_pad",
            "episode_index": 7152,
            "episode_id": "007152",
            "camera": "cam_high",
        },
        "frame_count": 15,
        "events": events.to_json(),
        "windows": {
            "operation": [3, 14],
            "target_0": [3, 9],
            "receiver_0": None,
            "gripper": [3, 14],
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    context = load_authoritative_loop_context(
        path,
        expected_task="move_pillbottle_pad",
        expected_episode_index=7152,
        expected_camera="cam_high",
    )

    assert context.annotation_mode.value == "target_only"
    assert context.events == events
    assert context.active_arm == "right"
    assert context.gripper_window == (3, 14)
    assert "t_open_start" not in context.events.to_json()
    assert "t_open_done" not in context.events.to_json()


def test_target_only_v2_rejects_gripper_window_that_stops_at_close_end(
    tmp_path: Path,
) -> None:
    path = tmp_path / "loop.json"
    payload = {
        "format_version": "robotwin_loop_context_v2",
        "annotation_mode": "target_only",
        "timeline_kind": "close_hold",
        "required_object_roles": ["target"],
        "episode": {
            "task": "move_pillbottle_pad",
            "episode_index": 7152,
            "episode_id": "007152",
            "camera": "cam_high",
        },
        "frame_count": 15,
        "events": {
            "active_arm": "right",
            "t_remove_start": 3,
            "t_close_start": 5,
            "t_close_end": 9,
        },
        "windows": {
            "operation": [3, 14],
            "target_0": [3, 9],
            "receiver_0": None,
            "gripper": [3, 9],
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(UrdfGripperDataError, match="window gripper"):
        load_authoritative_loop_context(
            path,
            expected_task="move_pillbottle_pad",
            expected_episode_index=7152,
            expected_camera="cam_high",
        )


def test_resolve_episode_paths_uses_zero_padding_and_depth_sidecar_tree(
    tmp_path: Path,
) -> None:
    paths = resolve_episode_paths(tmp_path, 7152, camera="cam_high")

    assert format_episode_id(7152) == "007152"
    assert paths.parquet == tmp_path / "data/chunk-007/episode_007152.parquet"
    assert paths.sidecar == tmp_path / "sidecars/episode_007152.hdf5"
    assert paths.rgb_video == (
        tmp_path
        / "videos/chunk-007/observation.images.cam_high/episode_007152.mp4"
    )
    assert paths.depth_video == (
        tmp_path
        / "sidecars/videos/chunk-007/observation.depths.cam_high/episode_007152.mkv"
    )


def test_load_episode_arrays_reads_joint_absolute_and_parquet_frame_count(
    tmp_path: Path,
) -> None:
    parquet = tmp_path / "episode_007152.parquet"
    observation_state, joint_absolute = _write_parquet(parquet)

    arrays = load_episode_arrays(parquet, expected_episode_index=7152)

    assert arrays.frame_count == len(observation_state)
    assert arrays.episode_index == 7152
    np.testing.assert_array_equal(arrays.observation_state, observation_state)
    np.testing.assert_array_equal(arrays.joint_absolute, joint_absolute)


def test_infer_active_loop_matches_existing_state_loop_contract() -> None:
    state = _one_right_arm_loop_state()
    expected = detect_episode_loop(
        EpisodeState(
            frame_count=len(state),
            task_text="test",
            gripper_states=state[:, (6, 13)],
            eef_states=np.stack((state[:, 0:6], state[:, 7:13]), axis=1),
            paths=EpisodePaths(Path("state.parquet"), Path("rgb.mp4"), Path("sidecar.h5")),
        )
    )

    actual = infer_active_loop(state)

    assert actual.active_arm == expected.active_arm == "right"
    assert actual.inclusive_window == (expected.t_move_start, expected.t_open_done)
    assert (
        actual.t_move_start,
        actual.t_close_start,
        actual.t_close_done,
        actual.t_open_start,
        actual.t_open_done,
    ) == (
        expected.t_move_start,
        expected.t_close_start,
        expected.t_close_done,
        expected.t_open_start,
        expected.t_open_done,
    )


def test_infer_active_loop_rejects_two_active_arms() -> None:
    state = _one_right_arm_loop_state()
    state[:, 6] = state[:, 13]
    state[:, 0:6] = state[:, 7:13]

    with pytest.raises(UrdfGripperDataError, match="exactly one active-arm loop"):
        infer_active_loop(state)


def test_load_authoritative_loop_context_preserves_all_source_boundaries(
    tmp_path: Path,
) -> None:
    expected = ActiveGripperLoop(
        active_arm="left",
        t_move_start=1,
        t_close_start=2,
        t_close_done=4,
        t_open_start=8,
        t_open_done=12,
    )
    path = tmp_path / "loop.json"
    _write_loop_context(path, expected)

    context = load_authoritative_loop_context(
        path,
        expected_task="move_pillbottle_pad",
        expected_episode_index=7152,
        expected_camera="cam_high",
    )

    assert context.events == expected
    assert context.frame_count == 15
    assert context.path == path.resolve()
    assert context.gripper_window == expected.inclusive_window


def test_load_episode_uses_authoritative_loop_without_recomputing_stage1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = resolve_episode_paths(tmp_path, 7152)
    for path in (paths.sidecar, paths.rgb_video, paths.depth_video):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    _write_parquet(paths.parquet)
    authoritative = ActiveGripperLoop(
        active_arm="left",
        t_move_start=1,
        t_close_start=2,
        t_close_done=4,
        t_open_start=8,
        t_open_done=12,
    )
    monkeypatch.setattr(
        data_module,
        "infer_active_loop",
        lambda *_args: pytest.fail("derived load must not recompute Stage-1"),
    )

    episode = load_urdf_gripper_episode(
        tmp_path,
        7152,
        authoritative_loop=authoritative,
    )

    assert episode.loop == authoritative
    assert episode.active_arm == "left"
    assert episode.active_window == (1, 12)


def test_load_episode_uses_target_only_normalized_window_through_final_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = resolve_episode_paths(tmp_path, 7152)
    for path in (paths.sidecar, paths.rgb_video, paths.depth_video):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    _write_parquet(paths.parquet)
    events = TargetOnlyEvents(
        active_arm="right",
        t_remove_start=3,
        t_close_start=5,
        t_close_end=9,
    )
    monkeypatch.setattr(
        data_module,
        "infer_active_loop",
        lambda *_args: pytest.fail("target_only must not run the five-event fallback"),
    )

    episode = load_urdf_gripper_episode(
        tmp_path,
        7152,
        authoritative_events=events,
        authoritative_gripper_window=(3, 14),
    )

    assert episode.events == events
    assert episode.active_arm == "right"
    assert episode.active_window == (3, 14)


def test_load_urdf_gripper_episode_validates_media_and_exposes_window(
    tmp_path: Path,
) -> None:
    paths = resolve_episode_paths(tmp_path, 7152)
    for path in (paths.sidecar, paths.rgb_video, paths.depth_video):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    _write_parquet(paths.parquet)

    episode = load_urdf_gripper_episode(tmp_path, 7152)

    assert episode.frame_count == 15
    assert episode.active_arm == "right"
    assert episode.active_window == (3, 14)
    assert episode.joint_absolute.shape == (15, 14)


def test_camera_calibration_is_lazy_and_crops_trailing_raw_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sidecar = tmp_path / "episode_007152.hdf5"
    sidecar.touch()
    intrinsic = np.arange(4 * 3 * 3, dtype=np.float64).reshape(4, 3, 3)
    extrinsic = np.arange(4 * 3 * 4, dtype=np.float64).reshape(4, 3, 4)
    cam2world = np.arange(4 * 4 * 4, dtype=np.float64).reshape(4, 4, 4)
    datasets = {
        "camera_params/cam_high/intrinsic_cv": intrinsic,
        "camera_params/cam_high/extrinsic_cv": extrinsic,
        "camera_params/cam_high/cam2world_gl": cam2world,
    }

    class FakeFile:
        def __init__(self, _path: Path, _mode: str) -> None:
            pass

        def __enter__(self) -> FakeFile:
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

        def __getitem__(self, key: str) -> np.ndarray:
            return datasets[key]

    fake_h5py = types.ModuleType("h5py")
    fake_h5py.File = FakeFile  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "h5py", fake_h5py)

    calibration = load_camera_calibration(sidecar, frame_count=3)

    assert calibration.frame_count == 3
    np.testing.assert_array_equal(calibration.intrinsic_cv, intrinsic[:3])
    np.testing.assert_array_equal(calibration.extrinsic_cv, extrinsic[:3])
    np.testing.assert_array_equal(calibration.cam2world_gl, cam2world[:3])
