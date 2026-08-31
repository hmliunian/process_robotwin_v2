from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from robotwin_annotation_v2.adapters.robotwin_dataset import DatasetError, RoboTwinDataset
from robotwin_annotation_v2.models import EpisodeRef


def _dataset(
    root: Path,
    *,
    manifest_data: dict[str, object] | None = None,
    task: str = "move_pillbottle_pad",
    camera: str = "cam_high",
) -> RoboTwinDataset:
    payload = {
        "task": task,
        "camera": camera,
        "dataset_root": "/DATA/merged/xuran/add_mask_robotwin/dataset/stale",
        "regression_episode_ids": [],
        "frame_shape_hw": [240, 320],
        "raw_video_frame_surplus": 1,
    }
    if manifest_data is not None:
        payload.update(manifest_data)
    return RoboTwinDataset(
        root,
        task=task,
        camera=camera,
        manifest_path=root / "EXTRACT_MANIFEST.json",
        manifest_data=payload,
    )


def test_binding_root_is_authoritative_when_manifest_root_is_stale(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)

    assert dataset.root == tmp_path.resolve()
    # Preserve provenance for callers that include the manifest in reports.
    assert dataset.manifest["dataset_root"] == (
        "/DATA/merged/xuran/add_mask_robotwin/dataset/stale"
    )


def test_missing_manifest_root_is_allowed(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, manifest_data={"dataset_root": None})

    assert dataset.root == tmp_path.resolve()
    assert dataset.manifest["dataset_root"] is None


@pytest.mark.parametrize("field", ("task", "camera"))
def test_manifest_identity_mismatch_is_still_rejected(tmp_path: Path, field: str) -> None:
    value = "other_task" if field == "task" else "cam_wrist"

    with pytest.raises(DatasetError, match="task/camera"):
        _dataset(tmp_path, manifest_data={field: value})


@pytest.mark.parametrize("field", ("task", "camera"))
def test_adapter_rejects_path_like_identity(tmp_path: Path, field: str) -> None:
    kwargs = {"task": "move_pillbottle_pad", "camera": "cam_high"}
    kwargs[field] = "../escape"
    with pytest.raises(DatasetError, match="single path component"):
        _dataset(tmp_path, **kwargs)  # type: ignore[arg-type]


def test_manifest_free_preflight_infers_video_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native path discovery can preflight without creating a manifest file."""

    dataset = _dataset(
        tmp_path,
        manifest_data={
            "regression_episode_ids": [0],
            "frame_shape_hw": None,
            "raw_video_frame_surplus": None,
        },
    )
    parquet = tmp_path / "data/chunk-000/episode_000000.parquet"
    video = tmp_path / "videos/chunk-000/observation.images.cam_high/episode_000000.mp4"
    sidecar = tmp_path / "sidecars/episode_000000.hdf5"
    for path in (parquet, video, sidecar):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    monkeypatch.setattr(dataset, "_metadata_index", lambda: {0: {"tasks": ["task"]}})
    monkeypatch.setattr(
        dataset,
        "load_state",
        lambda _ref: SimpleNamespace(frame_count=3),
    )
    monkeypatch.setattr(dataset, "video_info", lambda _ref: (4, (240, 320)))

    report = dataset.preflight((0,))

    assert report["passed"]
    assert dataset.manifest["frame_shape_hw"] == [240, 320]
    assert dataset.manifest["raw_video_frame_surplus"] == 1


def test_load_state_maps_declared_cora_single_right_arm_layout(tmp_path: Path) -> None:
    dataset = _dataset(
        tmp_path,
        task="open_microwave",
        camera="head_left",
        manifest_data={"regression_episode_ids": [0]},
    )
    state = np.zeros((15, 7), dtype=np.float32)
    state[:, :6] = np.arange(15, dtype=np.float32)[:, None]
    state[:, 6] = np.asarray(
        [0.0, 0.0, 0.8, 0.8, 0.8, 0.4, 0.4, 0.4, 0.8, 0.8, 0.8, 0.0, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )
    parquet = tmp_path / "data/chunk-000/episode_000000.parquet"
    parquet.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "frame_index": np.arange(len(state)),
            "episode_index": np.zeros(len(state), dtype=np.int64),
            "observation.state": list(state),
        }
    ).to_parquet(parquet, index=False)
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "episodes.jsonl").write_text(
        '{"episode_index":0,"tasks":["open the microwave door"]}\n',
        encoding="utf-8",
    )
    (meta / "info.json").write_text(
        """{
          "robot_type": "cora_cart_right_arm",
          "features": {
            "observation.state": {
              "shape": [7],
              "names": [
                "right_eef_x", "right_eef_y", "right_eef_z",
                "right_eef_roll", "right_eef_pitch", "right_eef_yaw",
                "right_gripper_open"
              ]
            }
          }
        }""",
        encoding="utf-8",
    )

    loaded = dataset.load_state(EpisodeRef("open_microwave", 0, "head_left"))

    assert loaded.task_text == "open the microwave door"
    assert loaded.gripper_states.shape == (15, 2)
    assert np.array_equal(loaded.gripper_states[:, 0], np.ones(15))
    assert np.array_equal(
        loaded.gripper_states[:, 1],
        np.asarray([1.0] * 5 + [0.0] * 3 + [1.0] * 7),
    )
    assert np.array_equal(loaded.eef_states[:, 0], np.zeros((15, 6)))
    assert np.array_equal(loaded.eef_states[:, 1], state[:, :6])


def test_load_state_rejects_undeclared_seven_dimensional_layout(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path, manifest_data={"regression_episode_ids": [0]})
    parquet = tmp_path / "data/chunk-000/episode_000000.parquet"
    parquet.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "frame_index": np.arange(8),
            "episode_index": np.zeros(8, dtype=np.int64),
            "observation.state": list(np.zeros((8, 7), dtype=np.float32)),
        }
    ).to_parquet(parquet, index=False)
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "episodes.jsonl").write_text(
        '{"episode_index":0,"tasks":["task"]}\n',
        encoding="utf-8",
    )
    (meta / "info.json").write_text(
        '{"robot_type":"unknown","features":{"observation.state":{"shape":[7]}}}',
        encoding="utf-8",
    )

    with pytest.raises(DatasetError, match="supported cora_cart_right_arm"):
        dataset.load_state(EpisodeRef("move_pillbottle_pad", 0, "cam_high"))
