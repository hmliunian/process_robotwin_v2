from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from robotwin_annotation_v2.adapters.robotwin_dataset import DatasetError, RoboTwinDataset


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
