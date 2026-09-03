from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from robotwin_annotation_v2.adapters.robotwin_dataset import RoboTwinDataset
from robotwin_annotation_v2.models import EpisodeRef


def _dataset(root: Path) -> RoboTwinDataset:
    return RoboTwinDataset(
        root,
        task="adjust_bottle",
        camera="cam_high",
        manifest_path=root / "EXTRACT_MANIFEST.json",
        manifest_data={
            "task": "adjust_bottle",
            "camera": "cam_high",
            "dataset_root": str(root.resolve()),
            "regression_episode_ids": [0],
        },
    )


def test_missing_video_contract_is_inferred_and_cached(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _dataset(tmp_path)
    ref = EpisodeRef("adjust_bottle", 0, "cam_high")
    for path in (
        dataset.paths(ref).parquet,
        dataset.paths(ref).video,
        dataset.paths(ref).sidecar,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    monkeypatch.setattr(dataset, "_metadata_index", lambda: {0: {}})
    monkeypatch.setattr(dataset, "load_state", lambda _ref: SimpleNamespace(frame_count=3))
    monkeypatch.setattr(dataset, "video_info", lambda _ref: (4, (240, 320)))

    assert dataset.preflight((0,))["passed"]
    assert dataset.frame_shape(ref) == (240, 320)
    assert dataset.manifest["frame_shape_hw"] == [240, 320]
    assert dataset.manifest["raw_video_frame_surplus"] == 1
