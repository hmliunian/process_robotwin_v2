from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from robotwin_annotation_v2.adapters.robotwin_dataset import RoboTwinDataset
from robotwin_annotation_v2.domain import TimelineSource
from robotwin_annotation_v2.models import EpisodeRef, TargetOnlyEvents


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


def test_load_target_only_timeline_needs_no_observation_state(tmp_path: Path) -> None:
    manifest = {
        "task": "pick_up_real",
        "camera": "cam_high",
        "dataset_root": str(tmp_path.resolve()),
        "regression_episode_ids": [0],
        "timeline_source": "episode_metadata",
    }
    dataset = RoboTwinDataset(
        tmp_path,
        task="pick_up_real",
        camera="cam_high",
        manifest_path=tmp_path / "EXTRACT_MANIFEST.json",
        manifest_data=manifest,
    )
    ref = EpisodeRef("pick_up_real", 0, "cam_high")
    parquet = dataset.paths(ref).parquet
    parquet.parent.mkdir(parents=True)
    pd.DataFrame(
        {"frame_index": range(12), "episode_index": [0] * 12}
    ).to_parquet(parquet, index=False)
    metadata = tmp_path / "meta" / "episodes.jsonl"
    metadata.parent.mkdir()
    metadata.write_text(
        json.dumps(
            {
                "episode_index": 0,
                "tasks": ["Pick up the object."],
                "length": 12,
                "target_only_events": {
                    "active_arm": "left",
                    "t_remove_start": 0,
                    "t_close_start": 5,
                    "t_close_end": 8,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    timeline = dataset.load_target_only_timeline(ref)

    assert dataset.timeline_source is TimelineSource.EPISODE_METADATA
    assert timeline.events == TargetOnlyEvents("left", 0, 5, 8)
    assert timeline.frame_count == 12
