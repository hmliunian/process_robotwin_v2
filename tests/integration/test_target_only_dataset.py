from __future__ import annotations

import json
from pathlib import Path

import pytest

from robotwin_annotation_v2.adapters import RoboTwinDataset
from robotwin_annotation_v2.models import EpisodeRef, TargetOnlyEvents
from robotwin_annotation_v2.pipeline import (
    StateLoopError,
    detect_episode_loop,
    detect_episode_target_only,
)

DATASET_PARENT = Path(
    "/DATA/disk8/xuran/add_mask_robotwin/dataset/target_only_20"
)
DATASET_ROOT = DATASET_PARENT / "adjust_bottle"
SELECTION_MANIFEST = DATASET_PARENT / "SELECTION_MANIFEST.json"


def _dataset() -> tuple[RoboTwinDataset, tuple[int, ...]]:
    if not DATASET_ROOT.is_dir() or not SELECTION_MANIFEST.is_file():
        pytest.skip(f"external target-only dataset unavailable: {DATASET_ROOT}")
    selection = json.loads(SELECTION_MANIFEST.read_text(encoding="utf-8"))
    episode_ids = tuple(int(value) for value in selection["tasks"][0]["episode_indices"])
    manifest = {
        "task": "adjust_bottle",
        "camera": "cam_high",
        "dataset_root": str(DATASET_ROOT),
        "regression_episode_ids": list(episode_ids),
        "smoke_episode_ids": [episode_ids[0]],
        "frame_shape_hw": [240, 320],
        "raw_video_frame_surplus": 1,
    }
    return (
        RoboTwinDataset(
            DATASET_ROOT,
            task="adjust_bottle",
            camera="cam_high",
            manifest_path=SELECTION_MANIFEST,
            manifest_data=manifest,
        ),
        episode_ids,
    )


def test_target_only_20_input_contract_is_complete() -> None:
    dataset, episode_ids = _dataset()

    report = dataset.preflight(episode_ids)

    assert report["passed"]
    assert report["episode_count"] == 20
    assert all(item["raw_video_frame_surplus"] == 1 for item in report["content"].values())
    for episode_index in episode_ids:
        depth = (
            DATASET_ROOT
            / "sidecars/videos/chunk-000/observation.depths.cam_high"
            / f"episode_{episode_index:06d}.mkv"
        )
        assert depth.is_file()


def test_target_only_20_has_exactly_one_close_and_hold_arm() -> None:
    dataset, episode_ids = _dataset()
    events: list[TargetOnlyEvents] = []

    for episode_index in episode_ids:
        ref = EpisodeRef("adjust_bottle", episode_index, "cam_high")
        state = dataset.load_state(ref)
        event = detect_episode_target_only(state)
        events.append(event)
        assert 0 <= event.t_remove_start <= event.t_close_start < event.t_close_end
        assert event.t_close_end < state.frame_count
        with pytest.raises(StateLoopError, match="expected exactly one active-arm loop"):
            detect_episode_loop(state)

    assert sum(item.active_arm == "left" for item in events) == 10
    assert sum(item.active_arm == "right" for item in events) == 10
    assert {item.t_remove_start for item in events} == {4}
    assert min(item.t_close_start for item in events) == 53
    assert max(item.t_close_start for item in events) == 61
    assert min(item.t_close_end for item in events) == 65
    assert max(item.t_close_end for item in events) == 73


def test_target_only_known_episode_boundaries() -> None:
    dataset, _episode_ids = _dataset()
    state = dataset.load_state(EpisodeRef("adjust_bottle", 0, "cam_high"))

    events = detect_episode_target_only(state)

    assert events == TargetOnlyEvents("left", 4, 53, 65)
