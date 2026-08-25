from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

DATASET_ROOT = Path("/DATA/disk8/xuran/add_mask_robotwin/dataset/target_only_20_v2")
EXPECTED_TASKS = (
    "adjust_bottle",
    "click_alarmclock",
    "click_bell",
    "move_playingcard_away",
    "open_laptop",
    "open_microwave",
    "press_stapler",
    "rotate_qrcode",
    "shake_bottle",
    "shake_bottle_horizontally",
    "turn_switch",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_target_only_v2_published_collection_matches_strict_single_arm_scope() -> None:
    selection_path = DATASET_ROOT / "SELECTION_MANIFEST.json"
    collection_path = DATASET_ROOT / "EXTRACT_MANIFEST.json"
    if not selection_path.is_file() or not collection_path.is_file():
        pytest.skip(f"external target-only v2 dataset unavailable: {DATASET_ROOT}")

    selection = _read_json(selection_path)
    collection = _read_json(collection_path)
    selection_tasks = tuple(str(item["task"]) for item in selection["tasks"])
    collection_tasks = tuple(str(item["task"]) for item in collection["datasets"])

    assert selection_tasks == EXPECTED_TASKS
    assert collection_tasks == EXPECTED_TASKS
    assert selection["scope"]["task_count"] == collection["task_count"] == 11
    assert selection["scope"]["episode_count"] == collection["episode_count"] == 220

    for task_record in selection["tasks"]:
        task = str(task_record["task"])
        task_root = DATASET_ROOT / task
        episode_ids = [int(value) for value in task_record["episode_indices"]]
        extract = _read_json(task_root / "EXTRACT_MANIFEST.json")
        metadata = [
            json.loads(line)
            for line in (task_root / "meta/episodes.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]

        assert len(episode_ids) == len(set(episode_ids)) == 20
        assert task_record["domain_counts"] == {"clean": 10, "randomized": 10}
        assert set(task_record["arm_counts"]) <= {"left", "right"}
        assert sum(int(value) for value in task_record["arm_counts"].values()) == 20
        assert extract["episode_indices"] == episode_ids
        assert [int(row["episode_index"]) for row in metadata] == episode_ids
        assert all(str(row["full_structured_tasks"][0]) == task for row in metadata)

        for episode_id in episode_ids:
            chunk = episode_id // 1000
            stem = f"episode_{episode_id:06d}"
            expected_files = (
                task_root / f"data/chunk-{chunk:03d}/{stem}.parquet",
                task_root / f"sidecars/{stem}.hdf5",
                task_root / f"videos/chunk-{chunk:03d}/observation.images.cam_high/{stem}.mp4",
                task_root
                / (f"sidecars/videos/chunk-{chunk:03d}/observation.depths.cam_high/{stem}.mkv"),
            )
            assert all(path.is_file() for path in expected_files)
