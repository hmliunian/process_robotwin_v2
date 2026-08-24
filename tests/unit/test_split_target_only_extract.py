from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from scripts.split_target_only_extract import materialize


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _source_fixture(root: Path) -> None:
    tasks = (
        ("click_alarmclock", "contact_action_site"),
        ("press_stapler", "contact_action_site"),
        ("open_laptop", "articulated_action_site"),
        ("adjust_bottle", "single_movable_target"),
    )
    selection_tasks: list[dict[str, object]] = []
    collection_tasks: list[dict[str, object]] = []
    for index, (task, task_kind) in enumerate(tasks):
        task_root = root / task
        payload = f"{task}\n".encode()
        data_path = task_root / "data" / "sample.bin"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data_path.write_bytes(payload)
        task_manifest = {
            "format": "test_extract",
            "profile": "target_only",
            "task": task,
            "task_kind": task_kind,
            "episode_indices": [index],
            "selection_manifest": str(root / "SELECTION_MANIFEST.json"),
            "copied_files": [
                {
                    "path": "data/sample.bin",
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            ],
        }
        _write_json(task_root / "EXTRACT_MANIFEST.json", task_manifest)
        selection_tasks.append(
            {
                "task": task,
                "task_kind": task_kind,
                "episode_count": 20,
                "episode_indices": [index],
                "materialization": "copy_from_source_dataset",
            }
        )
        collection_tasks.append(
            {
                "task": task,
                "task_kind": task_kind,
                "dataset_root": str(task_root),
                "extract_manifest": str(task_root / "EXTRACT_MANIFEST.json"),
                "episode_count": 20,
            }
        )
    _write_json(
        root / "SELECTION_MANIFEST.json",
        {"scope": {"task_count": 4, "episode_count": 80}, "tasks": selection_tasks},
    )
    _write_json(
        root / "EXTRACT_MANIFEST.json",
        {
            "format": "test_collection",
            "datasets": collection_tasks,
            "task_count": 4,
            "episode_count": 80,
        },
    )


def test_materialize_contact_press_subset_rewrites_manifests(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "contact_press"
    _source_fixture(source)

    summary = materialize(source, output)

    assert summary == {"task_count": 2, "episode_count": 40}
    assert sorted(path.name for path in output.iterdir() if path.is_dir()) == [
        "click_alarmclock",
        "press_stapler",
    ]
    selection = json.loads((output / "SELECTION_MANIFEST.json").read_text())
    collection = json.loads((output / "EXTRACT_MANIFEST.json").read_text())
    assert selection["selection_kind"] == "press_subset"
    assert selection["scope"]["episode_count"] == 40
    assert collection["selection_manifest"] == str(output / "SELECTION_MANIFEST.json")
    assert all(
        record["dataset_root"] == str(output / record["task"])
        for record in collection["datasets"]
    )


def test_materialize_rejects_non_contact_task(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _source_fixture(source)

    with pytest.raises(ValueError, match="not press"):
        materialize(source, tmp_path / "output", tasks=("adjust_bottle",))
