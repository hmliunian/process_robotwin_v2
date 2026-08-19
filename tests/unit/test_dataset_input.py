from __future__ import annotations

import json
from pathlib import Path

import pytest

from robotwin_annotation_v2.application.dataset_input import resolve_dataset_input
from robotwin_annotation_v2.domain import AnnotationMode


def _write_task(root: Path, task: str, *, profile: str = "target_only") -> None:
    for name in ("data", "videos", "sidecars", "meta"):
        (root / name).mkdir(parents=True, exist_ok=True)
    (root / "EXTRACT_MANIFEST.json").write_text(
        json.dumps(
            {
                "profile": profile,
                "task": task,
                "camera": "cam_high",
                "episode_indices": [1, 2],
            }
        ),
        encoding="utf-8",
    )


def test_resolve_single_task_dataset(tmp_path: Path) -> None:
    _write_task(tmp_path, "adjust_bottle")

    resolved = resolve_dataset_input(tmp_path, mode=AnnotationMode.TARGET_ONLY)

    assert not resolved.is_collection
    assert resolved.targets[0].root == tmp_path.resolve()
    assert resolved.targets[0].task == "adjust_bottle"
    assert resolved.targets[0].camera == "cam_high"
    assert resolved.targets[0].episode_ids == (1, 2)


def test_resolve_collection_uses_local_children_and_selects_task(tmp_path: Path) -> None:
    _write_task(tmp_path / "beta", "beta")
    _write_task(tmp_path / "alpha", "alpha")
    (tmp_path / "EXTRACT_MANIFEST.json").write_text(
        json.dumps(
            {
                "profile": "target_only",
                "datasets": [
                    {"task": "beta", "dataset_root": "/stale/beta"},
                    {"task": "alpha", "dataset_root": "/stale/alpha"},
                ],
            }
        ),
        encoding="utf-8",
    )

    all_tasks = resolve_dataset_input(tmp_path, mode=AnnotationMode.TARGET_ONLY)
    selected = resolve_dataset_input(
        tmp_path,
        mode=AnnotationMode.TARGET_ONLY,
        task="beta",
    )

    assert tuple(target.task for target in all_tasks.targets) == ("alpha", "beta")
    assert selected.targets[0].root == (tmp_path / "beta").resolve()


def test_resolve_rejects_profile_mismatch(tmp_path: Path) -> None:
    _write_task(tmp_path, "place_container_plate", profile="pick_place")

    with pytest.raises(ValueError, match="does not match --target-only"):
        resolve_dataset_input(tmp_path, mode=AnnotationMode.TARGET_ONLY)
