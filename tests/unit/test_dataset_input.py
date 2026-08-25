from __future__ import annotations

import json
from pathlib import Path

import pytest

from robotwin_annotation_v2.application.dataset_input import (
    read_dataset_task_kind,
    resolve_dataset_input,
)
from robotwin_annotation_v2.domain import (
    AnnotationMode,
    TargetOnlyTaskKind,
    TargetProfile,
    target_profile_for_task_kind,
)


def _write_task(
    root: Path,
    task: str,
    *,
    profile: str = "target_only",
    task_kind: str | None = None,
) -> None:
    for name in ("data", "videos", "sidecars", "meta"):
        (root / name).mkdir(parents=True, exist_ok=True)
    manifest = {
        "profile": profile,
        "task": task,
        "camera": "cam_high",
        "episode_indices": [1, 2],
    }
    if task_kind is not None:
        manifest["task_kind"] = task_kind
    (root / "EXTRACT_MANIFEST.json").write_text(
        json.dumps(manifest),
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
    assert resolved.targets[0].task_kind is None
    assert resolved.targets[0].profile is TargetProfile.GRASP_MANIPULATION


@pytest.mark.parametrize(
    ("task_kind", "expected_profile"),
    (
        ("single_movable_target", TargetProfile.GRASP_MANIPULATION),
        ("single_movable_target_conditional", TargetProfile.GRASP_MANIPULATION),
        ("contact_action_site", TargetProfile.CONTACT_PRESS),
        ("articulated_action_site", TargetProfile.GRASP_MANIPULATION),
    ),
)
def test_target_only_task_kind_selects_profile(
    tmp_path: Path,
    task_kind: str,
    expected_profile: TargetProfile,
) -> None:
    _write_task(tmp_path, "task", task_kind=task_kind)

    target = resolve_dataset_input(
        tmp_path,
        mode=AnnotationMode.TARGET_ONLY,
    ).targets[0]

    assert target.task_kind is TargetOnlyTaskKind(task_kind)
    assert target.profile is expected_profile
    assert read_dataset_task_kind(tmp_path) is TargetOnlyTaskKind(task_kind)


def test_target_only_v2_selection_routes_only_three_contact_tasks() -> None:
    selection_path = (
        Path(__file__).resolve().parents[2] / "configs/datasets/target_only_20_v2_selection.json"
    )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))

    contact_tasks = {
        str(record["task"])
        for record in selection["tasks"]
        if target_profile_for_task_kind(record["task_kind"]) is TargetProfile.CONTACT_PRESS
    }

    assert contact_tasks == {"click_alarmclock", "click_bell", "press_stapler"}


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


def test_resolve_rejects_unknown_target_only_task_kind(tmp_path: Path) -> None:
    _write_task(tmp_path, "task", task_kind="mystery")

    with pytest.raises(ValueError, match="unsupported task_kind"):
        resolve_dataset_input(tmp_path, mode=AnnotationMode.TARGET_ONLY)


def test_contact_task_kind_is_invalid_for_pick_place_extract(tmp_path: Path) -> None:
    _write_task(
        tmp_path,
        "task",
        profile="pick_place",
        task_kind="contact_action_site",
    )

    with pytest.raises(ValueError, match="requires target_only"):
        resolve_dataset_input(tmp_path, mode=AnnotationMode.PICK_PLACE)
