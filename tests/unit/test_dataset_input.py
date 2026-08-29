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


def test_collection_record_fallback_preserves_metadata_without_task_manifest(
    tmp_path: Path,
) -> None:
    """Use root-record selection/provenance when a child has no manifest."""

    _write_native_episode(tmp_path / "click_alarmclock", episode_id=2200)
    record = {
        "profile": "target_only",
        "task": "click_alarmclock",
        "task_kind": "contact_action_site",
        "dataset_root": "/stale/original/click_alarmclock",
        "episode_indices": [2200],
        "materialization": "copy_from_source_dataset",
    }
    (tmp_path / "EXTRACT_MANIFEST.json").write_text(
        json.dumps({"profile": "target_only", "datasets": [record]}),
        encoding="utf-8",
    )

    resolved = resolve_dataset_input(tmp_path, mode=AnnotationMode.TARGET_ONLY)

    target = resolved.targets[0]
    assert target.root == (tmp_path / "click_alarmclock").resolve()
    assert target.task == "click_alarmclock"
    assert target.episode_ids == (2200,)
    assert target.task_kind is TargetOnlyTaskKind.CONTACT_ACTION_SITE
    assert target.manifest_path is None
    assert target.manifest_data is not None
    assert target.manifest_data["materialization"] == "copy_from_source_dataset"
    # Preserve the source record for provenance; bind_dataset() later replaces
    # this stale path with the runtime root before adapters consume it.
    assert target.manifest_data["dataset_root"] == "/stale/original/click_alarmclock"
    assert target.manifest_data["profile"] == "target_only"
    assert target.manifest_data["task"] == "click_alarmclock"
    assert target.manifest_data["camera"] == "cam_high"


def test_collection_record_fallback_discovers_when_selection_is_absent(tmp_path: Path) -> None:
    _write_native_episode(tmp_path / "task", episode_id=3)
    _write_native_episode(tmp_path / "task", episode_id=7)
    (tmp_path / "EXTRACT_MANIFEST.json").write_text(
        json.dumps(
            {
                "profile": "pick_place",
                "datasets": [{"task": "task", "provenance": "legacy"}],
            }
        ),
        encoding="utf-8",
    )

    target = resolve_dataset_input(tmp_path, mode=AnnotationMode.PICK_PLACE).targets[0]

    assert target.episode_ids == (3, 7)
    assert target.manifest_data is not None
    assert target.manifest_data["provenance"] == "legacy"


def test_collection_record_profile_mismatch_is_rejected_without_child_manifest(
    tmp_path: Path,
) -> None:
    _write_native_episode(tmp_path / "task", episode_id=0)
    (tmp_path / "EXTRACT_MANIFEST.json").write_text(
        json.dumps(
            {
                "profile": "target_only",
                "datasets": [{"task": "task", "profile": "pick_place"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match --target-only"):
        resolve_dataset_input(tmp_path, mode=AnnotationMode.TARGET_ONLY)


def test_collection_without_top_level_profile_infers_homogeneous_record_profiles(
    tmp_path: Path,
) -> None:
    _write_native_episode(tmp_path / "alpha", episode_id=1)
    _write_native_episode(tmp_path / "beta", episode_id=2)
    (tmp_path / "EXTRACT_MANIFEST.json").write_text(
        json.dumps(
            {
                "datasets": [
                    {"profile": "pick_place", "task": "alpha", "episode_indices": [1]},
                    {"profile": "pick_place", "task": "beta", "episode_indices": [2]},
                ]
            }
        ),
        encoding="utf-8",
    )

    resolved = resolve_dataset_input(tmp_path, mode=AnnotationMode.PICK_PLACE)

    assert tuple(target.task for target in resolved.targets) == ("alpha", "beta")
    assert tuple(target.episode_ids for target in resolved.targets) == ((1,), (2,))


def test_collection_without_top_level_profile_requires_task_for_mixed_records(
    tmp_path: Path,
) -> None:
    _write_native_episode(tmp_path / "alpha", episode_id=1)
    _write_native_episode(tmp_path / "beta", episode_id=2)
    (tmp_path / "EXTRACT_MANIFEST.json").write_text(
        json.dumps(
            {
                "datasets": [
                    {"profile": "pick_place", "task": "alpha", "episode_indices": [1]},
                    {"profile": "target_only", "task": "beta", "episode_indices": [2]},
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="mixed profiles"):
        resolve_dataset_input(tmp_path, mode=AnnotationMode.PICK_PLACE)

    selected = resolve_dataset_input(
        tmp_path,
        mode=AnnotationMode.TARGET_ONLY,
        task="beta",
    )
    assert selected.targets[0].task == "beta"


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


def _write_native_episode(root: Path, *, episode_id: int = 0, camera: str = "cam_high") -> None:
    """Create the filesystem contract needed by manifest-free discovery."""

    for name in ("data", "videos", "sidecars", "meta"):
        (root / name).mkdir(parents=True, exist_ok=True)
    chunk = f"chunk-{episode_id // 1000:03d}"
    parquet_path = root / "data" / chunk / f"episode_{episode_id:06d}.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_path.touch()
    video_path = (
        root
        / "videos"
        / chunk
        / f"observation.images.{camera}"
        / f"episode_{episode_id:06d}.mp4"
    )
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.touch()
    (root / "sidecars" / f"episode_{episode_id:06d}.hdf5").touch()


def test_resolve_manifest_free_native_task_without_writing_manifest(tmp_path: Path) -> None:
    _write_native_episode(tmp_path, episode_id=7)

    resolved = resolve_dataset_input(tmp_path, mode=AnnotationMode.PICK_PLACE)

    assert not resolved.is_collection
    target = resolved.targets[0]
    assert target.task == tmp_path.name
    assert target.camera == "cam_high"
    assert target.episode_ids == (7,)
    assert target.manifest_path is None
    assert target.manifest_data is not None
    assert not (tmp_path / "EXTRACT_MANIFEST.json").exists()


def test_resolve_manifest_free_native_collection_uses_task_children(tmp_path: Path) -> None:
    _write_native_episode(tmp_path / "beta", episode_id=2)
    _write_native_episode(tmp_path / "alpha", episode_id=1)

    resolved = resolve_dataset_input(tmp_path, mode=AnnotationMode.PICK_PLACE)

    assert resolved.is_collection
    assert tuple(target.task for target in resolved.targets) == ("alpha", "beta")


def test_resolve_manifest_free_rejects_ambiguous_camera(tmp_path: Path) -> None:
    _write_native_episode(tmp_path, episode_id=0, camera="cam_high")
    _write_native_episode(tmp_path, episode_id=0, camera="cam_left")

    with pytest.raises(ValueError, match="camera is ambiguous"):
        resolve_dataset_input(tmp_path, mode=AnnotationMode.PICK_PLACE)

    selected = resolve_dataset_input(
        tmp_path,
        mode=AnnotationMode.PICK_PLACE,
        camera="cam_left",
    )
    assert selected.targets[0].camera == "cam_left"


def test_resolve_manifest_free_rejects_direct_data_and_task_children(tmp_path: Path) -> None:
    _write_native_episode(tmp_path, episode_id=0)
    _write_native_episode(tmp_path / "nested_task", episode_id=1)

    with pytest.raises(ValueError, match="dataset path is ambiguous"):
        resolve_dataset_input(tmp_path, mode=AnnotationMode.PICK_PLACE)


def test_resolve_manifest_free_rejects_raw_mcap_with_conversion_hint(tmp_path: Path) -> None:
    (tmp_path / "capture.mcap").write_bytes(b"not converted")

    with pytest.raises(ValueError, match="run just convert-real INPUT_ROOT OUTPUT_ROOT first"):
        resolve_dataset_input(tmp_path, mode=AnnotationMode.PICK_PLACE)


def test_resolve_task_manifest_falls_back_to_regression_episode_ids(tmp_path: Path) -> None:
    _write_task(tmp_path, "task")
    manifest_path = tmp_path / "EXTRACT_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("episode_indices")
    manifest["regression_episode_ids"] = [4, 9]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    target = resolve_dataset_input(tmp_path, mode=AnnotationMode.TARGET_ONLY).targets[0]

    assert target.episode_ids == (4, 9)


def test_resolve_task_manifest_falls_back_to_episode_ids_alias(tmp_path: Path) -> None:
    _write_task(tmp_path, "task")
    manifest_path = tmp_path / "EXTRACT_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("episode_indices")
    manifest["episode_ids"] = [4, 9]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    target = resolve_dataset_input(tmp_path, mode=AnnotationMode.TARGET_ONLY).targets[0]

    assert target.episode_ids == (4, 9)


def test_resolve_task_manifest_rejects_negative_episode_ids(tmp_path: Path) -> None:
    _write_task(tmp_path, "task")
    manifest_path = tmp_path / "EXTRACT_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["episode_indices"] = [-1]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="must be non-negative"):
        resolve_dataset_input(tmp_path, mode=AnnotationMode.TARGET_ONLY)


def test_collection_manifest_does_not_auto_include_unlisted_task(tmp_path: Path) -> None:
    _write_task(tmp_path / "listed", "listed")
    _write_task(tmp_path / "pick_and_place_real", "pick_and_place_real")
    (tmp_path / "EXTRACT_MANIFEST.json").write_text(
        json.dumps(
            {
                "profile": "target_only",
                "datasets": [{"task": "listed"}],
            }
        ),
        encoding="utf-8",
    )

    resolved = resolve_dataset_input(tmp_path, mode=AnnotationMode.TARGET_ONLY)

    assert tuple(target.task for target in resolved.targets) == ("listed",)


@pytest.mark.parametrize("field", ("task", "camera"))
def test_task_manifest_rejects_path_like_identity(tmp_path: Path, field: str) -> None:
    _write_task(tmp_path, "task")
    manifest_path = tmp_path / "EXTRACT_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = "../escape"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid task/camera"):
        resolve_dataset_input(tmp_path, mode=AnnotationMode.TARGET_ONLY)


@pytest.mark.parametrize("value", ("../escape", r"a\b", "/absolute/task"))
def test_collection_manifest_rejects_path_like_task_record(
    tmp_path: Path,
    value: str,
) -> None:
    (tmp_path / "EXTRACT_MANIFEST.json").write_text(
        json.dumps(
            {
                "profile": "target_only",
                "datasets": [{"task": value}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid task record"):
        resolve_dataset_input(tmp_path, mode=AnnotationMode.TARGET_ONLY)


def test_manifest_free_native_task_rejects_explicit_empty_identity(
    tmp_path: Path,
) -> None:
    _write_native_episode(tmp_path, episode_id=0)
    with pytest.raises(ValueError, match="non-empty name"):
        resolve_dataset_input(
            tmp_path,
            mode=AnnotationMode.PICK_PLACE,
            task="",
            camera="",
        )


def test_legacy_single_task_manifest_is_bound_in_memory(tmp_path: Path) -> None:
    """Old coverage extracts remain usable without generating a task config."""

    _write_native_episode(tmp_path, episode_id=7152)
    (tmp_path / "meta" / "episodes.jsonl").write_text(
        json.dumps({"full_structured_tasks": ["move_pillbottle_pad", "text", "success"]})
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "format": "robotwin_sparse_original_extract_v1",
        "episode_indices": [7152],
        "cameras": ["cam_high", "cam_left_wrist"],
    }
    manifest_path = tmp_path / "EXTRACT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    target = resolve_dataset_input(tmp_path, mode=AnnotationMode.PICK_PLACE).targets[0]

    assert target.task == "move_pillbottle_pad"
    assert target.camera == "cam_high"
    assert target.episode_ids == (7152,)
    assert target.manifest_path == manifest_path.resolve()
    assert target.manifest_data is not None
    assert target.manifest_data["profile"] == "pick_place"
    assert target.manifest_data["task"] == "move_pillbottle_pad"


def test_legacy_single_task_manifest_honors_explicit_camera_and_task(
    tmp_path: Path,
) -> None:
    _write_native_episode(tmp_path, episode_id=0)
    manifest_path = tmp_path / "EXTRACT_MANIFEST.json"
    manifest_path.write_text(
        json.dumps({"episode_indices": [0], "cameras": ["cam_high", "cam_left_wrist"]}),
        encoding="utf-8",
    )

    target = resolve_dataset_input(
        tmp_path,
        mode=AnnotationMode.PICK_PLACE,
        task="explicit_task",
        camera="cam_left_wrist",
    ).targets[0]

    assert target.task == "explicit_task"
    assert target.camera == "cam_left_wrist"


def test_collection_manifest_rejects_symlink_task_directory(tmp_path: Path) -> None:
    task = tmp_path / "task"
    outside = tmp_path / "outside"
    _write_task(outside, "task")
    try:
        task.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    (tmp_path / "EXTRACT_MANIFEST.json").write_text(
        json.dumps({"profile": "target_only", "datasets": [{"task": "task"}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must not be a symlink"):
        resolve_dataset_input(tmp_path, mode=AnnotationMode.TARGET_ONLY)


def test_manifest_free_collection_rejects_symlink_task_directory(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    _write_native_episode(outside, episode_id=1)
    try:
        (tmp_path / "task").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable")

    with pytest.raises(ValueError, match="must not be a symlink"):
        resolve_dataset_input(tmp_path, mode=AnnotationMode.PICK_PLACE)
