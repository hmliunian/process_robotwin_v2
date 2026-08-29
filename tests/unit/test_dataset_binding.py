from __future__ import annotations

from pathlib import Path

import pytest

from robotwin_annotation_v2.application.dataset_binding import (
    DatasetBinding,
    dataset_binding_from_target,
)
from robotwin_annotation_v2.application.dataset_input import DatasetTarget
from robotwin_annotation_v2.domain import TargetOnlyTaskKind


def _target(*, task_kind: TargetOnlyTaskKind | None = None) -> DatasetTarget:
    return DatasetTarget(
        root=Path("/dataset/task"),
        task="move_pillbottle_pad",
        camera="cam_high",
        episode_ids=(7, 8),
        task_kind=task_kind,
    )


def test_binding_uses_target_identity_and_default_episode_ids() -> None:
    binding = dataset_binding_from_target(_target())

    assert isinstance(binding, DatasetBinding)
    assert binding.root == Path("/dataset/task").resolve()
    assert binding.task == "move_pillbottle_pad"
    assert binding.camera == "cam_high"
    assert binding.episode_ids == (7, 8)


def test_explicit_episode_selection_is_deduplicated_in_order() -> None:
    binding = dataset_binding_from_target(_target(), episode_ids=[8, 7, 8, 9])

    assert binding.episode_ids == (8, 7, 9)


def test_task_kind_and_manifest_are_carried_without_aliasing() -> None:
    manifest = {"task_kind": "contact_action_site", "nested": {"value": 1}}
    binding = dataset_binding_from_target(
        _target(task_kind=TargetOnlyTaskKind.CONTACT_ACTION_SITE),
        manifest_data=manifest,
        manifest_path=Path("/dataset/EXTRACT_MANIFEST.json"),
    )

    assert binding.task_kind is TargetOnlyTaskKind.CONTACT_ACTION_SITE
    assert binding.manifest_data == manifest
    assert binding.manifest_data is not manifest
    assert binding.manifest_path == Path("/dataset/EXTRACT_MANIFEST.json").resolve()
    manifest["nested"]["value"] = 99
    assert binding.manifest_data["nested"]["value"] == 1


def test_manifest_task_kind_fills_an_older_target_without_kind() -> None:
    binding = dataset_binding_from_target(
        _target(),
        manifest_data={"task_kind": "contact_action_site"},
    )

    assert binding.task_kind is TargetOnlyTaskKind.CONTACT_ACTION_SITE


@pytest.mark.parametrize("value", ([True], ["7"], [-1]))
def test_invalid_episode_ids_fail_closed(value: list[object]) -> None:
    with pytest.raises(ValueError, match="episode_ids"):
        dataset_binding_from_target(_target(), episode_ids=value)  # type: ignore[arg-type]


def test_explicit_empty_episode_selection_is_preserved() -> None:
    # An empty explicit selection is distinct from ``None`` (which means
    # "use the target's declared IDs").  Execution workflows can reject it
    # at their boundary without changing this pure binding operation.
    binding = dataset_binding_from_target(_target(), episode_ids=[])

    assert binding.episode_ids == ()
