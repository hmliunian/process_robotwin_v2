from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from robotwin_annotation_v2.application import sam_workflow
from robotwin_annotation_v2.application.sam_workflow import (
    build_sam_dynamic_config,
    capture_sam_stage_output,
    default_sam_workflow_hooks,
    load_sam_runtime,
    read_json_object,
    render_sam_processed,
    summary_gripper_backend,
    validate_sam_run_id,
    validate_sam_run_ownership,
)
from robotwin_annotation_v2.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_default_hooks_bind_the_canonical_sam_support_api() -> None:
    hooks = default_sam_workflow_hooks()

    assert hooks.runtime_loader is load_sam_runtime
    assert hooks.build_dynamic_config is build_sam_dynamic_config
    assert hooks.capture_stage_output is capture_sam_stage_output
    assert hooks.render_processed is render_sam_processed
    assert hooks.validate_run_id is validate_sam_run_id
    assert hooks.validate_run_ownership is validate_sam_run_ownership

    expected_exports = {
        "build_sam_dynamic_config",
        "capture_sam_stage_output",
        "load_sam_runtime",
        "read_json_object",
        "render_sam_processed",
        "summary_gripper_backend",
        "validate_sam_run_id",
        "validate_sam_run_ownership",
    }
    assert expected_exports <= set(sam_workflow.__all__)


def test_default_hooks_keep_model_and_renderer_integrations_lazy() -> None:
    source = """
import sys
from robotwin_annotation_v2.application import sam_workflow

hooks = sam_workflow.default_sam_workflow_hooks()
assert hooks.runtime_loader is sam_workflow.load_sam_runtime
assert hooks.render_processed is sam_workflow.render_sam_processed
for name in (
    "robotwin_annotation_v2.adapters.qwen_client",
    "robotwin_annotation_v2.adapters.sam3_adapter",
    "robotwin_annotation_v2.adapters.rendering",
    "robotwin_annotation_v2.adapters.robotwin_dataset",
    "robotwin_annotation_v2.application.episode_pipeline",
):
    assert name not in sys.modules, name
"""
    completed = subprocess.run(
        [sys.executable, "-c", source],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_capture_sam_stage_output_forwards_captured_text() -> None:
    reporter = Mock()

    with capture_sam_stage_output(reporter):
        print("stage payload")

    reporter.detail.assert_called_once_with("stage payload")


def test_capture_sam_stage_output_leaves_stdout_visible_without_reporter(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with capture_sam_stage_output(None):
        print("stage payload")

    assert capsys.readouterr().out == "stage payload\n"


def test_build_sam_dynamic_config_binds_discovered_dataset(tmp_path: Path) -> None:
    config = load_config(PROJECT_ROOT / "configs/pilot_move_pillbottle_pad.yaml")
    manifest = {
        "smoke_episode_ids": [3],
        "regression_episode_ids": [3, 8],
        "episodes": [{"episode_index": 3}, {"episode_index": 8}],
    }
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "runs"

    dynamic = build_sam_dynamic_config(
        config,
        root=dataset_root,
        task="move_cube_pad",
        camera="cam_left",
        manifest=manifest,
        output_root=output_root,
    )

    assert dynamic.dataset.root == dataset_root.resolve()
    assert dynamic.dataset.task == "move_cube_pad"
    assert dynamic.dataset.camera == "cam_left"
    assert dynamic.dataset.smoke_episode_ids == (3,)
    assert dynamic.dataset.regression_episode_ids == (3, 8)
    assert dynamic.dataset.manifest_data is manifest
    assert dynamic.output_root == output_root.resolve()
    assert config.dataset.task == "move_pillbottle_pad"


@pytest.mark.parametrize(
    "run_id",
    ["", " ", ".", "..", "../run", "run/name", "run\\name", "run..name"],
)
def test_validate_sam_run_id_rejects_non_directory_names(run_id: str) -> None:
    with pytest.raises(ValueError, match="simple non-empty directory name"):
        validate_sam_run_id(run_id)


def test_validate_sam_run_id_returns_valid_name() -> None:
    assert validate_sam_run_id("sam-source-001") == "sam-source-001"


def test_read_json_object_reads_mapping_and_rejects_other_json(tmp_path: Path) -> None:
    object_path = tmp_path / "object.json"
    object_path.write_text('{"run_id": "sam-001"}', encoding="utf-8")
    array_path = tmp_path / "array.json"
    array_path.write_text("[]", encoding="utf-8")

    assert read_json_object(object_path, description="summary") == {
        "run_id": "sam-001"
    }
    with pytest.raises(ValueError, match="must contain one JSON object"):
        read_json_object(array_path, description="summary")
    with pytest.raises(FileNotFoundError, match="summary is missing"):
        read_json_object(tmp_path / "missing.json", description="summary")


@pytest.mark.parametrize(
    ("summary", "expected"),
    [
        ({}, None),
        ({"gripper_backend": "sam"}, "sam"),
        ({"backend": {"gripper": "sam"}}, "sam"),
        ({"backend": {"type": "urdf"}}, "urdf"),
    ],
)
def test_summary_gripper_backend_supports_both_summary_layouts(
    summary: dict[str, object],
    expected: str | None,
) -> None:
    assert summary_gripper_backend(summary) == expected


def test_summary_gripper_backend_rejects_conflicting_ownership() -> None:
    with pytest.raises(ValueError, match="conflicting backend ownership"):
        summary_gripper_backend(
            {"gripper_backend": "sam", "backend": {"gripper": "urdf"}}
        )


def test_validate_sam_run_ownership_accepts_unowned_and_sam_runs(
    tmp_path: Path,
) -> None:
    empty_run = tmp_path / "empty"
    validate_sam_run_ownership(empty_run, run_id="empty")

    sam_run = tmp_path / "sam-001"
    sam_run.mkdir()
    (sam_run / "process_summary.json").write_text(
        '{"run_id": "sam-001", "gripper_backend": "sam"}',
        encoding="utf-8",
    )
    validate_sam_run_ownership(sam_run, run_id="sam-001")


def test_validate_sam_run_ownership_rejects_foreign_runs(tmp_path: Path) -> None:
    urdf_marker_run = tmp_path / "marked"
    (urdf_marker_run / "_backend" / "urdf").mkdir(parents=True)
    with pytest.raises(ValueError, match="owned by the URDF backend"):
        validate_sam_run_ownership(urdf_marker_run, run_id="marked")

    mismatched_run = tmp_path / "mismatch"
    mismatched_run.mkdir()
    (mismatched_run / "process_summary.json").write_text(
        '{"run_id": "somewhere-else", "gripper_backend": "sam"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="run_id does not match"):
        validate_sam_run_ownership(mismatched_run, run_id="mismatch")

    foreign_backend_run = tmp_path / "foreign"
    foreign_backend_run.mkdir()
    (foreign_backend_run / "process_summary.json").write_text(
        '{"run_id": "foreign", "gripper_backend": "urdf"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="owned by backend 'urdf', not SAM"):
        validate_sam_run_ownership(foreign_backend_run, run_id="foreign")
