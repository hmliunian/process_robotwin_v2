from __future__ import annotations

import json
from pathlib import Path

import pytest

from robotwin_annotation_v2.config import (
    AnnotationConfig,
    ConfigError,
    GripperRoiConfig,
    MaskConfig,
    Sam3Config,
    load_config,
)
from robotwin_annotation_v2.domain import (
    AnnotationMode,
    GripperBackend,
    ObjectRole,
    annotation_spec,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_pilot_config_loads_new_pipeline_contract() -> None:
    config = load_config(PROJECT_ROOT / "configs/pilot_move_pillbottle_pad.yaml")

    assert config.dataset.task == "move_pillbottle_pad"
    assert config.annotation == AnnotationConfig(AnnotationMode.PICK_PLACE)
    assert config.annotation.spec.required_object_roles == (
        ObjectRole.TARGET,
        ObjectRole.RECEIVER,
    )
    assert config.annotation.spec.default_gripper_backend is GripperBackend.URDF
    assert config.dataset.camera == "cam_high"
    assert config.dataset.smoke_episode_ids == (7152,)
    assert len(config.dataset.regression_episode_ids) == 20
    assert config.qwen.query_selection == "first_recommended"
    assert not config.qwen.allow_query_fallback
    assert config.qwen.prompt_template.name == "target_receiver_semantic_open_set.txt"
    assert config.qwen.timeout_seconds == 600
    assert config.qwen.prompt_template.is_file()
    assert config.dataset.manifest.is_file()
    assert config.sam3.checkpoint.name == "sam3.pt"
    assert config.sam3.gpus == (2,)
    assert config.mask.qc_enabled
    assert config.mask.qc_prompt_template is not None
    assert config.mask.qc_prompt_template.is_file()
    assert config.mask.qc_max_candidates == 8
    assert config.mask.qc_max_attempts == 2
    assert config.mask.qc_query_fallback_enabled
    assert config.mask.qc_seed_fallback_enabled
    assert config.mask.qc_bbox_fallback_enabled
    assert config.mask.qc_bbox_prompt_template is not None
    assert config.mask.qc_bbox_prompt_template.name == "open_set_bbox_localization.txt"
    assert config.mask.qc_bbox_prompt_template.is_file()
    assert config.mask.qc_bbox_max_tokens == 180
    assert config.gripper_roi == GripperRoiConfig(
        prompt_axial_back_m=0.120,
        prompt_axial_front_m=0.060,
        hard_axial_back_m=0.120,
        hard_axial_front_m=0.045,
        fixed_half_width_m=0.085,
    )


def test_place_container_plate_config_pins_depth_complete_subset() -> None:
    config = load_config(PROJECT_ROOT / "configs/pilot_place_container_plate.yaml")
    manifest = json.loads(config.dataset.manifest.read_text(encoding="utf-8"))

    assert config.dataset.task == "place_container_plate"
    assert config.dataset.smoke_episode_ids == (14850,)
    assert len(config.dataset.regression_episode_ids) == 547
    assert config.dataset.regression_episode_ids == tuple(manifest["regression_episode_ids"])
    assert {int(value) for value in manifest["excluded_source_episodes"]} == {
        14941,
        15022,
        15360,
    }


def test_target_only_pilot_config_pins_close_and_hold_dataset() -> None:
    config = load_config(PROJECT_ROOT / "configs/pilot_adjust_bottle_target_only.yaml")
    manifest = json.loads(config.dataset.manifest.read_text(encoding="utf-8"))

    assert config.annotation == AnnotationConfig(AnnotationMode.TARGET_ONLY)
    assert config.annotation.spec.required_object_roles == (ObjectRole.TARGET,)
    assert config.annotation.spec.default_gripper_backend is GripperBackend.URDF
    assert config.dataset.task == "adjust_bottle"
    assert config.dataset.smoke_episode_ids == (0,)
    assert len(config.dataset.regression_episode_ids) == 20
    assert config.dataset.regression_episode_ids == tuple(manifest["regression_episode_ids"])
    assert config.qwen.prompt_template.name == "target_only_semantic_open_set.txt"
    assert config.qwen.timeout_seconds == 600
    assert config.qwen.max_tokens == 400
    assert config.mask.qc_prompt_template is not None
    assert config.mask.qc_prompt_template.name == (
        "target_only_mask_candidate_qc_open_set.txt"
    )
    assert config.mask.qc_max_candidates == 8
    assert config.mask.qc_query_fallback_enabled
    assert config.mask.qc_seed_fallback_enabled
    assert config.mask.qc_bbox_fallback_enabled
    assert config.mask.qc_bbox_prompt_template is not None
    assert config.mask.qc_bbox_prompt_template.name == "open_set_bbox_localization.txt"


def test_open_set_bbox_experiment_explicitly_enables_bbox_fallback() -> None:
    config = load_config(PROJECT_ROOT / "configs/open_set_mask_fallback_bbox.yaml")

    assert config.mask.qc_bbox_fallback_enabled
    assert config.mask.qc_bbox_prompt_template is not None
    assert config.mask.qc_bbox_prompt_template.name == "open_set_bbox_localization.txt"
    assert config.mask.qc_bbox_prompt_template.is_file()
    assert config.mask.qc_bbox_max_tokens == 180


def test_bbox_fallback_requires_qc_and_an_explicit_prompt() -> None:
    with pytest.raises(ConfigError, match="requires mask QC"):
        MaskConfig(qc_bbox_fallback_enabled=True)

    with pytest.raises(ConfigError, match="qc_bbox_prompt_template"):
        MaskConfig(
            qc_enabled=True,
            qc_prompt_template=Path("mask-qc.txt"),
            qc_bbox_fallback_enabled=True,
        )


@pytest.mark.parametrize(
    "field",
    [
        "temporal_envelope_guard_retry_enabled",
        "qc_bbox_directional_expand_enabled",
        "qc_border_retry_enabled",
        "qc_border_retry_prompt_template",
        "qc_border_retry_max_tokens",
    ],
)
def test_config_rejects_removed_s4_fields(tmp_path: Path, field: str) -> None:
    source = (PROJECT_ROOT / "configs/open_set_mask_fallback_bbox.yaml").read_text(encoding="utf-8")
    config_path = tmp_path / "removed-s4.yaml"
    config_path.write_text(source.replace("mask:\n", f"mask:\n  {field}: true\n"), encoding="utf-8")

    with pytest.raises(ConfigError, match="removed S4"):
        load_config(config_path)


@pytest.mark.parametrize(
    "field",
    ["qc_query_fallback_enabled", "qc_seed_fallback_enabled"],
)
def test_text_fallbacks_require_qc(field: str) -> None:
    with pytest.raises(ConfigError, match="requires mask QC"):
        MaskConfig(**{field: True})


def test_config_rejects_automatic_query_fallback(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        """
dataset:
  root: /tmp/data
  manifest: manifest.json
  task: task
  camera: cam_high
  smoke_episode_ids: [1]
  regression_episode_ids: [1]
qwen:
  endpoint: http://127.0.0.1:1/v1/chat/completions
  model: qwen
  prompt_template: prompt.txt
  allow_query_fallback: true
sam3:
  checkpoint: sam3.pt
gripper_roi:
  prompt:
    axial_back_m: 0.120
    axial_front_m: 0.060
  hard:
    axial_back_m: 0.120
    axial_front_m: 0.045
  fixed_half_width_m: 0.085
output:
  root: output
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="fallback"):
        load_config(config_path)


def test_annotation_specs_only_declare_roles_and_backend() -> None:
    pick_place = annotation_spec(AnnotationMode.PICK_PLACE)
    target_only = annotation_spec(AnnotationMode.TARGET_ONLY)

    assert pick_place.required_object_roles == (ObjectRole.TARGET, ObjectRole.RECEIVER)
    assert target_only.required_object_roles == (ObjectRole.TARGET,)
    assert target_only.canonical_object_roles == pick_place.canonical_object_roles
    assert not target_only.requires(ObjectRole.RECEIVER)
    assert target_only.default_gripper_backend is GripperBackend.URDF


def test_config_rejects_unknown_annotation_mode(tmp_path: Path) -> None:
    source = (PROJECT_ROOT / "configs/pilot_move_pillbottle_pad.yaml").read_text(encoding="utf-8")
    config_path = tmp_path / "bad-mode.yaml"
    config_path.write_text(source.replace("mode: pick_place", "mode: mystery"), encoding="utf-8")

    with pytest.raises(ConfigError, match="annotation.mode"):
        load_config(config_path)


def test_sam_config_requires_one_gpu() -> None:
    with pytest.raises(ConfigError, match="exactly one"):
        Sam3Config(Path("sam3.pt"), gpus=(0, 1))


def test_temporal_qc_thresholds_are_validated() -> None:
    with pytest.raises(ConfigError, match="IoU"):
        MaskConfig(temporal_qc_min_adjacent_iou_p05=1.1)
    with pytest.raises(ConfigError, match="signal count"):
        MaskConfig(temporal_qc_quarantine_signal_count=4)


@pytest.mark.parametrize("value", [0.0, -0.1, float("nan"), float("inf")])
def test_gripper_roi_requires_positive_finite_values(value: float) -> None:
    with pytest.raises(ConfigError, match="greater than zero"):
        GripperRoiConfig(
            prompt_axial_back_m=value,
            prompt_axial_front_m=0.060,
            hard_axial_back_m=0.120,
            hard_axial_front_m=0.045,
            fixed_half_width_m=0.085,
        )
