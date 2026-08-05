from __future__ import annotations

from pathlib import Path

import pytest

from robotwin_annotation_v2.config import (
    ConfigError,
    GripperRoiConfig,
    MaskConfig,
    Sam3Config,
    load_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_pilot_config_loads_new_pipeline_contract() -> None:
    config = load_config(PROJECT_ROOT / "configs/pilot_move_pillbottle_pad.yaml")

    assert config.dataset.task == "move_pillbottle_pad"
    assert config.dataset.camera == "cam_high"
    assert config.dataset.smoke_episode_ids == (7152,)
    assert len(config.dataset.regression_episode_ids) == 20
    assert config.qwen.query_selection == "first_recommended"
    assert not config.qwen.allow_query_fallback
    assert config.qwen.prompt_template.is_file()
    assert config.dataset.manifest.is_file()
    assert config.sam3.checkpoint.name == "sam3.pt"
    assert config.sam3.gpus == (2,)
    assert config.mask.qc_enabled
    assert config.mask.qc_prompt_template is not None
    assert config.mask.qc_prompt_template.is_file()
    assert config.mask.qc_max_candidates == 3
    assert config.mask.qc_max_attempts == 2
    assert config.gripper_roi == GripperRoiConfig(
        prompt_axial_back_m=0.120,
        prompt_axial_front_m=0.060,
        hard_axial_back_m=0.120,
        hard_axial_front_m=0.045,
        fixed_half_width_m=0.085,
    )


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
