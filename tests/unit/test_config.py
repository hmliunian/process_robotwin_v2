from __future__ import annotations

from pathlib import Path

import pytest

from robotwin_annotation_v2.config import ConfigError, Sam3Config, load_config


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
output:
  root: output
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="fallback"):
        load_config(config_path)


def test_sam_config_requires_one_gpu_and_same_frame_text() -> None:
    with pytest.raises(ConfigError, match="exactly one"):
        Sam3Config(Path("sam3.pt"), gpus=(0, 1))
    with pytest.raises(ConfigError, match="same-frame"):
        Sam3Config(Path("sam3.pt"), gpus=(0,), same_frame_text=False)
