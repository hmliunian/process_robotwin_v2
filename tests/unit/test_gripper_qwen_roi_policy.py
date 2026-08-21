from __future__ import annotations

import sys

import pytest

from robotwin_annotation_v2.application import episode_pipeline
from robotwin_annotation_v2.config import GripperRoiConfig
from robotwin_annotation_v2.pipeline.gripper.sam.annotator import (
    _roi_geometries,
    _roi_policy,
)


def _config(
    *,
    prompt_back: float = 0.120,
    prompt_front: float = 0.060,
    hard_back: float = 0.120,
    hard_front: float = 0.045,
    fixed_half_width: float = 0.085,
) -> GripperRoiConfig:
    return GripperRoiConfig(
        prompt_axial_back_m=prompt_back,
        prompt_axial_front_m=prompt_front,
        hard_axial_back_m=hard_back,
        hard_axial_front_m=hard_front,
        fixed_half_width_m=fixed_half_width,
    )


def test_final_front45_profile_keeps_prompt_front_longer_than_hard_crop() -> None:
    prompt, hard = _roi_geometries(_config())

    assert (prompt.axial_back_m, prompt.axial_front_m) == (0.120, 0.060)
    assert (hard.axial_back_m, hard.axial_front_m) == (0.120, 0.045)
    assert prompt.closed_half_width_m == prompt.open_half_width_m == 0.085
    assert hard.closed_half_width_m == hard.open_half_width_m == 0.085


def test_cli_help_has_no_roi_geometry_overrides(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_target_receiver.py", "gripper", "--help"],
    )
    with pytest.raises(SystemExit, match="0"):
        episode_pipeline.parse_args()

    help_text = capsys.readouterr().out
    assert "gripper ROI geometry" not in help_text
    assert "--roi-" not in help_text
    assert "--prompt-roi-" not in help_text
    assert "--hard-roi-" not in help_text


def test_configured_prompt_and_hard_extents_are_independent_and_traceable() -> None:
    config = _config(
        prompt_back=0.080,
        prompt_front=0.040,
        hard_back=0.065,
        hard_front=0.035,
    )

    prompt, hard = _roi_geometries(config)
    policy = _roi_policy(config)

    assert (prompt.axial_back_m, prompt.axial_front_m) == (0.080, 0.040)
    assert (hard.axial_back_m, hard.axial_front_m) == (0.065, 0.035)
    assert prompt.closed_half_width_m == hard.closed_half_width_m
    assert policy["prompt"]["usage"].startswith("SAM text-box/box-only")
    assert policy["hard"]["usage"].startswith("propagated native track crop")
    assert policy["legacy_roi_track_alias"] == "hard_roi_track"


def test_fixed_half_width_is_applied_to_both_geometries() -> None:
    config = _config(hard_front=0.060, fixed_half_width=0.085)
    prompt, hard = _roi_geometries(config)

    assert prompt.closed_half_width_m == prompt.open_half_width_m == 0.085
    assert hard.closed_half_width_m == hard.open_half_width_m == 0.085
    assert _roi_policy(config)["prompt"]["geometry"] == {
        "tcp_offset_m": 0.12,
        "axial_back_m": 0.12,
        "axial_front_m": 0.06,
        "closed_half_width_m": 0.085,
        "open_half_width_m": 0.085,
        "half_thickness_m": 0.05,
        "margin_px": 3.0,
    }
