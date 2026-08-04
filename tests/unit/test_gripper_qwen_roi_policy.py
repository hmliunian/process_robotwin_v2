from __future__ import annotations

import argparse
import importlib
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from robotwin_annotation_v2.experiments import DEFAULT_GRIPPER_ROI_GEOMETRY


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
QWEN_SCRIPT = importlib.import_module("generate_gripper_mask_video_qwen_qc")


def _args(
    *,
    prompt_back: float = 0.025,
    prompt_front: float = 0.060,
    hard_back: float = 0.025,
    hard_front: float = 0.060,
    fixed_half_width: float | None = None,
) -> Namespace:
    return Namespace(
        prompt_roi_axial_back_m=prompt_back,
        prompt_roi_axial_front_m=prompt_front,
        hard_roi_axial_back_m=hard_back,
        hard_roi_axial_front_m=hard_front,
        roi_fixed_half_width_m=fixed_half_width,
    )


def test_default_prompt_and_hard_roi_policy_preserves_single_roi_behavior() -> None:
    prompt, hard = QWEN_SCRIPT._roi_geometries(_args())

    assert prompt == DEFAULT_GRIPPER_ROI_GEOMETRY
    assert hard == DEFAULT_GRIPPER_ROI_GEOMETRY
    assert QWEN_SCRIPT._is_default_roi_policy(QWEN_SCRIPT._roi_policy(_args()))


def test_prompt_and_hard_axial_extents_are_independent_and_traceable() -> None:
    args = _args(
        prompt_back=0.080,
        prompt_front=0.040,
        hard_back=0.065,
        hard_front=0.035,
    )

    prompt, hard = QWEN_SCRIPT._roi_geometries(args)
    policy = QWEN_SCRIPT._roi_policy(args)

    assert (prompt.axial_back_m, prompt.axial_front_m) == (0.080, 0.040)
    assert (hard.axial_back_m, hard.axial_front_m) == (0.065, 0.035)
    assert prompt.closed_half_width_m == hard.closed_half_width_m
    assert policy["prompt"]["usage"] == "SAM box and selected-seed crop"
    assert policy["hard"]["usage"].startswith("per-frame propagated-track crop")
    assert policy["legacy_roi_track_alias"] == "hard_roi_track"
    assert not QWEN_SCRIPT._is_default_roi_policy(policy)


def test_fixed_half_width_disables_gripper_opening_interpolation() -> None:
    args = _args(
        prompt_back=0.120,
        prompt_front=0.060,
        hard_back=0.120,
        hard_front=0.060,
        fixed_half_width=0.085,
    )

    prompt, hard = QWEN_SCRIPT._roi_geometries(args)

    assert prompt.closed_half_width_m == prompt.open_half_width_m == 0.085
    assert hard.closed_half_width_m == hard.open_half_width_m == 0.085
    assert QWEN_SCRIPT._roi_policy(args)["prompt"]["geometry"] == {
        "tcp_offset_m": 0.12,
        "axial_back_m": 0.12,
        "axial_front_m": 0.06,
        "closed_half_width_m": 0.085,
        "open_half_width_m": 0.085,
        "half_thickness_m": 0.05,
        "margin_px": 3.0,
    }


@pytest.mark.parametrize("value", ["0", "-0.1", "nan", "inf", "-inf"])
def test_roi_cli_rejects_non_positive_or_non_finite_extents(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="greater than zero"):
        QWEN_SCRIPT._positive_finite_float(value)
