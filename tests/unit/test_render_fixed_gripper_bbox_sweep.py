from __future__ import annotations

import argparse
import importlib
import sys
from argparse import Namespace
from pathlib import Path

import pytest


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
SWEEP_SCRIPT = importlib.import_module("render_fixed_gripper_bbox_sweep")


def test_frame_spec_parses_episode_and_frame() -> None:
    assert SWEEP_SCRIPT._frame_spec("7188:124") == (7188, 124)


@pytest.mark.parametrize("value", ["7188", "a:1", "1:-2"])
def test_frame_spec_rejects_invalid_values(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        SWEEP_SCRIPT._frame_spec(value)


def test_geometry_arguments_must_be_positive_and_unique() -> None:
    with pytest.raises(ValueError, match="unique"):
        SWEEP_SCRIPT._validate_geometry_args(
            Namespace(
                axial_back_m=(0.08, 0.08),
                axial_front_m=0.06,
                fixed_half_width_m=0.085,
            )
        )
    with pytest.raises(ValueError, match="positive"):
        SWEEP_SCRIPT._validate_geometry_args(
            Namespace(
                axial_back_m=(0.08, 0.10),
                axial_front_m=0.0,
                fixed_half_width_m=0.085,
            )
        )
