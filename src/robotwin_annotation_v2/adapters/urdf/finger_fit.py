"""Pure depth evidence and ranking primitives for Aloha finger fitting."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating[Any]]
BoolArray = NDArray[np.bool_]
ArmSide = Literal["left", "right"]

GRIPPER_DRIVE_CLOSED_M = -0.01
GRIPPER_DRIVE_SPAN_M = 0.055
GRIPPER_KINEMATIC_MIN_M = 0.0
GRIPPER_KINEMATIC_MAX_M = 0.04765


@dataclass(frozen=True)
class DepthAgreement:
    """Robust rendered-vs-recorded depth statistics for a projected mask."""

    rendered_pixels: int
    comparable_pixels: int
    consistent_pixels: int
    consistent_fraction: float
    median_residual_mm: float | None
    p90_residual_mm: float | None
    median_signed_residual_mm: float | None = None
    rendered_in_front_pixels: int = 0
    rendered_behind_pixels: int = 0

    def as_dict(self) -> dict[str, int | float | None]:
        return {
            "rendered_pixels": self.rendered_pixels,
            "comparable_pixels": self.comparable_pixels,
            "consistent_pixels": self.consistent_pixels,
            "consistent_fraction": self.consistent_fraction,
            "median_residual_mm": self.median_residual_mm,
            "p90_residual_mm": self.p90_residual_mm,
            "median_signed_residual_mm": self.median_signed_residual_mm,
            "rendered_in_front_pixels": self.rendered_in_front_pixels,
            "rendered_behind_pixels": self.rendered_behind_pixels,
        }


@dataclass(frozen=True)
class FingerCandidateScore:
    """Evidence for one joint-q hypothesis shared by link7 and link8."""

    q_m: float
    agreement: DepthAgreement
    link7_agreement: DepthAgreement
    link8_agreement: DepthAgreement
    joint_name: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "q_m": self.q_m,
            "agreement": self.agreement.as_dict(),
            "link7_agreement": self.link7_agreement.as_dict(),
            "link8_agreement": self.link8_agreement.as_dict(),
            "joint_name": self.joint_name,
        }


def gripper_command_to_drive_target(command: float) -> float:
    """Map normalized RoboTwin command to the simulator drive target in metres."""

    result = GRIPPER_DRIVE_CLOSED_M + float(command) * GRIPPER_DRIVE_SPAN_M
    if not math.isfinite(result):
        raise ValueError("gripper command must be finite")
    return result


def gripper_command_to_kinematic_q(command: float) -> float:
    """Map and clamp a command to the representable URDF prismatic range."""

    return float(
        np.clip(
            gripper_command_to_drive_target(command),
            GRIPPER_KINEMATIC_MIN_M,
            GRIPPER_KINEMATIC_MAX_M,
        )
    )


def _validate_depth_inputs(
    mask: BoolArray | NDArray[Any],
    rendered_depth_mm: FloatArray | NDArray[Any],
    scene_depth_mm: FloatArray | NDArray[Any],
    tolerance_mm: float,
) -> tuple[BoolArray, FloatArray, FloatArray, float]:
    mask_array = np.asarray(mask, dtype=bool)
    rendered = np.asarray(rendered_depth_mm, dtype=np.float64)
    scene = np.asarray(scene_depth_mm, dtype=np.float64)
    tolerance = float(tolerance_mm)
    if (
        mask_array.ndim != 2
        or rendered.shape != mask_array.shape
        or scene.shape != mask_array.shape
    ):
        raise ValueError("mask, rendered depth, and scene depth must share one 2D shape")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance_mm must be finite and non-negative")
    return mask_array, rendered, scene, tolerance


def compute_visible_gripper_mask(
    active_gripper_mask: BoolArray | NDArray[Any],
    rendered_depth_mm: FloatArray | NDArray[Any],
    scene_depth_mm: FloatArray | NDArray[Any],
    tolerance_mm: float = 8.0,
) -> BoolArray:
    """Keep projected gripper pixels whose positive scene depth agrees."""

    mask, rendered, scene, tolerance = _validate_depth_inputs(
        active_gripper_mask,
        rendered_depth_mm,
        scene_depth_mm,
        tolerance_mm,
    )
    valid = mask & np.isfinite(rendered) & np.isfinite(scene) & (rendered > 0) & (scene > 0)
    visible: BoolArray = valid & (np.abs(rendered - scene) <= tolerance)
    return visible


def depth_agreement(
    mask: BoolArray | NDArray[Any],
    rendered_depth_mm: FloatArray | NDArray[Any],
    scene_depth_mm: FloatArray | NDArray[Any],
    tolerance_mm: float = 8.0,
) -> DepthAgreement:
    """Summarize depth support, treating missing scene depth as incomparable."""

    mask_array, rendered, scene, tolerance = _validate_depth_inputs(
        mask,
        rendered_depth_mm,
        scene_depth_mm,
        tolerance_mm,
    )
    rendered_mask = mask_array & np.isfinite(rendered) & (rendered > 0)
    comparable = rendered_mask & np.isfinite(scene) & (scene > 0)
    signed_residual = rendered[comparable] - scene[comparable]
    residual = np.abs(signed_residual)
    consistent_pixels = int(np.count_nonzero(residual <= tolerance))
    comparable_pixels = int(residual.size)
    return DepthAgreement(
        rendered_pixels=int(np.count_nonzero(rendered_mask)),
        comparable_pixels=comparable_pixels,
        consistent_pixels=consistent_pixels,
        consistent_fraction=(consistent_pixels / comparable_pixels if comparable_pixels else 0.0),
        median_residual_mm=(float(np.median(residual)) if comparable_pixels else None),
        p90_residual_mm=(float(np.percentile(residual, 90)) if comparable_pixels else None),
        median_signed_residual_mm=(
            float(np.median(signed_residual)) if comparable_pixels else None
        ),
        rendered_in_front_pixels=int(np.count_nonzero(signed_residual < -tolerance)),
        rendered_behind_pixels=int(np.count_nonzero(signed_residual > tolerance)),
    )


def _finite_metric(value: float | None) -> float:
    return float(value) if value is not None and math.isfinite(float(value)) else math.inf


def rank_finger_candidates(
    candidates: Sequence[FingerCandidateScore],
    *,
    command_prior_q_m: float | None = None,
    temporal_prior_q_m: float | None = None,
    maximum_median_residual_mm: float | None = None,
) -> tuple[FingerCandidateScore, ...]:
    """Rank q hypotheses by depth evidence, using priors only as tie-breaks."""

    for name, prior in (
        ("command_prior_q_m", command_prior_q_m),
        ("temporal_prior_q_m", temporal_prior_q_m),
    ):
        if prior is not None and not math.isfinite(float(prior)):
            raise ValueError(f"{name} must be finite when provided")
    if maximum_median_residual_mm is not None and (
        not math.isfinite(float(maximum_median_residual_mm)) or maximum_median_residual_mm < 0
    ):
        raise ValueError("maximum_median_residual_mm must be finite and non-negative")

    def key(candidate: FingerCandidateScore) -> tuple[float, ...]:
        agreement = candidate.agreement
        temporal_distance = (
            abs(candidate.q_m - float(temporal_prior_q_m))
            if temporal_prior_q_m is not None
            else 0.0
        )
        command_distance = (
            abs(candidate.q_m - float(command_prior_q_m))
            if command_prior_q_m is not None
            else 0.0
        )
        robust_residual = _finite_metric(agreement.median_residual_mm)
        robust_viable = maximum_median_residual_mm is None or robust_residual <= float(
            maximum_median_residual_mm
        )
        return (
            float(robust_viable),
            float(agreement.consistent_pixels),
            float(agreement.consistent_fraction),
            -robust_residual,
            -_finite_metric(agreement.p90_residual_mm),
            float(
                min(
                    candidate.link7_agreement.consistent_pixels,
                    candidate.link8_agreement.consistent_pixels,
                )
            ),
            -temporal_distance,
            -command_distance,
            -candidate.q_m,
        )

    return tuple(sorted(candidates, key=key, reverse=True))


def candidate_has_minimum_support(
    candidate: FingerCandidateScore,
    *,
    minimum_support_pixels: int,
    minimum_per_link_support_pixels: int,
    minimum_consistent_fraction: float,
) -> bool:
    """Return whether both fingers have enough scene-depth evidence."""

    if minimum_support_pixels < 0 or minimum_per_link_support_pixels < 0:
        raise ValueError("support thresholds must be non-negative")
    if not 0.0 <= minimum_consistent_fraction <= 1.0:
        raise ValueError("minimum_consistent_fraction must be within [0, 1]")
    return bool(
        agreement_has_minimum_support(
            candidate.agreement,
            minimum_support_pixels=minimum_support_pixels,
            minimum_consistent_fraction=minimum_consistent_fraction,
        )
        and candidate.link7_agreement.consistent_pixels >= minimum_per_link_support_pixels
        and candidate.link8_agreement.consistent_pixels >= minimum_per_link_support_pixels
    )


def agreement_has_minimum_support(
    agreement: DepthAgreement,
    *,
    minimum_support_pixels: int,
    minimum_consistent_fraction: float,
    maximum_median_residual_mm: float | None = None,
) -> bool:
    """Apply fail-closed pixel and fraction gates to one rendered component."""

    if minimum_support_pixels < 0:
        raise ValueError("minimum_support_pixels must be non-negative")
    if not 0.0 <= minimum_consistent_fraction <= 1.0:
        raise ValueError("minimum_consistent_fraction must be within [0, 1]")
    if maximum_median_residual_mm is not None and (
        not math.isfinite(float(maximum_median_residual_mm)) or maximum_median_residual_mm < 0
    ):
        raise ValueError("maximum_median_residual_mm must be finite and non-negative")
    median_residual = _finite_metric(agreement.median_residual_mm)
    return bool(
        agreement.consistent_pixels >= minimum_support_pixels
        and agreement.consistent_fraction >= minimum_consistent_fraction
        and (
            maximum_median_residual_mm is None
            or median_residual <= float(maximum_median_residual_mm)
        )
    )


def _normalize_active_side(active_side: str) -> tuple[ArmSide, str]:
    if active_side == "left":
        return "left", "fl"
    if active_side == "right":
        return "right", "fr"
    raise ValueError("active_side must be 'left' or 'right'")


def active_gripper_link_names(active_side: ArmSide) -> tuple[str, str, str]:
    """Return fixed wrist/palm link6 followed by finger links 7 and 8."""

    _, prefix = _normalize_active_side(active_side)
    return (f"{prefix}_link6", f"{prefix}_link7", f"{prefix}_link8")


def _grid(lower: float, upper: float, step: float) -> tuple[float, ...]:
    if not lower <= upper or step <= 0 or not all(map(math.isfinite, (lower, upper, step))):
        raise ValueError("invalid q search range or step")
    count = math.floor((upper - lower) / step)
    values = [lower + index * step for index in range(count + 1)]
    if not values or upper - values[-1] > 1e-12:
        values.append(upper)
    return tuple(float(np.clip(value, lower, upper)) for value in values)


__all__ = [
    "GRIPPER_DRIVE_CLOSED_M",
    "GRIPPER_DRIVE_SPAN_M",
    "GRIPPER_KINEMATIC_MAX_M",
    "GRIPPER_KINEMATIC_MIN_M",
    "ArmSide",
    "DepthAgreement",
    "FingerCandidateScore",
    "active_gripper_link_names",
    "agreement_has_minimum_support",
    "candidate_has_minimum_support",
    "compute_visible_gripper_mask",
    "depth_agreement",
    "gripper_command_to_drive_target",
    "gripper_command_to_kinematic_q",
    "rank_finger_candidates",
]
