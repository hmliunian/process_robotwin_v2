"""Candidate construction and mechanical screening for SAM gripper seeds."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import cv2
import numpy as np

from ....models.timeline import PickPlaceEvents
from .composition import ObjectExclusionResult, exclude_known_objects

NDArray = np.ndarray[Any, Any]


@dataclass(frozen=True)
class GripperSeedQualityGateConfig:
    """Mechanical seed-screen thresholds used before Qwen candidate selection."""

    minimum_pixels: int = 24
    minimum_dark_fraction: float = 0.80
    maximum_components: int = 3
    minimum_largest_component_fraction: float = 0.80
    maximum_tcp_distance_px: float = 20.0
    duplicate_iou_threshold: float = 0.98

    def __post_init__(self) -> None:
        if self.minimum_pixels < 1:
            raise ValueError("minimum_pixels must be positive")
        if not 0.0 <= self.minimum_dark_fraction <= 1.0:
            raise ValueError("minimum_dark_fraction must be in [0, 1]")
        if self.maximum_components < 1:
            raise ValueError("maximum_components must be positive")
        if not 0.0 <= self.minimum_largest_component_fraction <= 1.0:
            raise ValueError("minimum_largest_component_fraction must be in [0, 1]")
        if self.maximum_tcp_distance_px < 0.0:
            raise ValueError("maximum_tcp_distance_px must be non-negative")
        if not 0.0 <= self.duplicate_iou_threshold <= 1.0:
            raise ValueError("duplicate_iou_threshold must be in [0, 1]")


@dataclass(frozen=True)
class GripperSeedCandidate:
    candidate_id: str
    frame_id: int
    phase: str
    prompt_mode: str
    prompt_text: str | None
    raw_mask: NDArray
    roi_mask: NDArray
    cropped_mask: NDArray
    clean_mask: NDArray
    target_removed: NDArray
    receiver_removed: NDArray
    raw_pixels: int
    roi_pixels: int
    cropped_pixels: int
    clean_pixels: int
    target_removed_pixels: int
    receiver_removed_pixels: int
    dark_fraction: float | None
    component_count: int
    largest_component_fraction: float | None
    tcp_distance_px: float | None
    basic_valid: bool
    basic_reason: str | None = None
    duplicate_of: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "frame_id": self.frame_id,
            "phase": self.phase,
            "prompt_mode": self.prompt_mode,
            "prompt_text": self.prompt_text,
            "raw_pixels": self.raw_pixels,
            "roi_pixels": self.roi_pixels,
            "cropped_pixels": self.cropped_pixels,
            "clean_pixels": self.clean_pixels,
            "target_removed_pixels": self.target_removed_pixels,
            "receiver_removed_pixels": self.receiver_removed_pixels,
            "dark_fraction": self.dark_fraction,
            "component_count": self.component_count,
            "largest_component_fraction": self.largest_component_fraction,
            "tcp_distance_px": self.tcp_distance_px,
            "basic_valid": self.basic_valid,
            "basic_reason": self.basic_reason,
            "duplicate_of": self.duplicate_of,
        }


def phase_for_frame(frame_id: int, events: PickPlaceEvents) -> str:
    if frame_id < events.t_close_start:
        return "approach"
    if frame_id <= events.t_close_done:
        return "close"
    if frame_id < events.t_open_start:
        return "transport"
    return "release"


def _component_metrics(mask: NDArray) -> tuple[int, float | None]:
    value = np.asarray(mask, dtype=np.uint8)
    if not value.any():
        return 0, None
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        value,
        connectivity=8,
    )
    areas = stats[1:, cv2.CC_STAT_AREA]
    component_count = max(0, count - 1)
    return component_count, float(areas.max() / areas.sum())


def _tcp_distance(mask: NDArray, tcp_xy: NDArray) -> float | None:
    rows, columns = np.nonzero(mask)
    if not rows.size:
        return None
    x, y = (float(value) for value in tcp_xy)
    squared = (columns.astype(np.float64) - x) ** 2 + (rows.astype(np.float64) - y) ** 2
    return float(np.sqrt(squared.min()))


def build_gripper_seed_candidate(
    *,
    candidate_id: str,
    frame_id: int,
    events: PickPlaceEvents,
    prompt_mode: str,
    prompt_text: str | None,
    raw_mask: NDArray,
    roi_mask: NDArray,
    target_mask: NDArray,
    receiver_mask: NDArray,
    rgb: NDArray,
    tcp_pixel_xy: NDArray,
    minimum_pixels: int = 24,
) -> GripperSeedCandidate:
    raw = np.asarray(raw_mask, dtype=bool)
    roi = np.asarray(roi_mask, dtype=bool)
    target = np.asarray(target_mask, dtype=bool)
    receiver = np.asarray(receiver_mask, dtype=bool)
    image = np.asarray(rgb, dtype=np.uint8)
    if raw.shape != roi.shape or target.shape != raw.shape or receiver.shape != raw.shape:
        raise ValueError("gripper candidate and constraint masks must have identical shapes")
    if image.shape != (*raw.shape, 3):
        raise ValueError("gripper candidate RGB must match mask shape")
    cropped = raw & roi
    excluded: ObjectExclusionResult = exclude_known_objects(cropped, target, receiver)
    clean = excluded.gripper_mask
    clean_pixels = int(clean.sum())
    components, largest_fraction = _component_metrics(clean)
    dark_fraction = (
        None if not clean_pixels else float((image[clean].max(axis=1) < 70).mean())
    )
    if clean_pixels == 0:
        reason = "empty_after_pose_and_object_constraints"
    elif clean_pixels < minimum_pixels:
        reason = "too_few_pixels_after_constraints"
    else:
        reason = None
    return GripperSeedCandidate(
        candidate_id=candidate_id,
        frame_id=frame_id,
        phase=phase_for_frame(frame_id, events),
        prompt_mode=prompt_mode,
        prompt_text=prompt_text,
        raw_mask=raw,
        roi_mask=roi,
        cropped_mask=cropped,
        clean_mask=clean,
        target_removed=excluded.target_removed,
        receiver_removed=excluded.receiver_removed,
        raw_pixels=int(raw.sum()),
        roi_pixels=int(roi.sum()),
        cropped_pixels=int(cropped.sum()),
        clean_pixels=clean_pixels,
        target_removed_pixels=int(excluded.target_removed.sum()),
        receiver_removed_pixels=int(excluded.receiver_removed.sum()),
        dark_fraction=dark_fraction,
        component_count=components,
        largest_component_fraction=largest_fraction,
        tcp_distance_px=_tcp_distance(clean, tcp_pixel_xy),
        basic_valid=reason is None,
        basic_reason=reason,
    )


def apply_gripper_seed_quality_gate(
    candidate: GripperSeedCandidate,
    *,
    minimum_dark_fraction: float = 0.80,
    maximum_components: int = 3,
    minimum_largest_component_fraction: float = 0.80,
    maximum_tcp_distance_px: float = 20.0,
) -> GripperSeedCandidate:
    """Reject mechanically implausible black-gripper seed candidates."""

    if not 0.0 <= minimum_dark_fraction <= 1.0:
        raise ValueError("minimum_dark_fraction must be in [0, 1]")
    if maximum_components < 1:
        raise ValueError("maximum_components must be positive")
    if not 0.0 <= minimum_largest_component_fraction <= 1.0:
        raise ValueError("minimum_largest_component_fraction must be in [0, 1]")
    if maximum_tcp_distance_px < 0.0:
        raise ValueError("maximum_tcp_distance_px must be non-negative")
    if not candidate.basic_valid:
        return candidate

    failures: list[str] = []
    if candidate.dark_fraction is None or candidate.dark_fraction < minimum_dark_fraction:
        failures.append("low_dark_fraction")
    if candidate.component_count > maximum_components:
        failures.append("too_many_components")
    if (
        candidate.largest_component_fraction is None
        or candidate.largest_component_fraction < minimum_largest_component_fraction
    ):
        failures.append("small_largest_component")
    if (
        candidate.tcp_distance_px is None
        or candidate.tcp_distance_px > maximum_tcp_distance_px
    ):
        failures.append("too_far_from_tcp")
    if not failures:
        return candidate
    return replace(
        candidate,
        basic_valid=False,
        basic_reason="quality_gate:" + ",".join(failures),
    )


def mark_same_frame_duplicates(
    candidates: Sequence[GripperSeedCandidate],
    *,
    iou_threshold: float = 0.98,
) -> tuple[GripperSeedCandidate, ...]:
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("duplicate IoU threshold must be in [0, 1]")
    result: list[GripperSeedCandidate] = []
    for candidate in candidates:
        duplicate_of: str | None = None
        if candidate.basic_valid:
            for previous in result:
                if not previous.basic_valid or previous.frame_id != candidate.frame_id:
                    continue
                union = previous.clean_mask | candidate.clean_mask
                iou = 1.0 if not union.any() else float(
                    (previous.clean_mask & candidate.clean_mask).sum() / union.sum()
                )
                if iou >= iou_threshold:
                    duplicate_of = previous.candidate_id
                    break
        if duplicate_of is None:
            result.append(candidate)
        else:
            result.append(
                replace(
                    candidate,
                    basic_valid=False,
                    basic_reason="duplicate_same_frame_candidate",
                    duplicate_of=duplicate_of,
                )
            )
    return tuple(result)


__all__ = [
    "GripperSeedCandidate",
    "GripperSeedQualityGateConfig",
    "apply_gripper_seed_quality_gate",
    "build_gripper_seed_candidate",
    "mark_same_frame_duplicates",
    "phase_for_frame",
]
