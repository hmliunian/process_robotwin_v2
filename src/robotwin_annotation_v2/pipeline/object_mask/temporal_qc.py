"""Temporal quality checks and visible-window composition for object masks."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Literal

import numpy as np

from ...config import MaskConfig
from ...models import FrameWindow

NDArray = np.ndarray[Any, Any]


@dataclass(frozen=True)
class TemporalMaskQc:
    window: FrameWindow
    status: Literal["pass", "review", "quarantine"]
    window_frames: int
    nonempty_frames: int
    coverage: float
    presence_transitions: int
    internal_missing_frames: int
    adjacent_iou_mean: float | None
    adjacent_iou_p05: float | None
    centroid_jump_p95_px: float | None
    area_ratio_jump_p95: float | None
    max_reference_centroid_distance_px: float | None
    issues: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "format_version": "robotwin_temporal_mask_qc_v1",
            "window": self.window.to_json(),
            "status": self.status,
            "window_frames": self.window_frames,
            "nonempty_frames": self.nonempty_frames,
            "coverage": self.coverage,
            "presence_transitions": self.presence_transitions,
            "internal_missing_frames": self.internal_missing_frames,
            "adjacent_iou_mean": self.adjacent_iou_mean,
            "adjacent_iou_p05": self.adjacent_iou_p05,
            "centroid_jump_p95_px": self.centroid_jump_p95_px,
            "area_ratio_jump_p95": self.area_ratio_jump_p95,
            "max_reference_centroid_distance_px": (
                self.max_reference_centroid_distance_px
            ),
            "issues": list(self.issues),
        }


def compose_visible_mask(
    native_track: NDArray,
    output_window: FrameWindow,
) -> NDArray:
    """Crop a native identity track to the role's inclusive output window."""

    native = np.asarray(native_track, dtype=bool)
    if native.ndim != 3:
        raise ValueError("native mask must have [T,H,W] shape")
    if output_window.end >= native.shape[0]:
        raise ValueError("output window extends beyond the mask stack")
    visible = native.copy()
    visible[: output_window.start] = False
    visible[output_window.end + 1 :] = False
    return visible


def _centroid(mask: NDArray) -> NDArray | None:
    rows, columns = np.nonzero(mask)
    if not rows.size:
        return None
    return np.asarray([columns.mean(), rows.mean()], dtype=np.float64)


def _p95(values: list[float]) -> float | None:
    return None if not values else float(np.quantile(values, 0.95))


def evaluate_temporal_mask(
    mask_stack: NDArray,
    output_window: FrameWindow,
    mask_config: MaskConfig,
    *,
    reference_mask: NDArray | None = None,
) -> TemporalMaskQc:
    """Measure continuity and quarantine tracks with several severe jump signals."""

    masks = np.asarray(mask_stack, dtype=bool)
    if masks.ndim != 3:
        raise ValueError("temporal mask must have [T,H,W] shape")
    if output_window.end >= masks.shape[0]:
        raise ValueError("output window extends beyond the temporal mask")
    if reference_mask is not None and np.asarray(reference_mask).shape != masks.shape[1:]:
        raise ValueError("reference mask must match the temporal mask frame shape")

    window = masks[output_window.start : output_window.end + 1]
    flattened = window.reshape(window.shape[0], -1)
    present = flattened.any(axis=1)
    nonempty_frames = int(present.sum())
    presence_transitions = int(np.count_nonzero(present[1:] != present[:-1]))
    present_indices = np.flatnonzero(present)
    internal_missing_frames = (
        0
        if present_indices.size < 2
        else int((~present[present_indices[0] : present_indices[-1] + 1]).sum())
    )

    adjacent_ious: list[float] = []
    centroid_jumps: list[float] = []
    area_ratio_jumps: list[float] = []
    centroids: list[NDArray] = []
    previous_centroid: NDArray | None = None
    previous_area: int | None = None
    for frame in window:
        area = int(frame.sum())
        centroid = _centroid(frame)
        if centroid is not None:
            centroids.append(centroid)
        if previous_centroid is not None and centroid is not None:
            centroid_jumps.append(float(np.linalg.norm(centroid - previous_centroid)))
        if previous_area is not None and previous_area > 0 and area > 0:
            area_ratio_jumps.append(abs(area - previous_area) / previous_area)
        previous_centroid = centroid
        previous_area = area
    for left, right in pairwise(window):
        if not left.any() or not right.any():
            continue
        union = int((left | right).sum())
        adjacent_ious.append(int((left & right).sum()) / union)

    adjacent_iou_mean = None if not adjacent_ious else float(np.mean(adjacent_ious))
    adjacent_iou_p05 = (
        None if not adjacent_ious else float(np.quantile(adjacent_ious, 0.05))
    )
    centroid_jump_p95_px = _p95(centroid_jumps)
    area_ratio_jump_p95 = _p95(area_ratio_jumps)
    reference_centroid = (
        None if reference_mask is None else _centroid(np.asarray(reference_mask, dtype=bool))
    )
    max_reference_distance = (
        None
        if reference_centroid is None or not centroids
        else max(float(np.linalg.norm(value - reference_centroid)) for value in centroids)
    )

    severe_signals: list[str] = []
    if (
        adjacent_iou_p05 is not None
        and adjacent_iou_p05 < mask_config.temporal_qc_min_adjacent_iou_p05
    ):
        severe_signals.append("low_adjacent_iou_p05")
    if (
        centroid_jump_p95_px is not None
        and centroid_jump_p95_px > mask_config.temporal_qc_max_centroid_jump_p95_px
    ):
        severe_signals.append("large_centroid_jump_p95")
    if (
        area_ratio_jump_p95 is not None
        and area_ratio_jump_p95 > mask_config.temporal_qc_max_area_ratio_jump_p95
    ):
        severe_signals.append("large_area_ratio_jump_p95")

    issues = list(severe_signals)
    if nonempty_frames < len(window):
        issues.append("incomplete_window_coverage")
    if internal_missing_frames:
        issues.append("internal_missing_frames")
    if len(severe_signals) >= mask_config.temporal_qc_quarantine_signal_count:
        status: Literal["pass", "review", "quarantine"] = "quarantine"
    elif issues:
        status = "review"
    else:
        status = "pass"
    return TemporalMaskQc(
        window=output_window,
        status=status,
        window_frames=len(window),
        nonempty_frames=nonempty_frames,
        coverage=float(present.mean()),
        presence_transitions=presence_transitions,
        internal_missing_frames=internal_missing_frames,
        adjacent_iou_mean=adjacent_iou_mean,
        adjacent_iou_p05=adjacent_iou_p05,
        centroid_jump_p95_px=centroid_jump_p95_px,
        area_ratio_jump_p95=area_ratio_jump_p95,
        max_reference_centroid_distance_px=max_reference_distance,
        issues=tuple(issues),
    )


__all__ = [
    "TemporalMaskQc",
    "compose_visible_mask",
    "evaluate_temporal_mask",
]
