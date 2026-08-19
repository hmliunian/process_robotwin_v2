"""Pure visible-mask composition for the SAM gripper pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

NDArray = np.ndarray[Any, Any]


@dataclass(frozen=True)
class ObjectExclusionResult:
    """Exact partition of a candidate mask after known-object exclusion."""

    gripper_mask: NDArray
    removed_mask: NDArray
    target_removed: NDArray
    receiver_removed: NDArray


@dataclass(frozen=True)
class GripperTrackResult:
    """Exact visible-track partition after pose cropping and object exclusion."""

    roi_mask: NDArray
    candidate_mask: NDArray
    gripper_mask: NDArray
    removed_mask: NDArray
    target_removed: NDArray
    receiver_removed: NDArray


def exclude_known_objects(
    candidate_mask: NDArray,
    target_mask: NDArray,
    receiver_mask: NDArray,
) -> ObjectExclusionResult:
    """Subtract visible target/receiver masks without spatial post-processing.

    Target has priority only when attributing a pixel covered by both known
    object masks. The final gripper mask is independent of that attribution.
    """

    candidate = np.asarray(candidate_mask, dtype=bool)
    target = np.asarray(target_mask, dtype=bool)
    receiver = np.asarray(receiver_mask, dtype=bool)
    if candidate.ndim < 2:
        raise ValueError("candidate mask must have at least two dimensions")
    if target.shape != candidate.shape or receiver.shape != candidate.shape:
        raise ValueError("candidate, target, and receiver masks must have identical shapes")

    target_removed = candidate & target
    receiver_removed = candidate & receiver & ~target
    removed = target_removed | receiver_removed
    return ObjectExclusionResult(
        gripper_mask=candidate & ~removed,
        removed_mask=removed,
        target_removed=target_removed,
        receiver_removed=receiver_removed,
    )


def compose_gripper_track(
    native_track: NDArray,
    roi_track: NDArray,
    target_track: NDArray,
    receiver_track: NDArray,
    *,
    active_window: tuple[int, int],
) -> GripperTrackResult:
    """Crop one propagated gripper track and subtract visible known objects."""

    native = np.asarray(native_track, dtype=bool)
    roi = np.asarray(roi_track, dtype=bool)
    target = np.asarray(target_track, dtype=bool)
    receiver = np.asarray(receiver_track, dtype=bool)
    if native.ndim != 3:
        raise ValueError("gripper tracks must have [T,H,W] shape")
    if any(value.shape != native.shape for value in (roi, target, receiver)):
        raise ValueError("native, ROI, target, and receiver tracks must match")

    start, end = active_window
    if not 0 <= start <= end < native.shape[0]:
        raise ValueError("active_window must be inclusive and inside the track")
    active = np.zeros(native.shape[0], dtype=bool)
    active[start : end + 1] = True
    visible_roi = roi & active[:, None, None]
    candidate = native & visible_roi
    excluded = exclude_known_objects(candidate, target, receiver)
    return GripperTrackResult(
        roi_mask=visible_roi,
        candidate_mask=candidate,
        gripper_mask=excluded.gripper_mask,
        removed_mask=excluded.removed_mask,
        target_removed=excluded.target_removed,
        receiver_removed=excluded.receiver_removed,
    )


__all__ = [
    "GripperTrackResult",
    "ObjectExclusionResult",
    "compose_gripper_track",
    "exclude_known_objects",
]
