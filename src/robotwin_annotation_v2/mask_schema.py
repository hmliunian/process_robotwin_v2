"""Canonical visible-mask schema and per-frame rendering encodings.

The pixel masks stay as independent boolean instance tracks.  A compact
``frame_encoding`` array records which rendering semantics apply to each
instance on each frame without collapsing overlapping masks into one label
image.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from .models.timeline import TimelineEvents, derive_target_hold_window

MASK_FORMAT_VERSION = "robotwin_visible_masks_v3"
LEGACY_MASK_FORMAT_VERSION = "robotwin_visible_masks_v2"
# Shared renderer color for the post-close, pre-open target phase (RGB).
TARGET_HOLD_COLOR_RGB = (255, 255, 0)
MASK_BASE_KEYS = frozenset(
    {
        "format_version",
        "frame_count",
        "masks",
        "instance_names",
        "roles",
        "annotation_status",
        "qc_status",
    }
)
MASK_KEYS = MASK_BASE_KEYS | {"frame_encoding"}


class FrameEncoding(IntEnum):
    """Semantic rendering state for one instance on one frame."""

    ABSENT = 0
    VISIBLE = 1
    TARGET_GRASP_HOLD = 2


FRAME_ENCODING_LEGEND = {
    str(FrameEncoding.ABSENT.value): "absent",
    str(FrameEncoding.VISIBLE.value): "visible",
    str(FrameEncoding.TARGET_GRASP_HOLD.value): "target_grasp_hold",
}


def default_frame_encoding(masks: NDArray[Any]) -> NDArray[np.uint8]:
    """Encode every nonempty instance frame as ordinary visible mask data."""

    array = np.asarray(masks)
    if array.ndim != 4 or array.dtype != np.bool_:
        raise ValueError(
            "masks must have bool shape [N,T,H,W], "
            f"got {array.shape}/{array.dtype}"
        )
    present = array.reshape(array.shape[0], array.shape[1], -1).any(axis=2)
    return cast(
        NDArray[np.uint8],
        np.where(
            present,
            FrameEncoding.VISIBLE.value,
            FrameEncoding.ABSENT.value,
        ).astype(np.uint8),
    )


def target_hold_window(
    events: TimelineEvents,
    *,
    frame_count: int,
) -> tuple[int, int] | None:
    """Return the inclusive post-close/pre-open target hold interval."""

    hold = derive_target_hold_window(events, frame_count=frame_count)
    return None if hold is None else (hold.start, hold.end)


def build_frame_encoding(
    masks: NDArray[Any],
    events: TimelineEvents,
) -> NDArray[np.uint8]:
    """Build canonical encoding, marking held target frames with code ``2``."""

    array = np.asarray(masks)
    encoding = default_frame_encoding(array)
    hold_window = target_hold_window(events, frame_count=array.shape[1])
    if hold_window is None:
        return encoding
    start, end = hold_window
    target_present = encoding[0, start : end + 1] != FrameEncoding.ABSENT.value
    encoding[0, start : end + 1] = np.where(
        target_present,
        FrameEncoding.TARGET_GRASP_HOLD.value,
        FrameEncoding.ABSENT.value,
    )
    return encoding


def validate_frame_encoding(
    masks: NDArray[Any],
    frame_encoding: NDArray[Any],
) -> NDArray[np.uint8]:
    """Validate and return a canonical ``uint8 [N,T]`` encoding array."""

    array = np.asarray(masks)
    if array.ndim != 4 or array.dtype != np.bool_:
        raise ValueError(
            "masks must have bool shape [N,T,H,W], "
            f"got {array.shape}/{array.dtype}"
        )
    encoding = np.asarray(frame_encoding)
    if encoding.dtype != np.uint8 or encoding.shape != array.shape[:2]:
        raise ValueError(
            "frame_encoding must have uint8 shape "
            f"{array.shape[:2]}, got {encoding.shape}/{encoding.dtype}"
        )
    allowed_values = {item.value for item in FrameEncoding}
    allowed = np.asarray(sorted(allowed_values), dtype=np.uint8)
    if not bool(np.isin(encoding, allowed).all()):
        invalid = sorted(
            int(value)
            for value in np.unique(encoding)
            if int(value) not in allowed_values
        )
        raise ValueError(f"frame_encoding contains unsupported codes: {invalid}")
    present = array.reshape(array.shape[0], array.shape[1], -1).any(axis=2)
    encoded_present = encoding != FrameEncoding.ABSENT.value
    if not np.array_equal(encoded_present, present):
        raise ValueError("frame_encoding presence differs from mask presence")
    if encoding.shape[0] > 1 and np.any(
        encoding[1:] == FrameEncoding.TARGET_GRASP_HOLD.value
    ):
        raise ValueError("target_grasp_hold encoding is only valid for target_0")
    return cast(NDArray[np.uint8], encoding)


__all__ = [
    "FRAME_ENCODING_LEGEND",
    "LEGACY_MASK_FORMAT_VERSION",
    "MASK_BASE_KEYS",
    "MASK_FORMAT_VERSION",
    "MASK_KEYS",
    "TARGET_HOLD_COLOR_RGB",
    "FrameEncoding",
    "build_frame_encoding",
    "default_frame_encoding",
    "target_hold_window",
    "validate_frame_encoding",
]
