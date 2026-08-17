from __future__ import annotations

import numpy as np
import pytest

from robotwin_annotation_v2.mask_schema import (
    FrameEncoding,
    build_frame_encoding,
    default_frame_encoding,
    target_hold_window,
    validate_frame_encoding,
)
from robotwin_annotation_v2.models import LoopEvents, TargetOnlyEvents


def _masks(frame_count: int) -> np.ndarray:
    masks = np.zeros((4, frame_count, 2, 3), dtype=bool)
    masks[0, :, 0, 0] = True
    masks[1, :, 0, 1] = True
    return masks


def test_pick_place_encoding_changes_strictly_between_close_and_open() -> None:
    events = LoopEvents("right", 1, 2, 4, 8, 10)
    masks = _masks(12)
    masks[0] = False
    masks[0, 1:8, 0, 0] = True
    masks[0, 6] = False

    encoding = build_frame_encoding(masks, events)

    assert target_hold_window(events, frame_count=12) == (5, 7)
    assert encoding[0].tolist() == [0, 1, 1, 1, 1, 2, 0, 2, 0, 0, 0, 0]
    assert encoding[1].tolist() == [1] * 12


def test_target_only_encoding_holds_from_close_end_through_last_frame() -> None:
    events = TargetOnlyEvents("left", 1, 2, 4)
    masks = _masks(8)
    masks[0, 0] = False

    encoding = build_frame_encoding(masks, events)

    assert target_hold_window(events, frame_count=8) == (5, 7)
    assert encoding[0].tolist() == [0, 1, 1, 1, 1, 2, 2, 2]


def test_target_only_hold_window_can_be_empty_at_episode_end() -> None:
    events = TargetOnlyEvents("left", 1, 2, 4)
    assert events.target_hold_window(5) is None
    assert target_hold_window(events, frame_count=5) is None
    assert build_frame_encoding(_masks(5), events)[0].tolist() == [1] * 5


def test_default_encoding_supports_legacy_bool_masks() -> None:
    masks = _masks(3)
    masks[0, 1] = False

    encoding = default_frame_encoding(masks)

    assert encoding.dtype == np.uint8
    assert encoding[0].tolist() == [1, 0, 1]
    np.testing.assert_array_equal(validate_frame_encoding(masks, encoding), encoding)


def test_validation_rejects_encoding_presence_without_pixels() -> None:
    masks = _masks(3)
    masks[0, 1] = False
    encoding = default_frame_encoding(masks)
    encoding[0, 1] = FrameEncoding.TARGET_GRASP_HOLD.value

    with pytest.raises(ValueError, match="presence differs"):
        validate_frame_encoding(masks, encoding)
