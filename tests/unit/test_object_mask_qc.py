from __future__ import annotations

import numpy as np
import pytest

from robotwin_annotation_v2.pipeline.object_mask.qc import (
    MaskQCError,
    candidate_info,
    mask_iou,
)


def test_candidate_info_preserves_exact_mechanical_metrics() -> None:
    mask = np.asarray(
        [
            [1, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 0],
        ],
        dtype=bool,
    )

    info = candidate_info(
        "A",
        "category_query",
        "bottle",
        mask,
        min_area_fraction=3 / 16,
        max_area_fraction=3 / 16,
        seed_frame_id=7,
    )

    assert info.area_fraction == 3 / 16
    assert info.component_count == 2
    assert info.nonempty
    assert info.basic_valid
    assert info.basic_reason is None
    assert info.seed_frame_id == 7


@pytest.mark.parametrize(
    ("mask", "minimum", "maximum", "duplicate_of", "reason"),
    (
        (np.zeros((2, 2), dtype=bool), 0.0, 1.0, None, "empty_seed_mask"),
        (np.eye(2, dtype=bool), 0.75, 1.0, None, "seed_mask_too_small"),
        (np.ones((2, 2), dtype=bool), 0.0, 0.75, None, "seed_mask_too_large"),
        (np.eye(2, dtype=bool), 0.0, 1.0, "A", "duplicate_candidate_mask"),
    ),
)
def test_candidate_info_preserves_failure_priority(
    mask: np.ndarray,
    minimum: float,
    maximum: float,
    duplicate_of: str | None,
    reason: str,
) -> None:
    info = candidate_info(
        "B",
        "category_query",
        "bottle",
        mask,
        min_area_fraction=minimum,
        max_area_fraction=maximum,
        duplicate_of=duplicate_of,
    )

    assert not info.basic_valid
    assert info.basic_reason == reason


def test_candidate_info_rejects_non_image_masks_with_canonical_error() -> None:
    with pytest.raises(MaskQCError, match="candidate A mask must be 2-D"):
        candidate_info(
            "A",
            "category_query",
            "bottle",
            np.zeros((1, 2, 3), dtype=bool),
            min_area_fraction=0.0,
            max_area_fraction=1.0,
        )


def test_mask_iou_matches_fixed_binary_arrays() -> None:
    first = np.asarray([[1, 1], [0, 0]], dtype=bool)
    second = np.asarray([[0, 1], [1, 0]], dtype=bool)

    assert mask_iou(first, second) == 1 / 3
    assert mask_iou(np.zeros((2, 2)), np.zeros((2, 2))) == 1.0
