from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from robotwin_annotation_v2.pipeline.object_mask.proposals import (
    blue_planar_region,
    largest_component,
)
from robotwin_annotation_v2.pipeline.object_mask.qc import MaskQCError


def test_largest_component_preserves_first_component_on_size_tie() -> None:
    mask = np.asarray(
        [
            [1, 0, 1],
            [1, 0, 1],
        ],
        dtype=bool,
    )

    result = largest_component(mask)

    assert np.array_equal(
        result,
        np.asarray(
            [
                [1, 0, 0],
                [1, 0, 0],
            ],
            dtype=bool,
        ),
    )


def test_blue_planar_region_uses_exact_thresholds_and_largest_component() -> None:
    rgb = np.zeros((3, 4, 3), dtype=np.uint8)
    rgb[0, 0] = (50, 60, 80)
    rgb[1, 0] = (20, 40, 100)
    rgb[2, 3] = (10, 20, 90)
    rgb[1, 3] = (10, 20, 79)

    result = blue_planar_region(Image.fromarray(rgb), (3, 4))

    expected = np.zeros((3, 4), dtype=bool)
    expected[0:2, 0] = True
    assert np.array_equal(result, expected)


def test_blue_planar_region_rejects_shape_mismatch() -> None:
    image = Image.fromarray(np.zeros((2, 3, 3), dtype=np.uint8))

    with pytest.raises(MaskQCError, match="seed RGB shape .* expected"):
        blue_planar_region(image, (3, 2))
