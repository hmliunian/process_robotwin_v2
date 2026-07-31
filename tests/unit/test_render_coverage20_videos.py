from __future__ import annotations

import numpy as np
import pytest

from scripts.render_coverage20_videos import (
    HALO_COLOR,
    ROLE_COLORS,
    MaskArtifact,
    _external_outline_layers,
    overlay_frame,
)


def _artifact(mask: np.ndarray) -> MaskArtifact:
    return MaskArtifact(
        masks=mask[None, None, :, :],
        instance_names=("target_0",),
        roles=("target",),
        annotation_status=("valid",),
        frame_count=1,
        format_version="test",
    )


def test_overlay_uses_external_colored_outline_and_black_halo() -> None:
    frame = np.full((17, 17, 3), 200, dtype=np.uint8)
    mask = np.zeros((17, 17), dtype=bool)
    mask[7:10, 7:10] = True

    result = overlay_frame(
        frame,
        _artifact(mask),
        frame_id=0,
        alpha=0.25,
        outline_radius=3,
        halo_radius=5,
    )

    target_color = np.asarray(ROLE_COLORS["target"], dtype=np.uint8)
    expected_fill = (200 * 0.75 + target_color * 0.25).astype(np.uint8)
    np.testing.assert_array_equal(result[8, 8], expected_fill)
    np.testing.assert_array_equal(result[8, 12], target_color)
    np.testing.assert_array_equal(result[8, 13], HALO_COLOR)
    np.testing.assert_array_equal(result[8, 15], (200, 200, 200))
    np.testing.assert_array_equal(frame[8, 8], (200, 200, 200))


def test_outline_layers_never_consume_mask_pixels() -> None:
    mask = np.zeros((9, 9), dtype=bool)
    mask[3:6, 3:6] = True

    halo, outline = _external_outline_layers(mask, outline_radius=2, halo_radius=4)

    assert not (halo & mask).any()
    assert not (outline & mask).any()
    assert not (halo & outline).any()


def test_halo_must_cover_the_outline_radius() -> None:
    with pytest.raises(ValueError, match="at least outline_radius"):
        _external_outline_layers(
            np.ones((3, 3), dtype=bool),
            outline_radius=4,
            halo_radius=3,
        )
