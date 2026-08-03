from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.render_coverage20_videos import (
    HALO_COLOR,
    ROLE_COLORS,
    MaskArtifact,
    MaskCandidate,
    _external_outline_layers,
    overlay_frame,
    select_best_masks,
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


def _write_candidate(root: Path, run_id: str, statuses: tuple[str, ...]) -> None:
    path = (
        root
        / run_id
        / "move_pillbottle_pad"
        / "episode_007152"
        / "cam_high"
        / "masks.npz"
    )
    path.parent.mkdir(parents=True)
    np.savez_compressed(
        path,
        roles=np.asarray(("target", "receiver")),
        annotation_status=np.asarray(statuses),
    )


def test_select_best_masks_can_pin_an_exact_run(tmp_path: Path) -> None:
    _write_candidate(tmp_path, "old-valid", ("valid", "valid"))
    _write_candidate(tmp_path, "native-v1", ("valid", "quarantined"))

    selected = select_best_masks(
        tmp_path,
        task="move_pillbottle_pad",
        camera="cam_high",
        episode_ids=(7152,),
        run_id="native-v1",
    )

    assert selected[7152].run_id == "native-v1"


def test_fully_qc_verified_run_is_preferred_over_newer_unverified_run() -> None:
    verified = MaskCandidate(
        path=Path("verified.npz"),
        run_id="verified",
        role_status={"target": "valid", "receiver": "valid"},
        role_qc_status={"target": "passed", "receiver": "passed"},
        valid_role_count=2,
        qc_passed_role_count=2,
        modified_ns=1,
    )
    unverified = MaskCandidate(
        path=Path("unverified.npz"),
        run_id="unverified",
        role_status={"target": "valid", "receiver": "valid"},
        role_qc_status={"target": "not_run", "receiver": "not_run"},
        valid_role_count=2,
        qc_passed_role_count=0,
        modified_ns=999,
    )

    assert verified.score > unverified.score
