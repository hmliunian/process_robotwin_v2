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
    _merge_gripper_track,
    _output_video_name,
    _text_prompt_slug,
    overlay_frame,
    select_best_masks,
)


def test_text_prompt_filename_is_readable_portable_and_unique() -> None:
    prompt = "Use the right arm to set the flat-base orange bottle onto the pad."

    assert _text_prompt_slug(prompt) == (
        "use_the_right_arm_to_set_the_flat_base_orange_bottle_onto_the_pad"
    )
    assert _output_video_name(
        episode_id=7152,
        camera="cam_high",
        task_text=prompt,
        filename_mode="text_prompt",
    ) == (
        "use_the_right_arm_to_set_the_flat_base_orange_bottle_onto_the_pad"
        "__episode_007152_cam_high_overlay.mp4"
    )


def test_episode_filename_mode_preserves_existing_contract() -> None:
    assert _output_video_name(
        episode_id=7152,
        camera="cam_high",
        task_text="ignored",
        filename_mode="episode",
    ) == "episode_007152_cam_high_overlay.mp4"


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


def test_active_gripper_uses_same_fill_outline_and_halo_as_other_roles() -> None:
    frame = np.full((17, 17, 3), 200, dtype=np.uint8)
    masks = np.zeros((4, 1, 17, 17), dtype=bool)
    artifact = MaskArtifact(
        masks=masks,
        instance_names=("target_0", "receiver_0", "gripper_left", "gripper_right"),
        roles=("target", "receiver", "gripper", "gripper"),
        annotation_status=("valid", "valid", "not_annotated", "not_annotated"),
        frame_count=1,
        format_version="test",
        qc_status=("passed", "passed", "not_run", "not_run"),
    )
    gripper = np.zeros((1, 17, 17), dtype=bool)
    gripper[0, 7:10, 7:10] = True

    merged, removed = _merge_gripper_track(
        artifact,
        gripper,
        active_arm="left",
        qc_status="passed",
    )
    result = overlay_frame(
        frame,
        merged,
        frame_id=0,
        alpha=0.25,
        outline_radius=3,
        halo_radius=5,
    )

    gripper_color = np.asarray(ROLE_COLORS["gripper"], dtype=np.uint8)
    expected_fill = (200 * 0.75 + gripper_color * 0.25).astype(np.uint8)
    assert removed == 0
    assert merged.annotation_status[2] == "valid"
    assert merged.qc_status[2] == "passed"
    np.testing.assert_array_equal(result[8, 8], expected_fill)
    np.testing.assert_array_equal(result[8, 12], gripper_color)
    np.testing.assert_array_equal(result[8, 13], HALO_COLOR)


def test_object_masks_keep_priority_over_gripper_pixels() -> None:
    masks = np.zeros((4, 1, 7, 7), dtype=bool)
    masks[0, 0, 3, 3] = True
    artifact = MaskArtifact(
        masks=masks,
        instance_names=("target_0", "receiver_0", "gripper_left", "gripper_right"),
        roles=("target", "receiver", "gripper", "gripper"),
        annotation_status=("valid", "valid", "not_annotated", "not_annotated"),
        frame_count=1,
        format_version="test",
        qc_status=("passed", "passed", "not_run", "not_run"),
    )
    gripper = np.zeros((1, 7, 7), dtype=bool)
    gripper[0, 3, 3:5] = True

    merged, removed = _merge_gripper_track(
        artifact,
        gripper,
        active_arm="left",
        qc_status="passed",
    )

    assert removed == 1
    assert not merged.masks[2, 0, 3, 3]
    assert merged.masks[2, 0, 3, 4]


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
