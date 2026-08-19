from __future__ import annotations

import numpy as np
import pytest

import robotwin_annotation_v2.pipeline as public_pipeline
from robotwin_annotation_v2.pipeline import (
    CameraCalibration,
    GripperRoiGeometry,
    project_gripper_roi,
    rotation_from_rpy,
)
from robotwin_annotation_v2.pipeline import gripper_stage as legacy
from robotwin_annotation_v2.pipeline.gripper.sam import composition

compose_gripper_track = composition.compose_gripper_track
exclude_known_objects = composition.exclude_known_objects


def test_rotation_from_rpy_is_rz_ry_rx() -> None:
    rotation = rotation_from_rpy(0.0, 0.0, np.pi / 2)
    assert np.allclose(rotation @ np.asarray([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0])


def test_ep7150_frame51_reproduces_verified_projection() -> None:
    pose = np.asarray(
        [
            -0.0956539511680603,
            0.0383211150765419,
            0.9840234518051147,
            -0.002832249039784074,
            1.265782356262207,
            0.807780385017395,
        ]
    )
    roi = project_gripper_roi(pose, 1.0)

    assert np.allclose(roi.eef_pixel_xy, [121.02764577, 15.26118804], atol=1e-6)
    assert np.allclose(roi.tcp_pixel_xy, [139.94583300, 56.22266571], atol=1e-6)
    assert roi.hull_pixels_xy.shape[0] >= 4
    assert np.all(roi.corner_depths > 0)


def test_open_gripper_projects_a_wider_roi_than_closed_gripper() -> None:
    pose = np.asarray([-0.1, 0.0, 1.0, 0.0, 1.2, 0.8])
    closed = project_gripper_roi(pose, 0.0)
    opened = project_gripper_roi(pose, 1.0)

    assert opened.bbox_xyxy[2] - opened.bbox_xyxy[0] > closed.bbox_xyxy[2] - closed.bbox_xyxy[0]


def test_axial_back_can_expand_toward_palm_while_front_is_shortened() -> None:
    calibration = CameraCalibration(
        intrinsic_cv=np.eye(3),
        extrinsic_cv=np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1),
    )
    pose = np.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    baseline = project_gripper_roi(
        pose,
        0.0,
        calibration=calibration,
        geometry=GripperRoiGeometry(margin_px=0.0),
    )
    palm_focused = project_gripper_roi(
        pose,
        0.0,
        calibration=calibration,
        geometry=GripperRoiGeometry(
            axial_back_m=0.080,
            axial_front_m=0.040,
            margin_px=0.0,
        ),
    )

    assert palm_focused.bbox_xyxy[0] < baseline.bbox_xyxy[0]
    assert palm_focused.bbox_xyxy[2] < baseline.bbox_xyxy[2]


def test_known_object_exclusion_is_an_exact_disjoint_partition() -> None:
    candidate = np.asarray([[1, 1, 1, 1], [0, 1, 1, 0]], dtype=bool)
    target = np.asarray([[1, 1, 0, 0], [0, 1, 0, 0]], dtype=bool)
    receiver = np.asarray([[0, 1, 1, 0], [0, 1, 0, 0]], dtype=bool)

    result = exclude_known_objects(candidate, target, receiver)

    assert np.array_equal(
        result.gripper_mask,
        candidate & ~(target | receiver),
    )
    assert not (result.target_removed & result.receiver_removed).any()
    assert np.array_equal(
        result.gripper_mask | result.removed_mask,
        candidate,
    )


def test_known_object_exclusion_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="identical shapes"):
        exclude_known_objects(
            np.zeros((2, 2), dtype=bool),
            np.zeros((2, 3), dtype=bool),
            np.zeros((2, 2), dtype=bool),
        )


def test_compose_gripper_track_crops_window_roi_and_known_objects() -> None:
    native = np.ones((4, 2, 3), dtype=bool)
    roi = np.asarray(
        [
            [[1, 1, 1], [1, 1, 1]],
            [[1, 1, 0], [0, 0, 0]],
            [[1, 1, 1], [0, 0, 0]],
            [[1, 1, 1], [1, 1, 1]],
        ],
        dtype=bool,
    )
    target = np.zeros_like(native)
    receiver = np.zeros_like(native)
    target[1, 0, 0] = True
    receiver[2, 0, 1] = True

    result = compose_gripper_track(
        native,
        roi,
        target,
        receiver,
        active_window=(1, 2),
    )

    assert not result.gripper_mask[0].any()
    assert not result.gripper_mask[3].any()
    assert result.target_removed[1, 0, 0]
    assert result.receiver_removed[2, 0, 1]
    assert np.array_equal(
        result.gripper_mask | result.removed_mask,
        result.candidate_mask,
    )


def test_compose_gripper_track_rejects_invalid_inputs() -> None:
    track = np.zeros((3, 2, 2), dtype=bool)
    with pytest.raises(ValueError, match="must match"):
        compose_gripper_track(
            track,
            track[:, :, :1],
            track,
            track,
            active_window=(0, 2),
        )
    with pytest.raises(ValueError, match="active_window"):
        compose_gripper_track(
            track,
            track,
            track,
            track,
            active_window=(0, 3),
        )


@pytest.mark.parametrize(
    "name",
    (
        "GripperTrackResult",
        "ObjectExclusionResult",
        "compose_gripper_track",
        "exclude_known_objects",
    ),
)
def test_legacy_composition_exports_preserve_canonical_identity(name: str) -> None:
    canonical = getattr(composition, name)

    assert getattr(legacy, name) is canonical
    assert getattr(public_pipeline, name) is canonical
