from __future__ import annotations

import numpy as np

import pytest

from robotwin_annotation_v2.experiments import (
    exclude_known_objects,
    project_gripper_roi,
    rotation_from_rpy,
)


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
