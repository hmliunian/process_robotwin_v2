from __future__ import annotations

import numpy as np
import pytest

from robotwin_annotation_v2.pipeline import gripper_stage as legacy
from robotwin_annotation_v2.pipeline.gripper.sam import geometry

VERIFIED_POSE = np.asarray(
    [
        -0.0956539511680603,
        0.0383211150765419,
        0.9840234518051147,
        -0.002832249039784074,
        1.265782356262207,
        0.807780385017395,
    ]
)


def _roi_with_bbox(bbox_xyxy: tuple[float, float, float, float]) -> geometry.ProjectedGripperRoi:
    point = np.zeros(2, dtype=np.float64)
    points = np.zeros((0, 2), dtype=np.float64)
    return geometry.ProjectedGripperRoi(
        eef_pixel_xy=point,
        tcp_pixel_xy=point,
        corner_pixels_xy=points,
        hull_pixels_xy=points,
        bbox_xyxy=np.asarray(bbox_xyxy, dtype=np.float64),
        corner_depths=np.zeros(0, dtype=np.float64),
        open_fraction=0.0,
    )


def test_rotation_from_rpy_preserves_static_xyz_contract() -> None:
    rotation = geometry.rotation_from_rpy(0.0, 0.0, np.pi / 2)

    assert np.allclose(rotation @ np.asarray([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0])


def test_verified_episode_projection_preserves_full_numeric_contract() -> None:
    roi = geometry.project_gripper_roi(VERIFIED_POSE, 1.0)

    assert np.allclose(roi.eef_pixel_xy, [121.02764577, 15.26118804], atol=1e-6)
    assert np.allclose(roi.tcp_pixel_xy, [139.94583300, 56.22266571], atol=1e-6)
    assert np.allclose(
        roi.corner_pixels_xy,
        [
            [151.30013136, 91.54938377],
            [188.12354824, 51.49539434],
            [88.80406068, 46.24948530],
            [123.52589171, 11.24879147],
            [161.24476350, 111.92051119],
            [193.94422467, 75.59621119],
            [103.98148713, 69.21470271],
            [135.03298820, 37.06882594],
        ],
        atol=1e-6,
    )
    assert np.allclose(
        roi.hull_pixels_xy,
        [
            [88.80406068, 46.24948530],
            [123.52589171, 11.24879147],
            [188.12354824, 51.49539434],
            [193.94422467, 75.59621119],
            [161.24476350, 111.92051119],
            [103.98148713, 69.21470271],
        ],
        atol=1e-6,
    )
    assert np.allclose(
        roi.bbox_xyxy,
        [85.80406068, 8.24879147, 196.94422467, 114.92051119],
        atol=1e-6,
    )
    assert np.allclose(
        roi.corner_depths,
        [
            0.62670499,
            0.64416152,
            0.69711387,
            0.71457040,
            0.70263575,
            0.72009228,
            0.77304463,
            0.79050115,
        ],
        atol=1e-6,
    )
    assert roi.open_fraction == 1.0


def test_open_gripper_projects_wider_than_closed_gripper() -> None:
    pose = np.asarray([-0.1, 0.0, 1.0, 0.0, 1.2, 0.8])
    closed = geometry.project_gripper_roi(pose, 0.0)
    opened = geometry.project_gripper_roi(pose, 1.0)

    assert opened.bbox_xyxy[2] - opened.bbox_xyxy[0] > closed.bbox_xyxy[2] - closed.bbox_xyxy[0]


def test_axial_geometry_preserves_independent_back_and_front_extents() -> None:
    calibration = geometry.CameraCalibration(
        intrinsic_cv=np.eye(3),
        extrinsic_cv=np.concatenate((np.eye(3), np.zeros((3, 1))), axis=1),
    )
    pose = np.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    baseline = geometry.project_gripper_roi(
        pose,
        0.0,
        calibration=calibration,
        geometry=geometry.GripperRoiGeometry(margin_px=0.0),
    )
    palm_focused = geometry.project_gripper_roi(
        pose,
        0.0,
        calibration=calibration,
        geometry=geometry.GripperRoiGeometry(
            axial_back_m=0.080,
            axial_front_m=0.040,
            margin_px=0.0,
        ),
    )

    assert palm_focused.bbox_xyxy[0] < baseline.bbox_xyxy[0]
    assert palm_focused.bbox_xyxy[2] < baseline.bbox_xyxy[2]


def test_normalized_roi_box_rounds_and_clamps_to_frame() -> None:
    roi = _roi_with_bbox((-1.2, 2.2, 10.1, 12.0))

    assert geometry.normalized_roi_box(roi, (10, 20)) == (0.0, 0.2, 0.55, 1.0)


def test_normalized_roi_box_rejects_fully_offscreen_roi() -> None:
    roi = _roi_with_bbox((20.1, 1.0, 24.0, 4.0))

    assert geometry.normalized_roi_box(roi, (10, 20)) is None


@pytest.mark.parametrize(
    "name",
    (
        "CAM_HIGH_CALIBRATION",
        "DEFAULT_GRIPPER_ROI_GEOMETRY",
        "CameraCalibration",
        "GripperRoiGeometry",
        "ProjectedGripperRoi",
        "_convex_hull",
        "_project_world_points",
        "normalized_roi_box",
        "project_gripper_roi",
        "rotation_from_rpy",
    ),
)
def test_legacy_gripper_stage_reexports_canonical_geometry_by_identity(name: str) -> None:
    assert getattr(legacy, name) is getattr(geometry, name)
