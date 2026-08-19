"""Pure state-to-image geometry for the SAM gripper pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np

NDArray = np.ndarray[Any, Any]


@dataclass(frozen=True)
class CameraCalibration:
    intrinsic_cv: NDArray
    extrinsic_cv: NDArray

    def __post_init__(self) -> None:
        intrinsic = np.asarray(self.intrinsic_cv, dtype=np.float64)
        extrinsic = np.asarray(self.extrinsic_cv, dtype=np.float64)
        if intrinsic.shape != (3, 3):
            raise ValueError(f"intrinsic_cv must be [3,3], got {intrinsic.shape}")
        if extrinsic.shape != (3, 4):
            raise ValueError(f"extrinsic_cv must be [3,4], got {extrinsic.shape}")
        if not np.isfinite(intrinsic).all() or not np.isfinite(extrinsic).all():
            raise ValueError("camera calibration contains non-finite values")
        object.__setattr__(self, "intrinsic_cv", intrinsic)
        object.__setattr__(self, "extrinsic_cv", extrinsic)


@dataclass(frozen=True)
class GripperRoiGeometry:
    tcp_offset_m: float = 0.12
    axial_back_m: float = 0.025
    axial_front_m: float = 0.06
    closed_half_width_m: float = 0.045
    open_half_width_m: float = 0.085
    half_thickness_m: float = 0.05
    margin_px: float = 3.0

    def __post_init__(self) -> None:
        positive = (
            self.tcp_offset_m,
            self.axial_back_m,
            self.axial_front_m,
            self.closed_half_width_m,
            self.open_half_width_m,
            self.half_thickness_m,
        )
        if min(positive) <= 0 or self.margin_px < 0:
            raise ValueError("gripper ROI dimensions must be positive and margin non-negative")
        if self.open_half_width_m < self.closed_half_width_m:
            raise ValueError("open gripper width must not be smaller than closed width")


@dataclass(frozen=True)
class ProjectedGripperRoi:
    eef_pixel_xy: NDArray
    tcp_pixel_xy: NDArray
    corner_pixels_xy: NDArray
    hull_pixels_xy: NDArray
    bbox_xyxy: NDArray
    corner_depths: NDArray
    open_fraction: float


CAM_HIGH_CALIBRATION = CameraCalibration(
    intrinsic_cv=np.asarray(
        [
            [358.64218, 0.0, 160.0],
            [0.0, 358.64218, 120.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ),
    extrinsic_cv=np.asarray(
        [
            [1.0, 0.0, 0.0, 0.03200001],
            [0.0, -0.8, -0.6, 0.45],
            [0.0, 0.6, -0.8, 1.35],
        ],
        dtype=np.float64,
    ),
)

DEFAULT_GRIPPER_ROI_GEOMETRY = GripperRoiGeometry()


def rotation_from_rpy(roll: float, pitch: float, yaw: float) -> NDArray:
    """Return RoboTwin's static-XYZ rotation, equivalently ``Rz @ Ry @ Rx``."""

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.asarray([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.asarray([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    rotation: NDArray = rz @ ry @ rx
    return rotation


def _project_world_points(
    world_xyz: NDArray,
    calibration: CameraCalibration,
) -> tuple[NDArray, NDArray]:
    points = np.asarray(world_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"world_xyz must be [N,3], got {points.shape}")
    camera_xyz = (
        calibration.extrinsic_cv[:, :3] @ points.T
    ).T + calibration.extrinsic_cv[:, 3]
    depths = camera_xyz[:, 2]
    if np.any(depths <= 1e-8):
        raise ValueError("cannot project a gripper ROI at or behind the camera plane")
    homogeneous = (calibration.intrinsic_cv @ camera_xyz.T).T
    pixels: NDArray = homogeneous[:, :2] / homogeneous[:, 2:3]
    return pixels, depths


def _convex_hull(points_xy: NDArray) -> NDArray:
    points = sorted(set(map(tuple, np.asarray(points_xy, dtype=np.float64).tolist())))
    if len(points) <= 2:
        return np.asarray(points, dtype=np.float64)

    def cross(
        origin: tuple[float, float],
        a: tuple[float, float],
        b: tuple[float, float],
    ) -> float:
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (
            a[1] - origin[1]
        ) * (b[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def project_gripper_roi(
    eef_pose_xyzrpy: NDArray,
    gripper_open: float,
    *,
    calibration: CameraCalibration = CAM_HIGH_CALIBRATION,
    geometry: GripperRoiGeometry = DEFAULT_GRIPPER_ROI_GEOMETRY,
) -> ProjectedGripperRoi:
    """Project one state-derived compact ROI without inspecting RGB pixels."""

    pose = np.asarray(eef_pose_xyzrpy, dtype=np.float64)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError("eef_pose_xyzrpy must be one finite six-dimensional pose")
    if not np.isfinite(gripper_open):
        raise ValueError("gripper_open must be finite")

    rotation = rotation_from_rpy(*pose[3:6])
    eef_world = pose[:3]
    tcp_world = eef_world + rotation @ np.asarray([geometry.tcp_offset_m, 0.0, 0.0])
    open_fraction = float(np.clip(gripper_open, 0.0, 1.0))
    half_width = geometry.closed_half_width_m + open_fraction * (
        geometry.open_half_width_m - geometry.closed_half_width_m
    )
    local_corners = np.asarray(
        list(
            product(
                (-geometry.axial_back_m, geometry.axial_front_m),
                (-half_width, half_width),
                (-geometry.half_thickness_m, geometry.half_thickness_m),
            )
        ),
        dtype=np.float64,
    )
    corner_world = tcp_world + (rotation @ local_corners.T).T
    corner_pixels, corner_depths = _project_world_points(corner_world, calibration)
    centers, _depths = _project_world_points(
        np.stack((eef_world, tcp_world), axis=0),
        calibration,
    )
    hull = _convex_hull(corner_pixels)
    bbox = np.asarray(
        [
            np.min(corner_pixels[:, 0]) - geometry.margin_px,
            np.min(corner_pixels[:, 1]) - geometry.margin_px,
            np.max(corner_pixels[:, 0]) + geometry.margin_px,
            np.max(corner_pixels[:, 1]) + geometry.margin_px,
        ],
        dtype=np.float64,
    )
    return ProjectedGripperRoi(
        eef_pixel_xy=centers[0],
        tcp_pixel_xy=centers[1],
        corner_pixels_xy=corner_pixels,
        hull_pixels_xy=hull,
        bbox_xyxy=bbox,
        corner_depths=corner_depths,
        open_fraction=open_fraction,
    )


def normalized_roi_box(
    roi: ProjectedGripperRoi,
    frame_shape: tuple[int, int],
) -> tuple[float, float, float, float] | None:
    height, width = frame_shape
    x0 = max(0, min(width, int(np.floor(roi.bbox_xyxy[0]))))
    y0 = max(0, min(height, int(np.floor(roi.bbox_xyxy[1]))))
    x1 = max(0, min(width, int(np.ceil(roi.bbox_xyxy[2]))))
    y1 = max(0, min(height, int(np.ceil(roi.bbox_xyxy[3]))))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0 / width, y0 / height, x1 / width, y1 / height)


__all__ = [
    "CAM_HIGH_CALIBRATION",
    "DEFAULT_GRIPPER_ROI_GEOMETRY",
    "CameraCalibration",
    "GripperRoiGeometry",
    "ProjectedGripperRoi",
    "normalized_roi_box",
    "project_gripper_roi",
    "rotation_from_rpy",
]
