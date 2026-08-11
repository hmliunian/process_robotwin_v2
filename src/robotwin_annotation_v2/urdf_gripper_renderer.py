"""Deterministic Aloha URDF gripper rendering and depth-based finger fitting.

The module deliberately keeps its import surface lightweight.  URDF parsing,
forward kinematics, joint conversion, and candidate ranking require only the
standard library and NumPy.  ``trimesh`` and ``pyrender`` are imported lazily
when :class:`AlohaUrdfRenderer` is constructed.

RoboTwin records a normalized gripper *drive command*, not the realized
prismatic joint position.  During contact the fingers can be stopped by an
object well before the command target.  Consequently, the command conversion
helpers are suitable for open/free-space poses and priors; contact frames
should use :meth:`AlohaUrdfRenderer.fit_finger_q`.
"""

from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.floating[Any]]
BoolArray = NDArray[np.bool_]
UIntArray = NDArray[np.unsignedinteger[Any]]
ArmSide = Literal["left", "right"]

GRIPPER_DRIVE_CLOSED_M = -0.01
GRIPPER_DRIVE_SPAN_M = 0.055
GRIPPER_KINEMATIC_MIN_M = 0.0
GRIPPER_KINEMATIC_MAX_M = 0.04765

ALOHA_RENDER_LINKS = tuple(
    f"{prefix}_{suffix}"
    for prefix in ("fl", "fr")
    for suffix in (
        "base_link",
        "link1",
        "link2",
        "link3",
        "link4",
        "link5",
        "link6",
        "link7",
        "link8",
    )
)


class UrdfRendererError(RuntimeError):
    """The URDF model or rendering backend cannot satisfy the contract."""


@dataclass(frozen=True)
class UrdfVisual:
    """One mesh visual and its transform in the owning link frame."""

    mesh_filename: str
    origin: FloatArray
    scale: FloatArray


@dataclass(frozen=True)
class UrdfJoint:
    """Kinematic information needed from one URDF joint."""

    name: str
    joint_type: str
    parent: str
    child: str
    origin: FloatArray
    axis: FloatArray
    lower: float | None
    upper: float | None


@dataclass(frozen=True)
class UrdfModel:
    """Parsed mesh visuals and a validated, acyclic URDF kinematic tree."""

    path: Path
    links: tuple[str, ...]
    root_link: str
    joints: tuple[UrdfJoint, ...]
    children_by_link: Mapping[str, tuple[UrdfJoint, ...]]
    visuals_by_link: Mapping[str, tuple[UrdfVisual, ...]]


@dataclass(frozen=True)
class DepthAgreement:
    """Robust rendered-vs-recorded depth statistics for a projected mask."""

    rendered_pixels: int
    comparable_pixels: int
    consistent_pixels: int
    consistent_fraction: float
    median_residual_mm: float | None
    p90_residual_mm: float | None
    median_signed_residual_mm: float | None = None
    rendered_in_front_pixels: int = 0
    rendered_behind_pixels: int = 0

    def as_dict(self) -> dict[str, int | float | None]:
        return {
            "rendered_pixels": self.rendered_pixels,
            "comparable_pixels": self.comparable_pixels,
            "consistent_pixels": self.consistent_pixels,
            "consistent_fraction": self.consistent_fraction,
            "median_residual_mm": self.median_residual_mm,
            "p90_residual_mm": self.p90_residual_mm,
            "median_signed_residual_mm": self.median_signed_residual_mm,
            "rendered_in_front_pixels": self.rendered_in_front_pixels,
            "rendered_behind_pixels": self.rendered_behind_pixels,
        }


@dataclass(frozen=True)
class FingerCandidateScore:
    """Evidence for one joint-q hypothesis shared by link7 and link8."""

    q_m: float
    agreement: DepthAgreement
    link7_agreement: DepthAgreement
    link8_agreement: DepthAgreement
    joint_name: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "q_m": self.q_m,
            "agreement": self.agreement.as_dict(),
            "link7_agreement": self.link7_agreement.as_dict(),
            "link8_agreement": self.link8_agreement.as_dict(),
            "joint_name": self.joint_name,
        }


@dataclass(frozen=True)
class UrdfRenderResult:
    """A robot-only Z-buffer render for one camera and joint state.

    ``active_gripper_mask`` is amodal with respect to the recorded scene but
    respects self-occlusion by either robot arm.  It is exactly the union of
    active-side link6, link7, and link8 segmentation pixels.
    """

    active_side: ArmSide
    robot_mask: BoolArray
    robot_depth_mm: FloatArray
    active_gripper_mask: BoolArray
    active_gripper_depth_mm: FloatArray
    fixed_link6_mask: BoolArray
    finger_link7_mask: BoolArray
    finger_link8_mask: BoolArray
    per_link_masks: Mapping[str, BoolArray]
    per_link_depth_mm: Mapping[str, FloatArray]
    segmentation_ids: UIntArray
    joint_positions: Mapping[str, float]

    @property
    def amodal_gripper_mask(self) -> BoolArray:
        """Alias spelling out the scene-amodal semantics."""

        return self.active_gripper_mask

    @property
    def finger_mask(self) -> BoolArray:
        """The link7/link8 union, excluding fixed wrist/palm link6."""

        return self.finger_link7_mask | self.finger_link8_mask


@dataclass(frozen=True)
class FingerFitDiagnostics:
    """Serializable evidence and thresholds for a finger-q decision."""

    reason: str | None
    search_mode: Literal["prior_fast_path", "coordinate_sweep"]
    tolerance_mm: float
    command_q_m: float
    temporal_prior_q_by_joint: Mapping[str, float]
    temporal_max_delta_m: float | None
    minimum_support_pixels: int
    minimum_per_link_support_pixels: int
    minimum_consistent_fraction: float
    minimum_fast_path_fraction: float
    maximum_median_residual_mm: float
    minimum_fixed_support_pixels: int
    fixed_link6_agreement: DepthAgreement
    final_link7_agreement: DepthAgreement
    final_link8_agreement: DepthAgreement
    component_acceptance: Mapping[str, bool]
    selected_score_by_joint: Mapping[str, FingerCandidateScore]
    ranked_candidates_by_joint: Mapping[str, tuple[FingerCandidateScore, ...]]

    def as_dict(self) -> dict[str, object]:
        return {
            "reason": self.reason,
            "search_mode": self.search_mode,
            "tolerance_mm": self.tolerance_mm,
            "command_q_m": self.command_q_m,
            "temporal_prior_q_by_joint": dict(self.temporal_prior_q_by_joint),
            "temporal_max_delta_m": self.temporal_max_delta_m,
            "minimum_support_pixels": self.minimum_support_pixels,
            "minimum_per_link_support_pixels": self.minimum_per_link_support_pixels,
            "minimum_consistent_fraction": self.minimum_consistent_fraction,
            "minimum_fast_path_fraction": self.minimum_fast_path_fraction,
            "maximum_median_residual_mm": self.maximum_median_residual_mm,
            "minimum_fixed_support_pixels": self.minimum_fixed_support_pixels,
            "fixed_link6_agreement": self.fixed_link6_agreement.as_dict(),
            "final_link7_agreement": self.final_link7_agreement.as_dict(),
            "final_link8_agreement": self.final_link8_agreement.as_dict(),
            "component_acceptance": dict(self.component_acceptance),
            "selected_score_by_joint": {
                name: score.as_dict() for name, score in self.selected_score_by_joint.items()
            },
            "ranked_candidates_by_joint": {
                name: [candidate.as_dict() for candidate in candidates]
                for name, candidates in self.ranked_candidates_by_joint.items()
            },
        }


@dataclass(frozen=True)
class FingerFitResult:
    """Selected contact-aware finger pose and fail-closed visible mask."""

    accepted: bool
    selected_q_m: float | None
    selected_q_by_joint: Mapping[str, float]
    selected_render: UrdfRenderResult
    visible_mask: BoolArray
    component_visible_masks: Mapping[str, BoolArray]
    component_acceptance: Mapping[str, bool]
    diagnostics: FingerFitDiagnostics


def _parse_vector(text: str | None, default: Sequence[float]) -> FloatArray:
    if text is None:
        result = np.asarray(default, dtype=np.float64)
    else:
        try:
            result = np.asarray([float(value) for value in text.split()], dtype=np.float64)
        except ValueError as exc:
            raise UrdfRendererError(f"invalid URDF vector: {text!r}") from exc
    if result.shape != (len(default),) or not np.isfinite(result).all():
        raise UrdfRendererError(f"invalid URDF vector: {text!r}")
    return result


def xyz_rpy_matrix(xyz: Sequence[float], rpy: Sequence[float]) -> FloatArray:
    """Return URDF origin transform, using ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``."""

    xyz_array = np.asarray(xyz, dtype=np.float64)
    rpy_array = np.asarray(rpy, dtype=np.float64)
    if xyz_array.shape != (3,) or rpy_array.shape != (3,):
        raise ValueError("xyz and rpy must each have shape (3,)")
    if not np.isfinite(xyz_array).all() or not np.isfinite(rpy_array).all():
        raise ValueError("xyz and rpy must contain only finite values")
    roll, pitch, yaw = rpy_array
    cr, sr = math.cos(float(roll)), math.sin(float(roll))
    cp, sp = math.cos(float(pitch)), math.sin(float(pitch))
    cy, sy = math.cos(float(yaw)), math.sin(float(yaw))
    rx = np.array(((1, 0, 0), (0, cr, -sr), (0, sr, cr)), dtype=np.float64)
    ry = np.array(((cp, 0, sp), (0, 1, 0), (-sp, 0, cp)), dtype=np.float64)
    rz = np.array(((cy, -sy, 0), (sy, cy, 0), (0, 0, 1)), dtype=np.float64)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rz @ ry @ rx
    transform[:3, 3] = xyz_array
    return transform


def axis_angle_matrix(axis: Sequence[float], angle: float) -> FloatArray:
    """Return a homogeneous rotation about an arbitrary joint-frame axis."""

    axis_array = np.asarray(axis, dtype=np.float64)
    if axis_array.shape != (3,) or not np.isfinite(axis_array).all():
        raise ValueError("axis must be a finite shape-(3,) vector")
    norm = float(np.linalg.norm(axis_array))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("axis must be non-zero")
    if not math.isfinite(float(angle)):
        raise ValueError("angle must be finite")
    x, y, z = axis_array / norm
    skew = np.array(((0, -z, y), (z, 0, -x), (-y, x, 0)), dtype=np.float64)
    rotation = np.eye(3, dtype=np.float64)
    rotation += math.sin(float(angle)) * skew
    rotation += (1.0 - math.cos(float(angle))) * (skew @ skew)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    return transform


def wxyz_pose_matrix(position: Sequence[float], quaternion_wxyz: Sequence[float]) -> FloatArray:
    """Convert a SAPIEN-style ``(w, x, y, z)`` pose to a 4x4 matrix."""

    position_array = np.asarray(position, dtype=np.float64)
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    if position_array.shape != (3,) or quaternion.shape != (4,):
        raise ValueError("position must be (3,) and quaternion_wxyz must be (4,)")
    if not np.isfinite(position_array).all() or not np.isfinite(quaternion).all():
        raise ValueError("pose values must be finite")
    norm = float(np.linalg.norm(quaternion))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("quaternion must be non-zero")
    w, x, y, z = quaternion / norm
    rotation = np.array(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = position_array
    return transform


ALOHA_ROOT_POSE = wxyz_pose_matrix(
    (0.0, -0.65, 0.0),
    (0.707, 0.0, 0.0, 0.707),
)


def _origin_matrix(element: ET.Element | None) -> FloatArray:
    if element is None:
        return np.eye(4, dtype=np.float64)
    return xyz_rpy_matrix(
        _parse_vector(element.get("xyz"), (0.0, 0.0, 0.0)),
        _parse_vector(element.get("rpy"), (0.0, 0.0, 0.0)),
    )


def _optional_limit(element: ET.Element | None, name: str) -> float | None:
    if element is None or element.get(name) is None:
        return None
    try:
        result = float(element.get(name, ""))
    except ValueError as exc:
        raise UrdfRendererError(f"invalid joint {name} limit") from exc
    if not math.isfinite(result):
        raise UrdfRendererError(f"invalid joint {name} limit")
    return result


def load_urdf(path: str | Path) -> UrdfModel:
    """Parse the URDF subset needed for exact visual FK and rendering."""

    urdf_path = Path(path).expanduser().resolve()
    if not urdf_path.is_file():
        raise UrdfRendererError(f"URDF is missing: {urdf_path}")
    try:
        root = ET.parse(urdf_path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise UrdfRendererError(f"cannot parse URDF: {urdf_path}") from exc
    if root.tag != "robot":
        raise UrdfRendererError(f"expected <robot> root in {urdf_path}")

    link_elements = root.findall("link")
    links = tuple(element.get("name", "") for element in link_elements)
    if not links or any(not name for name in links) or len(set(links)) != len(links):
        raise UrdfRendererError("URDF links must have unique non-empty names")
    link_set = set(links)

    visuals_by_link: dict[str, tuple[UrdfVisual, ...]] = {}
    for link_element in link_elements:
        link_name = link_element.get("name", "")
        visuals: list[UrdfVisual] = []
        for visual_element in link_element.findall("visual"):
            mesh_element = visual_element.find("geometry/mesh")
            if mesh_element is None:
                continue
            filename = mesh_element.get("filename")
            if not filename:
                raise UrdfRendererError(f"mesh visual on {link_name} has no filename")
            scale = _parse_vector(mesh_element.get("scale"), (1.0, 1.0, 1.0))
            visuals.append(
                UrdfVisual(
                    mesh_filename=filename,
                    origin=_origin_matrix(visual_element.find("origin")),
                    scale=scale,
                )
            )
        visuals_by_link[link_name] = tuple(visuals)

    joints: list[UrdfJoint] = []
    child_links: set[str] = set()
    seen_joint_names: set[str] = set()
    children: dict[str, list[UrdfJoint]] = defaultdict(list)
    supported_types = {"fixed", "revolute", "continuous", "prismatic"}
    for joint_element in root.findall("joint"):
        name = joint_element.get("name", "")
        joint_type = joint_element.get("type", "")
        parent_element = joint_element.find("parent")
        child_element = joint_element.find("child")
        parent = "" if parent_element is None else parent_element.get("link", "")
        child = "" if child_element is None else child_element.get("link", "")
        if not name or name in seen_joint_names:
            raise UrdfRendererError("URDF joints must have unique non-empty names")
        if joint_type not in supported_types:
            raise UrdfRendererError(f"unsupported joint type {joint_type!r} for {name}")
        if parent not in link_set or child not in link_set:
            raise UrdfRendererError(f"joint {name} references an unknown link")
        if child in child_links:
            raise UrdfRendererError(f"link {child} has more than one parent")
        axis_element = joint_element.find("axis")
        axis = _parse_vector(
            None if axis_element is None else axis_element.get("xyz"),
            (1.0, 0.0, 0.0),
        )
        if joint_type != "fixed" and float(np.linalg.norm(axis)) <= np.finfo(np.float64).eps:
            raise UrdfRendererError(f"moving joint {name} has a zero axis")
        limit_element = joint_element.find("limit")
        joint = UrdfJoint(
            name=name,
            joint_type=joint_type,
            parent=parent,
            child=child,
            origin=_origin_matrix(joint_element.find("origin")),
            axis=axis,
            lower=_optional_limit(limit_element, "lower"),
            upper=_optional_limit(limit_element, "upper"),
        )
        joints.append(joint)
        children[parent].append(joint)
        child_links.add(child)
        seen_joint_names.add(name)

    root_links = sorted(link_set - child_links)
    if len(root_links) != 1:
        raise UrdfRendererError(f"URDF must have exactly one root link, got {root_links}")
    model = UrdfModel(
        path=urdf_path,
        links=links,
        root_link=root_links[0],
        joints=tuple(joints),
        children_by_link={name: tuple(value) for name, value in children.items()},
        visuals_by_link=visuals_by_link,
    )
    # Validate reachability and catch cycles before any rendering resources load.
    forward_kinematics(model, {})
    return model


def _validate_transform(value: FloatArray | Sequence[Sequence[float]], *, name: str) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (4, 4) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    return result


def forward_kinematics(
    model: UrdfModel,
    joint_positions: Mapping[str, float],
    root_pose: FloatArray | Sequence[Sequence[float]] = ALOHA_ROOT_POSE,
) -> dict[str, FloatArray]:
    """Compute link-to-world transforms with URDF joint-frame motion order."""

    root_transform = _validate_transform(root_pose, name="root_pose")
    unknown = set(joint_positions) - {joint.name for joint in model.joints}
    if unknown:
        raise ValueError(f"unknown URDF joints: {sorted(unknown)}")
    transforms: dict[str, FloatArray] = {model.root_link: root_transform.copy()}
    stack = [model.root_link]
    while stack:
        parent = stack.pop()
        for joint in model.children_by_link.get(parent, ()):
            if joint.child in transforms:
                raise UrdfRendererError(f"cycle reaches URDF link {joint.child}")
            value = float(joint_positions.get(joint.name, 0.0))
            if not math.isfinite(value):
                raise ValueError(f"joint {joint.name} must be finite")
            transform = transforms[parent] @ joint.origin
            if joint.joint_type in {"revolute", "continuous"}:
                transform = transform @ axis_angle_matrix(joint.axis, value)
            elif joint.joint_type == "prismatic":
                motion = np.eye(4, dtype=np.float64)
                motion[:3, 3] = joint.axis * value
                transform = transform @ motion
            transforms[joint.child] = transform
            stack.append(joint.child)
    missing = set(model.links) - set(transforms)
    if missing:
        raise UrdfRendererError(f"FK did not reach links: {sorted(missing)}")
    return transforms


def gripper_command_to_drive_target(command: float) -> float:
    """Map normalized RoboTwin command to the simulator drive target in metres."""

    result = GRIPPER_DRIVE_CLOSED_M + float(command) * GRIPPER_DRIVE_SPAN_M
    if not math.isfinite(result):
        raise ValueError("gripper command must be finite")
    return result


def gripper_command_to_kinematic_q(command: float) -> float:
    """Map and clamp a command to the representable URDF prismatic range."""

    return float(
        np.clip(
            gripper_command_to_drive_target(command),
            GRIPPER_KINEMATIC_MIN_M,
            GRIPPER_KINEMATIC_MAX_M,
        )
    )


def aloha_joint_positions(
    joint_absolute: Sequence[float] | FloatArray,
    finger_q_by_joint: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Map RoboTwin's 14D Aloha ordering to named URDF joints.

    The ordering is ``fl_joint1..6, fl_gripper, fr_joint1..6, fr_gripper``.
    Optional finger overrides are limited to ``fl/fr_joint7/8`` and are useful
    for contact-aware fitting.
    """

    values = np.asarray(joint_absolute, dtype=np.float64)
    if values.shape != (14,) or not np.isfinite(values).all():
        raise ValueError("joint_absolute must be a finite shape-(14,) vector")
    result: dict[str, float] = {}
    for prefix, offset in (("fl", 0), ("fr", 7)):
        for index in range(6):
            result[f"{prefix}_joint{index + 1}"] = float(values[offset + index])
        q_m = gripper_command_to_kinematic_q(float(values[offset + 6]))
        result[f"{prefix}_joint7"] = q_m
        result[f"{prefix}_joint8"] = q_m
    if finger_q_by_joint:
        valid_names = {f"{prefix}_joint{index}" for prefix in ("fl", "fr") for index in (7, 8)}
        unknown = set(finger_q_by_joint) - valid_names
        if unknown:
            raise ValueError(f"invalid finger joint overrides: {sorted(unknown)}")
        for name, raw_q in finger_q_by_joint.items():
            q_m = float(raw_q)
            if not math.isfinite(q_m):
                raise ValueError(f"finger joint {name} must be finite")
            if not GRIPPER_KINEMATIC_MIN_M <= q_m <= GRIPPER_KINEMATIC_MAX_M:
                raise ValueError(
                    f"finger joint {name}={q_m} is outside "
                    f"[{GRIPPER_KINEMATIC_MIN_M}, {GRIPPER_KINEMATIC_MAX_M}]"
                )
            result[name] = q_m
    return result


def _validate_depth_inputs(
    mask: BoolArray | NDArray[Any],
    rendered_depth_mm: FloatArray | NDArray[Any],
    scene_depth_mm: FloatArray | NDArray[Any],
    tolerance_mm: float,
) -> tuple[BoolArray, FloatArray, FloatArray, float]:
    mask_array = np.asarray(mask, dtype=bool)
    rendered = np.asarray(rendered_depth_mm, dtype=np.float64)
    scene = np.asarray(scene_depth_mm, dtype=np.float64)
    tolerance = float(tolerance_mm)
    if (
        mask_array.ndim != 2
        or rendered.shape != mask_array.shape
        or scene.shape != mask_array.shape
    ):
        raise ValueError("mask, rendered depth, and scene depth must share one 2D shape")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance_mm must be finite and non-negative")
    return mask_array, rendered, scene, tolerance


def compute_visible_gripper_mask(
    active_gripper_mask: BoolArray | NDArray[Any],
    rendered_depth_mm: FloatArray | NDArray[Any],
    scene_depth_mm: FloatArray | NDArray[Any],
    tolerance_mm: float = 8.0,
) -> BoolArray:
    """Keep projected gripper pixels whose positive scene depth agrees."""

    mask, rendered, scene, tolerance = _validate_depth_inputs(
        active_gripper_mask,
        rendered_depth_mm,
        scene_depth_mm,
        tolerance_mm,
    )
    valid = mask & np.isfinite(rendered) & np.isfinite(scene) & (rendered > 0) & (scene > 0)
    return valid & (np.abs(rendered - scene) <= tolerance)


def depth_agreement(
    mask: BoolArray | NDArray[Any],
    rendered_depth_mm: FloatArray | NDArray[Any],
    scene_depth_mm: FloatArray | NDArray[Any],
    tolerance_mm: float = 8.0,
) -> DepthAgreement:
    """Summarize depth support, treating missing scene depth as incomparable."""

    mask_array, rendered, scene, tolerance = _validate_depth_inputs(
        mask,
        rendered_depth_mm,
        scene_depth_mm,
        tolerance_mm,
    )
    rendered_mask = mask_array & np.isfinite(rendered) & (rendered > 0)
    comparable = rendered_mask & np.isfinite(scene) & (scene > 0)
    signed_residual = rendered[comparable] - scene[comparable]
    residual = np.abs(signed_residual)
    consistent_pixels = int(np.count_nonzero(residual <= tolerance))
    comparable_pixels = int(residual.size)
    return DepthAgreement(
        rendered_pixels=int(np.count_nonzero(rendered_mask)),
        comparable_pixels=comparable_pixels,
        consistent_pixels=consistent_pixels,
        consistent_fraction=(consistent_pixels / comparable_pixels if comparable_pixels else 0.0),
        median_residual_mm=(float(np.median(residual)) if comparable_pixels else None),
        p90_residual_mm=(float(np.percentile(residual, 90)) if comparable_pixels else None),
        median_signed_residual_mm=(
            float(np.median(signed_residual)) if comparable_pixels else None
        ),
        rendered_in_front_pixels=int(np.count_nonzero(signed_residual < -tolerance)),
        rendered_behind_pixels=int(np.count_nonzero(signed_residual > tolerance)),
    )


def _finite_metric(value: float | None) -> float:
    return float(value) if value is not None and math.isfinite(float(value)) else math.inf


def rank_finger_candidates(
    candidates: Sequence[FingerCandidateScore],
    *,
    command_prior_q_m: float | None = None,
    temporal_prior_q_m: float | None = None,
    maximum_median_residual_mm: float | None = None,
) -> tuple[FingerCandidateScore, ...]:
    """Rank q hypotheses by depth evidence, using priors only as tie-breaks.

    A configured median-residual limit is a viability gate that rejects broad
    object-surface coincidences.  Among viable candidates, total consistent
    pixels are deliberately the first evidence key.  Consistent fraction and
    robust residuals follow; temporal and command priors cannot overrule
    stronger observed depth evidence.
    """

    for name, prior in (
        ("command_prior_q_m", command_prior_q_m),
        ("temporal_prior_q_m", temporal_prior_q_m),
    ):
        if prior is not None and not math.isfinite(float(prior)):
            raise ValueError(f"{name} must be finite when provided")
    if maximum_median_residual_mm is not None and (
        not math.isfinite(float(maximum_median_residual_mm)) or maximum_median_residual_mm < 0
    ):
        raise ValueError("maximum_median_residual_mm must be finite and non-negative")

    def key(candidate: FingerCandidateScore) -> tuple[float, ...]:
        agreement = candidate.agreement
        temporal_distance = (
            abs(candidate.q_m - float(temporal_prior_q_m))
            if temporal_prior_q_m is not None
            else 0.0
        )
        command_distance = (
            abs(candidate.q_m - float(command_prior_q_m)) if command_prior_q_m is not None else 0.0
        )
        robust_residual = _finite_metric(agreement.median_residual_mm)
        robust_viable = maximum_median_residual_mm is None or robust_residual <= float(
            maximum_median_residual_mm
        )
        return (
            float(robust_viable),
            float(agreement.consistent_pixels),
            float(agreement.consistent_fraction),
            -robust_residual,
            -_finite_metric(agreement.p90_residual_mm),
            float(
                min(
                    candidate.link7_agreement.consistent_pixels,
                    candidate.link8_agreement.consistent_pixels,
                )
            ),
            -temporal_distance,
            -command_distance,
            -candidate.q_m,
        )

    return tuple(sorted(candidates, key=key, reverse=True))


def candidate_has_minimum_support(
    candidate: FingerCandidateScore,
    *,
    minimum_support_pixels: int,
    minimum_per_link_support_pixels: int,
    minimum_consistent_fraction: float,
) -> bool:
    """Return whether both fingers have enough scene-depth evidence."""

    if minimum_support_pixels < 0 or minimum_per_link_support_pixels < 0:
        raise ValueError("support thresholds must be non-negative")
    if not 0.0 <= minimum_consistent_fraction <= 1.0:
        raise ValueError("minimum_consistent_fraction must be within [0, 1]")
    return bool(
        agreement_has_minimum_support(
            candidate.agreement,
            minimum_support_pixels=minimum_support_pixels,
            minimum_consistent_fraction=minimum_consistent_fraction,
        )
        and candidate.link7_agreement.consistent_pixels >= minimum_per_link_support_pixels
        and candidate.link8_agreement.consistent_pixels >= minimum_per_link_support_pixels
    )


def agreement_has_minimum_support(
    agreement: DepthAgreement,
    *,
    minimum_support_pixels: int,
    minimum_consistent_fraction: float,
    maximum_median_residual_mm: float | None = None,
) -> bool:
    """Apply fail-closed pixel and fraction gates to one rendered component."""

    if minimum_support_pixels < 0:
        raise ValueError("minimum_support_pixels must be non-negative")
    if not 0.0 <= minimum_consistent_fraction <= 1.0:
        raise ValueError("minimum_consistent_fraction must be within [0, 1]")
    if maximum_median_residual_mm is not None and (
        not math.isfinite(float(maximum_median_residual_mm)) or maximum_median_residual_mm < 0
    ):
        raise ValueError("maximum_median_residual_mm must be finite and non-negative")
    median_residual = _finite_metric(agreement.median_residual_mm)
    return bool(
        agreement.consistent_pixels >= minimum_support_pixels
        and agreement.consistent_fraction >= minimum_consistent_fraction
        and (
            maximum_median_residual_mm is None
            or median_residual <= float(maximum_median_residual_mm)
        )
    )


def _normalize_active_side(active_side: str) -> tuple[ArmSide, str]:
    if active_side == "left":
        return "left", "fl"
    if active_side == "right":
        return "right", "fr"
    raise ValueError("active_side must be 'left' or 'right'")


def active_gripper_link_names(active_side: ArmSide) -> tuple[str, str, str]:
    """Return fixed wrist/palm link6 followed by finger links 7 and 8."""

    _, prefix = _normalize_active_side(active_side)
    return (f"{prefix}_link6", f"{prefix}_link7", f"{prefix}_link8")


def _validate_intrinsics(value: FloatArray | NDArray[Any]) -> FloatArray:
    intrinsic = np.asarray(value, dtype=np.float64)
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise ValueError("intrinsic_cv must be a finite 3x3 matrix")
    if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
        raise ValueError("camera focal lengths must be positive")
    return intrinsic


def _grid(lower: float, upper: float, step: float) -> tuple[float, ...]:
    if not lower <= upper or step <= 0 or not all(map(math.isfinite, (lower, upper, step))):
        raise ValueError("invalid q search range or step")
    count = int(math.floor((upper - lower) / step))
    values = [lower + index * step for index in range(count + 1)]
    if not values or upper - values[-1] > 1e-12:
        values.append(upper)
    return tuple(float(np.clip(value, lower, upper)) for value in values)


class AlohaUrdfRenderer:
    """Reusable EGL renderer for exact RoboTwin Aloha visual meshes."""

    def __init__(
        self,
        urdf_path: str | Path,
        *,
        mesh_root: str | Path | None = None,
        width: int = 320,
        height: int = 240,
        root_pose: FloatArray | Sequence[Sequence[float]] = ALOHA_ROOT_POSE,
        znear: float = 0.01,
        zfar: float = 10.0,
    ) -> None:
        if isinstance(width, bool) or isinstance(height, bool) or width <= 0 or height <= 0:
            raise ValueError("width and height must be positive integers")
        if not (math.isfinite(znear) and math.isfinite(zfar) and 0 < znear < zfar):
            raise ValueError("camera clipping planes must satisfy 0 < znear < zfar")
        self.model = load_urdf(urdf_path)
        missing_links = set(ALOHA_RENDER_LINKS) - set(self.model.links)
        if missing_links:
            raise UrdfRendererError(f"Aloha URDF is missing render links: {sorted(missing_links)}")
        self.width = int(width)
        self.height = int(height)
        self.root_pose = _validate_transform(root_pose, name="root_pose").copy()
        self.znear = float(znear)
        self.zfar = float(zfar)
        self.mesh_root = (
            Path(mesh_root).expanduser().resolve()
            if mesh_root is not None
            else self.model.path.parent
        )
        self._closed = False
        self._load_backend()
        try:
            self._build_scene()
        except Exception:
            renderer = getattr(self, "_renderer", None)
            if renderer is not None:
                renderer.delete()
            raise

    def _load_backend(self) -> None:
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        os.environ.setdefault("MESA_SHADER_CACHE_DISABLE", "true")
        try:
            import pyrender
            import trimesh
        except ImportError as exc:
            raise UrdfRendererError(
                "URDF rendering requires the optional trimesh and pyrender packages"
            ) from exc
        self._pyrender = pyrender
        self._trimesh = trimesh

    def _resolve_mesh_path(self, filename: str) -> Path:
        normalized = filename
        if normalized.startswith("package://"):
            normalized = normalized[len("package://") :]
        candidate = Path(normalized)
        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            resolved = (self.mesh_root / candidate).resolve()
        if not resolved.is_file():
            raise UrdfRendererError(f"URDF visual mesh is missing: {resolved}")
        return resolved

    def _load_visual_mesh(self, path: Path, scale: FloatArray) -> Any:
        loaded = self._trimesh.load(path, force="scene", process=False)
        if isinstance(loaded, self._trimesh.Trimesh):
            mesh = loaded
        else:
            # Applies transforms embedded in Collada before concatenation.
            mesh = (
                loaded.to_geometry()
                if hasattr(loaded, "to_geometry")
                else loaded.dump(concatenate=True)
            )
        if not np.allclose(scale, 1.0):
            mesh.apply_scale(scale)
        return mesh

    @staticmethod
    def _id_color(link_id: int) -> tuple[int, int, int]:
        return (link_id & 255, (link_id >> 8) & 255, (link_id >> 16) & 255)

    def _build_scene(self) -> None:
        pyrender = self._pyrender
        self._scene = pyrender.Scene(
            bg_color=np.array((0, 0, 0, 0), dtype=np.uint8),
            ambient_light=np.zeros(3, dtype=np.float32),
        )
        self._nodes_by_link: dict[str, list[tuple[Any, FloatArray]]] = defaultdict(list)
        self._seg_node_map: dict[Any, tuple[int, int, int]] = {}
        self._link_ids: dict[str, int] = {}
        for link_id, link_name in enumerate(ALOHA_RENDER_LINKS, start=1):
            visuals = self.model.visuals_by_link.get(link_name, ())
            if not visuals:
                raise UrdfRendererError(f"render link has no mesh visual: {link_name}")
            self._link_ids[link_name] = link_id
            for visual_index, visual in enumerate(visuals):
                mesh_path = self._resolve_mesh_path(visual.mesh_filename)
                tri_mesh = self._load_visual_mesh(mesh_path, visual.scale)
                render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth=False)
                node = self._scene.add(
                    render_mesh,
                    pose=np.eye(4, dtype=np.float64),
                    name=f"{link_name}:{visual_index}",
                )
                self._nodes_by_link[link_name].append((node, visual.origin))
                self._seg_node_map[node] = self._id_color(link_id)
        initial_camera = pyrender.IntrinsicsCamera(
            fx=1.0,
            fy=1.0,
            cx=0.0,
            cy=0.0,
            znear=self.znear,
            zfar=self.zfar,
        )
        self._camera_node = self._scene.add(
            initial_camera,
            pose=np.eye(4, dtype=np.float64),
            name="robotwin_camera",
        )
        self._renderer = pyrender.OffscreenRenderer(
            viewport_width=self.width,
            viewport_height=self.height,
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise UrdfRendererError("renderer is closed")

    def render(
        self,
        joint_absolute: Sequence[float] | FloatArray,
        intrinsic_cv: FloatArray | NDArray[Any],
        cam2world_gl: FloatArray | NDArray[Any],
        *,
        active_side: ArmSide,
        finger_q_by_joint: Mapping[str, float] | None = None,
    ) -> UrdfRenderResult:
        """Render all follower-arm links and select one link6+7+8 gripper."""

        self._ensure_open()
        side, _ = _normalize_active_side(active_side)
        intrinsic = _validate_intrinsics(intrinsic_cv)
        camera_pose = _validate_transform(cam2world_gl, name="cam2world_gl")
        joint_positions = aloha_joint_positions(joint_absolute, finger_q_by_joint)
        link_to_world = forward_kinematics(self.model, joint_positions, self.root_pose)
        for link_name, nodes in self._nodes_by_link.items():
            for node, visual_origin in nodes:
                self._scene.set_pose(node, link_to_world[link_name] @ visual_origin)

        self._camera_node.camera = self._pyrender.IntrinsicsCamera(
            fx=float(intrinsic[0, 0]),
            fy=float(intrinsic[1, 1]),
            cx=float(intrinsic[0, 2]),
            cy=float(intrinsic[1, 2]),
            znear=self.znear,
            zfar=self.zfar,
        )
        self._scene.set_pose(self._camera_node, camera_pose)
        flags = self._pyrender.RenderFlags.SEG | self._pyrender.RenderFlags.SKIP_CULL_FACES
        segmentation_rgb, depth_m = self._renderer.render(
            self._scene,
            flags=flags,
            seg_node_map=self._seg_node_map,
        )
        colors = np.asarray(segmentation_rgb, dtype=np.uint32)[..., :3]
        segmentation_ids = (colors[..., 0] + (colors[..., 1] << 8) + (colors[..., 2] << 16)).astype(
            np.uint32, copy=False
        )
        robot_depth_mm = np.asarray(depth_m, dtype=np.float32) * np.float32(1000.0)
        per_link_masks = {
            name: np.asarray(segmentation_ids == link_id, dtype=bool)
            for name, link_id in self._link_ids.items()
        }
        per_link_depth_mm = {
            name: np.where(mask, robot_depth_mm, np.float32(0.0)).astype(np.float32, copy=False)
            for name, mask in per_link_masks.items()
        }
        link6_name, link7_name, link8_name = active_gripper_link_names(side)
        fixed_link6_mask = per_link_masks[link6_name]
        finger_link7_mask = per_link_masks[link7_name]
        finger_link8_mask = per_link_masks[link8_name]
        active_gripper_mask = fixed_link6_mask | finger_link7_mask | finger_link8_mask
        return UrdfRenderResult(
            active_side=side,
            robot_mask=np.asarray(segmentation_ids != 0, dtype=bool),
            robot_depth_mm=robot_depth_mm,
            active_gripper_mask=active_gripper_mask,
            active_gripper_depth_mm=np.where(
                active_gripper_mask, robot_depth_mm, np.float32(0.0)
            ).astype(np.float32, copy=False),
            fixed_link6_mask=fixed_link6_mask,
            finger_link7_mask=finger_link7_mask,
            finger_link8_mask=finger_link8_mask,
            per_link_masks=per_link_masks,
            per_link_depth_mm=per_link_depth_mm,
            segmentation_ids=segmentation_ids,
            joint_positions=dict(joint_positions),
        )

    def _score_candidate(
        self,
        rendered: UrdfRenderResult,
        scene_depth_mm: FloatArray,
        tolerance_mm: float,
        q_m: float,
        joint_name: str,
    ) -> FingerCandidateScore:
        link7 = depth_agreement(
            rendered.finger_link7_mask,
            rendered.robot_depth_mm,
            scene_depth_mm,
            tolerance_mm,
        )
        link8 = depth_agreement(
            rendered.finger_link8_mask,
            rendered.robot_depth_mm,
            scene_depth_mm,
            tolerance_mm,
        )
        return FingerCandidateScore(
            q_m=q_m,
            agreement=link7 if joint_name.endswith("joint7") else link8,
            link7_agreement=link7,
            link8_agreement=link8,
            joint_name=joint_name,
        )

    def fit_finger_q(
        self,
        joint_absolute: Sequence[float] | FloatArray,
        intrinsic_cv: FloatArray | NDArray[Any],
        cam2world_gl: FloatArray | NDArray[Any],
        scene_depth_mm: FloatArray | NDArray[Any],
        *,
        active_side: ArmSide,
        tolerance_mm: float = 8.0,
        q_min_m: float = GRIPPER_KINEMATIC_MIN_M,
        q_max_m: float = GRIPPER_KINEMATIC_MAX_M,
        coarse_step_m: float = 0.005,
        fine_step_m: float = 0.0005,
        minimum_support_pixels: int = 64,
        minimum_per_link_support_pixels: int = 24,
        minimum_consistent_fraction: float = 0.15,
        minimum_fast_path_fraction: float = 0.7,
        maximum_median_residual_mm: float | None = None,
        minimum_fixed_support_pixels: int = 16,
        temporal_prior_q_m: float | None = None,
        temporal_prior_q_by_joint: Mapping[str, float] | None = None,
        temporal_max_delta_m: float | None = 0.01,
        minimum_searchable_pixels: int = 4,
    ) -> FingerFitResult:
        """Fit contact-constrained link7/link8 q with a coarse-to-fine sweep.

        Link7 and link8 are fitted independently by coordinate sweeps.  Every
        hypothesis is rendered with both arms, so segmentation and depth
        respect robot self-occlusion.  When a previous per-joint q is supplied,
        each search is physically bounded to ``prior +/- temporal_max_delta_m``;
        temporal continuity therefore has units and cannot be bypassed by a
        one-pixel score difference.

        Link6, link7, and link8 pass depth-support gates independently.  The
        published mask is the union of supported components only: failure of
        one finger never erases a verified palm or the other finger.
        """

        self._ensure_open()
        side, prefix = _normalize_active_side(active_side)
        joints = np.asarray(joint_absolute, dtype=np.float64)
        if joints.shape != (14,) or not np.isfinite(joints).all():
            raise ValueError("joint_absolute must be a finite shape-(14,) vector")
        scene_depth = np.asarray(scene_depth_mm, dtype=np.float64)
        expected_shape = (self.height, self.width)
        if scene_depth.shape != expected_shape:
            raise ValueError(f"scene_depth_mm must have shape {expected_shape}")
        if not math.isfinite(float(tolerance_mm)) or tolerance_mm < 0:
            raise ValueError("tolerance_mm must be finite and non-negative")
        if (
            minimum_support_pixels < 0
            or minimum_fixed_support_pixels < 0
            or minimum_searchable_pixels < 0
        ):
            raise ValueError("support thresholds must be non-negative")
        finger_support_threshold = max(
            minimum_per_link_support_pixels,
            math.ceil(minimum_support_pixels / 2),
        )
        # Validate per-component gates before doing GPU work.
        dummy = DepthAgreement(0, 0, 0, 0.0, None, None)
        agreement_has_minimum_support(
            dummy,
            minimum_support_pixels=finger_support_threshold,
            minimum_consistent_fraction=minimum_consistent_fraction,
        )
        if not 0.0 <= minimum_fast_path_fraction <= 1.0:
            raise ValueError("minimum_fast_path_fraction must be within [0, 1]")
        fast_path_fraction = max(
            minimum_consistent_fraction,
            minimum_fast_path_fraction,
        )
        if maximum_median_residual_mm is None:
            robust_median_limit = min(
                float(tolerance_mm),
                max(2.0, float(tolerance_mm) * 0.25),
            )
        else:
            robust_median_limit = float(maximum_median_residual_mm)
        agreement_has_minimum_support(
            dummy,
            minimum_support_pixels=0,
            minimum_consistent_fraction=0.0,
            maximum_median_residual_mm=robust_median_limit,
        )
        q_lower = max(float(q_min_m), GRIPPER_KINEMATIC_MIN_M)
        q_upper = min(float(q_max_m), GRIPPER_KINEMATIC_MAX_M)
        # Validate the global range and both step sizes.
        _grid(q_lower, q_upper, float(coarse_step_m))
        _grid(q_lower, q_upper, float(fine_step_m))
        command_index = 6 if side == "left" else 13
        command_q = gripper_command_to_kinematic_q(float(joints[command_index]))
        command_q = float(np.clip(command_q, q_lower, q_upper))
        finger_joint_names = (f"{prefix}_joint7", f"{prefix}_joint8")
        if temporal_prior_q_m is not None and temporal_prior_q_by_joint is not None:
            raise ValueError("provide temporal_prior_q_m or temporal_prior_q_by_joint, not both")
        temporal_priors: dict[str, float] = {}
        if temporal_prior_q_m is not None:
            shared_prior = float(temporal_prior_q_m)
            if not math.isfinite(shared_prior):
                raise ValueError("temporal_prior_q_m must be finite when provided")
            temporal_priors = {name: shared_prior for name in finger_joint_names}
        elif temporal_prior_q_by_joint is not None:
            unknown_priors = set(temporal_prior_q_by_joint) - set(finger_joint_names)
            if unknown_priors:
                raise ValueError(f"unexpected temporal prior joints: {sorted(unknown_priors)}")
            for name, value in temporal_prior_q_by_joint.items():
                prior = float(value)
                if not math.isfinite(prior):
                    raise ValueError(f"temporal prior {name} must be finite")
                temporal_priors[name] = prior
        temporal_delta = None if temporal_max_delta_m is None else float(temporal_max_delta_m)
        if temporal_delta is not None and (not math.isfinite(temporal_delta) or temporal_delta < 0):
            raise ValueError("temporal_max_delta_m must be finite and non-negative")
        temporal_priors = {
            name: float(np.clip(value, q_lower, q_upper)) for name, value in temporal_priors.items()
        }
        working_q = {name: temporal_priors.get(name, command_q) for name in finger_joint_names}
        selected_scores: dict[str, FingerCandidateScore] = {}
        ranked_by_joint: dict[str, tuple[FingerCandidateScore, ...]] = {}

        def finish(
            selected_render: UrdfRenderResult,
            *,
            search_mode: Literal["prior_fast_path", "coordinate_sweep"],
        ) -> FingerFitResult:
            fixed_agreement = depth_agreement(
                selected_render.fixed_link6_mask,
                selected_render.robot_depth_mm,
                scene_depth,
                float(tolerance_mm),
            )
            link7_agreement = depth_agreement(
                selected_render.finger_link7_mask,
                selected_render.robot_depth_mm,
                scene_depth,
                float(tolerance_mm),
            )
            link8_agreement = depth_agreement(
                selected_render.finger_link8_mask,
                selected_render.robot_depth_mm,
                scene_depth,
                float(tolerance_mm),
            )
            component_agreements = (fixed_agreement, link7_agreement, link8_agreement)
            component_thresholds = (
                minimum_fixed_support_pixels,
                finger_support_threshold,
                finger_support_threshold,
            )
            link_names = active_gripper_link_names(side)
            component_acceptance = {
                name: agreement_has_minimum_support(
                    agreement,
                    minimum_support_pixels=threshold,
                    minimum_consistent_fraction=minimum_consistent_fraction,
                )
                for name, agreement, threshold in zip(
                    link_names,
                    component_agreements,
                    component_thresholds,
                    strict=True,
                )
            }
            component_masks = {
                link_names[0]: selected_render.fixed_link6_mask,
                link_names[1]: selected_render.finger_link7_mask,
                link_names[2]: selected_render.finger_link8_mask,
            }
            component_visible_masks = {
                name: (
                    compute_visible_gripper_mask(
                        mask,
                        selected_render.robot_depth_mm,
                        scene_depth,
                        float(tolerance_mm),
                    )
                    if component_acceptance[name]
                    else np.zeros(expected_shape, dtype=bool)
                )
                for name, mask in component_masks.items()
            }
            visible_mask = np.zeros(expected_shape, dtype=bool)
            for component_mask in component_visible_masks.values():
                visible_mask |= component_mask
            failed_components = [
                name for name, is_supported in component_acceptance.items() if not is_supported
            ]
            reason = (
                "insufficient_depth_support:" + ",".join(failed_components)
                if failed_components
                else None
            )
            diagnostics = FingerFitDiagnostics(
                reason=reason,
                search_mode=search_mode,
                tolerance_mm=float(tolerance_mm),
                command_q_m=command_q,
                temporal_prior_q_by_joint=temporal_priors,
                temporal_max_delta_m=temporal_delta,
                minimum_support_pixels=minimum_support_pixels,
                minimum_per_link_support_pixels=minimum_per_link_support_pixels,
                minimum_consistent_fraction=minimum_consistent_fraction,
                minimum_fast_path_fraction=fast_path_fraction,
                maximum_median_residual_mm=robust_median_limit,
                minimum_fixed_support_pixels=minimum_fixed_support_pixels,
                fixed_link6_agreement=fixed_agreement,
                final_link7_agreement=link7_agreement,
                final_link8_agreement=link8_agreement,
                component_acceptance=component_acceptance,
                selected_score_by_joint=dict(selected_scores),
                ranked_candidates_by_joint=dict(ranked_by_joint),
            )
            selected_q_by_joint = dict(working_q)
            q7, q8 = (selected_q_by_joint[name] for name in finger_joint_names)
            return FingerFitResult(
                accepted=not failed_components,
                selected_q_m=(q7 if math.isclose(q7, q8, abs_tol=1e-12) else None),
                selected_q_by_joint=selected_q_by_joint,
                selected_render=selected_render,
                visible_mask=visible_mask,
                component_visible_masks=component_visible_masks,
                component_acceptance=component_acceptance,
                diagnostics=diagnostics,
            )

        baseline_render = self.render(
            joints,
            intrinsic_cv,
            cam2world_gl,
            active_side=side,
            finger_q_by_joint=working_q,
        )
        for joint_name in finger_joint_names:
            baseline_score = self._score_candidate(
                baseline_render,
                scene_depth,
                float(tolerance_mm),
                working_q[joint_name],
                joint_name,
            )
            selected_scores[joint_name] = baseline_score
            ranked_by_joint[joint_name] = (baseline_score,)
        baseline_result = finish(baseline_render, search_mode="prior_fast_path")
        baseline_agreements = (
            baseline_result.diagnostics.fixed_link6_agreement,
            baseline_result.diagnostics.final_link7_agreement,
            baseline_result.diagnostics.final_link8_agreement,
        )
        baseline_thresholds = (
            minimum_fixed_support_pixels,
            finger_support_threshold,
            finger_support_threshold,
        )
        baseline_fast_acceptance = tuple(
            agreement_has_minimum_support(
                agreement,
                minimum_support_pixels=threshold,
                minimum_consistent_fraction=fast_path_fraction,
                maximum_median_residual_mm=robust_median_limit,
            )
            for agreement, threshold in zip(
                baseline_agreements,
                baseline_thresholds,
                strict=True,
            )
        )
        if all(baseline_fast_acceptance):
            return baseline_result
        baseline_finger_agreements = (
            baseline_result.diagnostics.final_link7_agreement,
            baseline_result.diagnostics.final_link8_agreement,
        )
        if all(
            agreement.rendered_pixels < minimum_searchable_pixels
            and agreement.comparable_pixels < minimum_searchable_pixels
            for agreement in baseline_finger_agreements
        ):
            return baseline_result

        def fit_one_joint(joint_name: str) -> None:
            prior = temporal_priors.get(joint_name)
            search_lower, search_upper = q_lower, q_upper
            if prior is not None and temporal_delta is not None:
                search_lower = max(search_lower, prior - temporal_delta)
                search_upper = min(search_upper, prior + temporal_delta)
            coarse_values = list(_grid(search_lower, search_upper, float(coarse_step_m)))
            coarse_values.append(float(np.clip(working_q[joint_name], search_lower, search_upper)))
            if search_lower <= command_q <= search_upper:
                coarse_values.append(command_q)
            scores_by_q: dict[float, FingerCandidateScore] = {}

            def evaluate(raw_q: float) -> None:
                q_m = round(float(raw_q), 12)
                if q_m in scores_by_q:
                    return
                overrides = dict(working_q)
                overrides[joint_name] = q_m
                rendered = self.render(
                    joints,
                    intrinsic_cv,
                    cam2world_gl,
                    active_side=side,
                    finger_q_by_joint=overrides,
                )
                scores_by_q[q_m] = self._score_candidate(
                    rendered,
                    scene_depth,
                    float(tolerance_mm),
                    q_m,
                    joint_name,
                )

            for q_m in coarse_values:
                evaluate(q_m)
            ranked_coarse = rank_finger_candidates(
                tuple(scores_by_q.values()),
                command_prior_q_m=command_q,
                temporal_prior_q_m=prior,
                maximum_median_residual_mm=robust_median_limit,
            )
            coarse_winner = ranked_coarse[0].q_m
            fine_lower = max(search_lower, coarse_winner - float(coarse_step_m))
            fine_upper = min(search_upper, coarse_winner + float(coarse_step_m))
            for q_m in _grid(fine_lower, fine_upper, float(fine_step_m)):
                evaluate(q_m)
            ranked = rank_finger_candidates(
                tuple(scores_by_q.values()),
                command_prior_q_m=command_q,
                temporal_prior_q_m=prior,
                maximum_median_residual_mm=robust_median_limit,
            )
            best = ranked[0]
            best_is_supported = agreement_has_minimum_support(
                best.agreement,
                minimum_support_pixels=finger_support_threshold,
                minimum_consistent_fraction=minimum_consistent_fraction,
                maximum_median_residual_mm=robust_median_limit,
            )
            bounded_by_prior = search_lower > q_lower or search_upper < q_upper
            if not best_is_supported and bounded_by_prior:
                # A prior can become stale while one finger is occluded. If
                # its metric window contains no supported hypothesis, retry
                # once over the physical range so the finger can reacquire
                # after a large opening/closing change.
                for q_m in _grid(q_lower, q_upper, float(coarse_step_m)):
                    evaluate(q_m)
                ranked_global_coarse = rank_finger_candidates(
                    tuple(scores_by_q.values()),
                    command_prior_q_m=command_q,
                    temporal_prior_q_m=prior,
                    maximum_median_residual_mm=robust_median_limit,
                )
                global_coarse_winner = ranked_global_coarse[0].q_m
                global_fine_lower = max(
                    q_lower,
                    global_coarse_winner - float(coarse_step_m),
                )
                global_fine_upper = min(
                    q_upper,
                    global_coarse_winner + float(coarse_step_m),
                )
                for q_m in _grid(
                    global_fine_lower,
                    global_fine_upper,
                    float(fine_step_m),
                ):
                    evaluate(q_m)
                ranked = rank_finger_candidates(
                    tuple(scores_by_q.values()),
                    command_prior_q_m=command_q,
                    temporal_prior_q_m=prior,
                    maximum_median_residual_mm=robust_median_limit,
                )
                best = ranked[0]
                best_is_supported = agreement_has_minimum_support(
                    best.agreement,
                    minimum_support_pixels=finger_support_threshold,
                    minimum_consistent_fraction=minimum_consistent_fraction,
                    maximum_median_residual_mm=robust_median_limit,
                )
            if best_is_supported:
                working_q[joint_name] = best.q_m
                selected_scores[joint_name] = best
            else:
                # Evidence is too weak to move this component away from its
                # bounded temporal/command baseline.
                fallback_q = round(float(working_q[joint_name]), 12)
                working_q[joint_name] = fallback_q
                selected_scores[joint_name] = scores_by_q.get(fallback_q, best)
            ranked_by_joint[joint_name] = ranked

        # Coordinate fitting avoids an intractable Cartesian product while
        # still allowing asymmetric contact-constrained finger positions.  A
        # component already supported by the prior render does not need a sweep.
        if not baseline_fast_acceptance[1]:
            fit_one_joint(finger_joint_names[0])
        if not baseline_fast_acceptance[2]:
            fit_one_joint(finger_joint_names[1])
        selected_render = self.render(
            joints,
            intrinsic_cv,
            cam2world_gl,
            active_side=side,
            finger_q_by_joint=working_q,
        )
        return finish(selected_render, search_mode="coordinate_sweep")

    def close(self) -> None:
        """Release the offscreen GL context; safe to call more than once."""

        if not self._closed:
            self._renderer.delete()
            self._closed = True

    def __enter__(self) -> AlohaUrdfRenderer:
        self._ensure_open()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
