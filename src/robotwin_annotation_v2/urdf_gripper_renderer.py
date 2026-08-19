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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray

from .adapters.urdf import finger_fit as _finger_fit

FloatArray = NDArray[np.floating[Any]]
BoolArray = NDArray[np.bool_]
UIntArray = NDArray[np.unsignedinteger[Any]]
VectorInput = Sequence[float] | FloatArray

GRIPPER_DRIVE_CLOSED_M = _finger_fit.GRIPPER_DRIVE_CLOSED_M
GRIPPER_DRIVE_SPAN_M = _finger_fit.GRIPPER_DRIVE_SPAN_M
GRIPPER_KINEMATIC_MAX_M = _finger_fit.GRIPPER_KINEMATIC_MAX_M
GRIPPER_KINEMATIC_MIN_M = _finger_fit.GRIPPER_KINEMATIC_MIN_M
ArmSide = _finger_fit.ArmSide
DepthAgreement = _finger_fit.DepthAgreement
FingerCandidateScore = _finger_fit.FingerCandidateScore
FingerFitDiagnostics = _finger_fit.FingerFitDiagnostics
FingerFitResult = _finger_fit.FingerFitResult
FingerPoseFitter = _finger_fit.FingerPoseFitter
_grid = _finger_fit._grid
_normalize_active_side = _finger_fit._normalize_active_side
active_gripper_link_names = _finger_fit.active_gripper_link_names
agreement_has_minimum_support = _finger_fit.agreement_has_minimum_support
candidate_has_minimum_support = _finger_fit.candidate_has_minimum_support
compute_visible_gripper_mask = _finger_fit.compute_visible_gripper_mask
depth_agreement = _finger_fit.depth_agreement
gripper_command_to_drive_target = _finger_fit.gripper_command_to_drive_target
gripper_command_to_kinematic_q = _finger_fit.gripper_command_to_kinematic_q
rank_finger_candidates = _finger_fit.rank_finger_candidates

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


def xyz_rpy_matrix(xyz: VectorInput, rpy: VectorInput) -> FloatArray:
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


def axis_angle_matrix(axis: VectorInput, angle: float) -> FloatArray:
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


def _validate_intrinsics(value: FloatArray | NDArray[Any]) -> FloatArray:
    intrinsic = np.asarray(value, dtype=np.float64)
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise ValueError("intrinsic_cv must be a finite 3x3 matrix")
    if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
        raise ValueError("camera focal lengths must be positive")
    return intrinsic


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
        normalized = normalized.removeprefix("package://")
        candidate = Path(normalized)
        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            resolved = (self.mesh_root / candidate).resolve()
        if not resolved.is_file():
            raise UrdfRendererError(f"URDF visual mesh is missing: {resolved}")
        return resolved

    def _load_visual_mesh(self, path: Path, scale: FloatArray) -> Any:
        loaded: Any = self._trimesh.load(path, force="scene", process=False)
        mesh: Any
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
        """Delegate contact-aware finger fitting to the canonical fitter."""

        return FingerPoseFitter(self).fit_finger_q(
            joint_absolute,
            intrinsic_cv,
            cam2world_gl,
            scene_depth_mm,
            active_side=active_side,
            tolerance_mm=tolerance_mm,
            q_min_m=q_min_m,
            q_max_m=q_max_m,
            coarse_step_m=coarse_step_m,
            fine_step_m=fine_step_m,
            minimum_support_pixels=minimum_support_pixels,
            minimum_per_link_support_pixels=minimum_per_link_support_pixels,
            minimum_consistent_fraction=minimum_consistent_fraction,
            minimum_fast_path_fraction=minimum_fast_path_fraction,
            maximum_median_residual_mm=maximum_median_residual_mm,
            minimum_fixed_support_pixels=minimum_fixed_support_pixels,
            temporal_prior_q_m=temporal_prior_q_m,
            temporal_prior_q_by_joint=temporal_prior_q_by_joint,
            temporal_max_delta_m=temporal_max_delta_m,
            minimum_searchable_pixels=minimum_searchable_pixels,
        )
    def close(self) -> None:
        """Release the offscreen GL context; safe to call more than once."""

        if not self._closed:
            self._renderer.delete()
            self._closed = True

    def __enter__(self) -> Self:
        self._ensure_open()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
