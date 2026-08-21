"""Legacy gripper-stage exports and object-track reader compatibility."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..domain import ObjectRole
from ..models.loop_context import EpisodeRef
from .gripper.sam import annotator as _annotator
from .gripper.sam import candidates as _candidates
from .gripper.sam import composition as _composition
from .gripper.sam import geometry as _geometry
from .gripper.sam import qc as _qc

NDArray = np.ndarray[Any, Any]

GripperSamBackend = _annotator.GripperSamBackend
GripperStageError = _annotator.GripperStageError
GripperStageResult = _annotator.GripperStageResult
_build_gripper_candidates = _annotator._build_gripper_candidates
_build_roi_track = _annotator._build_roi_track
_context_frame_ids = _annotator._context_frame_ids
_load_resource_image = _annotator._load_resource_image
_load_state_arrays = _annotator._load_state_arrays
_polygon_mask = _annotator._polygon_mask
_roi_geometries = _annotator._roi_geometries
_roi_policy = _annotator._roi_policy
_track_summary = _annotator._track_summary
gripper_keyframes = _annotator.gripper_keyframes
run_gripper_stage = _annotator.run_gripper_stage

GripperSeedCandidate = _candidates.GripperSeedCandidate
GripperSeedQualityGateConfig = _candidates.GripperSeedQualityGateConfig
_component_metrics = _candidates._component_metrics
_tcp_distance = _candidates._tcp_distance
apply_gripper_seed_quality_gate = _candidates.apply_gripper_seed_quality_gate
build_gripper_seed_candidate = _candidates.build_gripper_seed_candidate
mark_same_frame_duplicates = _candidates.mark_same_frame_duplicates
phase_for_frame = _candidates.phase_for_frame

GripperTrackResult = _composition.GripperTrackResult
ObjectExclusionResult = _composition.ObjectExclusionResult
compose_gripper_track = _composition.compose_gripper_track
exclude_known_objects = _composition.exclude_known_objects

CAM_HIGH_CALIBRATION = _geometry.CAM_HIGH_CALIBRATION
DEFAULT_GRIPPER_ROI_GEOMETRY = _geometry.DEFAULT_GRIPPER_ROI_GEOMETRY
CameraCalibration = _geometry.CameraCalibration
GripperRoiGeometry = _geometry.GripperRoiGeometry
ProjectedGripperRoi = _geometry.ProjectedGripperRoi
_convex_hull = _geometry._convex_hull
_project_world_points = _geometry._project_world_points
normalized_roi_box = _geometry.normalized_roi_box
project_gripper_roi = _geometry.project_gripper_roi
rotation_from_rpy = _geometry.rotation_from_rpy

BLUE = _qc.BLUE
CYAN = _qc.CYAN
ORANGE = _qc.ORANGE
RED_RGB = _qc.RED_RGB
YELLOW_RGB = _qc.YELLOW_RGB
GripperQwenClient = _qc.GripperQwenClient
GripperSeedQCResult = _qc.GripperSeedQCResult
_candidate_record = _qc._candidate_record
_fallback_candidate = _qc._fallback_candidate
_outline = _qc._outline
build_gripper_qwen_request = _qc.build_gripper_qwen_request
render_gripper_candidate_panel = _qc.render_gripper_candidate_panel
render_gripper_candidate_sheet = _qc.render_gripper_candidate_sheet
run_gripper_seed_qc = _qc.run_gripper_seed_qc


@dataclass(frozen=True)
class KnownObjectTracks:
    tracks: Mapping[ObjectRole, NDArray]
    seed_masks: Mapping[ObjectRole, NDArray]
    seed_frames: Mapping[ObjectRole, int]
    provenance: dict[str, Any]

    def track(self, role: ObjectRole, *, empty_like: NDArray | None = None) -> NDArray:
        value = self.tracks.get(role)
        if value is not None:
            return np.asarray(value, dtype=bool)
        if empty_like is None:
            raise KeyError(f"known object track has no applicable role {role.value}")
        return np.zeros_like(np.asarray(empty_like, dtype=bool))

    @property
    def target(self) -> NDArray:
        return self.track(ObjectRole.TARGET)

    @property
    def receiver(self) -> NDArray:
        return self.track(ObjectRole.RECEIVER, empty_like=self.target)

    @property
    def target_seed_mask(self) -> NDArray:
        return np.asarray(self.seed_masks[ObjectRole.TARGET], dtype=bool)

    @property
    def receiver_seed_mask(self) -> NDArray:
        return np.asarray(
            self.seed_masks.get(ObjectRole.RECEIVER, np.zeros_like(self.target_seed_mask)),
            dtype=bool,
        )

    @property
    def target_seed_frame(self) -> int:
        return self.seed_frames[ObjectRole.TARGET]

    @property
    def receiver_seed_frame(self) -> int | None:
        return self.seed_frames.get(ObjectRole.RECEIVER)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_qc_native_object_tracks(
    run_root: Path,
    ref: EpisodeRef,
    *,
    expected_shape: tuple[int, int, int],
    required_roles: tuple[ObjectRole, ...] = (ObjectRole.TARGET, ObjectRole.RECEIVER),
) -> KnownObjectTracks:
    """Load identity-QC-passed full native tracks by structured role metadata."""

    episode_dir = (
        run_root.expanduser().resolve()
        / ref.task
        / f"episode_{ref.episode_id}"
        / ref.camera
    )
    manifest_path = episode_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"object run manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    episode = manifest.get("episode", {})
    expected_episode = {
        "task": ref.task,
        "episode_index": ref.episode_index,
        "episode_id": ref.episode_id,
        "camera": ref.camera,
    }
    for key, expected in expected_episode.items():
        if episode.get(key) != expected:
            raise ValueError(
                f"object manifest has {key}={episode.get(key)!r}, expected {expected!r}"
            )
    if manifest.get("frame_count") != expected_shape[0]:
        raise ValueError("object manifest frame_count does not match the episode")

    role_records = manifest.get("roles")
    if not isinstance(role_records, list):
        raise ValueError(  # noqa: TRY004 - preserve the artifact input contract
            "object manifest roles must be a list"
        )
    tracks: dict[str, NDArray] = {}
    seed_masks: dict[str, NDArray] = {}
    seed_frames: dict[str, int] = {}
    sources: dict[str, Any] = {}
    root = episode_dir.resolve()
    for object_role in required_roles:
        role = object_role.value
        matches = [record for record in role_records if record.get("role") == role]
        if len(matches) != 1:
            raise ValueError(f"object manifest must contain exactly one {role} role")
        record = matches[0]
        if record.get("status") != "ok" or record.get("qc_status") != "passed":
            raise ValueError(f"object manifest {role} is not status=ok and qc_status=passed")
        relative_path = record.get("native_track_path")
        if not isinstance(relative_path, str) or not relative_path.strip():
            raise ValueError(f"object manifest {role} native_track_path is invalid")
        track_path = (episode_dir / relative_path).resolve()
        try:
            track_path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"object manifest {role} track escapes episode directory") from exc
        if not track_path.is_file():
            raise FileNotFoundError(f"object native track does not exist: {track_path}")
        with np.load(track_path, allow_pickle=False) as archive:
            if "masks" not in archive.files:
                raise ValueError(f"object native track has no masks array: {track_path}")
            track = np.asarray(archive["masks"], dtype=bool)
        if track.shape != expected_shape:
            raise ValueError(
                f"object native track {track_path} has shape {track.shape}, "
                f"expected {expected_shape}"
            )
        tracks[role] = track
        seed_frame = record.get("seed_frame_id")
        seed_relative_path = record.get("seed_mask_path")
        if not isinstance(seed_frame, int) or not 0 <= seed_frame < expected_shape[0]:
            raise ValueError(f"object manifest {role} seed_frame_id is invalid")
        if not isinstance(seed_relative_path, str) or not seed_relative_path.strip():
            raise ValueError(f"object manifest {role} seed_mask_path is invalid")
        seed_path = (episode_dir / seed_relative_path).resolve()
        try:
            seed_path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"object manifest {role} seed escapes episode directory") from exc
        if not seed_path.is_file():
            raise FileNotFoundError(f"object seed mask does not exist: {seed_path}")
        with Image.open(seed_path) as image:
            seed_mask = np.asarray(image.convert("L"), dtype=np.uint8) != 0
        if seed_mask.shape != expected_shape[1:] or not seed_mask.any():
            raise ValueError(f"object {role} seed mask is empty or has the wrong shape")
        seed_masks[role] = seed_mask
        seed_frames[role] = seed_frame
        sources[role] = {
            "native_track_path": str(track_path),
            "native_track_sha256": _sha256(track_path),
            "seed_frame_id": seed_frame,
            "seed_mask_path": str(seed_path),
            "seed_mask_sha256": _sha256(seed_path),
            "seed_pixels": int(seed_mask.sum()),
            "primary_query": record.get("primary_query"),
            "qc_selected_candidate": record.get("qc_selected_candidate"),
            "qc_reason": record.get("qc_reason"),
        }
    return KnownObjectTracks(
        tracks={ObjectRole(role): track for role, track in tracks.items()},
        seed_masks={ObjectRole(role): mask for role, mask in seed_masks.items()},
        seed_frames={ObjectRole(role): frame for role, frame in seed_frames.items()},
        provenance={
            "run_id": manifest.get("run_id"),
            "manifest": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "roles": sources,
        },
    )
