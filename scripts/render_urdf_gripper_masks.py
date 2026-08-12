#!/usr/bin/env python3
"""Render visible Aloha gripper masks from URDF geometry and scene depth."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import uuid
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import numpy as np

from robotwin_annotation_v2.urdf_gripper_data import (
    ActiveGripperLoop,
    CameraCalibrationSeries,
    UrdfGripperEpisodeData,
    load_authoritative_loop_context,
    load_camera_calibration,
    load_urdf_gripper_episode,
)
from robotwin_annotation_v2.urdf_gripper_publisher import (
    validate_derivation_source_episode,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = Path(
    "/DATA/disk8/xuran/add_mask_robotwin/dataset/"
    "move_pillbottle_pad_coverage20_original"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "urdf_gripper_mask_coverage20"
INSTANCE_NAMES = ("target_0", "receiver_0", "gripper_left", "gripper_right")
ROLES = ("target", "receiver", "gripper", "gripper")
ROLE_COLORS = np.asarray(
    (
        (36, 180, 92),
        (35, 116, 224),
        (232, 67, 55),
        (232, 67, 55),
    ),
    dtype=np.float32,
)
RUN_FORMAT_VERSION = "robotwin_urdf_gripper_run_v2"
DIAGNOSTICS_FORMAT_VERSION = "robotwin_urdf_gripper_diagnostics_v2"
PRODUCT_FORMAT_VERSION = "robotwin_urdf_gripper_masks_v2"


class UrdfMaskRunError(RuntimeError):
    """The integration run cannot preserve its declared artifact contract."""


class UrdfBatchIncompleteError(UrdfMaskRunError):
    """The batch finished, but one or more episode results are incomplete."""

    def __init__(self, message: str, *, result: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.result = result


@dataclass(frozen=True)
class FourChannelMasks:
    """One existing target/receiver/gripper mask artifact and its full payload."""

    path: Path
    payload: dict[str, np.ndarray]
    masks: np.ndarray
    annotation_status: tuple[str, ...]
    qc_status: tuple[str, ...]

    @property
    def frame_count(self) -> int:
        return int(self.masks.shape[1])

    @property
    def frame_shape(self) -> tuple[int, int]:
        return (int(self.masks.shape[2]), int(self.masks.shape[3]))


@dataclass(frozen=True)
class UrdfMaskProduct:
    """Episode-length renderer output before it is merged with object masks."""

    gripper_track: np.ndarray
    rendered_amodal_track: np.ndarray
    depth_evaluable_track: np.ndarray
    depth_consistent_track: np.ndarray
    frame_diagnostics: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class EpisodePlan:
    episode_index: int
    frame_count: int
    frame_shape: tuple[int, int]
    loop: ActiveGripperLoop
    source_masks: Path
    source_loop: Path
    parquet: Path
    sidecar: Path
    rgb_video: Path
    depth_video: Path
    input_identities: Mapping[str, Mapping[str, Any]]
    source_lineage: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.loop.end >= self.frame_count:
            raise ValueError(
                f"active window {self.loop.inclusive_window} exceeds frame count "
                f"{self.frame_count}"
            )

    @property
    def active_arm(self) -> str:
        return self.loop.active_arm

    @property
    def active_window(self) -> tuple[int, int]:
        return self.loop.inclusive_window

    def to_json(self) -> dict[str, Any]:
        return {
            "episode_index": self.episode_index,
            "frame_count": self.frame_count,
            "frame_shape_hw": list(self.frame_shape),
            "active_arm": self.active_arm,
            "active_window": list(self.active_window),
            "events": self.loop.to_json(),
            "source_masks": str(self.source_masks),
            "source_loop": str(self.source_loop),
            "parquet": str(self.parquet),
            "sidecar": str(self.sidecar),
            "rgb_video": str(self.rgb_video),
            "depth_video": str(self.depth_video),
            "inputs": _jsonable(self.input_identities),
            "source_lineage": _jsonable(self.source_lineage),
        }


@dataclass(frozen=True)
class RunConfig:
    dataset_root: Path
    source_run_dir: Path
    output_root: Path
    run_id: str
    urdf_path: Path
    mesh_root: Path | None
    episode_ids: tuple[int, ...]
    task: str = "move_pillbottle_pad"
    camera: str = "cam_high"
    depth_tolerance_mm: float = 8.0
    minimum_eligible_nonempty_fraction: float = 0.90
    fit_config_json: Path | None = None
    overlay_alpha: float = 0.36
    overlay_crf: int = 18
    overlay_preset: str = "medium"
    skip_overlay: bool = False
    dry_run: bool = False
    resume: bool = False
    egl_device_id: int | None = None

    @property
    def run_dir(self) -> Path:
        return self.output_root / self.run_id


def new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--source-run-dir",
        type=Path,
        required=True,
        help="Exact immutable run directory containing task/episode/camera/masks.npz",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", help="New output run id; generated when omitted")
    parser.add_argument("--task", default="move_pillbottle_pad")
    parser.add_argument("--camera", default="cam_high")
    episodes = parser.add_mutually_exclusive_group(required=True)
    episodes.add_argument("--episode-id", type=int)
    episodes.add_argument("--episode-ids", type=int, nargs="+")
    parser.add_argument("--urdf-path", type=Path, required=True)
    parser.add_argument("--mesh-root", type=Path)
    parser.add_argument("--depth-tolerance-mm", type=float, default=8.0)
    parser.add_argument(
        "--minimum-eligible-nonempty-fraction",
        type=float,
        default=0.90,
        help="Minimum fraction of depth-evaluable active frames with a nonempty mask",
    )
    parser.add_argument(
        "--fit-config-json",
        type=Path,
        help="Optional JSON keyword arguments passed to the renderer's finger fitter",
    )
    parser.add_argument("--overlay-alpha", type=float, default=0.36)
    parser.add_argument("--overlay-crf", type=int, default=18)
    parser.add_argument("--overlay-preset", default="medium")
    parser.add_argument("--skip-overlay", action="store_true")
    parser.add_argument(
        "--egl-device-id",
        type=int,
        help="Physical GPU id exposed to EGL while constructing the renderer",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume this run id; only fully validated episode directories are skipped",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate all inputs and print the plan without creating an output run",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> RunConfig:
    args = _parser().parse_args(argv)
    if args.resume and not args.run_id:
        raise ValueError("--resume requires an explicit --run-id")
    if args.dry_run and args.resume:
        raise ValueError("--dry-run and --resume cannot be used together")
    raw_episode_ids = (
        (args.episode_id,) if args.episode_id is not None else tuple(args.episode_ids)
    )
    if any(value < 0 for value in raw_episode_ids):
        raise ValueError("episode ids must be non-negative")
    if len(set(raw_episode_ids)) != len(raw_episode_ids):
        raise ValueError("episode ids must be unique")
    if not args.task.strip() or not args.camera.strip():
        raise ValueError("task and camera must be non-empty")
    if args.depth_tolerance_mm < 0:
        raise ValueError("--depth-tolerance-mm must be non-negative")
    if not math.isfinite(args.minimum_eligible_nonempty_fraction) or not (
        0.0 <= args.minimum_eligible_nonempty_fraction <= 1.0
    ):
        raise ValueError("--minimum-eligible-nonempty-fraction must be finite and in [0, 1]")
    if not 0.0 <= args.overlay_alpha <= 1.0:
        raise ValueError("--overlay-alpha must be between zero and one")
    if not 0 <= args.overlay_crf <= 51:
        raise ValueError("--overlay-crf must be between 0 and 51")
    if args.egl_device_id is not None and args.egl_device_id < 0:
        raise ValueError("--egl-device-id must be non-negative")
    return RunConfig(
        dataset_root=args.dataset_root.expanduser().resolve(),
        source_run_dir=args.source_run_dir.expanduser().resolve(),
        output_root=args.output_root.expanduser().resolve(),
        run_id=args.run_id or new_run_id(),
        urdf_path=args.urdf_path.expanduser().resolve(),
        mesh_root=None if args.mesh_root is None else args.mesh_root.expanduser().resolve(),
        episode_ids=tuple(raw_episode_ids),
        task=args.task,
        camera=args.camera,
        depth_tolerance_mm=float(args.depth_tolerance_mm),
        minimum_eligible_nonempty_fraction=float(
            args.minimum_eligible_nonempty_fraction
        ),
        fit_config_json=(
            None
            if args.fit_config_json is None
            else args.fit_config_json.expanduser().resolve()
        ),
        overlay_alpha=float(args.overlay_alpha),
        overlay_crf=int(args.overlay_crf),
        overlay_preset=args.overlay_preset,
        skip_overlay=bool(args.skip_overlay),
        dry_run=bool(args.dry_run),
        resume=bool(args.resume),
        egl_device_id=args.egl_device_id,
    )


def _is_within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def validate_run_config(
    config: RunConfig,
    *,
    allow_existing_output: bool = False,
) -> None:
    if config.dry_run and config.resume:
        raise ValueError("dry_run and resume cannot be enabled together")
    if not config.episode_ids:
        raise ValueError("at least one episode id is required")
    if any(value < 0 for value in config.episode_ids):
        raise ValueError("episode ids must be non-negative")
    if len(set(config.episode_ids)) != len(config.episode_ids):
        raise ValueError("episode ids must be unique")
    if not config.run_id or "/" in config.run_id or "\\" in config.run_id:
        raise ValueError("run_id must be a simple non-empty directory name")
    if config.run_id in {".", ".."} or ".." in Path(config.run_id).parts:
        raise ValueError("run_id must not contain parent traversal")
    if not config.dataset_root.is_dir():
        raise FileNotFoundError(f"dataset root is missing: {config.dataset_root}")
    if not config.source_run_dir.is_dir():
        raise FileNotFoundError(f"source run directory is missing: {config.source_run_dir}")
    if not config.urdf_path.is_file():
        raise FileNotFoundError(f"Aloha URDF is missing: {config.urdf_path}")
    if config.mesh_root is not None and not config.mesh_root.is_dir():
        raise FileNotFoundError(f"mesh root is missing: {config.mesh_root}")
    if config.fit_config_json is not None and not config.fit_config_json.is_file():
        raise FileNotFoundError(f"fit config JSON is missing: {config.fit_config_json}")
    if not math.isfinite(config.minimum_eligible_nonempty_fraction) or not (
        0.0 <= config.minimum_eligible_nonempty_fraction <= 1.0
    ):
        raise ValueError("minimum_eligible_nonempty_fraction must be finite and in [0, 1]")
    if config.egl_device_id is not None and (
        isinstance(config.egl_device_id, bool) or config.egl_device_id < 0
    ):
        raise ValueError("egl_device_id must be a non-negative integer")
    if _is_within(config.output_root, config.dataset_root):
        raise ValueError("output_root must be outside the source dataset")
    if _is_within(config.output_root, config.source_run_dir):
        raise ValueError("output_root must be outside the immutable source run")
    if config.run_dir.exists() and not config.resume and not allow_existing_output:
        raise FileExistsError(f"output run already exists: {config.run_dir}")
    if config.run_dir.exists() and not config.run_dir.is_dir():
        raise FileExistsError(f"output run path is not a directory: {config.run_dir}")
    if config.resume and not config.run_dir.is_dir() and not allow_existing_output:
        raise FileNotFoundError(f"resume run directory is missing: {config.run_dir}")


def resolve_source_masks(
    source_run_dir: Path,
    *,
    task: str,
    episode_index: int,
    camera: str,
) -> Path:
    return (
        source_run_dir
        / task
        / f"episode_{episode_index:06d}"
        / camera
        / "masks.npz"
    )


def resolve_source_loop(
    source_run_dir: Path,
    *,
    task: str,
    episode_index: int,
    camera: str,
) -> Path:
    return (
        source_run_dir
        / task
        / f"episode_{episode_index:06d}"
        / camera
        / "loop.json"
    )


def _small_strings(value: np.ndarray) -> tuple[str, ...]:
    return tuple(str(item) for item in value.tolist())


def load_four_channel_masks(path: Path, *, frame_count: int) -> FourChannelMasks:
    source = path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"source four-channel masks are missing: {source}")
    with np.load(source, allow_pickle=False) as archive:
        required = {
            "format_version",
            "frame_count",
            "masks",
            "instance_names",
            "roles",
            "annotation_status",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise UrdfMaskRunError(f"source masks are missing keys {missing}: {source}")
        payload = {key: np.asarray(archive[key]).copy() for key in archive.files}

    masks = np.asarray(payload["masks"], dtype=bool)
    names = _small_strings(payload["instance_names"])
    roles = _small_strings(payload["roles"])
    statuses = _small_strings(payload["annotation_status"])
    qc_statuses = (
        _small_strings(payload["qc_status"])
        if "qc_status" in payload
        else tuple("not_run" for _ in INSTANCE_NAMES)
    )
    stored_frame_count = int(payload["frame_count"])
    if names != INSTANCE_NAMES or roles != ROLES:
        raise UrdfMaskRunError(
            f"source channel contract mismatch: names={names}, roles={roles}"
        )
    if masks.ndim != 4 or masks.shape[0] != 4:
        raise UrdfMaskRunError(f"source masks must have shape [4,T,H,W], got {masks.shape}")
    if stored_frame_count != frame_count or masks.shape[1] != frame_count:
        raise UrdfMaskRunError(
            f"source mask frame count mismatch: stored={stored_frame_count}, "
            f"shape={masks.shape[1]}, parquet={frame_count}"
        )
    if len(statuses) != 4 or len(qc_statuses) != 4:
        raise UrdfMaskRunError("source annotation/QC status must contain four entries")
    return FourChannelMasks(
        path=source,
        payload=payload,
        masks=masks,
        annotation_status=statuses,
        qc_status=qc_statuses,
    )


def validate_product(
    product: UrdfMaskProduct,
    episode: UrdfGripperEpisodeData,
    *,
    frame_shape: tuple[int, int],
) -> None:
    expected_shape = (episode.frame_count, *frame_shape)
    for name in (
        "gripper_track",
        "rendered_amodal_track",
        "depth_evaluable_track",
        "depth_consistent_track",
    ):
        value = np.asarray(getattr(product, name))
        if value.shape != expected_shape:
            raise UrdfMaskRunError(f"{name} must have shape {expected_shape}, got {value.shape}")
    visible = np.asarray(product.gripper_track, dtype=bool)
    amodal = np.asarray(product.rendered_amodal_track, dtype=bool)
    evaluable = np.asarray(product.depth_evaluable_track, dtype=bool)
    consistent = np.asarray(product.depth_consistent_track, dtype=bool)
    if np.any(visible & ~amodal):
        raise UrdfMaskRunError("visible gripper pixels must be a subset of rendered amodal pixels")
    if np.any(visible & ~consistent):
        raise UrdfMaskRunError("visible gripper pixels must be a subset of depth-consistent pixels")
    if np.any(consistent & ~evaluable) or np.any(evaluable & ~amodal):
        raise UrdfMaskRunError(
            "depth tracks must satisfy consistent subset evaluable subset amodal"
        )
    start, end = episode.active_window
    for name, value in (
        ("gripper_track", visible),
        ("rendered_amodal_track", amodal),
        ("depth_evaluable_track", evaluable),
        ("depth_consistent_track", consistent),
    ):
        if value[:start].any() or value[end + 1 :].any():
            raise UrdfMaskRunError(
                f"{name} must be empty outside the inclusive active window"
            )
    if not visible[start : end + 1].any():
        raise UrdfMaskRunError("gripper_track must be nonempty inside the active window")


def compose_four_channel_payload(
    source: FourChannelMasks,
    *,
    active_arm: str,
    gripper_track: np.ndarray,
) -> dict[str, np.ndarray]:
    if active_arm not in {"left", "right"}:
        raise ValueError(f"active_arm must be left or right, got {active_arm!r}")
    track = np.asarray(gripper_track, dtype=bool)
    if track.shape != source.masks.shape[1:]:
        raise UrdfMaskRunError(
            f"gripper/source shape mismatch: {track.shape} != {source.masks.shape[1:]}"
        )
    payload = {key: value.copy() for key, value in source.payload.items()}
    masks = source.masks.copy()
    masks[2:4] = False
    active_index = 2 if active_arm == "left" else 3
    masks[active_index] = track
    statuses = list(source.annotation_status)
    statuses[2:4] = ["not_annotated", "not_annotated"]
    statuses[active_index] = "valid"
    qc_statuses = list(source.qc_status)
    qc_statuses[2:4] = ["not_run", "not_run"]
    payload.update(
        {
            "format_version": np.asarray("robotwin_visible_masks_urdf_gripper_v1"),
            "masks": masks,
            "annotation_status": np.asarray(statuses),
            "qc_status": np.asarray(qc_statuses),
        }
    )
    return payload


def _atomic_write_npz(path: Path, payload: Mapping[str, np.ndarray]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".npz",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **payload)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(_jsonable(payload), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_identity(path: Path, *, relative_path: str | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"artifact is missing: {path}")
    return {
        "path": relative_path if relative_path is not None else str(path),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _load_json_object(path: Path, *, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{description} is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise UrdfMaskRunError(f"cannot read {description}: {path}") from exc
    if not isinstance(payload, dict):
        raise UrdfMaskRunError(f"{description} must contain a JSON object: {path}")
    return payload


def _resolve_urdf_mesh_path(uri: str, *, mesh_root: Path) -> Path:
    normalized = uri[len("package://") :] if uri.startswith("package://") else uri
    candidate = Path(normalized)
    resolved = candidate.resolve() if candidate.is_absolute() else (mesh_root / candidate).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"URDF visual mesh is missing: {resolved}")
    return resolved


def collect_asset_identity(urdf_path: Path, mesh_root: Path | None) -> dict[str, Any]:
    """Hash the exact URDF and every visual mesh file it references."""

    resolved_urdf = urdf_path.expanduser().resolve()
    try:
        root = ET.parse(resolved_urdf).getroot()
    except (ET.ParseError, OSError) as exc:
        raise UrdfMaskRunError(f"cannot parse URDF asset: {resolved_urdf}") from exc
    if root.tag != "robot":
        raise UrdfMaskRunError(f"expected <robot> root in URDF: {resolved_urdf}")
    resolved_mesh_root = (
        mesh_root.expanduser().resolve() if mesh_root is not None else resolved_urdf.parent
    )
    references: dict[Path, dict[str, set[str]]] = {}
    for link in root.findall("link"):
        link_name = link.get("name", "")
        for mesh in link.findall("visual/geometry/mesh"):
            uri = mesh.get("filename")
            if not uri:
                raise UrdfMaskRunError(
                    f"visual mesh on link {link_name!r} has no filename: {resolved_urdf}"
                )
            path = _resolve_urdf_mesh_path(uri, mesh_root=resolved_mesh_root)
            entry = references.setdefault(path, {"uris": set(), "links": set()})
            entry["uris"].add(uri)
            entry["links"].add(link_name)
    meshes = []
    for path in sorted(references, key=str):
        identity = _file_identity(path)
        identity.update(
            {
                "uris": sorted(references[path]["uris"]),
                "links": sorted(references[path]["links"]),
            }
        )
        meshes.append(identity)
    return {
        "urdf": _file_identity(resolved_urdf),
        "mesh_root": str(resolved_mesh_root),
        "visual_meshes": meshes,
    }


def _ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise UrdfMaskRunError("ffmpeg is required to decode depth/RGB and encode overlays")
    return executable


def _ffprobe() -> str:
    executable = shutil.which("ffprobe")
    if executable is None:
        raise UrdfMaskRunError("ffprobe is required to preserve the source video rate")
    return executable


def _decode_raw_video(
    path: Path,
    *,
    frame_count: int,
    frame_shape: tuple[int, int],
    pixel_format: str,
    bytes_per_pixel: int,
) -> bytes:
    command = [
        _ffmpeg(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-frames:v",
        str(frame_count),
        "-f",
        "rawvideo",
        "-pix_fmt",
        pixel_format,
        "-",
    ]
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if completed.returncode != 0:
        message = completed.stderr.decode(errors="replace").strip()
        raise UrdfMaskRunError(f"ffmpeg failed to decode {path}: {message}")
    height, width = frame_shape
    expected_bytes = frame_count * height * width * bytes_per_pixel
    if len(completed.stdout) != expected_bytes:
        raise UrdfMaskRunError(
            f"decoded byte count mismatch for {path}: "
            f"{len(completed.stdout)} != {expected_bytes}"
        )
    return completed.stdout


def decode_depth_video(
    path: Path,
    *,
    frame_count: int,
    frame_shape: tuple[int, int],
) -> np.ndarray:
    """Decode the first Parquet-aligned FFV1 gray16le frames in one ffmpeg call."""

    raw = _decode_raw_video(
        path,
        frame_count=frame_count,
        frame_shape=frame_shape,
        pixel_format="gray16le",
        bytes_per_pixel=2,
    )
    return np.frombuffer(raw, dtype="<u2").reshape(frame_count, *frame_shape).copy()


def decode_rgb_video(
    path: Path,
    *,
    frame_count: int,
    frame_shape: tuple[int, int],
) -> np.ndarray:
    raw = _decode_raw_video(
        path,
        frame_count=frame_count,
        frame_shape=frame_shape,
        pixel_format="rgb24",
        bytes_per_pixel=3,
    )
    return np.frombuffer(raw, dtype=np.uint8).reshape(frame_count, *frame_shape, 3).copy()


def _video_rate(path: Path) -> str:
    command = [
        _ffprobe(),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,r_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if completed.returncode != 0:
        message = completed.stderr.decode(errors="replace").strip()
        raise UrdfMaskRunError(f"ffprobe failed for {path}: {message}")
    try:
        stream = json.loads(completed.stdout)["streams"][0]
        raw_rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate")
        rate = Fraction(str(raw_rate))
    except (IndexError, KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise UrdfMaskRunError(f"video has no valid frame rate: {path}") from exc
    if rate <= 0:
        raise UrdfMaskRunError(f"video has non-positive frame rate: {path}")
    return f"{rate.numerator}/{rate.denominator}"


def _probe_video(path: Path) -> dict[str, Any]:
    command = [
        _ffprobe(),
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,nb_read_frames,nb_frames",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if completed.returncode != 0:
        message = completed.stderr.decode(errors="replace").strip()
        raise UrdfMaskRunError(f"ffprobe failed for {path}: {message}")
    try:
        stream = json.loads(completed.stdout)["streams"][0]
        width = int(stream["width"])
        height = int(stream["height"])
        raw_count = stream.get("nb_read_frames") or stream.get("nb_frames")
        frame_count = int(raw_count)
        raw_rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate")
        rate = Fraction(str(raw_rate))
    except (IndexError, KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise UrdfMaskRunError(f"video has invalid stream metadata: {path}") from exc
    if width <= 0 or height <= 0 or frame_count <= 0 or rate <= 0:
        raise UrdfMaskRunError(f"video has non-positive stream metadata: {path}")
    return {
        "frame_count": frame_count,
        "frame_shape_hw": [height, width],
        "frame_rate": f"{rate.numerator}/{rate.denominator}",
    }


def overlay_frame(
    rgb: np.ndarray,
    masks: np.ndarray,
    annotation_status: Sequence[str],
    *,
    frame_id: int,
    alpha: float,
) -> np.ndarray:
    frame = np.asarray(rgb, dtype=np.uint8)
    tracks = np.asarray(masks, dtype=bool)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"RGB frame must have shape [H,W,3], got {frame.shape}")
    if tracks.ndim != 4 or tracks.shape[0] != 4 or tracks.shape[2:] != frame.shape[:2]:
        raise ValueError(f"masks must have shape [4,T,{frame.shape[0]},{frame.shape[1]}]")
    if not 0 <= frame_id < tracks.shape[1]:
        raise IndexError(f"frame_id is outside masks: {frame_id}")
    output = frame.astype(np.float32)
    for channel, status in enumerate(annotation_status):
        if status != "valid":
            continue
        mask = tracks[channel, frame_id]
        if mask.any():
            output[mask] = output[mask] * (1.0 - alpha) + ROLE_COLORS[channel] * alpha
    return np.clip(output, 0, 255).astype(np.uint8)


def render_overlay_video(
    rgb_path: Path,
    combined_masks: FourChannelMasks,
    output_path: Path,
    *,
    alpha: float,
    crf: int,
    preset: str,
) -> dict[str, Any]:
    rgb = decode_rgb_video(
        rgb_path,
        frame_count=combined_masks.frame_count,
        frame_shape=combined_masks.frame_shape,
    )
    for frame_id in range(combined_masks.frame_count):
        rgb[frame_id] = overlay_frame(
            rgb[frame_id],
            combined_masks.masks,
            combined_masks.annotation_status,
            frame_id=frame_id,
            alpha=alpha,
        )
    rate = _video_rate(rgb_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.",
        suffix=".mp4",
        dir=output_path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink(missing_ok=True)
    height, width = combined_masks.frame_shape
    command = [
        _ffmpeg(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        rate,
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    try:
        completed = subprocess.run(
            command,
            input=rgb.tobytes(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if completed.returncode != 0:
            message = completed.stderr.decode(errors="replace").strip()
            raise UrdfMaskRunError(f"ffmpeg overlay encode failed: {message}")
        os.replace(temporary, output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    probe = _probe_video(output_path)
    if probe["frame_count"] != combined_masks.frame_count:
        raise UrdfMaskRunError(
            f"overlay frame count mismatch: {probe['frame_count']} != "
            f"{combined_masks.frame_count}"
        )
    if probe["frame_shape_hw"] != [height, width]:
        raise UrdfMaskRunError(
            f"overlay frame shape mismatch: {probe['frame_shape_hw']} != {[height, width]}"
        )
    return {
        **probe,
        "sha256": _sha256(output_path),
        "bytes": output_path.stat().st_size,
    }


def _fit_config(path: Path | None) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if path is not None:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("fit config JSON must contain a keyword-argument object")
        payload = raw
    reserved = {
        "joint_absolute",
        "intrinsic_cv",
        "cam2world_gl",
        "scene_depth_mm",
        "active_side",
        "tolerance_mm",
        "temporal_prior_q_m",
        "temporal_prior_q_by_joint",
    }
    conflict = sorted(reserved & set(payload))
    if conflict:
        raise ValueError(f"fit config JSON cannot override runner arguments: {conflict}")
    return payload


def create_renderer(
    config: RunConfig,
    frame_shape: tuple[int, int],
) -> tuple[Any, dict[str, Any]]:
    from robotwin_annotation_v2.urdf_gripper_renderer import AlohaUrdfRenderer

    height, width = frame_shape
    previous_egl_device = os.environ.get("EGL_DEVICE_ID")
    if config.egl_device_id is not None:
        os.environ["EGL_DEVICE_ID"] = str(config.egl_device_id)
    try:
        renderer = AlohaUrdfRenderer(
            config.urdf_path,
            mesh_root=config.mesh_root,
            width=width,
            height=height,
        )
    finally:
        if config.egl_device_id is not None:
            if previous_egl_device is None:
                os.environ.pop("EGL_DEVICE_ID", None)
            else:
                os.environ["EGL_DEVICE_ID"] = previous_egl_device
    return renderer, _fit_config(config.fit_config_json)


def _selected_q(fit: Any) -> dict[str, float] | None:
    value = getattr(fit, "selected_q_by_joint", None)
    if value is None:
        return None
    return {str(key): float(item) for key, item in value.items()}


def render_episode_product(
    renderer: Any,
    fit_config: Mapping[str, Any],
    episode: UrdfGripperEpisodeData,
    calibration: CameraCalibrationSeries,
    scene_depth_mm: np.ndarray,
    *,
    frame_shape: tuple[int, int],
    tolerance_mm: float,
) -> UrdfMaskProduct:
    """Drive the frame-level renderer and preserve fit evidence for every active frame."""

    expected_depth_shape = (episode.frame_count, *frame_shape)
    depth = np.asarray(scene_depth_mm)
    if depth.shape != expected_depth_shape:
        raise UrdfMaskRunError(
            f"scene depth must have shape {expected_depth_shape}, got {depth.shape}"
        )
    gripper = np.zeros(expected_depth_shape, dtype=bool)
    amodal = np.zeros(expected_depth_shape, dtype=bool)
    evaluable = np.zeros(expected_depth_shape, dtype=bool)
    consistent = np.zeros(expected_depth_shape, dtype=bool)
    frame_records: list[dict[str, Any]] = []
    temporal_prior_q_by_joint: dict[str, float] = {}
    start, end = episode.active_window
    for frame_id in range(start, end + 1):
        fit = renderer.fit_finger_q(
            episode.joint_absolute[frame_id],
            calibration.intrinsic_cv[frame_id],
            calibration.cam2world_gl[frame_id],
            depth[frame_id],
            active_side=episode.active_arm,
            tolerance_mm=tolerance_mm,
            temporal_prior_q_by_joint=(
                dict(temporal_prior_q_by_joint) or None
            ),
            **fit_config,
        )
        render = fit.selected_render
        visible = np.asarray(fit.visible_mask, dtype=bool)
        rendered_amodal = np.asarray(render.active_gripper_mask, dtype=bool)
        rendered_depth_mm = np.asarray(render.active_gripper_depth_mm, dtype=np.float64)
        if (
            visible.shape != frame_shape
            or rendered_amodal.shape != frame_shape
            or rendered_depth_mm.shape != frame_shape
        ):
            raise UrdfMaskRunError(
                f"renderer frame shape mismatch at frame {frame_id}: "
                f"visible={visible.shape}, amodal={rendered_amodal.shape}, "
                f"depth={rendered_depth_mm.shape}"
            )
        valid_scene_depth = np.isfinite(depth[frame_id]) & (depth[frame_id] > 0)
        valid_render_depth = np.isfinite(rendered_depth_mm) & (rendered_depth_mm > 0)
        depth_evaluable = rendered_amodal & valid_scene_depth & valid_render_depth
        depth_consistent = depth_evaluable & (
            np.abs(rendered_depth_mm - depth[frame_id]) <= tolerance_mm
        )
        if np.any(visible & ~depth_consistent):
            raise UrdfMaskRunError(
                "visible mask is not a subset of raw depth-consistent render at "
                f"frame {frame_id}"
            )
        gripper[frame_id] = visible
        evaluable[frame_id] = depth_evaluable
        consistent[frame_id] = depth_consistent
        amodal[frame_id] = rendered_amodal
        selected_q = _selected_q(fit)
        component_acceptance = {
            str(key): bool(value)
            for key, value in getattr(fit, "component_acceptance", {}).items()
        }
        if selected_q is not None:
            for joint_name, value in selected_q.items():
                component_name = joint_name.replace("_joint", "_link")
                if component_acceptance.get(component_name, False):
                    temporal_prior_q_by_joint[joint_name] = value
        frame_records.append(
            {
                "frame_id": frame_id,
                "accepted": bool(fit.accepted),
                "selected_q_by_joint": selected_q,
                "component_acceptance": component_acceptance,
                "visible_pixels": int(visible.sum()),
                "amodal_pixels": int(rendered_amodal.sum()),
                "depth_evaluable_pixels": int(depth_evaluable.sum()),
                "diagnostics": _jsonable(fit.diagnostics),
            }
        )
    product = UrdfMaskProduct(
        gripper_track=gripper,
        rendered_amodal_track=amodal,
        depth_evaluable_track=evaluable,
        depth_consistent_track=consistent,
        frame_diagnostics=tuple(frame_records),
    )
    validate_product(product, episode, frame_shape=frame_shape)
    return product


def _product_payload(
    product: UrdfMaskProduct,
    episode: UrdfGripperEpisodeData,
    *,
    tolerance_mm: float,
) -> dict[str, np.ndarray]:
    return {
        "format_version": np.asarray(PRODUCT_FORMAT_VERSION),
        "frame_count": np.asarray(episode.frame_count, dtype=np.int64),
        "gripper_track": np.asarray(product.gripper_track, dtype=bool),
        "rendered_amodal_track": np.asarray(product.rendered_amodal_track, dtype=bool),
        "depth_evaluable_track": np.asarray(product.depth_evaluable_track, dtype=bool),
        "depth_consistent_track": np.asarray(product.depth_consistent_track, dtype=bool),
        "active_arm": np.asarray(episode.active_arm),
        "active_window": np.asarray(episode.active_window, dtype=np.int64),
        "depth_tolerance_mm": np.asarray(tolerance_mm, dtype=np.float64),
    }


def _fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def _quality_summary(
    product: UrdfMaskProduct,
    episode: UrdfGripperEpisodeData | EpisodePlan,
) -> dict[str, Any]:
    visible_counts = np.asarray(product.gripper_track, dtype=bool).reshape(
        episode.frame_count, -1
    ).sum(axis=1)
    amodal_counts = np.asarray(product.rendered_amodal_track, dtype=bool).reshape(
        episode.frame_count, -1
    ).sum(axis=1)
    evaluable_counts = np.asarray(product.depth_evaluable_track, dtype=bool).reshape(
        episode.frame_count, -1
    ).sum(axis=1)
    start, end = episode.active_window
    active_counts = visible_counts[start : end + 1]
    eligible = (
        (amodal_counts[start : end + 1] > 0)
        & (evaluable_counts[start : end + 1] > 0)
    )
    eligible_count = int(np.count_nonzero(eligible))
    eligible_nonempty_count = int(
        np.count_nonzero(eligible & (active_counts > 0))
    )
    component_counts: dict[str, dict[str, int]] = {}
    previous_q: dict[str, float] = {}
    maximum_jump_by_joint: dict[str, float] = {}
    records_by_frame: dict[int, Mapping[str, Any]] = {}
    for raw_record in product.frame_diagnostics:
        if not isinstance(raw_record, Mapping):
            raise UrdfMaskRunError("frame diagnostics entries must be mappings")
        raw_frame_id = raw_record.get("frame_id")
        if isinstance(raw_frame_id, bool) or not isinstance(raw_frame_id, int):
            raise UrdfMaskRunError("frame diagnostics frame_id must be an integer")
        if raw_frame_id in records_by_frame:
            raise UrdfMaskRunError(f"duplicate frame diagnostics for frame {raw_frame_id}")
        records_by_frame[raw_frame_id] = raw_record
    expected_frame_ids = set(range(start, end + 1))
    if set(records_by_frame) != expected_frame_ids:
        raise UrdfMaskRunError(
            "frame diagnostics must contain exactly the inclusive active window"
        )
    for frame_id in range(start, end + 1):
        record = records_by_frame[frame_id]
        expected_counts = {
            "visible_pixels": int(visible_counts[frame_id]),
            "amodal_pixels": int(amodal_counts[frame_id]),
            "depth_evaluable_pixels": int(evaluable_counts[frame_id]),
        }
        for key, expected in expected_counts.items():
            actual = record.get(key)
            if isinstance(actual, bool) or not isinstance(actual, int) or actual != expected:
                raise UrdfMaskRunError(
                    f"frame {frame_id} diagnostics {key} mismatch: {actual!r} != {expected}"
                )
        raw_acceptance = record.get("component_acceptance", {})
        if isinstance(raw_acceptance, Mapping):
            for component, accepted in raw_acceptance.items():
                counts = component_counts.setdefault(
                    str(component), {"accepted_frames": 0, "evaluated_frames": 0}
                )
                counts["evaluated_frames"] += 1
                if bool(accepted):
                    counts["accepted_frames"] += 1
        raw_q = record.get("selected_q_by_joint")
        if not isinstance(raw_q, Mapping):
            continue
        for joint, raw_value in raw_q.items():
            value = float(raw_value)
            if not math.isfinite(value):
                raise UrdfMaskRunError(f"frame diagnostics contain non-finite q for {joint}")
            name = str(joint)
            if name in previous_q:
                jump = abs(value - previous_q[name])
                maximum_jump_by_joint[name] = max(
                    jump, maximum_jump_by_joint.get(name, 0.0)
                )
            previous_q[name] = value
    active_nonempty_count = int(np.count_nonzero(active_counts))
    quality = {
        "active_frame_count": int(active_counts.size),
        "active_nonempty_frame_count": active_nonempty_count,
        "active_nonempty_fraction": _fraction(
            active_nonempty_count, int(active_counts.size)
        ),
        "eligible_frame_count": eligible_count,
        "eligible_nonempty_frame_count": eligible_nonempty_count,
        "eligible_nonempty_fraction": _fraction(
            eligible_nonempty_count, eligible_count
        ),
        "visible_pixels_total": int(active_counts.sum()),
        "component_acceptance": component_counts,
        "maximum_q_jump_m": (
            max(maximum_jump_by_joint.values()) if maximum_jump_by_joint else 0.0
        ),
        "maximum_q_jump_by_joint_m": maximum_jump_by_joint,
    }
    for key, value in quality.items():
        if key.endswith("_fraction") and (
            not isinstance(value, float)
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):
            raise UrdfMaskRunError(f"quality fraction must be finite and in [0, 1]: {key}")
    return quality


def _enforce_quality_gate(
    quality: Mapping[str, Any],
    *,
    minimum_eligible_nonempty_fraction: float,
) -> None:
    fraction = quality.get("eligible_nonempty_fraction")
    if not isinstance(fraction, (int, float)) or not math.isfinite(float(fraction)):
        raise UrdfMaskRunError("eligible_nonempty_fraction must be finite")
    if float(fraction) < minimum_eligible_nonempty_fraction:
        raise UrdfMaskRunError(
            "eligible nonempty fraction is below the run threshold: "
            f"{float(fraction):.6f} < {minimum_eligible_nonempty_fraction:.6f}"
        )


def save_episode_artifacts(
    output_dir: Path,
    episode: UrdfGripperEpisodeData,
    source: FourChannelMasks,
    product: UrdfMaskProduct,
    *,
    tolerance_mm: float,
    minimum_eligible_nonempty_fraction: float = 0.90,
) -> tuple[FourChannelMasks, dict[str, Any]]:
    """Atomically save standalone and merged masks without touching source artifacts."""

    validate_product(product, episode, frame_shape=source.frame_shape)
    quality = _quality_summary(product, episode)
    _enforce_quality_gate(
        quality,
        minimum_eligible_nonempty_fraction=minimum_eligible_nonempty_fraction,
    )
    gripper_path = _atomic_write_npz(
        output_dir / "gripper_masks.npz",
        _product_payload(product, episode, tolerance_mm=tolerance_mm),
    )
    combined_payload = compose_four_channel_payload(
        source,
        active_arm=episode.active_arm,
        gripper_track=product.gripper_track,
    )
    combined_path = _atomic_write_npz(output_dir / "masks.npz", combined_payload)
    combined = load_four_channel_masks(combined_path, frame_count=episode.frame_count)
    diagnostics: dict[str, Any] = {
        "format_version": DIAGNOSTICS_FORMAT_VERSION,
        "status": "incomplete",
        "episode_index": episode.paths.episode_index,
        "active_arm": episode.active_arm,
        "active_window": list(episode.active_window),
        "frame_count": episode.frame_count,
        "depth_tolerance_mm": tolerance_mm,
        "minimum_eligible_nonempty_fraction": minimum_eligible_nonempty_fraction,
        "source_masks": str(source.path),
        "source_masks_sha256": _sha256(source.path),
        "quality": quality,
        "frame_diagnostics": list(product.frame_diagnostics),
        "artifacts": {
            "gripper_masks": _file_identity(
                gripper_path, relative_path=gripper_path.name
            ),
            "masks": _file_identity(combined_path, relative_path=combined_path.name),
            "overlay": None,
        },
    }
    _atomic_write_json(output_dir / "diagnostics.json", diagnostics)
    return combined, diagnostics


def finalize_episode_diagnostics(
    output_dir: Path,
    diagnostics: Mapping[str, Any],
    *,
    overlay: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Publish complete diagnostics only after all optional artifacts exist."""

    completed = copy.deepcopy(dict(diagnostics))
    artifacts = completed.get("artifacts")
    if not isinstance(artifacts, dict):
        raise UrdfMaskRunError("episode diagnostics have no artifact map")
    if overlay is None:
        artifacts["overlay"] = None
    else:
        overlay_path = output_dir / "overlay.mp4"
        identity = _file_identity(overlay_path, relative_path=overlay_path.name)
        identity.update(
            {
                key: _jsonable(value)
                for key, value in overlay.items()
                if key not in {"path", "sha256", "bytes"}
            }
        )
        artifacts["overlay"] = identity
    completed["status"] = "complete"
    completed["completed_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_write_json(output_dir / "diagnostics.json", completed)
    return completed


def _validated_artifact(
    output_dir: Path,
    artifacts: Mapping[str, Any],
    key: str,
) -> tuple[Path, dict[str, Any]]:
    entry = artifacts.get(key)
    if not isinstance(entry, Mapping):
        raise UrdfMaskRunError(f"diagnostics artifact entry is invalid: {key}")
    relative = entry.get("path")
    if not isinstance(relative, str) or not relative:
        raise UrdfMaskRunError(f"diagnostics artifact path is invalid: {key}")
    path = (output_dir / relative).resolve()
    if path.parent != output_dir.resolve():
        raise UrdfMaskRunError(f"diagnostics artifact escapes episode directory: {key}")
    actual = _file_identity(path, relative_path=relative)
    if entry.get("sha256") != actual["sha256"] or entry.get("bytes") != actual["bytes"]:
        raise UrdfMaskRunError(f"diagnostics artifact hash/size mismatch: {key}")
    return path, actual


def validate_completed_episode(
    output_dir: Path,
    plan: EpisodePlan,
    config: RunConfig,
) -> dict[str, Any]:
    """Validate a published episode completely before it may be skipped on resume."""

    if not output_dir.is_dir():
        raise UrdfMaskRunError(f"episode output directory is missing: {output_dir}")
    diagnostics_path = output_dir / "diagnostics.json"
    diagnostics = _load_json_object(
        diagnostics_path, description="episode diagnostics"
    )
    expected_fields = {
        "format_version": DIAGNOSTICS_FORMAT_VERSION,
        "status": "complete",
        "episode_index": plan.episode_index,
        "active_arm": plan.active_arm,
        "active_window": list(plan.active_window),
        "frame_count": plan.frame_count,
        "depth_tolerance_mm": config.depth_tolerance_mm,
        "minimum_eligible_nonempty_fraction": (
            config.minimum_eligible_nonempty_fraction
        ),
        "source_masks": str(plan.source_masks),
        "source_masks_sha256": _sha256(plan.source_masks),
    }
    for key, expected in expected_fields.items():
        actual = diagnostics.get(key)
        if key in {
            "depth_tolerance_mm",
            "minimum_eligible_nonempty_fraction",
        }:
            try:
                matches = bool(np.isclose(float(actual), float(expected), rtol=0, atol=1e-12))
            except (TypeError, ValueError):
                matches = False
        else:
            matches = actual == expected
        if not matches:
            raise UrdfMaskRunError(
                f"episode diagnostics field mismatch for {key}: {actual!r} != {expected!r}"
            )
    recorded_quality = diagnostics.get("quality")
    if not isinstance(recorded_quality, Mapping):
        raise UrdfMaskRunError("episode diagnostics have no quality summary")
    raw_frame_diagnostics = diagnostics.get("frame_diagnostics")
    if not isinstance(raw_frame_diagnostics, list):
        raise UrdfMaskRunError("episode diagnostics frame_diagnostics must be a list")
    artifacts = diagnostics.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise UrdfMaskRunError("episode diagnostics have no artifact map")
    gripper_path, gripper_identity = _validated_artifact(
        output_dir, artifacts, "gripper_masks"
    )
    masks_path, masks_identity = _validated_artifact(output_dir, artifacts, "masks")

    with np.load(gripper_path, allow_pickle=False) as archive:
        required = {
            "format_version",
            "frame_count",
            "gripper_track",
            "rendered_amodal_track",
            "depth_evaluable_track",
            "depth_consistent_track",
            "active_arm",
            "active_window",
            "depth_tolerance_mm",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise UrdfMaskRunError(f"gripper masks are missing keys {missing}")
        if str(archive["format_version"].item()) != PRODUCT_FORMAT_VERSION:
            raise UrdfMaskRunError("gripper mask format version mismatch")
        frame_count = int(archive["frame_count"].item())
        active_arm = str(archive["active_arm"].item())
        active_window = tuple(int(value) for value in archive["active_window"].tolist())
        tolerance = float(archive["depth_tolerance_mm"].item())
        visible = np.asarray(archive["gripper_track"])
        amodal = np.asarray(archive["rendered_amodal_track"])
        evaluable = np.asarray(archive["depth_evaluable_track"])
        consistent = np.asarray(archive["depth_consistent_track"])
    expected_shape = (plan.frame_count, *plan.frame_shape)
    if frame_count != plan.frame_count or active_arm != plan.active_arm:
        raise UrdfMaskRunError("gripper mask episode metadata mismatch")
    if active_window != plan.active_window or not np.isclose(
        tolerance, config.depth_tolerance_mm, rtol=0, atol=1e-12
    ):
        raise UrdfMaskRunError("gripper mask window/tolerance mismatch")
    for name, value in (
        ("gripper_track", visible),
        ("rendered_amodal_track", amodal),
        ("depth_evaluable_track", evaluable),
        ("depth_consistent_track", consistent),
    ):
        if value.shape != expected_shape or value.dtype != np.bool_:
            raise UrdfMaskRunError(
                f"{name} must have bool shape {expected_shape}, got {value.shape}/{value.dtype}"
            )
    if np.any(visible & ~consistent) or np.any(consistent & ~evaluable):
        raise UrdfMaskRunError("visible/depth-consistent/evaluable subset contract is violated")
    if np.any(evaluable & ~amodal):
        raise UrdfMaskRunError("depth-evaluable pixels must be a subset of amodal pixels")
    start, end = plan.active_window
    if visible[:start].any() or visible[end + 1 :].any():
        raise UrdfMaskRunError("published gripper mask is nonempty outside active window")
    if not visible[start : end + 1].any():
        raise UrdfMaskRunError("published gripper mask is empty inside active window")
    for name, value in (
        ("rendered_amodal_track", amodal),
        ("depth_evaluable_track", evaluable),
        ("depth_consistent_track", consistent),
    ):
        if value[:start].any() or value[end + 1 :].any():
            raise UrdfMaskRunError(f"{name} is nonempty outside the active window")

    frame_diagnostics: list[dict[str, Any]] = []
    for raw_record in raw_frame_diagnostics:
        if not isinstance(raw_record, dict):
            raise UrdfMaskRunError("frame diagnostics entries must be JSON objects")
        frame_diagnostics.append(raw_record)
    product = UrdfMaskProduct(
        gripper_track=visible,
        rendered_amodal_track=amodal,
        depth_evaluable_track=evaluable,
        depth_consistent_track=consistent,
        frame_diagnostics=tuple(frame_diagnostics),
    )
    recomputed_quality = _quality_summary(product, plan)
    if _jsonable(recorded_quality) != _jsonable(recomputed_quality):
        raise UrdfMaskRunError("episode diagnostics quality does not match NPZ/frame records")
    _enforce_quality_gate(
        recomputed_quality,
        minimum_eligible_nonempty_fraction=(
            config.minimum_eligible_nonempty_fraction
        ),
    )

    source = load_four_channel_masks(plan.source_masks, frame_count=plan.frame_count)
    combined = load_four_channel_masks(masks_path, frame_count=plan.frame_count)
    expected_payload = compose_four_channel_payload(
        source, active_arm=plan.active_arm, gripper_track=visible
    )
    if set(combined.payload) != set(expected_payload):
        raise UrdfMaskRunError("combined mask payload keys differ from the source contract")
    for key, expected in expected_payload.items():
        if not np.array_equal(combined.payload[key], expected):
            raise UrdfMaskRunError(f"combined mask payload mismatch: {key}")

    artifact_summary: dict[str, Any] = {
        "gripper_masks": gripper_identity,
        "masks": masks_identity,
        "overlay": None,
    }
    if config.skip_overlay:
        if artifacts.get("overlay") is not None:
            raise UrdfMaskRunError("skip-overlay run unexpectedly declares an overlay")
    else:
        overlay_path, overlay_identity = _validated_artifact(
            output_dir, artifacts, "overlay"
        )
        if overlay_identity["bytes"] <= 0:
            raise UrdfMaskRunError("overlay video is empty")
        overlay_probe = _probe_video(overlay_path)
        expected_overlay_probe = {
            "frame_count": plan.frame_count,
            "frame_shape_hw": list(plan.frame_shape),
        }
        for key, expected in expected_overlay_probe.items():
            if overlay_probe.get(key) != expected or artifacts["overlay"].get(key) != expected:
                raise UrdfMaskRunError(
                    f"overlay {key} mismatch: probe={overlay_probe.get(key)!r}, "
                    f"diagnostics={artifacts['overlay'].get(key)!r}, expected={expected!r}"
                )
        artifact_summary["overlay"] = overlay_identity
    artifact_summary["diagnostics"] = _file_identity(
        diagnostics_path, relative_path="diagnostics.json"
    )
    return {
        **plan.to_json(),
        "status": "complete",
        "output_dir": output_dir.name,
        "quality": _jsonable(recomputed_quality),
        "artifacts": artifact_summary,
    }


def _build_episode_plan(config: RunConfig, episode_index: int) -> EpisodePlan:
    source_path = resolve_source_masks(
        config.source_run_dir,
        task=config.task,
        episode_index=episode_index,
        camera=config.camera,
    )
    source_loop_path = resolve_source_loop(
        config.source_run_dir,
        task=config.task,
        episode_index=episode_index,
        camera=config.camera,
    )
    source_loop = load_authoritative_loop_context(
        source_loop_path,
        expected_task=config.task,
        expected_episode_index=episode_index,
        expected_camera=config.camera,
    )
    episode = load_urdf_gripper_episode(
        config.dataset_root,
        episode_index,
        camera=config.camera,
        authoritative_loop=source_loop.events,
    )
    if source_loop.frame_count != episode.frame_count:
        raise UrdfMaskRunError(
            f"source loop frame count mismatch: loop={source_loop.frame_count}, "
            f"parquet={episode.frame_count}"
        )
    source = load_four_channel_masks(source_path, frame_count=episode.frame_count)
    try:
        validated_source = validate_derivation_source_episode(
            source_path.parent,
            task=config.task,
            camera=config.camera,
            episode_index=episode_index,
            expected_frame_count=episode.frame_count,
            expected_dataset_root=config.dataset_root,
        )
    except Exception as exc:
        raise UrdfMaskRunError(
            f"source lineage validation failed for episode {episode_index}: {exc}"
        ) from exc
    lineage_masks = validated_source.lineage["control_artifacts"]["masks"]
    lineage_loop = validated_source.lineage["control_artifacts"]["loop"]
    current_masks = _file_identity(source_path)
    current_loop = _file_identity(source_loop_path)
    if any(
        current.get(key) != frozen.get(key)
        for current, frozen in (
            (current_masks, lineage_masks),
            (current_loop, lineage_loop),
        )
        for key in ("sha256", "bytes")
    ):
        raise UrdfMaskRunError(
            f"source artifacts changed while planning episode {episode_index}"
        )
    return EpisodePlan(
        episode_index=episode_index,
        frame_count=episode.frame_count,
        frame_shape=source.frame_shape,
        loop=source_loop.events,
        source_masks=source.path,
        source_loop=source_loop.path,
        parquet=episode.paths.parquet,
        sidecar=episode.paths.sidecar,
        rgb_video=episode.paths.rgb_video,
        depth_video=episode.paths.depth_video,
        source_lineage=validated_source.lineage,
        input_identities={
            "source_masks": _file_identity(source.path),
            "source_loop": _file_identity(source_loop.path),
            "parquet": _file_identity(episode.paths.parquet),
            "sidecar": _file_identity(episode.paths.sidecar),
            "rgb_video": _file_identity(episode.paths.rgb_video),
            "depth_video": _file_identity(episode.paths.depth_video),
        },
    )


def build_episode_plan(config: RunConfig, episode_index: int) -> EpisodePlan:
    """Build one source-ready episode plan without preflighting later episodes."""

    validate_run_config(config, allow_existing_output=True)
    if episode_index not in config.episode_ids:
        raise ValueError(f"episode {episode_index} is not declared by this run")
    return _build_episode_plan(config, episode_index)


def build_plan(config: RunConfig) -> tuple[EpisodePlan, ...]:
    validate_run_config(config)
    plans = tuple(_build_episode_plan(config, value) for value in config.episode_ids)
    frame_shapes = {plan.frame_shape for plan in plans}
    if len(frame_shapes) > 1:
        raise UrdfMaskRunError(
            f"all episodes must share one frame shape: {sorted(frame_shapes)}"
        )
    return plans


def _git_revision() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    revision = completed.stdout.strip()
    return revision if completed.returncode == 0 and revision else None


def _implementation_identity() -> dict[str, Any]:
    paths = (
        Path(__file__).resolve(),
        PROJECT_ROOT / "src/robotwin_annotation_v2/urdf_gripper_data.py",
        PROJECT_ROOT / "src/robotwin_annotation_v2/urdf_gripper_renderer.py",
        PROJECT_ROOT / "src/robotwin_annotation_v2/urdf_gripper_publisher.py",
    )
    return {
        "git_revision": _git_revision(),
        "files": [
            _file_identity(path, relative_path=str(path.relative_to(PROJECT_ROOT)))
            for path in paths
        ],
    }


def _run_contract(
    config: RunConfig,
    plans: Sequence[EpisodePlan],
    *,
    fit_config: Mapping[str, Any],
) -> dict[str, Any]:
    fit_identity = (
        None
        if config.fit_config_json is None
        else _file_identity(config.fit_config_json)
    )
    episode_plans = [plan.to_json() for plan in plans]
    return {
        "run_id": config.run_id,
        "dataset_root": str(config.dataset_root),
        "source_run_dir": str(config.source_run_dir),
        "output_run_dir": str(config.run_dir),
        "task": config.task,
        "camera": config.camera,
        "depth_tolerance_mm": config.depth_tolerance_mm,
        "minimum_eligible_nonempty_fraction": (
            config.minimum_eligible_nonempty_fraction
        ),
        "egl_device_id": config.egl_device_id,
        "fit_config": {
            "file": fit_identity,
            "kwargs": _jsonable(dict(fit_config)),
        },
        "overlay": {
            "enabled": not config.skip_overlay,
            "alpha": config.overlay_alpha,
            "crf": config.overlay_crf,
            "preset": config.overlay_preset,
        },
        "episode_plans": episode_plans,
        "assets": collect_asset_identity(config.urdf_path, config.mesh_root),
        "implementation": _implementation_identity(),
    }


def _checkpoint_manifest(path: Path, manifest: dict[str, Any]) -> None:
    episodes = manifest.get("episodes", [])
    if not isinstance(episodes, list):
        raise UrdfMaskRunError("run manifest episodes must be a list")
    complete = sum(
        1
        for record in episodes
        if isinstance(record, Mapping) and record.get("status") == "complete"
    )
    failed = sum(
        1 for record in episodes if isinstance(record, Mapping) and record.get("status") == "failed"
    )
    failures = manifest.get("failures", [])
    if not isinstance(failures, list):
        raise UrdfMaskRunError("run manifest failures must be a list")
    manifest["successful_episode_count"] = complete
    manifest["failed_episode_count"] = failed
    manifest["failure_attempt_count"] = len(failures)
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_write_json(path, manifest)


def _episode_record_map(manifest: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    records = manifest.get("episodes", [])
    if not isinstance(records, list):
        raise UrdfMaskRunError("run manifest episodes must be a list")
    result: dict[int, dict[str, Any]] = {}
    for raw in records:
        if not isinstance(raw, dict) or "episode_index" not in raw:
            raise UrdfMaskRunError("run manifest contains an invalid episode record")
        episode_index = int(raw["episode_index"])
        if episode_index in result:
            raise UrdfMaskRunError(
                f"run manifest repeats episode {episode_index}"
            )
        result[episode_index] = raw
    return result


def _ordered_episode_records(
    plans: Sequence[EpisodePlan],
    records: Mapping[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        records.get(
            plan.episode_index,
            {**plan.to_json(), "status": "pending"},
        )
        for plan in plans
    ]


def _artifact_identity_anchor(
    record: Mapping[str, Any],
    *,
    key: str,
) -> dict[str, Any] | None:
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, Mapping) or key not in artifacts:
        raise UrdfMaskRunError(f"complete manifest record has no {key} artifact identity")
    entry = artifacts[key]
    if entry is None:
        return None
    if not isinstance(entry, Mapping):
        raise UrdfMaskRunError(f"complete manifest {key} artifact identity is invalid")
    identity = {name: entry.get(name) for name in ("path", "sha256", "bytes")}
    if (
        not isinstance(identity["path"], str)
        or not isinstance(identity["sha256"], str)
        or isinstance(identity["bytes"], bool)
        or not isinstance(identity["bytes"], int)
    ):
        raise UrdfMaskRunError(f"complete manifest {key} artifact identity is incomplete")
    return identity


def _anchor_completed_manifest_record(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    skip_overlay: bool,
) -> None:
    for key in ("gripper_masks", "masks", "diagnostics", "overlay"):
        previous_identity = _artifact_identity_anchor(previous, key=key)
        current_identity = _artifact_identity_anchor(current, key=key)
        if key == "overlay" and skip_overlay:
            if previous_identity is not None or current_identity is not None:
                raise UrdfMaskRunError(
                    "skip-overlay complete record must anchor a null overlay"
                )
            continue
        if previous_identity != current_identity:
            raise UrdfMaskRunError(
                f"complete manifest artifact identity changed for {key}"
            )


def _new_run_manifest(
    config: RunConfig,
    plans: Sequence[EpisodePlan],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": RUN_FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": config.run_id,
        "status": "running",
        "dry_run": False,
        "episode_count": len(plans),
        "run_contract": _jsonable(contract),
        "assets": _jsonable(contract["assets"]),
        "episodes": [
            {**plan.to_json(), "status": "pending"} for plan in plans
        ],
        "failures": [],
        "successful_episode_count": 0,
        "failed_episode_count": 0,
        "failure_attempt_count": 0,
    }


def _resume_manifest(
    path: Path,
    *,
    config: RunConfig,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = _load_json_object(path, description="run manifest")
    if manifest.get("format_version") != RUN_FORMAT_VERSION:
        raise UrdfMaskRunError(
            f"cannot resume unsupported manifest format: {manifest.get('format_version')!r}"
        )
    if manifest.get("run_id") != config.run_id:
        raise UrdfMaskRunError("resume manifest run id does not match the output directory")
    if manifest.get("run_contract") != _jsonable(contract):
        raise UrdfMaskRunError(
            "resume configuration/assets differ from the immutable run contract"
        )
    if not isinstance(manifest.get("failures"), list):
        raise UrdfMaskRunError("resume manifest failures must be a list")
    manifest["status"] = "running"
    manifest["resumed_at"] = datetime.now(timezone.utc).isoformat()
    return manifest


def _failure_record(plan: EpisodePlan, exc: Exception) -> dict[str, Any]:
    return {
        **plan.to_json(),
        "status": "failed",
        "failed_at": datetime.now(timezone.utc).isoformat(),
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


class IncrementalUrdfEpisodeWorker:
    """Render source-ready episodes one at a time with one persistent renderer."""

    def __init__(
        self,
        config: RunConfig,
        *,
        renderer: Any | None = None,
        fit_config: Mapping[str, Any] | None = None,
    ) -> None:
        if config.dry_run:
            raise ValueError("incremental URDF workers do not support dry_run")
        validate_run_config(config)
        configured_fit = _fit_config(config.fit_config_json)
        self._fit_config = (
            configured_fit if fit_config is None else dict(fit_config)
        )
        self.config = config
        self._renderer = renderer
        self._owns_renderer = renderer is None
        self._renderer_frame_shape: tuple[int, int] | None = None
        self._closed = False
        self._finalized = False
        self._manifest_path = config.run_dir / "manifest.json"
        self._plans: dict[int, EpisodePlan] = {}
        self._plan_contracts: dict[int, dict[str, Any]] = {}
        base_contract = _run_contract(config, (), fit_config=self._fit_config)

        if config.resume:
            manifest = _load_json_object(
                self._manifest_path,
                description="run manifest",
            )
            if manifest.get("format_version") != RUN_FORMAT_VERSION:
                raise UrdfMaskRunError("cannot resume unsupported manifest format")
            if manifest.get("run_id") != config.run_id:
                raise UrdfMaskRunError("resume manifest run id does not match")
            previous_contract = manifest.get("run_contract")
            if not isinstance(previous_contract, dict):
                raise UrdfMaskRunError("resume manifest has no run contract")
            previous_stable = {
                key: value
                for key, value in previous_contract.items()
                if key != "episode_plans"
            }
            expected_stable = {
                key: value for key, value in base_contract.items() if key != "episode_plans"
            }
            if previous_stable != expected_stable:
                raise UrdfMaskRunError(
                    "resume configuration/assets differ from the immutable run contract"
                )
            raw_plans = previous_contract.get("episode_plans")
            if not isinstance(raw_plans, list):
                raise UrdfMaskRunError("resume run contract episode_plans must be a list")
            for raw in raw_plans:
                if not isinstance(raw, dict) or "episode_index" not in raw:
                    raise UrdfMaskRunError("resume run contract has an invalid episode plan")
                episode_index = int(raw["episode_index"])
                if episode_index in self._plan_contracts:
                    raise UrdfMaskRunError(
                        f"resume run contract repeats episode {episode_index}"
                    )
                self._plan_contracts[episode_index] = raw
            frame_shapes = {
                tuple(int(value) for value in raw.get("frame_shape_hw", ()))
                for raw in self._plan_contracts.values()
            }
            if any(len(shape) != 2 for shape in frame_shapes) or len(frame_shapes) > 1:
                raise UrdfMaskRunError(
                    "resume run contract does not declare one common frame shape"
                )
            if frame_shapes:
                self._renderer_frame_shape = next(iter(frame_shapes))
            manifest["status"] = "running"
            manifest["resumed_at"] = datetime.now(timezone.utc).isoformat()
            self._manifest = manifest
        else:
            config.run_dir.mkdir(parents=True, exist_ok=False)
            self._manifest = _new_run_manifest(config, (), base_contract)
            self._manifest["episode_count"] = len(config.episode_ids)
            self._manifest["episodes"] = [
                {"episode_index": value, "status": "pending"}
                for value in config.episode_ids
            ]
        self._record_map = _episode_record_map(self._manifest)
        unexpected = set(self._record_map) - set(config.episode_ids)
        if unexpected:
            raise UrdfMaskRunError(
                f"run manifest contains undeclared episodes: {sorted(unexpected)}"
            )
        for episode_index in config.episode_ids:
            self._record_map.setdefault(
                episode_index,
                {"episode_index": episode_index, "status": "pending"},
            )
        self._checkpoint()

    def _ordered_records(self) -> list[dict[str, Any]]:
        return [self._record_map[value] for value in self.config.episode_ids]

    def _checkpoint(self) -> None:
        position = {value: index for index, value in enumerate(self.config.episode_ids)}
        contract = dict(self._manifest["run_contract"])
        contract["episode_plans"] = [
            self._plan_contracts[value]
            for value in sorted(self._plan_contracts, key=position.__getitem__)
        ]
        self._manifest["run_contract"] = contract
        self._manifest["episodes"] = self._ordered_records()
        _checkpoint_manifest(self._manifest_path, self._manifest)

    def _register_plan(self, plan: EpisodePlan) -> None:
        plan_payload = plan.to_json()
        previous = self._plan_contracts.get(plan.episode_index)
        if previous is not None and previous != plan_payload:
            raise UrdfMaskRunError(
                f"episode {plan.episode_index} differs from its immutable run plan"
            )
        if self._renderer_frame_shape is not None and (
            plan.frame_shape != self._renderer_frame_shape
        ):
            raise UrdfMaskRunError(
                "all episodes must share the renderer frame shape: "
                f"{plan.frame_shape} != {self._renderer_frame_shape}"
            )
        self._plans[plan.episode_index] = plan
        self._plan_contracts[plan.episode_index] = plan_payload
        self._checkpoint()

    def _ensure_renderer(self, frame_shape: tuple[int, int]) -> Any:
        if self._renderer is None:
            self._renderer, _ = create_renderer(self.config, frame_shape)
        if self._renderer_frame_shape is None:
            self._renderer_frame_shape = frame_shape
        elif self._renderer_frame_shape != frame_shape:
            raise UrdfMaskRunError(
                f"renderer frame shape changed: {frame_shape} != "
                f"{self._renderer_frame_shape}"
            )
        return self._renderer

    def _validate_or_render(self, plan: EpisodePlan) -> dict[str, Any]:
        output_dir = self.config.run_dir / f"episode_{plan.episode_index:06d}"
        previous_record = self._record_map[plan.episode_index]
        previous_status = previous_record.get("status")
        if output_dir.exists():
            completed = validate_completed_episode(output_dir, plan, self.config)
            if previous_status == "complete":
                _anchor_completed_manifest_record(
                    previous_record,
                    completed,
                    skip_overlay=self.config.skip_overlay,
                )
                completed["resume_action"] = "validated_skip"
            elif previous_status in {"pending", "failed"}:
                completed["resume_action"] = "crash_recovered"
            else:
                raise UrdfMaskRunError(
                    f"manifest episode status is invalid: {previous_status!r}"
                )
            return completed
        if previous_status == "complete":
            raise UrdfMaskRunError(
                "manifest marks episode complete but its published directory is missing: "
                f"{output_dir}"
            )

        temporary_dir = self.config.run_dir / (
            f".episode_{plan.episode_index:06d}.{uuid.uuid4().hex}.tmp"
        )
        temporary_dir.mkdir(parents=False, exist_ok=False)
        try:
            episode = load_urdf_gripper_episode(
                self.config.dataset_root,
                plan.episode_index,
                camera=self.config.camera,
                authoritative_loop=plan.loop,
            )
            source = load_four_channel_masks(
                plan.source_masks,
                frame_count=episode.frame_count,
            )
            calibration = load_camera_calibration(
                episode.paths.sidecar,
                camera=self.config.camera,
                frame_count=episode.frame_count,
            )
            depth = decode_depth_video(
                episode.paths.depth_video,
                frame_count=episode.frame_count,
                frame_shape=source.frame_shape,
            )
            product = render_episode_product(
                self._ensure_renderer(plan.frame_shape),
                self._fit_config,
                episode,
                calibration,
                depth,
                frame_shape=source.frame_shape,
                tolerance_mm=self.config.depth_tolerance_mm,
            )
            combined, diagnostics = save_episode_artifacts(
                temporary_dir,
                episode,
                source,
                product,
                tolerance_mm=self.config.depth_tolerance_mm,
                minimum_eligible_nonempty_fraction=(
                    self.config.minimum_eligible_nonempty_fraction
                ),
            )
            overlay: dict[str, Any] | None = None
            if not self.config.skip_overlay:
                overlay = render_overlay_video(
                    episode.paths.rgb_video,
                    combined,
                    temporary_dir / "overlay.mp4",
                    alpha=self.config.overlay_alpha,
                    crf=self.config.overlay_crf,
                    preset=self.config.overlay_preset,
                )
            finalize_episode_diagnostics(
                temporary_dir,
                diagnostics,
                overlay=overlay,
            )
            validate_completed_episode(temporary_dir, plan, self.config)
            if output_dir.exists():
                raise UrdfMaskRunError(
                    "episode output appeared during rendering; refusing overwrite: "
                    f"{output_dir}"
                )
            os.replace(temporary_dir, output_dir)
            return validate_completed_episode(output_dir, plan, self.config)
        finally:
            if temporary_dir.exists():
                shutil.rmtree(temporary_dir)

    def process_episode(self, episode_index: int) -> dict[str, Any]:
        """Plan, render, validate, and checkpoint one newly source-ready episode."""

        if self._closed:
            raise RuntimeError("incremental URDF worker is closed")
        if self._finalized:
            raise RuntimeError("incremental URDF worker is finalized")
        if episode_index not in self.config.episode_ids:
            raise ValueError(f"episode {episode_index} is not declared by this run")
        plan: EpisodePlan | None = None
        try:
            plan = build_episode_plan(self.config, episode_index)
            self._register_plan(plan)
            record = self._validate_or_render(plan)
        except Exception as exc:  # noqa: BLE001 - one episode failure is checkpointed
            if plan is None:
                record = {
                    "episode_index": episode_index,
                    "status": "failed",
                    "failed_at": datetime.now(timezone.utc).isoformat(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            else:
                record = _failure_record(plan, exc)
            self._manifest["failures"].append(record)
        self._record_map[episode_index] = record
        self._checkpoint()
        return cast("dict[str, Any]", _jsonable(record))

    def snapshot(self) -> dict[str, Any]:
        snapshot = cast("dict[str, Any]", _jsonable(self._manifest))
        snapshot["manifest_path"] = str(self._manifest_path)
        return snapshot

    def finalize(self) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("incremental URDF worker is closed")
        for episode_index in self.config.episode_ids:
            record = self._record_map[episode_index]
            if record.get("status") != "pending":
                continue
            failure = {
                "episode_index": episode_index,
                "status": "failed",
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "error_type": "SourceEpisodeUnavailable",
                "error": "source episode did not become ready",
            }
            self._record_map[episode_index] = failure
            self._manifest["failures"].append(failure)
        incomplete_ids = [
            value
            for value in self.config.episode_ids
            if self._record_map[value].get("status") != "complete"
        ]
        self._manifest["status"] = "failed" if incomplete_ids else "complete"
        self._finalized = True
        self._checkpoint()
        result = self.snapshot()
        if incomplete_ids:
            raise UrdfBatchIncompleteError(
                f"URDF gripper run failed for episodes {incomplete_ids}; "
                f"see {self._manifest_path}",
                result=result,
            )
        return result

    def close(self) -> None:
        if self._closed:
            return
        if self._owns_renderer and self._renderer is not None:
            close = getattr(self._renderer, "close", None)
            if callable(close):
                close()
        self._closed = True


def run_experiment(
    config: RunConfig,
    *,
    renderer: Any | None = None,
    fit_config: Any | None = None,
) -> dict[str, Any]:
    plans = build_plan(config)
    configured_fit = _fit_config(config.fit_config_json)
    if fit_config is None:
        effective_fit = configured_fit
    elif isinstance(fit_config, Mapping):
        effective_fit = dict(fit_config)
    else:
        raise TypeError("fit_config must be a keyword-argument mapping")
    contract = _run_contract(config, plans, fit_config=effective_fit)
    base_manifest: dict[str, Any] = {
        "format_version": RUN_FORMAT_VERSION,
        "run_id": config.run_id,
        "dry_run": config.dry_run,
        "episode_count": len(plans),
        "run_contract": contract,
        "assets": contract["assets"],
        "episodes": [plan.to_json() for plan in plans],
    }
    if config.dry_run:
        return base_manifest

    manifest_path = config.run_dir / "manifest.json"
    if config.resume:
        manifest = _resume_manifest(
            manifest_path,
            config=config,
            contract=contract,
        )
    else:
        config.run_dir.mkdir(parents=True, exist_ok=False)
        manifest = _new_run_manifest(config, plans, contract)
    record_map = _episode_record_map(manifest)
    manifest["episodes"] = _ordered_episode_records(plans, record_map)
    _checkpoint_manifest(manifest_path, manifest)

    pending: list[EpisodePlan] = []
    skipped = 0
    for plan in plans:
        output_dir = config.run_dir / f"episode_{plan.episode_index:06d}"
        previous_record = record_map[plan.episode_index]
        previous_status = previous_record.get("status")
        if not output_dir.exists():
            if previous_status == "complete":
                raise UrdfMaskRunError(
                    "manifest marks episode complete but its published directory is missing: "
                    f"{output_dir}"
                )
            pending.append(plan)
            continue
        try:
            completed = validate_completed_episode(output_dir, plan, config)
            if previous_status == "complete":
                _anchor_completed_manifest_record(
                    previous_record,
                    completed,
                    skip_overlay=config.skip_overlay,
                )
                completed["resume_action"] = "validated_skip"
            elif previous_status in {"pending", "failed"}:
                completed["resume_action"] = "crash_recovered"
            else:
                raise UrdfMaskRunError(
                    f"manifest episode status is invalid: {previous_status!r}"
                )
            record_map[plan.episode_index] = completed
            skipped += 1
        except Exception as exc:
            if previous_status == "complete":
                raise UrdfMaskRunError(
                    "manifest-complete episode failed immutable resume validation; "
                    f"refusing recovery or overwrite: {output_dir}: {exc}"
                ) from exc
            guarded = UrdfMaskRunError(
                f"existing episode output is incomplete or invalid; refusing overwrite: "
                f"{output_dir}: {exc}"
            )
            failure = _failure_record(plan, guarded)
            record_map[plan.episode_index] = failure
            manifest["failures"].append(failure)
        manifest["episodes"] = _ordered_episode_records(plans, record_map)
        _checkpoint_manifest(manifest_path, manifest)

    owns_renderer = False
    try:
        if pending and renderer is None:
            renderer, _ = create_renderer(config, pending[0].frame_shape)
            owns_renderer = True
        for position, plan in enumerate(plans, start=1):
            if plan not in pending:
                status = record_map[plan.episode_index]["status"]
                print(
                    json.dumps(
                        {
                            "progress": f"{position}/{len(plans)}",
                            "episode_index": plan.episode_index,
                            "status": "skipped" if status == "complete" else status,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                continue
            output_dir = config.run_dir / f"episode_{plan.episode_index:06d}"
            temporary_dir = config.run_dir / (
                f".episode_{plan.episode_index:06d}.{uuid.uuid4().hex}.tmp"
            )
            temporary_dir.mkdir(parents=False, exist_ok=False)
            status = "complete"
            try:
                episode = load_urdf_gripper_episode(
                    config.dataset_root,
                    plan.episode_index,
                    camera=config.camera,
                    authoritative_loop=plan.loop,
                )
                source = load_four_channel_masks(
                    plan.source_masks,
                    frame_count=episode.frame_count,
                )
                calibration = load_camera_calibration(
                    episode.paths.sidecar,
                    camera=config.camera,
                    frame_count=episode.frame_count,
                )
                depth = decode_depth_video(
                    episode.paths.depth_video,
                    frame_count=episode.frame_count,
                    frame_shape=source.frame_shape,
                )
                product = render_episode_product(
                    renderer,
                    effective_fit,
                    episode,
                    calibration,
                    depth,
                    frame_shape=source.frame_shape,
                    tolerance_mm=config.depth_tolerance_mm,
                )
                combined, diagnostics = save_episode_artifacts(
                    temporary_dir,
                    episode,
                    source,
                    product,
                    tolerance_mm=config.depth_tolerance_mm,
                    minimum_eligible_nonempty_fraction=(
                        config.minimum_eligible_nonempty_fraction
                    ),
                )
                overlay: dict[str, Any] | None = None
                if not config.skip_overlay:
                    overlay = render_overlay_video(
                        episode.paths.rgb_video,
                        combined,
                        temporary_dir / "overlay.mp4",
                        alpha=config.overlay_alpha,
                        crf=config.overlay_crf,
                        preset=config.overlay_preset,
                    )
                finalize_episode_diagnostics(
                    temporary_dir,
                    diagnostics,
                    overlay=overlay,
                )
                validate_completed_episode(temporary_dir, plan, config)
                if output_dir.exists():
                    raise UrdfMaskRunError(
                        "episode output appeared during rendering; refusing overwrite: "
                        f"{output_dir}"
                    )
                os.replace(temporary_dir, output_dir)
                completed = validate_completed_episode(output_dir, plan, config)
                record_map[plan.episode_index] = completed
            except Exception as exc:
                status = "failed"
                failure = _failure_record(plan, exc)
                record_map[plan.episode_index] = failure
                manifest["failures"].append(failure)
            finally:
                if temporary_dir.exists():
                    shutil.rmtree(temporary_dir)
            manifest["episodes"] = _ordered_episode_records(plans, record_map)
            _checkpoint_manifest(manifest_path, manifest)
            print(
                json.dumps(
                    {
                        "progress": f"{position}/{len(plans)}",
                        "episode_index": plan.episode_index,
                        "status": status,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    finally:
        if owns_renderer and renderer is not None:
            close = getattr(renderer, "close", None)
            if callable(close):
                close()

    manifest["episodes"] = _ordered_episode_records(plans, record_map)
    incomplete_ids = [
        plan.episode_index
        for plan in plans
        if record_map[plan.episode_index].get("status") != "complete"
    ]
    manifest["resume_skipped_episode_count"] = skipped
    manifest["status"] = "failed" if incomplete_ids else "complete"
    _checkpoint_manifest(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    if incomplete_ids:
        raise UrdfBatchIncompleteError(
            f"URDF gripper run failed for episodes {incomplete_ids}; see {manifest_path}",
            result=manifest,
        )
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    config = parse_args(argv)
    result = run_experiment(config)
    print(json.dumps(_jsonable(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
