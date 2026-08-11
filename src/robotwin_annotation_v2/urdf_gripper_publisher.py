"""Publish URDF gripper results through the canonical mask-run contract."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import tempfile
import uuid
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

MASK_FORMAT_VERSION = "robotwin_visible_masks_v2"
MASK_RUN_FORMAT_VERSION = "robotwin_mask_run_v2"
PROVENANCE_FORMAT_VERSION = "robotwin_frame_provenance_v2"
URDF_PRODUCT_FORMAT_VERSION = "robotwin_urdf_gripper_masks_v2"
PROCESS_SUMMARY_FORMAT_VERSION = "robotwin_process_dataset_summary_v1"
LOOP_FORMAT_VERSION = "robotwin_loop_context_v1"
DERIVATION_SOURCE_LINEAGE_FORMAT_VERSION = "robotwin_derivation_source_lineage_v1"
DERIVATION_FORMAT_VERSION = "robotwin_urdf_gripper_derivation_v1"
PUBLISHER_IMPLEMENTATION_FORMAT_VERSION = (
    "robotwin_urdf_gripper_publisher_implementation_v1"
)
INSTANCE_NAMES = ("target_0", "receiver_0", "gripper_left", "gripper_right")
ROLES = ("target", "receiver", "gripper", "gripper")
ROLE_ARTIFACT_PATH_KEYS = (
    "seed_rgb_path",
    "seed_mask_path",
    "canonical_envelope_path",
    "native_track_path",
    "temporal_qc_path",
)
SOURCE_EXCLUDED_NAMES = {
    "masks.npz",
    "run_manifest.json",
    "frame_provenance.json",
    "gripper_left",
    "gripper_right",
    "gripper_failure.json",
}
GENERATED_FILENAMES = {
    "masks.npz",
    "run_manifest.json",
    "frame_provenance.json",
}


class UrdfGripperPublishError(RuntimeError):
    """A URDF episode cannot satisfy the canonical public artifact contract."""


@dataclass(frozen=True)
class DerivationSourceEpisode:
    """Validated, content-addressed source inputs for one derived episode."""

    source_run_dir: Path
    episode_dir: Path
    summary: Mapping[str, Any]
    loop: Mapping[str, Any]
    manifest: Mapping[str, Any]
    provenance: Mapping[str, Any]
    source_masks: Mapping[str, Any]
    source_roles: tuple[dict[str, Any], ...]
    source_algorithm: Mapping[str, Any]
    source_material: Mapping[str, Path]
    lineage: Mapping[str, Any]

    @property
    def frame_count(self) -> int:
        return int(self.source_masks["frame_count"])


def _json_clone(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _read_json_object(path: Path, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{description} is missing: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise UrdfGripperPublishError(f"cannot read {description}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise UrdfGripperPublishError(f"{description} must be one JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_identity(path: Path, *, relative_path: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise UrdfGripperPublishError(f"artifact must be a regular file: {path}")
    return {
        "path": relative_path,
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _source_file_identity(path: Path, *, source_run_dir: Path) -> dict[str, Any]:
    try:
        relative = path.resolve().relative_to(source_run_dir.resolve()).as_posix()
    except ValueError as exc:
        raise UrdfGripperPublishError(
            f"source artifact escapes its source run: {path}"
        ) from exc
    return _file_identity(path, relative_path=relative)


def publisher_implementation_identity() -> dict[str, Any]:
    """Return a dirty-worktree-safe identity for the canonical publisher."""

    project_root = Path(__file__).resolve().parents[2]
    path = Path(__file__).resolve()
    return {
        "format_version": PUBLISHER_IMPLEMENTATION_FORMAT_VERSION,
        "files": [
            _file_identity(
                path,
                relative_path=path.relative_to(project_root).as_posix(),
            )
        ],
    }


def _validate_identity(path: Path, identity: Mapping[str, Any], *, label: str) -> None:
    expected_sha = identity.get("sha256")
    expected_bytes = identity.get("bytes")
    if not isinstance(expected_sha, str) or not expected_sha:
        raise UrdfGripperPublishError(f"{label} has no valid sha256 identity")
    if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int):
        raise UrdfGripperPublishError(f"{label} has no valid byte-size identity")
    actual = _file_identity(path, relative_path=str(identity.get("path", path.name)))
    if actual["sha256"] != expected_sha or actual["bytes"] != expected_bytes:
        raise UrdfGripperPublishError(f"{label} hash/size differs from its backend record")


def _safe_child(root: Path, relative: Any, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative.strip():
        raise UrdfGripperPublishError(f"{label} path is missing")
    root_resolved = root.resolve()
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise UrdfGripperPublishError(f"{label} escapes its episode directory") from exc
    return candidate


def _small_strings(array: np.ndarray) -> tuple[str, ...]:
    return tuple(str(item) for item in array.tolist())


def _scalar_int(value: np.ndarray, *, label: str) -> int:
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in {"i", "u"}:
        raise UrdfGripperPublishError(f"{label} must be one integer scalar")
    return int(array.item())


def _load_source_masks(path: Path) -> dict[str, Any]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "format_version",
                "frame_count",
                "masks",
                "instance_names",
                "roles",
                "annotation_status",
                "qc_status",
            }
            actual_keys = set(archive.files)
            missing = sorted(required - actual_keys)
            extra = sorted(actual_keys - required)
            if missing or extra:
                raise UrdfGripperPublishError(
                    "source masks must contain exactly the canonical seven keys: "
                    f"missing={missing}, extra={extra}: {path}"
                )
            format_version = np.asarray(archive["format_version"])
            masks = np.asarray(archive["masks"])
            frame_count = _scalar_int(archive["frame_count"], label="source frame_count")
            names = _small_strings(np.asarray(archive["instance_names"]))
            roles = _small_strings(np.asarray(archive["roles"]))
            statuses = _small_strings(np.asarray(archive["annotation_status"]))
            qc_statuses = _small_strings(np.asarray(archive["qc_status"]))
    except (EOFError, OSError, ValueError, zipfile.BadZipFile) as exc:
        if isinstance(exc, UrdfGripperPublishError):
            raise
        raise UrdfGripperPublishError(f"cannot load source masks {path}: {exc}") from exc
    if (
        format_version.ndim != 0
        or str(format_version.item()) != MASK_FORMAT_VERSION
    ):
        raise UrdfGripperPublishError(
            f"unsupported source masks format: {format_version!r}"
        )
    if masks.dtype != np.bool_ or masks.ndim != 4 or masks.shape[0] != 4:
        raise UrdfGripperPublishError(
            f"source masks must have bool shape [4,T,H,W], got {masks.shape}/{masks.dtype}"
        )
    if frame_count < 1 or masks.shape[1] != frame_count:
        raise UrdfGripperPublishError("source masks and frame_count disagree")
    if names != INSTANCE_NAMES or roles != ROLES:
        raise UrdfGripperPublishError("source mask channel names/roles are not canonical")
    if statuses[:2] != ("valid", "valid") or qc_statuses[:2] != ("passed", "passed"):
        raise UrdfGripperPublishError(
            "source target and receiver must both be valid and QC-passed"
        )
    return {
        "format_version": MASK_FORMAT_VERSION,
        "frame_count": frame_count,
        "masks": masks.copy(),
        "annotation_status": statuses,
        "qc_status": qc_statuses,
    }


def _parse_window(value: Any, *, frame_count: int, label: str) -> tuple[int, int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise UrdfGripperPublishError(f"{label} must contain two integer frame ids")
    start, end = (int(item) for item in value)
    if start < 0 or end < start or end >= frame_count:
        raise UrdfGripperPublishError(f"{label} lies outside the episode frame range")
    return start, end


def _artifact_from_record(
    backend_episode_dir: Path,
    record: Mapping[str, Any],
    key: str,
) -> tuple[Path, dict[str, Any]]:
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise UrdfGripperPublishError("backend episode record has no artifact map")
    raw_identity = artifacts.get(key)
    if not isinstance(raw_identity, Mapping):
        raise UrdfGripperPublishError(f"backend episode record has no {key} identity")
    identity = dict(raw_identity)
    path = _safe_child(backend_episode_dir, identity.get("path"), label=key)
    _validate_identity(path, identity, label=f"backend {key}")
    return path, identity


def _load_product(
    path: Path,
    *,
    frame_count: int,
    frame_shape: tuple[int, int],
    active_arm: str,
    active_window: tuple[int, int],
) -> dict[str, Any]:
    try:
        with np.load(path, allow_pickle=False) as archive:
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
                raise UrdfGripperPublishError(
                    f"URDF product is missing keys {missing}: {path}"
                )
            format_version = str(np.asarray(archive["format_version"]).item())
            product_frame_count = _scalar_int(
                archive["frame_count"], label="URDF product frame_count"
            )
            product_arm = str(np.asarray(archive["active_arm"]).item())
            product_window = tuple(
                int(item) for item in np.asarray(archive["active_window"]).tolist()
            )
            tolerance = float(np.asarray(archive["depth_tolerance_mm"]).item())
            tracks = {
                key: np.asarray(archive[key]).copy()
                for key in (
                    "gripper_track",
                    "rendered_amodal_track",
                    "depth_evaluable_track",
                    "depth_consistent_track",
                )
            }
    except (EOFError, OSError, ValueError, zipfile.BadZipFile) as exc:
        if isinstance(exc, UrdfGripperPublishError):
            raise
        raise UrdfGripperPublishError(f"cannot load URDF product {path}: {exc}") from exc
    if format_version != URDF_PRODUCT_FORMAT_VERSION:
        raise UrdfGripperPublishError(f"unsupported URDF product format: {format_version}")
    if product_frame_count != frame_count or product_arm != active_arm:
        raise UrdfGripperPublishError("URDF product episode metadata differs from its record")
    if product_window != active_window or not np.isfinite(tolerance) or tolerance < 0:
        raise UrdfGripperPublishError("URDF product window/tolerance is invalid")
    expected_shape = (frame_count, *frame_shape)
    for name, track in tracks.items():
        if track.dtype != np.bool_ or track.shape != expected_shape:
            raise UrdfGripperPublishError(
                f"{name} must have bool shape {expected_shape}, got {track.shape}/{track.dtype}"
            )
    visible = tracks["gripper_track"]
    amodal = tracks["rendered_amodal_track"]
    evaluable = tracks["depth_evaluable_track"]
    consistent = tracks["depth_consistent_track"]
    if np.any(visible & ~consistent) or np.any(consistent & ~evaluable):
        raise UrdfGripperPublishError("URDF visible/depth subset contract is violated")
    if np.any(evaluable & ~amodal):
        raise UrdfGripperPublishError("URDF evaluable/amodal subset contract is violated")
    start, end = active_window
    for name, track in tracks.items():
        if track[:start].any() or track[end + 1 :].any():
            raise UrdfGripperPublishError(f"{name} is nonempty outside the active window")
    if not visible[start : end + 1].any():
        raise UrdfGripperPublishError("URDF gripper track is empty inside the active window")
    return {"tracks": tracks, "depth_tolerance_mm": tolerance}


def _validate_backend_combined_masks(
    path: Path,
    *,
    source_masks: np.ndarray,
    gripper_track: np.ndarray,
    active_arm: str,
) -> None:
    try:
        with np.load(path, allow_pickle=False) as archive:
            masks = np.asarray(archive["masks"])
            names = _small_strings(np.asarray(archive["instance_names"]))
            roles = _small_strings(np.asarray(archive["roles"]))
    except (EOFError, KeyError, OSError, ValueError, zipfile.BadZipFile) as exc:
        raise UrdfGripperPublishError(
            f"cannot validate backend combined masks {path}: {exc}"
        ) from exc
    if masks.dtype != np.bool_ or masks.shape != source_masks.shape:
        raise UrdfGripperPublishError("backend combined masks have an invalid shape/dtype")
    if names != INSTANCE_NAMES or roles != ROLES:
        raise UrdfGripperPublishError("backend combined masks have invalid channels")
    active_index = 2 if active_arm == "left" else 3
    inactive_index = 3 if active_index == 2 else 2
    if not np.array_equal(masks[:2], source_masks[:2]):
        raise UrdfGripperPublishError("backend changed source target/receiver pixels")
    if not np.array_equal(masks[active_index], gripper_track) or masks[inactive_index].any():
        raise UrdfGripperPublishError("backend combined gripper channels differ from product")


def _source_material_files(source_episode_dir: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}

    def visit(directory: Path, relative_dir: Path) -> None:
        for child in sorted(directory.iterdir(), key=lambda item: item.name):
            relative = relative_dir / child.name
            if not relative_dir.parts and child.name in SOURCE_EXCLUDED_NAMES:
                continue
            if child.is_symlink():
                raise UrdfGripperPublishError(
                    f"source episode material must not contain symlinks: {child}"
                )
            if child.is_dir():
                visit(child, relative)
            elif child.is_file():
                result[relative.as_posix()] = child
            else:
                raise UrdfGripperPublishError(f"unsupported source artifact type: {child}")

    visit(source_episode_dir, Path())
    if not any(path.startswith("target_0/") for path in result):
        raise UrdfGripperPublishError("source episode has no target_0 artifacts")
    if not any(path.startswith("receiver_0/") for path in result):
        raise UrdfGripperPublishError("source episode has no receiver_0 artifacts")
    return result


def _materialize_source_files(files: Mapping[str, Path], destination: Path) -> None:
    for relative, source in files.items():
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, target)
        except OSError as exc:
            if exc.errno not in {
                errno.EXDEV,
                errno.EPERM,
                errno.EACCES,
                errno.EMLINK,
                errno.ENOSYS,
                errno.EOPNOTSUPP,
            }:
                raise
            shutil.copy2(source, target)


def _source_material_identities(files: Mapping[str, Path]) -> list[dict[str, Any]]:
    return [
        _file_identity(path, relative_path=relative)
        for relative, path in sorted(files.items())
    ]


def _validate_source_manifest(
    manifest: Mapping[str, Any],
    *,
    task: str,
    camera: str,
    episode_index: int,
    frame_count: int,
    source_episode_dir: Path,
    source_run_id: str,
    role_windows: Mapping[str, tuple[int, int]],
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, dict[str, Path]],
]:
    if manifest.get("format_version") != MASK_RUN_FORMAT_VERSION:
        raise UrdfGripperPublishError("source manifest format is unsupported")
    if manifest.get("run_id") != source_run_id:
        raise UrdfGripperPublishError(
            "source manifest run_id differs from source process summary"
        )
    episode = manifest.get("episode")
    expected_episode = {
        "task": task,
        "episode_index": episode_index,
        "episode_id": f"{episode_index:06d}",
        "camera": camera,
    }
    if not isinstance(episode, Mapping) or any(
        episode.get(key) != value for key, value in expected_episode.items()
    ):
        raise UrdfGripperPublishError("source manifest episode identity is not canonical")
    if manifest.get("frame_count") != frame_count:
        raise UrdfGripperPublishError("source manifest frame_count differs from source masks")
    raw_roles = manifest.get("roles")
    if not isinstance(raw_roles, list):
        raise UrdfGripperPublishError("source manifest roles must be a list")
    role_map: dict[str, dict[str, Any]] = {}
    for raw in raw_roles:
        if not isinstance(raw, dict):
            continue
        role = raw.get("role")
        if role not in {"target", "receiver"}:
            continue
        if role in role_map:
            raise UrdfGripperPublishError(f"source manifest repeats role {role}")
        role_map[str(role)] = _json_clone(raw)
    if set(role_map) != {"target", "receiver"}:
        raise UrdfGripperPublishError("source manifest lacks target or receiver")
    role_artifacts: dict[str, dict[str, Path]] = {}
    for role in ("target", "receiver"):
        record = role_map[role]
        if record.get("status") != "ok" or record.get("qc_status") != "passed":
            raise UrdfGripperPublishError(f"source {role} is not status=ok/qc_status=passed")
        output_window = _parse_window(
            record.get("output_window"), frame_count=frame_count, label=role
        )
        if output_window != role_windows[role]:
            raise UrdfGripperPublishError(
                f"source {role} output window differs from source loop"
            )
        references: dict[str, Path] = {}
        for key in ROLE_ARTIFACT_PATH_KEYS:
            relative = record.get(key)
            if relative is not None:
                path = _safe_child(source_episode_dir, relative, label=f"{role} {key}")
                if not path.is_file() or path.is_symlink():
                    raise UrdfGripperPublishError(f"source {role} artifact is missing: {path}")
                references[key] = path
        role_artifacts[role] = references
    algorithm = manifest.get("algorithm")
    if not isinstance(algorithm, Mapping):
        raise UrdfGripperPublishError("source manifest has no algorithm metadata")
    return (
        [role_map["target"], role_map["receiver"]],
        _json_clone(algorithm),
        role_artifacts,
    )


def _validate_source_summary(
    summary: Mapping[str, Any],
    *,
    source_run_dir: Path,
    task: str,
    camera: str,
    episode_index: int,
    expected_dataset_root: Path | None,
) -> Path:
    if summary.get("format_version") != PROCESS_SUMMARY_FORMAT_VERSION:
        raise UrdfGripperPublishError("source process summary format is unsupported")
    if summary.get("run_id") != source_run_dir.name:
        raise UrdfGripperPublishError(
            "source process summary run_id differs from its directory"
        )
    if summary.get("task") != task or summary.get("camera") != camera:
        raise UrdfGripperPublishError(
            "source process summary task/camera differs from the requested episode"
        )
    raw_dataset_root = summary.get("dataset_root")
    if not isinstance(raw_dataset_root, str) or not raw_dataset_root.strip():
        raise UrdfGripperPublishError(
            "source process summary has no valid dataset_root"
        )
    dataset_root = Path(raw_dataset_root).expanduser().resolve()
    if (
        expected_dataset_root is not None
        and dataset_root != expected_dataset_root.expanduser().resolve()
    ):
        raise UrdfGripperPublishError(
            "source process summary dataset_root differs from the requested dataset"
        )

    records = summary.get("records")
    if not isinstance(records, list):
        raise UrdfGripperPublishError("source process summary records must be a list")
    records_by_episode: dict[int, Mapping[str, Any]] = {}
    for raw_record in records:
        if not isinstance(raw_record, Mapping) or "episode" not in raw_record:
            continue
        raw_id = raw_record.get("episode")
        if isinstance(raw_id, bool):
            raise UrdfGripperPublishError(
                "source process summary contains an invalid episode record"
            )
        try:
            record_id = int(raw_id)
        except (TypeError, ValueError) as exc:
            raise UrdfGripperPublishError(
                "source process summary contains an invalid episode record"
            ) from exc
        if record_id in records_by_episode:
            raise UrdfGripperPublishError(
                f"source process summary repeats episode {record_id}"
            )
        records_by_episode[record_id] = raw_record
    record = records_by_episode.get(episode_index)
    if record is None:
        raise UrdfGripperPublishError(
            "source process summary has no record for the requested episode"
        )
    if record.get("status") not in {"completed", "skipped_complete"}:
        raise UrdfGripperPublishError(
            "source process summary episode record is not complete"
        )

    dynamic = summary.get("dynamic_manifest")
    if not isinstance(dynamic, Mapping):
        raise UrdfGripperPublishError(
            "source process summary has no dynamic_manifest object"
        )
    expected_dynamic = {
        "task": task,
        "camera": camera,
        "dataset_root": str(dataset_root),
    }
    for key, expected in expected_dynamic.items():
        if dynamic.get(key) != expected:
            raise UrdfGripperPublishError(
                f"source dynamic manifest {key} differs from the source run"
            )
    regression_ids = dynamic.get("regression_episode_ids")
    if not isinstance(regression_ids, list) or episode_index not in regression_ids:
        raise UrdfGripperPublishError(
            "source dynamic manifest does not contain the requested episode"
        )
    return dataset_root


def _validate_source_loop(
    path: Path,
    *,
    dataset_root: Path,
    task: str,
    camera: str,
    episode_index: int,
    frame_count: int,
) -> tuple[dict[str, Any], dict[str, tuple[int, int]]]:
    from robotwin_annotation_v2.urdf_gripper_data import (
        UrdfGripperDataError,
        load_authoritative_loop_context,
    )

    try:
        context = load_authoritative_loop_context(
            path,
            expected_task=task,
            expected_episode_index=episode_index,
            expected_camera=camera,
        )
    except UrdfGripperDataError as exc:
        raise UrdfGripperPublishError(str(exc)) from exc
    if context.frame_count != frame_count:
        raise UrdfGripperPublishError(
            "source loop frame_count differs from source masks"
        )
    loop = _read_json_object(path, description="source loop")
    sources = loop.get("sources")
    if not isinstance(sources, Mapping):
        raise UrdfGripperPublishError("source loop has no source path map")
    chunk = f"chunk-{episode_index // 1000:03d}"
    episode_id = f"{episode_index:06d}"
    expected_sources = {
        "state": dataset_root / "data" / chunk / f"episode_{episode_id}.parquet",
        "video": (
            dataset_root
            / "videos"
            / chunk
            / f"observation.images.{camera}"
            / f"episode_{episode_id}.mp4"
        ),
    }
    for key, expected in expected_sources.items():
        raw = sources.get(key)
        if not isinstance(raw, str) or Path(raw).expanduser().resolve() != expected:
            raise UrdfGripperPublishError(
                f"source loop {key} path differs from the source dataset"
            )
    return loop, {
        "target": (context.events.t_move_start, context.events.t_close_done),
        "receiver": (context.events.t_close_done, context.events.t_open_done),
    }


def _validate_source_provenance(
    provenance: Mapping[str, Any],
    *,
    role_windows: Mapping[str, tuple[int, int]],
) -> None:
    if provenance.get("format_version") != PROVENANCE_FORMAT_VERSION:
        raise UrdfGripperPublishError("source frame provenance format is unsupported")
    channels = provenance.get("channels")
    if not isinstance(channels, Mapping):
        raise UrdfGripperPublishError("source frame provenance has no channel map")
    for role, channel_name in (("target", "target_0"), ("receiver", "receiver_0")):
        channel = channels.get(channel_name)
        if not isinstance(channel, Mapping):
            raise UrdfGripperPublishError(
                f"source frame provenance lacks {channel_name}"
            )
        if channel.get("status") != "ok" or channel.get("qc_status") != "passed":
            raise UrdfGripperPublishError(
                f"source frame provenance {channel_name} is not QC-passed"
            )
        raw_window = channel.get("output_window")
        if (
            not isinstance(raw_window, list)
            or len(raw_window) != 2
            or tuple(raw_window) != role_windows[role]
        ):
            raise UrdfGripperPublishError(
                f"source frame provenance {channel_name} window differs from source loop"
            )


def validate_derivation_source_episode(
    source_episode_dir: Path,
    *,
    task: str,
    camera: str,
    episode_index: int,
    expected_frame_count: int | None = None,
    expected_dataset_root: Path | None = None,
) -> DerivationSourceEpisode:
    """Validate and content-address every inherited source dependency."""

    episode_dir = source_episode_dir.expanduser().resolve()
    if not episode_dir.is_dir() or episode_dir.is_symlink():
        raise FileNotFoundError(f"source episode directory is missing: {episode_dir}")
    if len(episode_dir.parents) < 3:
        raise UrdfGripperPublishError(
            "source episode must be nested below one source run directory"
        )
    source_run_dir = episode_dir.parents[2]
    expected_episode_dir = (
        source_run_dir / task / f"episode_{episode_index:06d}" / camera
    )
    if expected_episode_dir != episode_dir:
        raise UrdfGripperPublishError(
            "source episode path is not <run>/<task>/episode_<id>/<camera>"
        )

    summary_path = source_run_dir / "process_summary.json"
    summary = _read_json_object(
        summary_path,
        description="source process summary",
    )
    dataset_root = _validate_source_summary(
        summary,
        source_run_dir=source_run_dir,
        task=task,
        camera=camera,
        episode_index=episode_index,
        expected_dataset_root=expected_dataset_root,
    )
    source = _load_source_masks(episode_dir / "masks.npz")
    frame_count = int(source["frame_count"])
    if expected_frame_count is not None and frame_count != expected_frame_count:
        raise UrdfGripperPublishError(
            "source masks frame_count differs from the expected dataset frame count"
        )

    loop, role_windows = _validate_source_loop(
        episode_dir / "loop.json",
        dataset_root=dataset_root,
        task=task,
        camera=camera,
        episode_index=episode_index,
        frame_count=frame_count,
    )
    manifest = _read_json_object(
        episode_dir / "run_manifest.json",
        description="source episode manifest",
    )
    source_roles, source_algorithm, role_artifacts = _validate_source_manifest(
        manifest,
        task=task,
        camera=camera,
        episode_index=episode_index,
        frame_count=frame_count,
        source_episode_dir=episode_dir,
        source_run_id=str(summary["run_id"]),
        role_windows=role_windows,
    )
    provenance = _read_json_object(
        episode_dir / "frame_provenance.json",
        description="source frame provenance",
    )
    _validate_source_provenance(provenance, role_windows=role_windows)

    dynamic = summary["dynamic_manifest"]
    raw_shape = dynamic.get("frame_shape_hw")
    frame_shape = tuple(int(item) for item in np.asarray(source["masks"]).shape[2:])
    if (
        not isinstance(raw_shape, list)
        or len(raw_shape) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in raw_shape)
        or tuple(raw_shape) != frame_shape
    ):
        raise UrdfGripperPublishError(
            "source dynamic manifest frame shape differs from source masks"
        )

    source_material = _source_material_files(episode_dir)
    role_identities = {
        role: {
            key: _source_file_identity(path, source_run_dir=source_run_dir)
            for key, path in sorted(references.items())
        }
        for role, references in sorted(role_artifacts.items())
    }
    control_paths = {
        "loop": episode_dir / "loop.json",
        "run_manifest": episode_dir / "run_manifest.json",
        "frame_provenance": episode_dir / "frame_provenance.json",
        "masks": episode_dir / "masks.npz",
    }
    lineage: dict[str, Any] = {
        "format_version": DERIVATION_SOURCE_LINEAGE_FORMAT_VERSION,
        "source_run": {
            "run_id": summary["run_id"],
            "path": str(source_run_dir),
            "dataset_root": str(dataset_root),
            "process_summary": _source_file_identity(
                summary_path,
                source_run_dir=source_run_dir,
            ),
        },
        "episode": {
            "task": task,
            "episode_index": episode_index,
            "episode_id": f"{episode_index:06d}",
            "camera": camera,
        },
        "frame_count": frame_count,
        "frame_shape_hw": list(frame_shape),
        "control_artifacts": {
            key: _source_file_identity(path, source_run_dir=source_run_dir)
            for key, path in control_paths.items()
        },
        "role_artifacts": role_identities,
    }
    lineage["lineage_sha256"] = _canonical_json_sha256(lineage)
    return DerivationSourceEpisode(
        source_run_dir=source_run_dir,
        episode_dir=episode_dir,
        summary=_json_clone(summary),
        loop=_json_clone(loop),
        manifest=_json_clone(manifest),
        provenance=_json_clone(provenance),
        source_masks=source,
        source_roles=tuple(source_roles),
        source_algorithm=source_algorithm,
        source_material=source_material,
        lineage=lineage,
    )


def _backend_provenance(
    backend_episode_dir: Path,
    record: Mapping[str, Any],
    artifact_identities: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    root_manifest_path = backend_episode_dir.parent / "manifest.json"
    root_manifest = _read_json_object(
        root_manifest_path,
        description="URDF backend run manifest",
    )
    format_version = root_manifest.get("format_version")
    if not isinstance(format_version, str) or not format_version.startswith(
        "robotwin_urdf_gripper_run_"
    ):
        raise UrdfGripperPublishError("unsupported URDF backend run manifest format")
    run_contract = root_manifest.get("run_contract")
    if not isinstance(run_contract, Mapping):
        raise UrdfGripperPublishError("URDF backend manifest has no immutable run contract")
    if root_manifest.get("run_id") != backend_episode_dir.parent.name:
        raise UrdfGripperPublishError(
            "URDF backend manifest run_id differs from its directory"
        )
    raw_episodes = root_manifest.get("episodes")
    if not isinstance(raw_episodes, list):
        raise UrdfGripperPublishError("URDF backend manifest has no episode records")
    episode_index = record.get("episode_index")
    recorded = [
        item
        for item in raw_episodes
        if isinstance(item, Mapping) and item.get("episode_index") == episode_index
    ]
    if len(recorded) != 1 or _json_clone(recorded[0]) != _json_clone(record):
        raise UrdfGripperPublishError(
            "backend episode record differs from the anchored run manifest"
        )
    raw_plans = run_contract.get("episode_plans")
    if not isinstance(raw_plans, list):
        raise UrdfGripperPublishError(
            "URDF backend immutable contract has no episode plans"
        )
    plans = [
        item
        for item in raw_plans
        if isinstance(item, Mapping) and item.get("episode_index") == episode_index
    ]
    if len(plans) != 1:
        raise UrdfGripperPublishError(
            "URDF backend immutable contract does not anchor the episode plan"
        )
    for key in (
        "frame_count",
        "frame_shape_hw",
        "active_arm",
        "active_window",
        "events",
        "inputs",
        "source_lineage",
    ):
        if _json_clone(plans[0].get(key)) != _json_clone(record.get(key)):
            raise UrdfGripperPublishError(
                f"backend episode {key} differs from its immutable plan"
            )
    assets = root_manifest.get("assets")
    if not isinstance(assets, Mapping):
        raise UrdfGripperPublishError("URDF backend manifest has no asset identities")
    stable_contract_keys = (
        "dataset_root",
        "source_run_dir",
        "task",
        "camera",
        "depth_tolerance_mm",
        "minimum_eligible_nonempty_fraction",
        "fit_config",
        "implementation",
    )
    stable_contract = {
        key: _json_clone(run_contract[key])
        for key in stable_contract_keys
        if key in run_contract
    }
    return {
        "format_version": format_version,
        "run_id": root_manifest.get("run_id"),
        "run_contract": stable_contract,
        "assets": _json_clone(assets),
        "inputs": _json_clone(record.get("inputs", {})),
        "artifacts": _json_clone(artifact_identities),
    }


def _validate_backend_publisher_identity(
    backend_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Require the backend immutable contract to anchor this publisher file."""

    run_contract = backend_provenance.get("run_contract")
    implementation = (
        run_contract.get("implementation")
        if isinstance(run_contract, Mapping)
        else None
    )
    files = implementation.get("files") if isinstance(implementation, Mapping) else None
    if not isinstance(files, list):
        raise UrdfGripperPublishError(
            "URDF backend implementation does not anchor publisher files"
        )
    publisher = publisher_implementation_identity()
    expected = publisher["files"][0]
    matches = [
        item
        for item in files
        if isinstance(item, Mapping) and item.get("path") == expected["path"]
    ]
    if len(matches) != 1:
        raise UrdfGripperPublishError(
            "URDF backend implementation must anchor exactly one publisher file"
        )
    actual = matches[0]
    if any(actual.get(key) != expected[key] for key in ("sha256", "bytes")):
        raise UrdfGripperPublishError(
            "URDF backend publisher identity differs from the current publisher"
        )
    return publisher


def _build_public_payloads(
    *,
    source_episode_dir: Path,
    backend_episode_dir: Path,
    destination_dir: Path,
    run_id: str,
    task: str,
    camera: str,
    record: Mapping[str, Any],
    validated_source: DerivationSourceEpisode,
    product: Mapping[str, Any],
    artifact_identities: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    source = validated_source.source_masks
    source_manifest = validated_source.manifest
    source_provenance = validated_source.provenance
    source_material = validated_source.source_material
    episode_index = int(record["episode_index"])
    frame_count = int(source["frame_count"])
    active_arm = str(record["active_arm"])
    active_window = _parse_window(
        record.get("active_window"), frame_count=frame_count, label="active_window"
    )
    active_index = 2 if active_arm == "left" else 3
    active_role = f"gripper_{active_arm}"
    visible = np.asarray(product["tracks"]["gripper_track"], dtype=bool)
    masks = np.zeros_like(np.asarray(source["masks"], dtype=bool))
    masks[:2] = np.asarray(source["masks"], dtype=bool)[:2]
    masks[active_index] = visible
    annotation_status = np.asarray(
        ["valid", "valid", "not_annotated", "not_annotated"]
    )
    annotation_status[active_index] = "valid"
    qc_status = np.asarray(["passed", "passed", "not_run", "not_run"])
    qc_status[active_index] = "passed"
    masks_payload = {
        "format_version": np.asarray(MASK_FORMAT_VERSION),
        "frame_count": np.asarray(frame_count, dtype=np.int64),
        "masks": masks,
        "instance_names": np.asarray(INSTANCE_NAMES),
        "roles": np.asarray(ROLES),
        "annotation_status": annotation_status,
        "qc_status": qc_status,
    }

    source_roles = [_json_clone(record) for record in validated_source.source_roles]
    source_algorithm = _json_clone(validated_source.source_algorithm)
    quality = record.get("quality")
    if not isinstance(quality, Mapping):
        raise UrdfGripperPublishError("backend episode record has no quality summary")
    nonempty_frames = int(visible.reshape(frame_count, -1).any(axis=1).sum())
    gripper_role = {
        "role": active_role,
        "status": "ok",
        "seed_frame_id": None,
        "primary_query": None,
        "output_window": list(active_window),
        "seed_rgb_path": None,
        "seed_mask_path": None,
        "canonical_envelope_path": None,
        "native_track_path": f"{active_role}/native_track.npz",
        "temporal_qc_path": None,
        "nonempty_frames": nonempty_frames,
        "failure": None,
        "qc_status": "passed",
        "qc_selected_candidate": None,
        "qc_reason": "URDF geometry/depth visibility quality gate passed",
    }
    backend_provenance = _backend_provenance(
        backend_episode_dir,
        record,
        artifact_identities,
    )
    backend_contract = backend_provenance["run_contract"]
    expected_backend_contract = {
        "dataset_root": validated_source.lineage["source_run"]["dataset_root"],
        "source_run_dir": str(validated_source.source_run_dir),
        "task": task,
        "camera": camera,
    }
    for key, expected in expected_backend_contract.items():
        if backend_contract.get(key) != expected:
            raise UrdfGripperPublishError(
                f"URDF backend {key} differs from the validated source contract"
            )
    publisher_identity = _validate_backend_publisher_identity(backend_provenance)
    source_mask_identity = _file_identity(
        source_episode_dir / "masks.npz", relative_path="masks.npz"
    )
    source_material_identities = _source_material_identities(source_material)
    gripper_stage = {
        "backend": "urdf",
        "producer": "robotwin_urdf_visual_geometry_depth_visibility",
        "seed": None,
        "propagation": None,
        "visibility": "URDF visual projection clipped by recorded scene depth",
        "active_arm": active_arm,
        "active_window": list(active_window),
        "qc_status": "passed",
        "qc_reason": gripper_role["qc_reason"],
        "quality": _json_clone(quality),
        "depth_tolerance_mm": product["depth_tolerance_mm"],
        "source_run_id": source_manifest.get("run_id"),
        "source_masks": source_mask_identity,
        "source_material": source_material_identities,
        "source_lineage_sha256": validated_source.lineage["lineage_sha256"],
        "backend_provenance": backend_provenance,
        "artifacts": {
            "native_track": f"{active_role}/native_track.npz",
            "product": f"{active_role}/urdf_product.npz",
            "diagnostics": f"{active_role}/urdf_diagnostics.json",
        },
    }
    source_algorithm["gripper_stage"] = gripper_stage
    source_algorithm["amodal_completion"] = False
    derivation = {
        "format_version": DERIVATION_FORMAT_VERSION,
        "source": _json_clone(validated_source.lineage),
        "publisher": publisher_identity,
    }
    run_manifest = {
        "format_version": MASK_RUN_FORMAT_VERSION,
        "gripper_backend": "urdf",
        "run_id": run_id,
        "episode": {
            "task": task,
            "episode_index": episode_index,
            "episode_id": f"{episode_index:06d}",
            "camera": camera,
        },
        "frame_count": frame_count,
        "roles": [*source_roles, gripper_role],
        "artifact_dir": str(destination_dir),
        "channels": {
            "target_0": 0,
            "receiver_0": 1,
            "gripper_left": 2 if active_arm == "left" else "not_annotated",
            "gripper_right": 3 if active_arm == "right" else "not_annotated",
        },
        "semantic_prompt_sha256": source_manifest.get("semantic_prompt_sha256"),
        "algorithm": source_algorithm,
        "roi_policy": None,
        "gripper_qc": {
            "backend": "urdf",
            "status": "ok",
            "qc_status": "passed",
            "active_arm": active_arm,
            "selected_candidate": None,
            "confidence": None,
            "reason": gripper_role["qc_reason"],
            "forced_fallback": False,
            "nonempty_frames": nonempty_frames,
            "quality": _json_clone(quality),
        },
        "artifacts": {
            "masks": "masks.npz",
            "frame_provenance": "frame_provenance.json",
            "gripper_native_track": f"{active_role}/native_track.npz",
            "urdf_product": f"{active_role}/urdf_product.npz",
            "urdf_diagnostics": f"{active_role}/urdf_diagnostics.json",
        },
        "derivation": derivation,
    }

    source_channels = source_provenance.get("channels")
    if not isinstance(source_channels, Mapping):
        raise UrdfGripperPublishError("source frame provenance has no channel map")
    if not all(isinstance(source_channels.get(key), Mapping) for key in ("target_0", "receiver_0")):
        raise UrdfGripperPublishError("source frame provenance lacks target/receiver")
    provenance_channels: dict[str, Any] = {
        "target_0": _json_clone(source_channels["target_0"]),
        "receiver_0": _json_clone(source_channels["receiver_0"]),
        "gripper_left": {"status": "not_annotated"},
        "gripper_right": {"status": "not_annotated"},
    }
    provenance_channels[active_role] = {
        "status": "ok",
        "backend": "urdf",
        "active_arm": active_arm,
        "active_window": list(active_window),
        "nonempty_frame_ids": [
            int(item)
            for item in np.flatnonzero(visible.reshape(frame_count, -1).any(axis=1))
        ],
        "qc_status": "passed",
        "qc_reason": gripper_role["qc_reason"],
        "quality": _json_clone(quality),
        "native_track_path": f"{active_role}/native_track.npz",
        "product_path": f"{active_role}/urdf_product.npz",
        "diagnostics_path": f"{active_role}/urdf_diagnostics.json",
        "source_masks_sha256": source_mask_identity["sha256"],
        "source_lineage_sha256": validated_source.lineage["lineage_sha256"],
        "backend_run_id": backend_provenance["run_id"],
    }
    frame_provenance = {
        "format_version": PROVENANCE_FORMAT_VERSION,
        "gripper_backend": "urdf",
        "derivation": _json_clone(derivation),
        "composition": (
            "target/receiver source native tracks clipped to role output windows; "
            "gripper from URDF visual geometry clipped by recorded scene depth"
        ),
        "channels": provenance_channels,
    }
    return masks_payload, run_manifest, frame_provenance


def _validate_canonical_path(
    destination_dir: Path,
    *,
    run_id: str,
    task: str,
    camera: str,
    episode_index: int,
) -> None:
    if not run_id or "/" in run_id or "\\" in run_id or ".." in run_id:
        raise ValueError("run_id must be a simple non-empty directory name")
    expected_parts = (
        destination_dir.name,
        destination_dir.parent.name,
        destination_dir.parent.parent.name,
        destination_dir.parent.parent.parent.name,
    )
    expected = (camera, f"episode_{episode_index:06d}", task, run_id)
    if expected_parts != expected:
        raise UrdfGripperPublishError(
            "destination must be <run_id>/<task>/episode_<id>/<camera>: "
            f"got {destination_dir}"
        )


def _prepare_contract(
    *,
    source_episode_dir: Path,
    backend_episode_dir: Path,
    destination_dir: Path,
    run_id: str,
    task: str,
    camera: str,
    backend_episode_record: Mapping[str, Any],
) -> dict[str, Any]:
    source_episode_dir = source_episode_dir.expanduser().resolve()
    backend_episode_dir = backend_episode_dir.expanduser().resolve()
    destination_dir = destination_dir.expanduser().resolve()
    if not source_episode_dir.is_dir():
        raise FileNotFoundError(f"source episode directory is missing: {source_episode_dir}")
    if not backend_episode_dir.is_dir():
        raise FileNotFoundError(f"backend episode directory is missing: {backend_episode_dir}")
    if backend_episode_record.get("status") != "complete":
        raise UrdfGripperPublishError("only a complete URDF backend episode may be published")
    raw_episode_index = backend_episode_record.get("episode_index")
    if isinstance(raw_episode_index, bool) or not isinstance(raw_episode_index, int):
        raise UrdfGripperPublishError("backend episode_index must be an integer")
    episode_index = int(raw_episode_index)
    if episode_index < 0:
        raise UrdfGripperPublishError("backend episode_index must be non-negative")
    _validate_canonical_path(
        destination_dir,
        run_id=run_id,
        task=task,
        camera=camera,
        episode_index=episode_index,
    )
    raw_frame_count = backend_episode_record.get("frame_count")
    if isinstance(raw_frame_count, bool) or not isinstance(raw_frame_count, int):
        raise UrdfGripperPublishError("backend frame_count must be an integer")
    validated_source = validate_derivation_source_episode(
        source_episode_dir,
        task=task,
        camera=camera,
        episode_index=episode_index,
        expected_frame_count=raw_frame_count,
    )
    source = validated_source.source_masks
    frame_count = validated_source.frame_count
    backend_lineage = backend_episode_record.get("source_lineage")
    if not isinstance(backend_lineage, Mapping):
        raise UrdfGripperPublishError(
            "backend episode record has no frozen source lineage"
        )
    if _json_clone(backend_lineage) != _json_clone(validated_source.lineage):
        raise UrdfGripperPublishError(
            "backend source lineage differs from the current source episode"
        )
    frame_shape = tuple(int(item) for item in np.asarray(source["masks"]).shape[2:])
    raw_shape = backend_episode_record.get("frame_shape_hw")
    if raw_shape is not None and tuple(raw_shape) != frame_shape:
        raise UrdfGripperPublishError("backend and source frame shapes differ")
    active_arm = backend_episode_record.get("active_arm")
    if active_arm not in {"left", "right"}:
        raise UrdfGripperPublishError("backend active_arm must be left or right")
    active_window = _parse_window(
        backend_episode_record.get("active_window"),
        frame_count=frame_count,
        label="backend active_window",
    )
    source_events = validated_source.loop.get("events")
    if not isinstance(source_events, Mapping):
        raise UrdfGripperPublishError("source loop has no event map")
    if active_arm != source_events.get("active_arm"):
        raise UrdfGripperPublishError(
            "backend active arm differs from the authoritative source loop"
        )
    expected_active_window = (
        source_events.get("t_move_start"),
        source_events.get("t_open_done"),
    )
    if active_window != expected_active_window:
        raise UrdfGripperPublishError(
            "backend active window differs from the authoritative source loop"
        )
    product_path, product_identity = _artifact_from_record(
        backend_episode_dir, backend_episode_record, "gripper_masks"
    )
    combined_path, combined_identity = _artifact_from_record(
        backend_episode_dir, backend_episode_record, "masks"
    )
    diagnostics_path, diagnostics_identity = _artifact_from_record(
        backend_episode_dir, backend_episode_record, "diagnostics"
    )
    product = _load_product(
        product_path,
        frame_count=frame_count,
        frame_shape=frame_shape,
        active_arm=str(active_arm),
        active_window=active_window,
    )
    _validate_backend_combined_masks(
        combined_path,
        source_masks=np.asarray(source["masks"]),
        gripper_track=np.asarray(product["tracks"]["gripper_track"]),
        active_arm=str(active_arm),
    )
    diagnostics = _read_json_object(
        diagnostics_path,
        description="URDF backend episode diagnostics",
    )
    if diagnostics.get("status") != "complete":
        raise UrdfGripperPublishError("URDF backend diagnostics are not complete")
    expected_diagnostic_fields = {
        "episode_index": episode_index,
        "frame_count": frame_count,
        "active_arm": active_arm,
        "active_window": list(active_window),
        "quality": backend_episode_record.get("quality"),
    }
    for key, expected in expected_diagnostic_fields.items():
        if diagnostics.get(key) != expected:
            raise UrdfGripperPublishError(
                f"URDF diagnostics {key} differs from the backend record"
            )
    raw_inputs = backend_episode_record.get("inputs")
    source_identity = (
        raw_inputs.get("source_masks") if isinstance(raw_inputs, Mapping) else None
    )
    if not isinstance(source_identity, Mapping):
        raise UrdfGripperPublishError(
            "backend episode record has no source_masks identity"
        )
    _validate_identity(
        source_episode_dir / "masks.npz",
        source_identity,
        label="backend source_masks",
    )
    artifact_identities = {
        "gripper_masks": product_identity,
        "masks": combined_identity,
        "diagnostics": diagnostics_identity,
    }
    payloads = _build_public_payloads(
        source_episode_dir=source_episode_dir,
        backend_episode_dir=backend_episode_dir,
        destination_dir=destination_dir,
        run_id=run_id,
        task=task,
        camera=camera,
        record=backend_episode_record,
        validated_source=validated_source,
        product=product,
        artifact_identities=artifact_identities,
    )
    return {
        "source_episode_dir": source_episode_dir,
        "backend_episode_dir": backend_episode_dir,
        "destination_dir": destination_dir,
        "episode_index": episode_index,
        "active_arm": str(active_arm),
        "active_window": active_window,
        "frame_count": frame_count,
        "frame_shape": frame_shape,
        "source": source,
        "validated_source": validated_source,
        "source_lineage": validated_source.lineage,
        "source_material": validated_source.source_material,
        "product": product,
        "product_path": product_path,
        "diagnostics_path": diagnostics_path,
        "masks_payload": payloads[0],
        "run_manifest": payloads[1],
        "frame_provenance": payloads[2],
    }


def _write_stage(stage_dir: Path, contract: Mapping[str, Any]) -> None:
    _materialize_source_files(contract["source_material"], stage_dir)
    active_role = f"gripper_{contract['active_arm']}"
    gripper_dir = stage_dir / active_role
    gripper_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        gripper_dir / "native_track.npz",
        masks=np.asarray(contract["product"]["tracks"]["gripper_track"], dtype=bool),
    )
    shutil.copy2(contract["product_path"], gripper_dir / "urdf_product.npz")
    shutil.copy2(contract["diagnostics_path"], gripper_dir / "urdf_diagnostics.json")
    np.savez_compressed(stage_dir / "masks.npz", **contract["masks_payload"])
    _write_json(stage_dir / "run_manifest.json", contract["run_manifest"])
    _write_json(stage_dir / "frame_provenance.json", contract["frame_provenance"])


def _all_regular_files(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise UrdfGripperPublishError(f"published episode contains a symlink: {path}")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = path
        elif not path.is_dir():
            raise UrdfGripperPublishError(f"published episode has unsupported entry: {path}")
    return result


def _validate_public_masks(path: Path, contract: Mapping[str, Any]) -> None:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(contract["masks_payload"]):
                raise UrdfGripperPublishError(
                    "public masks.npz must contain exactly the canonical seven keys"
                )
            for key, expected in contract["masks_payload"].items():
                actual = np.asarray(archive[key])
                if not np.array_equal(actual, expected):
                    raise UrdfGripperPublishError(f"public masks payload differs for {key}")
    except (EOFError, OSError, ValueError, zipfile.BadZipFile) as exc:
        if isinstance(exc, UrdfGripperPublishError):
            raise
        raise UrdfGripperPublishError(f"cannot validate public masks: {path}: {exc}") from exc


def _validate_tree(root: Path, contract: Mapping[str, Any]) -> None:
    if not root.is_dir() or root.is_symlink():
        raise UrdfGripperPublishError(f"published episode directory is missing: {root}")
    active_role = f"gripper_{contract['active_arm']}"
    generated = {
        *GENERATED_FILENAMES,
        f"{active_role}/native_track.npz",
        f"{active_role}/urdf_product.npz",
        f"{active_role}/urdf_diagnostics.json",
    }
    expected_files = set(contract["source_material"]) | generated
    actual_files = _all_regular_files(root)
    if set(actual_files) != expected_files:
        missing = sorted(expected_files - set(actual_files))
        extra = sorted(set(actual_files) - expected_files)
        raise UrdfGripperPublishError(
            f"published episode file set differs: missing={missing}, extra={extra}"
        )
    for relative, source in contract["source_material"].items():
        destination = actual_files[relative]
        if source.stat().st_size != destination.stat().st_size:
            raise UrdfGripperPublishError(f"materialized source size differs: {relative}")
        if _sha256(source) != _sha256(destination):
            raise UrdfGripperPublishError(f"materialized source hash differs: {relative}")
    _validate_public_masks(root / "masks.npz", contract)
    active_dir = root / active_role
    native_path = active_dir / "native_track.npz"
    try:
        with np.load(native_path, allow_pickle=False) as archive:
            if archive.files != ["masks"]:
                raise UrdfGripperPublishError("native_track.npz must contain only masks")
            native = np.asarray(archive["masks"])
    except (EOFError, OSError, ValueError, zipfile.BadZipFile) as exc:
        if isinstance(exc, UrdfGripperPublishError):
            raise
        raise UrdfGripperPublishError(f"cannot validate native track: {exc}") from exc
    expected_track = np.asarray(contract["product"]["tracks"]["gripper_track"])
    if native.dtype != np.bool_ or not np.array_equal(native, expected_track):
        raise UrdfGripperPublishError("native track differs from the URDF visible product")
    if _sha256(active_dir / "urdf_product.npz") != _sha256(contract["product_path"]):
        raise UrdfGripperPublishError("published URDF product differs from backend product")
    if _sha256(active_dir / "urdf_diagnostics.json") != _sha256(
        contract["diagnostics_path"]
    ):
        raise UrdfGripperPublishError(
            "published URDF diagnostics differ from backend diagnostics"
        )
    run_manifest = _read_json_object(
        root / "run_manifest.json", description="published run manifest"
    )
    if run_manifest != contract["run_manifest"]:
        raise UrdfGripperPublishError("published run manifest differs from its contract")
    frame_provenance = _read_json_object(
        root / "frame_provenance.json",
        description="published frame provenance",
    )
    if frame_provenance != contract["frame_provenance"]:
        raise UrdfGripperPublishError("published frame provenance differs from its contract")


def _process_record(contract: Mapping[str, Any], *, status: str) -> dict[str, Any]:
    destination = Path(contract["destination_dir"])
    return {
        "episode": int(contract["episode_index"]),
        "status": status,
        "artifact": str(destination / "run_manifest.json"),
        "gripper_backend": "urdf",
        "active_arm": str(contract["active_arm"]),
        "active_window": list(contract["active_window"]),
        "quality": _json_clone(contract["run_manifest"]["gripper_qc"]["quality"]),
        "source_lineage_sha256": contract["source_lineage"]["lineage_sha256"],
        "source_lineage": _json_clone(contract["source_lineage"]),
    }


def validate_published_urdf_episode(
    source_episode_dir: Path,
    backend_episode_dir: Path,
    destination_dir: Path,
    *,
    run_id: str,
    task: str,
    camera: str,
    backend_episode_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Fully validate one canonical URDF-backed episode and return its process record."""

    contract = _prepare_contract(
        source_episode_dir=source_episode_dir,
        backend_episode_dir=backend_episode_dir,
        destination_dir=destination_dir,
        run_id=run_id,
        task=task,
        camera=camera,
        backend_episode_record=backend_episode_record,
    )
    _validate_tree(contract["destination_dir"], contract)
    return _process_record(contract, status="completed")


def publish_urdf_episode(
    source_episode_dir: Path,
    backend_episode_dir: Path,
    destination_dir: Path,
    *,
    run_id: str,
    task: str,
    camera: str,
    backend_episode_record: Mapping[str, Any],
    resume: bool = False,
) -> dict[str, Any]:
    """Atomically publish or validated-skip one canonical URDF-backed episode."""

    contract = _prepare_contract(
        source_episode_dir=source_episode_dir,
        backend_episode_dir=backend_episode_dir,
        destination_dir=destination_dir,
        run_id=run_id,
        task=task,
        camera=camera,
        backend_episode_record=backend_episode_record,
    )
    destination = Path(contract["destination_dir"])
    if destination.exists():
        if not resume:
            raise FileExistsError(f"destination episode already exists: {destination}")
        _validate_tree(destination, contract)
        return _process_record(contract, status="skipped_complete")

    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.{uuid.uuid4().hex}.",
            suffix=".tmp",
            dir=destination.parent,
        )
    )
    try:
        _write_stage(stage, contract)
        _validate_tree(stage, contract)
        if destination.exists():
            raise FileExistsError(
                f"destination appeared while the episode was staged: {destination}"
            )
        stage.rename(destination)
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    _validate_tree(destination, contract)
    return _process_record(contract, status="completed")


__all__ = [
    "DerivationSourceEpisode",
    "UrdfGripperPublishError",
    "publish_urdf_episode",
    "publisher_implementation_identity",
    "validate_derivation_source_episode",
    "validate_published_urdf_episode",
]
