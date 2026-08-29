"""Resolve task datasets and collections from extract manifests."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robotwin_annotation_v2.config import validate_dataset_component
from robotwin_annotation_v2.domain import (
    AnnotationMode,
    TargetOnlyTaskKind,
    TargetProfile,
    target_profile_for_task_kind,
)

from .discovery import discover_episodes


@dataclass(frozen=True)
class DatasetTarget:
    root: Path
    task: str
    camera: str
    episode_ids: tuple[int, ...]
    task_kind: TargetOnlyTaskKind | None = None
    # The extract manifest is optional for native RoboTwin directories.  Keep
    # it in memory when present so the profile binder never has to create a
    # per-task config/JSON just to carry provenance downstream.
    manifest_path: Path | None = None
    manifest_data: dict[str, Any] | None = None

    @property
    def profile(self) -> TargetProfile:
        """Semantic profile derived solely from manifest task kind."""

        return target_profile_for_task_kind(self.task_kind)


@dataclass(frozen=True)
class DatasetInput:
    root: Path
    targets: tuple[DatasetTarget, ...]
    is_collection: bool


def _read_manifest(root: Path) -> dict[str, Any]:
    path = root / "EXTRACT_MANIFEST.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"dataset extract manifest is missing: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read dataset extract manifest: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"dataset extract manifest must contain one object: {path}")
    return payload


def _read_optional_manifest(root: Path) -> tuple[Path | None, dict[str, Any] | None]:
    """Read an extract manifest when one exists, without requiring it."""

    path = root / "EXTRACT_MANIFEST.json"
    if not path.is_file():
        return None, None
    return path, _read_manifest(root)


def _camera_from_layout(root: Path, requested: str | None) -> str:
    if requested is not None:
        return validate_dataset_component(requested, field="dataset camera")
    candidates = sorted(
        {
            path.name.removeprefix("observation.images.")
            for path in (root / "videos").glob("chunk-*/observation.images.*")
            if path.is_dir() and path.name.startswith("observation.images.")
        }
    )
    if not candidates:
        raise ValueError(f"cannot infer camera: no observation.images.* under {root / 'videos'}")
    if len(candidates) > 1:
        raise ValueError(
            f"camera is ambiguous under {root / 'videos'}; pass --camera "
            f"(found: {', '.join(candidates)})"
        )
    return validate_dataset_component(candidates[0], field="dataset camera")


def _native_task_children(root: Path) -> tuple[Path, ...]:
    """Return immediate children that look like native task directories.

    A native RoboTwin task is identified by its ``data/`` directory.  Keeping
    this check deliberately shallow is important: collection inputs should be
    resolved from their declared task list (when a collection manifest exists)
    and must not accidentally walk unrelated descendants.
    """

    children: list[Path] = []
    for child in sorted(root.iterdir()):
        # A symlink can make a seemingly safe one-component task name resolve
        # outside the collection root.  Do not follow it implicitly; callers
        # should provide a real task directory (or bind that directory
        # explicitly as the top-level input).
        if child.is_symlink():
            if child.is_dir() and (child / "data").is_dir():
                raise ValueError(f"native task directory must not be a symlink: {child}")
            continue
        if child.is_dir() and (child / "data").is_dir():
            children.append(child)
    return tuple(children)


def _task_child_path(collection_root: Path, task: str) -> Path:
    """Resolve one manifest task below a collection without following links."""

    candidate = collection_root / task
    if candidate.is_symlink():
        raise ValueError(f"collection task directory must not be a symlink: {candidate}")
    root_resolved = collection_root.resolve()
    candidate_resolved = candidate.resolve(strict=False)
    if candidate_resolved.parent != root_resolved:
        raise ValueError(f"collection task path escapes collection root: {task!r}")
    return candidate


def _reject_raw_mcap_input(root: Path) -> None:
    """Raise an actionable error for an unconverted real-data directory."""

    # The real-MCAP converter consumes a directory of top-level ``*.mcap``
    # files.  Do not attempt to infer a RoboTwin task from that layout (or
    # silently create a manifest); conversion is an explicit, reproducible
    # preprocessing step.
    if any(path.is_file() and path.suffix.lower() == ".mcap" for path in root.iterdir()):
        raise ValueError(
            "input looks like raw MCAP; run just convert-real INPUT_ROOT OUTPUT_ROOT first"
        )


def _native_task_target(
    root: Path,
    *,
    mode: AnnotationMode,
    task: str | None,
    camera: str | None,
    manifest_override: Mapping[str, Any] | None = None,
) -> DatasetTarget:
    """Resolve a native RoboTwin task without a local extract manifest.

    A collection manifest can carry the metadata for a task even when the
    task directory itself has no ``EXTRACT_MANIFEST.json``.  In that case the
    record is passed as ``manifest_override`` and acts as an in-memory
    manifest.  This keeps the root collection manifest authoritative without
    writing a new task-specific file or rediscovering a different episode
    set.
    """

    if manifest_override is not None and not isinstance(manifest_override, Mapping):
        raise TypeError("manifest_override must be a mapping")

    # Validate the identity declared by a collection record before falling
    # back to directory names/layout discovery.  ``dataset_root`` is only
    # provenance and is intentionally not checked: extracts are often moved
    # or mounted at a different path after they are generated.
    declared_task = (
        manifest_override.get("task") if manifest_override is not None else None
    )
    if declared_task is not None:
        try:
            declared_task = validate_dataset_component(
                declared_task,
                field="collection task record task",
            )
        except ValueError as exc:
            raise ValueError(f"collection task record has an invalid task: {root}") from exc
    if task is not None and declared_task is not None and task != declared_task:
        raise ValueError(
            f"requested task {task!r} does not match collection task {declared_task!r}"
        )

    declared_camera = (
        manifest_override.get("camera") if manifest_override is not None else None
    )
    if declared_camera is not None:
        try:
            declared_camera = validate_dataset_component(
                declared_camera,
                field="collection task record camera",
            )
        except ValueError as exc:
            raise ValueError(f"collection task record has an invalid camera: {root}") from exc
    if camera is not None and declared_camera is not None and camera != declared_camera:
        raise ValueError(
            f"requested camera {camera!r} does not match collection camera "
            f"{declared_camera!r}"
        )

    declared_profile = (
        manifest_override.get("profile") if manifest_override is not None else None
    )
    if declared_profile is not None and declared_profile != mode.value:
        raise ValueError(
            f"dataset profile {declared_profile!r} does not match "
            f"--{mode.value.replace('_', '-')}"
        )

    for directory in ("data", "videos", "sidecars", "meta"):
        if not (root / directory).is_dir():
            raise FileNotFoundError(
                f"dataset manifest is missing and native task is missing {directory}/: {root}"
            )
    if task is not None:
        # An explicit CLI/API value should retain the validator's actionable
        # diagnostic (for example, ``task=""`` is a non-empty-name error),
        # rather than being mislabeled as a failed path inference.
        resolved_task = validate_dataset_component(task, field="dataset task")
    else:
        try:
            resolved_task = validate_dataset_component(
                declared_task or root.name,
                field="dataset task",
            )
        except ValueError as exc:
            raise ValueError(f"cannot infer task name from dataset path: {root}") from exc

    resolved_camera = _camera_from_layout(
        root,
        camera if camera is not None else declared_camera,
    )

    # Collection records use the same aliases as task manifests.  Prefer the
    # canonical ``episode_indices`` spelling, then the runtime regression and
    # compatibility aliases.  If no selection is declared, discover all
    # complete episodes from the native layout as before.
    declared_episode_ids: Any = None
    if manifest_override is not None:
        for key in ("episode_indices", "regression_episode_ids", "episode_ids"):
            if manifest_override.get(key) is not None:
                declared_episode_ids = manifest_override.get(key)
                break
    discovery = None
    if declared_episode_ids is None:
        discovery = discover_episodes(root, camera=resolved_camera, require_depth=False)
        if not discovery.episodes:
            raise ValueError(f"no complete episodes found under {root}")
        episode_ids = discovery.episode_ids
    else:
        if not isinstance(declared_episode_ids, list) or not declared_episode_ids:
            raise ValueError(f"collection task record has no episode_indices: {root}")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in declared_episode_ids
        ):
            raise ValueError(
                f"collection task record episode_indices must be integers: {root}"
            )
        if any(value < 0 for value in declared_episode_ids):
            raise ValueError(
                f"collection task record episode_indices must be non-negative: {root}"
            )
        episode_ids = tuple(dict.fromkeys(declared_episode_ids))
        if len(episode_ids) != len(declared_episode_ids):
            raise ValueError(
                f"collection task record contains duplicate episode_indices: {root}"
            )

    task_kind = (
        _parse_task_kind(dict(manifest_override), root=root)
        if manifest_override is not None
        else None
    )
    target_profile = target_profile_for_task_kind(task_kind)
    if (
        target_profile is TargetProfile.CONTACT_PRESS
        and mode is not AnnotationMode.TARGET_ONLY
    ):
        raise ValueError("contact_press task_kind requires target_only dataset profile")

    # Keep all record provenance in memory.  Identity and selection fields are
    # normalized for downstream bind_dataset(), while an existing declared
    # dataset_root (even if stale) remains untouched.
    manifest_data: dict[str, Any]
    if manifest_override is None:
        manifest_data = {}
    else:
        manifest_data = deepcopy(dict(manifest_override))
    manifest_data.setdefault(
        "format_version",
        (
            "robotwin_dataset_manifest_discovered_v1"
            if manifest_override is None
            else "robotwin_dataset_manifest_bound_v1"
        ),
    )
    # Historical collection records sometimes omit ``profile`` (or encode it
    # as null).  The selected CLI mode is the only safe default in that case;
    # never pass a null profile into the downstream binding contract.
    if manifest_data.get("profile") is None:
        manifest_data["profile"] = mode.value
    manifest_data["task"] = resolved_task
    manifest_data["camera"] = resolved_camera
    manifest_data.setdefault("dataset_root", str(root))
    manifest_data["episode_indices"] = list(episode_ids)
    manifest_data["regression_episode_ids"] = list(episode_ids)
    manifest_data.setdefault("smoke_episode_ids", [episode_ids[0]])
    if discovery is not None:
        manifest_data["discovery_skipped"] = list(discovery.skipped)
    if task_kind is not None:
        manifest_data["task_kind"] = task_kind.value

    return DatasetTarget(
        root=root,
        task=resolved_task,
        camera=resolved_camera,
        episode_ids=episode_ids,
        task_kind=task_kind,
        manifest_data=manifest_data,
    )


def _parse_task_kind(
    manifest: dict[str, Any],
    *,
    root: Path,
) -> TargetOnlyTaskKind | None:
    raw_task_kind = manifest.get("task_kind")
    if raw_task_kind is None:
        return None
    try:
        return TargetOnlyTaskKind(raw_task_kind)
    except (TypeError, ValueError) as exc:
        choices = ", ".join(item.value for item in TargetOnlyTaskKind)
        raise ValueError(
            f"dataset manifest has unsupported task_kind {raw_task_kind!r}: "
            f"{root}; choose {choices}"
        ) from exc


def _is_legacy_single_task_manifest(manifest: Mapping[str, Any]) -> bool:
    """Recognize the narrow pre-profile extract-manifest shape.

    Older coverage extracts describe only ``episode_indices`` and ``cameras``.
    Keep compatibility deliberately constrained to that single-task shape so
    incomplete modern manifests still fail closed instead of being guessed.
    """

    return (
        "datasets" not in manifest
        and manifest.get("profile") is None
        and manifest.get("task") is None
        and manifest.get("camera") is None
        and isinstance(manifest.get("episode_indices"), list)
        and isinstance(manifest.get("cameras"), list)
    )


def _infer_legacy_task_name(root: Path, task: str | None) -> str:
    """Infer a task from old metadata, with a deterministic directory fallback."""

    if task is not None:
        return validate_dataset_component(task, field="dataset task")

    candidates: set[str] = set()
    metadata_path = root / "meta" / "episodes.jsonl"
    if metadata_path.is_file():
        try:
            with metadata_path.open(encoding="utf-8") as handle:
                for line in handle:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(record, Mapping):
                        continue
                    structured = record.get("full_structured_tasks")
                    if isinstance(structured, list) and structured:
                        value = structured[0]
                        if isinstance(value, str) and value.strip():
                            candidates.add(value.strip())
                    for key in ("task", "task_name", "dataset_task"):
                        value = record.get(key)
                        if isinstance(value, str) and value.strip():
                            candidates.add(value.strip())
        except OSError:
            pass
    if len(candidates) == 1:
        return validate_dataset_component(next(iter(candidates)), field="dataset task")
    # The suffix was used by the original coverage extraction recipe; retain
    # it as a fallback only when metadata cannot identify one task.
    for suffix in ("_coverage20_original", "_coverage20"):
        if root.name.endswith(suffix):
            candidate = root.name[: -len(suffix)]
            if candidate:
                return validate_dataset_component(candidate, field="dataset task")
    return validate_dataset_component(root.name, field="dataset task")


def _infer_legacy_camera(
    root: Path,
    manifest: Mapping[str, Any],
    camera: str | None,
) -> str:
    """Select the historical default camera while honoring explicit overrides."""

    if camera is not None:
        return validate_dataset_component(camera, field="dataset camera")
    raw_cameras = manifest.get("cameras")
    if isinstance(raw_cameras, list):
        cameras = tuple(
            validate_dataset_component(value, field="dataset manifest camera")
            for value in raw_cameras
        )
        if "cam_high" in cameras:
            return "cam_high"
        if len(cameras) == 1:
            return cameras[0]
    return _camera_from_layout(root, None)


def _legacy_task_target(
    root: Path,
    *,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    mode: AnnotationMode,
    task: str | None,
    camera: str | None,
) -> DatasetTarget:
    """Adapt one historical extract manifest without writing a replacement."""

    task_name = _infer_legacy_task_name(root, task)
    camera_name = _infer_legacy_camera(root, manifest, camera)
    raw_episode_ids = manifest.get("episode_indices")
    if not isinstance(raw_episode_ids, list) or not raw_episode_ids:
        raise ValueError(f"legacy dataset manifest has no episode_indices: {root}")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in raw_episode_ids):
        raise ValueError(f"legacy dataset manifest episode_indices must be integers: {root}")
    if any(value < 0 for value in raw_episode_ids):
        raise ValueError(
            f"legacy dataset manifest episode_indices must be non-negative: {root}"
        )
    episode_ids = tuple(dict.fromkeys(raw_episode_ids))
    if len(episode_ids) != len(raw_episode_ids):
        raise ValueError(f"legacy dataset manifest contains duplicate episode_indices: {root}")
    for directory in ("data", "videos", "sidecars", "meta"):
        if not (root / directory).is_dir():
            raise ValueError(f"task dataset is missing {directory}/: {root}")

    manifest_data = deepcopy(dict(manifest))
    manifest_data.update(
        {
            "format_version": "robotwin_dataset_manifest_legacy_bound_v1",
            "profile": mode.value,
            "task": task_name,
            "camera": camera_name,
            "dataset_root": str(root),
            "episode_indices": list(episode_ids),
            "regression_episode_ids": list(episode_ids),
            "smoke_episode_ids": [episode_ids[0]],
        }
    )
    return DatasetTarget(
        root=root,
        task=task_name,
        camera=camera_name,
        episode_ids=episode_ids,
        manifest_path=manifest_path,
        manifest_data=manifest_data,
    )


def read_dataset_task_kind(root: Path) -> TargetOnlyTaskKind | None:
    """Read optional task-kind provenance from one task extract manifest."""

    resolved = root.expanduser().resolve()
    if not (resolved / "EXTRACT_MANIFEST.json").is_file():
        return None
    return _parse_task_kind(_read_manifest(resolved), root=resolved)


def _task_target(
    root: Path,
    mode: AnnotationMode,
    *,
    task_override: str | None = None,
    camera_override: str | None = None,
    manifest_override: Mapping[str, Any] | None = None,
) -> DatasetTarget:
    manifest_path, manifest = _read_optional_manifest(root)
    if manifest is None:
        return _native_task_target(
            root,
            mode=mode,
            task=task_override,
            camera=camera_override,
            manifest_override=manifest_override,
        )
    if manifest_override is not None:
        # A task-local manifest is more specific than the collection record.
        # Keep the local file authoritative when both are present.
        manifest_override = None
    if _is_legacy_single_task_manifest(manifest):
        return _legacy_task_target(
            root,
            manifest_path=manifest_path or (root / "EXTRACT_MANIFEST.json"),
            manifest=manifest,
            mode=mode,
            task=task_override,
            camera=camera_override,
        )
    declared_profile = manifest.get("profile")
    if declared_profile != mode.value:
        raise ValueError(
            f"dataset profile {declared_profile!r} does not match "
            f"--{mode.value.replace('_', '-')}"
        )
    task = manifest.get("task")
    camera = manifest.get("camera")
    episode_ids = manifest.get("episode_indices")
    if episode_ids is None:
        episode_ids = manifest.get("regression_episode_ids")
    if episode_ids is None:
        episode_ids = manifest.get("episode_ids")
    if not isinstance(task, str) or not task:
        raise ValueError(f"dataset manifest does not define a task: {root}")
    if not isinstance(camera, str) or not camera:
        raise ValueError(f"dataset manifest does not define a camera: {root}")
    try:
        task = validate_dataset_component(task, field="dataset manifest task")
        camera = validate_dataset_component(camera, field="dataset manifest camera")
    except ValueError as exc:
        raise ValueError(f"dataset manifest has invalid task/camera: {root}") from exc
    if camera_override is not None and camera != camera_override:
        raise ValueError(
            f"requested camera {camera_override!r} does not match dataset camera {camera!r}"
        )
    if task_override is not None and task_override != task:
        raise ValueError(
            f"requested task {task_override!r} does not match dataset task {task!r}"
        )
    if not isinstance(episode_ids, list) or not episode_ids:
        raise ValueError(f"dataset manifest has no episode_indices: {root}")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in episode_ids):
        raise ValueError(f"dataset manifest episode_indices must be integers: {root}")
    if any(value < 0 for value in episode_ids):
        raise ValueError(
            f"dataset manifest episode_indices must be non-negative: {root}"
        )
    unique_ids = tuple(dict.fromkeys(episode_ids))
    if len(unique_ids) != len(episode_ids):
        raise ValueError(f"dataset manifest contains duplicate episode_indices: {root}")
    task_kind = _parse_task_kind(manifest, root=root)
    target_profile = target_profile_for_task_kind(task_kind)
    if (
        target_profile is TargetProfile.CONTACT_PRESS
        and mode is not AnnotationMode.TARGET_ONLY
    ):
        raise ValueError("contact_press task_kind requires target_only dataset profile")
    for directory in ("data", "videos", "sidecars", "meta"):
        if not (root / directory).is_dir():
            raise ValueError(f"task dataset is missing {directory}/: {root}")
    return DatasetTarget(
        root,
        task,
        camera,
        unique_ids,
        task_kind=task_kind,
        manifest_path=manifest_path,
        manifest_data=dict(manifest),
    )


def resolve_dataset_input(
    path: Path,
    *,
    mode: AnnotationMode,
    task: str | None = None,
    camera: str | None = None,
) -> DatasetInput:
    """Resolve and validate one task dataset or a task collection."""

    root = path.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"dataset path is not a directory: {root}")
    # Normalize CLI/API overrides once at the resolver boundary.  Besides
    # keeping comparisons consistent (e.g. ``" cam_high "``), this prevents
    # path-like task/camera values from reaching any child lookup.
    if task is None:
        requested_task = None
    elif isinstance(task, str) and not task.strip():
        # Preserve an explicitly empty task for _native_task_target(), which
        # emits the resolver-specific "cannot infer task name" diagnostic.
        requested_task = task
    else:
        requested_task = validate_dataset_component(task, field="dataset task override")
    if camera is None:
        requested_camera = None
    elif isinstance(camera, str) and not camera.strip():
        # Keep the explicit empty value long enough for task identity
        # validation to produce the historical task diagnostic first.
        requested_camera = camera
    else:
        requested_camera = validate_dataset_component(camera, field="dataset camera override")
    _manifest_path, manifest = _read_optional_manifest(root)
    if manifest is None:
        # A directory containing the native data layout is one task.  Otherwise
        # treat immediate task-shaped children as a collection; no manifest is
        # written or synthesized on disk.
        if (root / "data").is_dir():
            task_children = _native_task_children(root)
            if task_children:
                child_names = ", ".join(child.name for child in task_children)
                raise ValueError(
                    "dataset path is ambiguous: it contains both direct native "
                    f"data/ and task subdirectories ({child_names}); pass a task directory "
                    "or a collection root with EXTRACT_MANIFEST.json"
                )
            target = _native_task_target(
                root,
                mode=mode,
                task=requested_task,
                camera=requested_camera,
            )
            return DatasetInput(root, (target,), False)
        children = _native_task_children(root)
        if not children:
            _reject_raw_mcap_input(root)
            raise FileNotFoundError(
                f"dataset manifest is missing and no native task directories were found: {root}"
            )
        selected = (
            children
            if requested_task is None
            else tuple(child for child in children if child.name == requested_task)
        )
        if requested_task is not None and not selected:
            raise ValueError(f"requested collection task is absent: {requested_task}")
        targets = tuple(
            _task_target(
                child,
                mode,
                task_override=(
                    requested_task if len(selected) == 1 and requested_task is not None else None
                ),
                camera_override=requested_camera,
            )
            for child in selected
        )
        return DatasetInput(root, targets, True)
    datasets = manifest.get("datasets")
    if datasets is None:
        target = _task_target(
            root,
            mode,
            task_override=requested_task,
            camera_override=requested_camera,
        )
        return DatasetInput(root, (target,), False)

    collection_profile = manifest.get("profile")
    if collection_profile is not None and collection_profile != mode.value:
        raise ValueError(
            f"dataset profile {collection_profile!r} does not match "
            f"--{mode.value.replace('_', '-')}"
        )
    if not isinstance(datasets, list) or not datasets:
        raise ValueError(f"collection manifest has no datasets: {root}")
    names: list[str] = []
    records_by_name: dict[str, Mapping[str, Any]] = {}
    for record in datasets:
        name = record.get("task") if isinstance(record, Mapping) else None
        if not isinstance(name, str):
            raise ValueError(  # noqa: TRY004 - preserve resolver's ValueError contract
                f"collection manifest contains an invalid task record: {root}"
            )
        try:
            name = validate_dataset_component(name, field="collection task")
        except ValueError as exc:
            raise ValueError(
                f"collection manifest contains an invalid task record: {root}"
            ) from exc
        names.append(name)
        # Keep the complete record, not just its task name.  Some historical
        # collections omit per-task EXTRACT_MANIFEST.json files and rely on
        # these records for episode selection, task kind, and provenance.
        records_by_name[name] = record
    if len(set(names)) != len(names):
        raise ValueError(f"collection manifest contains duplicate tasks: {root}")
    selected_names: list[str] = names if requested_task is None else [requested_task]
    unknown = sorted(set(selected_names) - set(names))
    if unknown:
        raise ValueError(f"requested collection task is absent: {unknown}")

    # Some older mixed collections (for example profile-compat extracts) do
    # not have a top-level profile.  Infer it from the records when the
    # selected set is homogeneous.  A mixed, unfiltered request is ambiguous
    # and must name one task explicitly; otherwise a target-only run could
    # accidentally process pick-and-place data (or vice versa).
    if collection_profile is None:
        selected_profile_values = [
            records_by_name[name].get("profile") for name in selected_names
        ]
        selected_profiles = {
            record_profile
            for record_profile in selected_profile_values
            if record_profile is not None
        }
        if any(value is None for value in selected_profile_values):
            raise ValueError(
                "collection manifest has no top-level profile and task records "
                "do not declare one consistently; add a top-level profile or "
                "select a task with an explicit profile"
            )
        if requested_task is None and len(selected_profiles) > 1:
            choices = ", ".join(sorted(str(value) for value in selected_profiles))
            raise ValueError(
                "collection manifest has mixed profiles "
                f"({choices}); pass --task to select one task"
            )
        if selected_profiles and any(
            record_profile != mode.value for record_profile in selected_profiles
        ):
            declared = ", ".join(sorted(str(value) for value in selected_profiles))
            raise ValueError(
                f"dataset profile {declared!r} does not match "
                f"--{mode.value.replace('_', '-')}"
            )
    targets = tuple(
        _task_target(
            _task_child_path(root, name),
            mode,
            task_override=name,
            camera_override=requested_camera,
            manifest_override=records_by_name[name],
        )
        for name in sorted(selected_names)
    )
    return DatasetInput(root, targets, True)


__all__ = [
    "DatasetInput",
    "DatasetTarget",
    "read_dataset_task_kind",
    "resolve_dataset_input",
]
