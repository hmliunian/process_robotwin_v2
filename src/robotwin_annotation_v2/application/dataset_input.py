"""Resolve task datasets and collections from extract manifests."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
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
from .provenance import (
    prompt_bundle_from_manifest,
    validate_profile_provenance_pair,
)

_EPISODE_SELECTION_ALIASES = (
    "episode_indices",
    "regression_episode_ids",
    "episode_ids",
)


def _parse_episode_ids(value: Any, *, field: str) -> tuple[int, ...]:
    """Validate one manifest episode selection and preserve its order."""

    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise ValueError(f"{field} must be a list of integers")
    if any(item < 0 for item in value):
        raise ValueError(f"{field} must be non-negative")
    normalized = tuple(value)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field} contains duplicate episode ids")
    return normalized


def _resolve_episode_selection(
    manifest: Mapping[str, Any] | None,
    *,
    context: str,
) -> tuple[tuple[int, ...] | None, tuple[int, ...] | None]:
    """Resolve selection aliases and validate the smoke subset contract.

    Extracts in the wild use three names for the same regression selection.
    Accept one spelling for compatibility, but never let two declarations
    silently disagree.  ``None`` means that a field was not declared; an
    explicitly empty list remains a declaration and is rejected by callers as
    an unusable regression selection.
    """

    declared: list[tuple[str, tuple[int, ...]]] = []
    if manifest is not None:
        for key in _EPISODE_SELECTION_ALIASES:
            value = manifest.get(key)
            if value is None:
                continue
            declared.append(
                (key, _parse_episode_ids(value, field=f"{context} {key}"))
            )

    regression: tuple[int, ...] | None = None
    if declared:
        first_key, regression = declared[0]
        for key, candidate in declared[1:]:
            if candidate != regression:
                raise ValueError(
                    f"{context} {key} differs from {first_key}; "
                    "episode selection aliases must match exactly"
                )

    smoke: tuple[int, ...] | None = None
    if manifest is not None and manifest.get("smoke_episode_ids") is not None:
        smoke = _parse_episode_ids(
            manifest["smoke_episode_ids"],
            field=f"{context} smoke_episode_ids",
        )
        if not smoke:
            raise ValueError(f"{context} smoke_episode_ids must be non-empty")
        if regression is not None and not set(smoke).issubset(regression):
            raise ValueError(
                f"{context} smoke_episode_ids must be included in the regression selection"
            )
    return regression, smoke


def _validate_smoke_selection(
    smoke: tuple[int, ...] | None,
    regression: Sequence[int],
    *,
    context: str,
) -> None:
    """Validate a smoke selection after discovery supplies regression IDs."""

    if smoke is None:
        return
    if not smoke:
        raise ValueError(f"{context} smoke_episode_ids must be non-empty")
    if not set(smoke).issubset(regression):
        raise ValueError(
            f"{context} smoke_episode_ids must be included in the regression selection"
        )


def _validate_parent_selection(
    parent_manifest: Mapping[str, Any] | None,
    child_manifest: Mapping[str, Any] | None,
    *,
    context: str,
) -> None:
    """Reject contradictory parent/child episode contracts."""

    parent_regression, parent_smoke = _resolve_episode_selection(
        parent_manifest,
        context=f"{context} parent",
    )
    child_regression, child_smoke = _resolve_episode_selection(
        child_manifest,
        context=f"{context} child",
    )
    if (
        parent_regression is not None
        and child_regression is not None
        and parent_regression != child_regression
    ):
        raise ValueError(
            f"{context} child episode selection differs from parent selection"
        )
    if parent_smoke is not None and child_smoke is not None and parent_smoke != child_smoke:
        raise ValueError(f"{context} child smoke selection differs from parent selection")


def _validate_parent_semantics(
    parent_manifest: Mapping[str, Any],
    child_manifest: Mapping[str, Any],
    *,
    root: Path,
) -> None:
    """Reject a child manifest that changes a collection semantic contract."""

    parent_profile = _declared_target_profile(parent_manifest)
    child_profile = _declared_target_profile(child_manifest)
    if parent_profile is not None and child_profile is not None and parent_profile is not child_profile:
        raise ValueError(
            "dataset child target_profile differs from collection target_profile: "
            f"{child_profile.value} != {parent_profile.value}: {root}"
        )
    parent_kind = _parse_task_kind(dict(parent_manifest), root=root)
    child_kind = _parse_task_kind(dict(child_manifest), root=root)
    if parent_kind is not None and child_kind is not None and parent_kind is not child_kind:
        raise ValueError(
            "dataset child task_kind differs from collection task_kind: "
            f"{child_kind.value} != {parent_kind.value}: {root}"
        )
    if parent_profile is not None and child_kind is not None:
        inferred = target_profile_for_task_kind(child_kind)
        if inferred is not parent_profile:
            raise ValueError(
                "dataset child task_kind conflicts with collection target_profile: "
                f"{inferred.value} != {parent_profile.value}: {root}"
            )
    if child_profile is not None and parent_kind is not None:
        inferred = target_profile_for_task_kind(parent_kind)
        if inferred is not child_profile:
            raise ValueError(
                "dataset child target_profile conflicts with collection task_kind: "
                f"{child_profile.value} != {inferred.value}: {root}"
            )

    parent_bundle = prompt_bundle_from_manifest(parent_manifest, strict=True)
    child_bundle = prompt_bundle_from_manifest(child_manifest, strict=True)
    # A present, malformed bundle must not be interpreted as an omitted one.
    # ``prompt_bundle_from_manifest`` intentionally returns ``None`` for both
    # cases at the public compatibility boundary, so inspect the raw values
    # before comparing them here.
    for label, candidate in (("collection", parent_manifest), ("child", child_manifest)):
        if (
            "prompt_bundle" in candidate
            and candidate.get("prompt_bundle") is not None
            and not isinstance(candidate.get("prompt_bundle"), Mapping)
        ):
            raise ValueError(f"{label} prompt_bundle must be an object: {root}")
    if parent_bundle is not None and child_bundle is not None:
        try:
            validate_profile_provenance_pair(
                {"target_profile": parent_profile, "prompt_bundle": parent_bundle},
                {
                    "target_profile": child_profile or parent_profile,
                    "prompt_bundle": child_bundle,
                },
                label=f"collection/child prompt provenance ({root})",
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(str(exc)) from exc

    parent_raw = parent_manifest.get("profile")
    child_raw = child_manifest.get("profile")
    if not isinstance(parent_raw, str) or not isinstance(child_raw, str):
        return
    parent_alias = parent_raw.strip().lower().replace("-", "_")
    child_alias = child_raw.strip().lower().replace("-", "_")
    semantic_aliases = {
        "origin",
        "grasp_manipulation",
        "graspmanipulation",
        "contact_press",
        "contactpress",
        "door_open",
        "dooropen",
    }
    workflow_aliases = {
        AnnotationMode.PICK_PLACE.value,
        AnnotationMode.TARGET_ONLY.value,
    }
    if parent_alias in semantic_aliases and child_alias in workflow_aliases:
        raise ValueError(
            "dataset child workflow profile overrides collection semantic profile: "
            f"{child_raw!r} over {parent_raw!r}: {root}"
        )


def _canonical_profile(value: Any) -> str | None:
    """Normalize historical profile spellings to the workflow contract."""

    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(  # noqa: TRY004 - one manifest validation contract
            "dataset profile must be a non-empty string"
        )
    if not value.strip():
        raise ValueError("dataset profile must be a non-empty string")
    normalized = value.strip().lower().replace("-", "_")
    # Semantic profile aliases are target-only workflow selectors, not a
    # third timeline.  ``origin`` is the historical spelling for the default
    # grasp-manipulation target-only profile.
    if normalized in {"contact_press", "contactpress", "origin"}:
        return AnnotationMode.TARGET_ONLY.value
    if normalized in {
        "door_open",
        "dooropen",
        "grasp_manipulation",
        "graspmanipulation",
    }:
        return AnnotationMode.TARGET_ONLY.value
    if normalized in {"targetonly", "target_only"}:
        return AnnotationMode.TARGET_ONLY.value
    if normalized in {"pickplace", "pick_place"}:
        return AnnotationMode.PICK_PLACE.value
    return normalized


def _profile_matches_mode(value: Any, mode: AnnotationMode) -> bool:
    return _canonical_profile(value) == mode.value


def _manifest_workflow_mode(
    manifest: Mapping[str, Any] | None,
    *,
    root: Path,
) -> AnnotationMode | None:
    """Infer the timeline mode declared by one manifest envelope.

    ``profile`` is overloaded in historical extracts: ordinary workflow
    values (``pick_place``/``target_only``) select a timeline, while semantic
    aliases (``origin``/``contact_press``/``door_open``) select a target-only
    prompt profile.  Newer manifests may omit that field and declare only a
    ``task_kind`` or ``target_profile``.  Those declarations still imply the
    target-only timeline.  Returning ``None`` for an envelope with no such
    declaration lets collection resolution fail closed instead of guessing.
    """

    if not isinstance(manifest, Mapping):
        return None

    mode: AnnotationMode | None = None
    raw_profile = manifest.get("profile")
    if raw_profile is not None:
        normalized = _canonical_profile(raw_profile)
        if normalized in {item.value for item in AnnotationMode}:
            mode = AnnotationMode(normalized)
        else:
            # Validate a semantic/unknown value through the canonical parser.
            # Known semantic aliases imply target-only; unknown values raise a
            # useful field-level error rather than being treated as omission.
            _declared_target_profile(manifest)
            mode = AnnotationMode.TARGET_ONLY

    if manifest.get("target_profile") is not None:
        _declared_target_profile(manifest)
        if mode is None:
            mode = AnnotationMode.TARGET_ONLY

    if manifest.get("task_kind") is not None:
        _parse_task_kind(dict(manifest), root=root)
        if mode is None:
            mode = AnnotationMode.TARGET_ONLY

    return mode


def _declared_target_profile(manifest: Mapping[str, Any] | None) -> TargetProfile | None:
    """Resolve explicit semantic profile metadata from a task record.

    ``profile`` is a workflow field in modern manifests but was also used as
    a semantic selector by early contact/door extracts.  Treat only known
    target-profile values as semantic metadata; ordinary ``pick_place`` and
    ``target_only`` values remain workflow aliases.
    """

    if manifest is None:
        return None

    def parse(raw: Any, *, field: str) -> TargetProfile:
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f"dataset {field} must be a non-empty string")
        normalized = raw.strip().lower().replace("-", "_")
        normalized = {
            "origin": TargetProfile.GRASP_MANIPULATION.value,
            "graspmanipulation": TargetProfile.GRASP_MANIPULATION.value,
            "contactpress": TargetProfile.CONTACT_PRESS.value,
            "dooropen": TargetProfile.DOOR_OPEN.value,
        }.get(normalized, normalized)
        try:
            return TargetProfile(normalized)
        except ValueError as exc:
            choices = ", ".join(profile.value for profile in TargetProfile)
            raise ValueError(
                f"dataset {field} must be one of: {choices}"
            ) from exc

    explicit_raw = manifest.get("target_profile")
    explicit = (
        None if explicit_raw is None else parse(explicit_raw, field="target_profile")
    )
    legacy_raw = manifest.get("profile")
    legacy: TargetProfile | None = None
    if legacy_raw is not None:
        if not isinstance(legacy_raw, str) or not legacy_raw.strip():
            raise ValueError("dataset profile must be a non-empty string")
        normalized = legacy_raw.strip().lower().replace("-", "_")
        if normalized not in {
            AnnotationMode.PICK_PLACE.value,
            AnnotationMode.TARGET_ONLY.value,
            "pickplace",
            "targetonly",
        }:
            legacy = parse(legacy_raw, field="profile")
    if explicit is not None and legacy is not None and explicit is not legacy:
        raise ValueError("dataset target_profile conflicts with semantic profile alias")
    return explicit or legacy


def _resolve_target_profile(
    manifest: Mapping[str, Any] | None,
    task_kind: TargetOnlyTaskKind | None,
) -> TargetProfile:
    """Combine task-kind and explicit profile metadata with conflict checks."""

    inferred = target_profile_for_task_kind(task_kind)
    declared = _declared_target_profile(manifest)
    if declared is not None and task_kind is not None and declared is not inferred:
        raise ValueError(
            "dataset target_profile differs from task_kind-derived profile: "
            f"{declared.value} != {inferred.value}"
        )
    return inferred if declared is None else declared


def _validate_target_only_mode(
    task_kind: TargetOnlyTaskKind | None,
    target_profile: TargetProfile,
    mode: AnnotationMode,
) -> None:
    """Ensure target-only semantic declarations use the target-only timeline."""

    if task_kind is not None and mode is not AnnotationMode.TARGET_ONLY:
        raise ValueError(
            f"{task_kind.value} task_kind requires target_only dataset profile"
        )
    if (
        target_profile in {TargetProfile.CONTACT_PRESS, TargetProfile.DOOR_OPEN}
        and mode is not AnnotationMode.TARGET_ONLY
    ):
        raise ValueError(
            f"{target_profile.value} profile requires target_only dataset profile"
        )


@dataclass(frozen=True)
class DatasetTarget:
    root: Path
    task: str
    camera: str
    episode_ids: tuple[int, ...]
    task_kind: TargetOnlyTaskKind | None = None
    target_profile: TargetProfile | None = None
    discovered_all_episodes: bool = False
    # The extract manifest is optional for native RoboTwin directories.  Keep
    # it in memory when present so the profile binder never has to create a
    # per-task config/JSON just to carry provenance downstream.
    manifest_path: Path | None = None
    manifest_data: dict[str, Any] | None = None

    @property
    def profile(self) -> TargetProfile:
        """Semantic profile derived solely from manifest task kind."""

        if self.target_profile is not None:
            return self.target_profile
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
    if declared_profile is not None and not _profile_matches_mode(declared_profile, mode):
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

    # Collection records use the same aliases as task manifests.  Resolve all
    # declarations together so conflicting aliases cannot select different
    # episode universes in the resolver and binder.
    declared_episode_ids, declared_smoke_ids = _resolve_episode_selection(
        manifest_override,
        context="collection task record",
    )
    episode_tasks, metadata_tasks = _native_episode_tasks(root)
    discovery = None
    if declared_episode_ids is None:
        discovery = discover_episodes(root, camera=resolved_camera, require_depth=False)
        if not discovery.episodes:
            raise ValueError(f"no complete episodes found under {root}")
        episode_ids = discovery.episode_ids
        # A native full export may contain all tasks under one data/videos
        # root.  Use episodes.jsonl as the task authority whenever present;
        # never silently process the entire mixed root for an explicit task.
        if metadata_tasks:
            _require_native_task_coverage(root, discovery.episode_ids, episode_tasks)
            has_declared_task = task is not None or declared_task is not None
            if not has_declared_task and len(metadata_tasks) > 1:
                choices = ", ".join(sorted(metadata_tasks))
                raise ValueError(
                    "native dataset contains multiple tasks "
                    f"({choices}); pass --task to select one task"
                )
            selected_task = (
                resolved_task if has_declared_task else next(iter(metadata_tasks))
            )
            if selected_task not in metadata_tasks:
                raise ValueError(
                    f"requested task {selected_task!r} is absent from native metadata: {root}"
                )
            filtered_ids = tuple(
                episode_id
                for episode_id in episode_ids
                if episode_tasks.get(episode_id) == selected_task
            )
            if not filtered_ids:
                raise ValueError(
                    f"no complete episodes found for task {selected_task!r} under {root}"
                )
            episode_ids = filtered_ids
            resolved_task = selected_task
    else:
        if not declared_episode_ids:
            raise ValueError(f"collection task record has no episode_indices: {root}")
        episode_ids = declared_episode_ids
        if metadata_tasks:
            _require_native_task_coverage(root, episode_ids, episode_tasks)
            if resolved_task not in metadata_tasks:
                raise ValueError(
                    f"requested task {resolved_task!r} is absent from native metadata: {root}"
                )
            foreign_ids = tuple(
                episode_id
                for episode_id in episode_ids
                if episode_tasks[episode_id] != resolved_task
            )
            if foreign_ids:
                rendered = ", ".join(str(value) for value in foreign_ids)
                raise ValueError(
                    "collection task record selects episode ids assigned to another task: "
                    f"{rendered} for {resolved_task!r}"
                )

    _validate_smoke_selection(
        declared_smoke_ids,
        episode_ids,
        context=f"collection task record {resolved_task!r}",
    )

    task_kind = (
        _parse_task_kind(dict(manifest_override), root=root)
        if manifest_override is not None
        else None
    )
    target_profile = _resolve_target_profile(manifest_override, task_kind)
    _validate_target_only_mode(task_kind, target_profile, mode)

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
    if "episode_ids" in manifest_data:
        manifest_data["episode_ids"] = list(episode_ids)
    manifest_data.setdefault("smoke_episode_ids", [episode_ids[0]])
    if discovery is not None:
        manifest_data["discovery_skipped"] = list(discovery.skipped)
    if task_kind is not None:
        manifest_data["task_kind"] = task_kind.value
    if target_profile is not TargetProfile.GRASP_MANIPULATION or _declared_target_profile(
        manifest_override
    ) is not None:
        manifest_data["target_profile"] = target_profile.value

    return DatasetTarget(
        root=root,
        task=resolved_task,
        camera=resolved_camera,
        episode_ids=episode_ids,
        task_kind=task_kind,
        target_profile=target_profile,
        discovered_all_episodes=declared_episode_ids is None,
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

    Older coverage extracts describe only an episode selection
    (historically ``episode_indices``) and ``cameras``.
    Keep compatibility deliberately constrained to that single-task shape so
    incomplete modern manifests still fail closed instead of being guessed.
    """

    has_episode_selection = any(
        isinstance(manifest.get(key), list) for key in _EPISODE_SELECTION_ALIASES
    )
    return (
        "datasets" not in manifest
        and manifest.get("profile") is None
        and manifest.get("task") is None
        and manifest.get("camera") is None
        and has_episode_selection
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


def _native_episode_tasks(root: Path) -> tuple[dict[int, str], set[str]]:
    """Read episode-to-task authority from native ``meta/episodes.jsonl``.

    RoboTwin's full export stores many tasks in one root.  Directory names are
    then insufficient to resolve ``--task``; this helper keeps the metadata
    authoritative.  Valid single-task exports may omit every coarse-task
    field; once any record declares one, every record must provide a valid
    episode-to-task mapping.
    """

    path = root / "meta" / "episodes.jsonl"
    if not path.is_file():
        return {}, set()
    mapping: dict[int, str] = {}
    conflicting_ids: set[int] = set()
    tasks: set[str] = set()
    missing_task_records: list[tuple[int, int]] = []
    declares_task_authority = False
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"native metadata has invalid JSON at {path}:{line_number}: {exc.msg}"
                    ) from exc
                if not isinstance(record, Mapping):
                    raise ValueError(  # noqa: TRY004 - one invalid metadata contract
                        f"native metadata record must be an object at {path}:{line_number}"
                    )
                raw_id = record.get("episode_index")
                if isinstance(raw_id, bool) or not isinstance(raw_id, int) or raw_id < 0:
                    raise ValueError(
                        "native metadata has invalid episode_index at "
                        f"{path}:{line_number}: {raw_id!r}"
                    )
                has_task_field = any(
                    key in record
                    for key in (
                        "full_structured_tasks",
                        "task",
                        "task_name",
                        "dataset_task",
                    )
                )
                declares_task_authority = declares_task_authority or has_task_field
                candidate: Any = None
                structured = record.get("full_structured_tasks")
                if isinstance(structured, list) and structured:
                    candidate = structured[0]
                if not isinstance(candidate, str) or not candidate.strip():
                    for key in ("task", "task_name", "dataset_task"):
                        value = record.get(key)
                        if isinstance(value, str) and value.strip():
                            candidate = value
                            break
                if not isinstance(candidate, str) or not candidate.strip():
                    missing_task_records.append((line_number, raw_id))
                    continue
                try:
                    task_name = validate_dataset_component(candidate.strip(), field="dataset task")
                except ValueError as exc:
                    raise ValueError(
                        "native metadata has invalid task for episode "
                        f"{raw_id} at {path}:{line_number}: {candidate!r}"
                    ) from exc
                previous = mapping.get(raw_id)
                if previous is not None and previous != task_name:
                    conflicting_ids.add(raw_id)
                    continue
                mapping[raw_id] = task_name
                tasks.add(task_name)
    except OSError as exc:
        raise ValueError(f"cannot read native metadata: {path}: {exc}") from exc
    if not declares_task_authority:
        return {}, set()
    if missing_task_records:
        line_number, episode_id = missing_task_records[0]
        raise ValueError(
            "native metadata is missing task metadata for episode "
            f"{episode_id} at {path}:{line_number}"
        )
    if conflicting_ids:
        rendered = ", ".join(str(value) for value in sorted(conflicting_ids))
        raise ValueError(
            "native metadata assigns multiple tasks to episode ids: "
            f"{rendered}"
        )
    return mapping, tasks


def _require_native_task_coverage(
    root: Path,
    episode_ids: tuple[int, ...],
    episode_tasks: Mapping[int, str],
) -> None:
    """Reject partial task authority before it can silently drop episodes."""

    missing = tuple(episode_id for episode_id in episode_ids if episode_id not in episode_tasks)
    if missing:
        rendered = ", ".join(str(value) for value in missing[:10])
        suffix = " ..." if len(missing) > 10 else ""
        raise ValueError(
            "native metadata has no task mapping for discovered episode ids: "
            f"{rendered}{suffix} under {root}"
        )


def discover_task_episode_ids(
    root: Path,
    *,
    task: str,
    camera: str,
    require_depth: bool = False,
) -> tuple[int, ...]:
    """Discover all complete episodes while retaining mixed-root task scope."""

    discovery = discover_episodes(root, camera=camera, require_depth=require_depth)
    episode_tasks, metadata_tasks = _native_episode_tasks(root)
    if not episode_tasks:
        return discovery.episode_ids
    _require_native_task_coverage(root, discovery.episode_ids, episode_tasks)
    if task not in metadata_tasks:
        raise ValueError(f"requested task {task!r} is absent from native metadata: {root}")
    selected = tuple(
        episode_id
        for episode_id in discovery.episode_ids
        if episode_tasks.get(episode_id) == task
    )
    if not selected:
        raise ValueError(f"no complete episodes found for task {task!r} under {root}")
    return selected


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
    raw_episode_ids, smoke_ids = _resolve_episode_selection(
        manifest,
        context="legacy dataset manifest",
    )
    if not raw_episode_ids:
        raise ValueError(f"legacy dataset manifest has no episode_indices: {root}")
    episode_ids = raw_episode_ids
    _validate_smoke_selection(
        smoke_ids,
        episode_ids,
        context=f"legacy dataset manifest {root}",
    )
    task_kind = _parse_task_kind(dict(manifest), root=root)
    target_profile = _resolve_target_profile(manifest, task_kind)
    _validate_target_only_mode(task_kind, target_profile, mode)
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
            "smoke_episode_ids": list(smoke_ids or (episode_ids[0],)),
        }
    )
    for key in _EPISODE_SELECTION_ALIASES:
        if key in manifest_data:
            manifest_data[key] = list(episode_ids)
    if task_kind is not None:
        manifest_data["task_kind"] = task_kind.value
    if target_profile is not TargetProfile.GRASP_MANIPULATION or _declared_target_profile(
        manifest
    ) is not None:
        manifest_data["target_profile"] = target_profile.value
    return DatasetTarget(
        root=root,
        task=task_name,
        camera=camera_name,
        episode_ids=episode_ids,
        task_kind=task_kind,
        target_profile=target_profile,
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
        _validate_parent_selection(
            manifest_override,
            manifest,
            context=f"dataset task {root.name!r}",
        )
        _validate_parent_semantics(
            manifest_override,
            manifest,
            root=root,
        )
    effective_manifest = manifest
    if manifest_override is not None:
        # A task-local manifest is more specific than the collection record.
        # Keep its values authoritative, but inherit collection-level semantic
        # metadata when the local file predates target profiles.
        merged_override = dict(manifest_override)
        effective_manifest = dict(manifest)
        for key in (
            "profile",
            "target_profile",
            "task_kind",
            "prompt_bundle",
            "source_dataset_root",
            "source_manifest_path",
        ):
            if (
                key not in effective_manifest or effective_manifest[key] is None
            ) and key in merged_override:
                effective_manifest[key] = deepcopy(merged_override[key])
        manifest_override = merged_override
    if _is_legacy_single_task_manifest(manifest):
        return _legacy_task_target(
            root,
            manifest_path=manifest_path or (root / "EXTRACT_MANIFEST.json"),
            manifest=effective_manifest,
            mode=mode,
            task=task_override,
            camera=camera_override,
        )
    # Resolve semantic metadata before checking the workflow selector.  Newer
    # manifests may intentionally omit the overloaded legacy ``profile`` key
    # and carry only ``target_profile``/``task_kind``; those fields still give
    # us enough information to require the correct target-only timeline.
    task_kind = _parse_task_kind(dict(effective_manifest), root=root)
    target_profile = _resolve_target_profile(effective_manifest, task_kind)
    declared_profile = effective_manifest.get("profile")
    if declared_profile is not None and not _profile_matches_mode(declared_profile, mode):
        raise ValueError(
            f"dataset profile {declared_profile!r} does not match "
            f"--{mode.value.replace('_', '-')}"
        )
    _validate_target_only_mode(task_kind, target_profile, mode)
    task = effective_manifest.get("task")
    camera = effective_manifest.get("camera")
    episode_ids, smoke_ids = _resolve_episode_selection(
        effective_manifest,
        context="dataset manifest",
    )
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
    if not episode_ids:
        raise ValueError(f"dataset manifest has no episode_indices: {root}")
    unique_ids = episode_ids
    _validate_smoke_selection(
        smoke_ids,
        unique_ids,
        context=f"dataset manifest {root}",
    )
    for directory in ("data", "videos", "sidecars", "meta"):
        if not (root / directory).is_dir():
            raise ValueError(f"task dataset is missing {directory}/: {root}")
    manifest_data = dict(effective_manifest)
    # Normalize the workflow field in the in-memory copy so downstream
    # binding/adapters can consume a single contract even when the source
    # manifest used only the modern semantic profile fields.
    manifest_data.setdefault("profile", mode.value)
    if target_profile is not TargetProfile.GRASP_MANIPULATION or _declared_target_profile(
        effective_manifest
    ) is not None:
        manifest_data["target_profile"] = target_profile.value
    return DatasetTarget(
        root,
        task,
        camera,
        unique_ids,
        task_kind=task_kind,
        target_profile=target_profile,
        manifest_path=manifest_path,
        manifest_data=manifest_data,
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

    declared_collection_mode = _manifest_workflow_mode(manifest, root=root)
    if (
        declared_collection_mode is not None
        and declared_collection_mode is not mode
    ):
        raise ValueError(
            "dataset collection metadata does not match "
            f"--{mode.value.replace('_', '-')}"
        )
    if not isinstance(datasets, list) or not datasets:
        raise ValueError(f"collection manifest has no datasets: {root}")
    names: list[str] = []
    records_by_name: dict[str, Mapping[str, Any]] = {}
    inherited_keys = (
        "profile",
        "target_profile",
        "task_kind",
        "camera",
        "source_dataset_root",
        "source_manifest_path",
        "prompt_bundle",
    )
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
        if not isinstance(record, Mapping):  # guarded above; keeps mypy precise
            raise ValueError(  # noqa: TRY004 - the public resolver uses ValueError
                f"collection manifest contains an invalid task record: {root}"
            )
        # Validate the parent contract against the raw record before filling
        # inherited fields.  Otherwise an explicit child ``profile`` or
        # ``target_profile: null`` could hide a collection-level semantic
        # contract and route the task through the generic origin prompts.
        _validate_parent_selection(
            manifest,
            record,
            context=f"collection task {name!r}",
        )
        _validate_parent_semantics(
            manifest,
            record,
            root=root / name,
        )
        # Keep the complete record, not just its task name.  Some historical
        # collections omit per-task EXTRACT_MANIFEST.json files and rely on
        # these records for episode selection, task kind, and provenance.
        merged_record = dict(record)
        # A collection-level contract applies to child records unless a child
        # explicitly overrides it.  In particular, a legacy collection with
        # ``profile: contact_press`` must not silently route children through
        # the generic origin prompts merely because it omitted task_kind on
        # every record.
        for key in inherited_keys:
            if (key not in merged_record or merged_record[key] is None) and key in manifest:
                merged_record[key] = deepcopy(manifest[key])
        records_by_name[name] = merged_record
    if len(set(names)) != len(names):
        raise ValueError(f"collection manifest contains duplicate tasks: {root}")
    selected_names: list[str] = names if requested_task is None else [requested_task]
    unknown = sorted(set(selected_names) - set(names))
    if unknown:
        raise ValueError(f"requested collection task is absent: {unknown}")

    # Some older mixed collections (for example profile-compat extracts) do
    # not have a top-level profile.  Infer the *workflow* from the records
    # when the selected set is homogeneous.  Semantic target-only profiles are
    # intentionally allowed to differ (origin/contact/door): they all share
    # one target-only timeline and are routed per target below.  A mixed
    # pick-place/target-only request remains ambiguous and must name one task
    # explicitly; otherwise a target-only run could accidentally process
    # pick-and-place data (or vice versa).
    if declared_collection_mode is None:
        selected_profile_values = [records_by_name[name].get("profile") for name in selected_names]
        selected_workflow_modes: list[AnnotationMode | None] = []
        for name in selected_names:
            record = records_by_name[name]
            selected_workflow_modes.append(
                _manifest_workflow_mode(record, root=root / name)
            )
        if any(value is None for value in selected_workflow_modes):
            raise ValueError(
                "collection manifest has no top-level profile and task records "
                "do not declare a workflow or semantic profile consistently; "
                "add a top-level profile or select a task with explicit metadata"
            )
        selected_workflows = {
            value.value for value in selected_workflow_modes if value is not None
        }
        if requested_task is None and len(selected_workflows) > 1:
            # Keep the historical diagnostic useful by rendering the raw
            # profile spellings when available, while semantic aliases that
            # normalize to one target-only workflow do not trigger this path.
            choices = ", ".join(
                sorted(
                    str(value)
                    for value in {
                        raw
                        for raw in selected_profile_values
                        if raw is not None
                    }
                )
            )
            if not choices:
                choices = ", ".join(sorted(selected_workflows))
            raise ValueError(
                "collection manifest has mixed profiles "
                f"({choices}); pass --task to select one task"
            )
        if selected_workflows and any(value != mode.value for value in selected_workflows):
            declared = ", ".join(sorted(str(value) for value in selected_workflows))
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
    "discover_task_episode_ids",
    "read_dataset_task_kind",
    "resolve_dataset_input",
]
