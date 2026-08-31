"""Configuration loading for the small target/receiver experiment."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any
from urllib.parse import urlsplit

import yaml

from .domain import (
    AnnotationMode,
    AnnotationSpec,
    TargetOnlyTaskKind,
    TargetProfile,
    annotation_spec,
    target_profile_for_task_kind,
)

_REMOVED_S4_MASK_FIELDS = frozenset(
    {
        "temporal_envelope_guard_retry_enabled",
        "qc_bbox_directional_expand_enabled",
        "qc_border_retry_enabled",
        "qc_border_retry_prompt_template",
        "qc_border_retry_max_tokens",
    }
)


class ConfigError(ValueError):
    """Configuration is missing or violates the pipeline contract."""


def validate_dataset_component(value: Any, *, field: str) -> str:
    """Validate a task/camera name before it is interpolated into a path.

    Dataset task and camera values are *names*, rather than arbitrary paths.
    They are used in paths such as ``<run>/<task>/episode/...`` and
    ``videos/observation.images.<camera>/...``.  Accepting a separator here
    would therefore allow a manifest or CLI override to escape the intended
    directory (and would behave differently on POSIX and Windows).  Keep the
    contract deliberately portable: reject both slash styles, NUL bytes,
    dot-components, and Windows drive/UNC prefixes.

    The normalized value is returned so callers can consistently strip
    incidental surrounding whitespace while retaining the original spelling
    of valid names.
    """

    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field} must be a non-empty name")
    normalized = value.strip()
    windows = PureWindowsPath(normalized)
    if (
        normalized in {".", ".."}
        or "/" in normalized
        or "\\" in normalized
        or "\x00" in normalized
        or windows.is_absolute()
        or bool(windows.drive)
    ):
        raise ConfigError(f"{field} must be a single path component")
    return normalized


def _required(mapping: dict[str, Any], key: str, *, section: str = "config") -> Any:
    if key not in mapping:
        raise ConfigError(f"{section}.{key} is required")
    return mapping[key]


def _path(value: Any, *, base_dir: Path, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field} must be a non-empty path")
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else (base_dir / candidate).resolve()


def _integers(value: Any, *, field: str) -> tuple[int, ...]:
    """Parse a YAML integer list without implicit coercion.

    YAML makes it deceptively easy to pass values such as ``"2"`` or ``1.9``
    into a numeric option.  Coercing those values here can silently change GPU
    or episode selection, so configuration parsing accepts only real Python
    integers (while still rejecting ``bool``, which is an ``int`` subclass).
    """

    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise ConfigError(f"{field} must be a list of integers")
    return tuple(value)


def parse_gpu_list(value: str, *, field: str = "GPU list") -> tuple[int, ...]:
    """Parse a comma-separated CLI GPU list using the config GPU contract."""

    if not isinstance(value, str):
        raise ConfigError(f"{field} must be a comma-separated list of integers")
    stripped = value.strip()
    if not stripped:
        return ()
    parts = tuple(part.strip() for part in stripped.split(","))
    if any(not part for part in parts):
        raise ConfigError(f"{field} must be a comma-separated list of integers")
    try:
        gpus = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise ConfigError(f"{field} must be a comma-separated list of integers") from exc
    _validate_gpu_list(gpus, field=field)
    return gpus


def _validate_gpu_list(gpus: tuple[int, ...], *, field: str) -> None:
    if any(isinstance(gpu, bool) or not isinstance(gpu, int) for gpu in gpus):
        raise ConfigError(f"{field} must contain integers")
    if any(gpu < 0 for gpu in gpus):
        raise ConfigError(f"{field} must contain non-negative integers")
    if len(set(gpus)) != len(gpus):
        raise ConfigError(f"{field} must not contain duplicate GPUs")


def _gpu_ids(value: Any, *, field: str) -> tuple[int, ...]:
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        raise ConfigError(f"{field} must be a list of integers")
    gpus = tuple(value)
    _validate_gpu_list(gpus, field=field)
    return gpus


def _integer(value: Any, *, field: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise ConfigError(f"{field} must be a {qualifier} integer")
    return value


def _strict_float(value: Any, *, field: str) -> float:
    """Parse a finite YAML number while rejecting strings and booleans."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{field} must be a finite number")
    try:
        parsed = float(value)
    except (TypeError, OverflowError, ValueError) as exc:
        # The type guard above handles YAML's built-in scalar types.  Keep the
        # conversion guarded as callers may provide an int subclass whose
        # value cannot be represented as a Python float (for example, a very
        # large integer).
        raise ConfigError(f"{field} must be a finite number") from exc
    if not math.isfinite(parsed):
        raise ConfigError(f"{field} must be a finite number")
    return parsed


def _strict_bool(value: Any, *, field: str) -> bool:
    """Parse a YAML boolean without treating non-empty strings as true."""

    if not isinstance(value, bool):
        raise ConfigError(f"{field} must be a boolean")
    return value


def _string(value: Any, *, field: str) -> str:
    """Parse a required textual option without stringifying arbitrary values."""

    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field} must be a non-empty string")
    return value.strip()


def _positive_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{field} must be a finite number greater than zero")
    try:
        parsed = float(value)
    except (TypeError, OverflowError, ValueError) as exc:
        raise ConfigError(f"{field} must be a finite number greater than zero") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ConfigError(f"{field} must be a finite number greater than zero")
    return parsed


@dataclass(frozen=True)
class DatasetConfig:
    root: Path
    manifest: Path
    task: str
    camera: str
    smoke_episode_ids: tuple[int, ...]
    regression_episode_ids: tuple[int, ...]
    manifest_data: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        # These values are later interpolated into dataset and artifact paths;
        # reject path-like input even when a caller constructs the DTO
        # directly rather than going through ``load_config``.
        object.__setattr__(
            self,
            "task",
            validate_dataset_component(self.task, field="dataset.task"),
        )
        object.__setattr__(
            self,
            "camera",
            validate_dataset_component(self.camera, field="dataset.camera"),
        )


@dataclass(frozen=True)
class DatasetBinding:
    """Runtime dataset identity bound to an algorithm :class:`PipelineProfile`.

    Dataset identity is intentionally kept out of reusable profile YAML.  The
    ``episode_ids`` field is a compatibility alias for
    ``regression_episode_ids``; callers should prefer the explicit smoke and
    regression fields when they have separate selections.
    """

    root: Path
    task: str
    camera: str
    smoke_episode_ids: tuple[int, ...] | Sequence[int] | None = None
    regression_episode_ids: tuple[int, ...] | Sequence[int] | None = None
    manifest_path: Path | None = None
    manifest_data: Mapping[str, Any] | None = None
    episode_ids: tuple[int, ...] | Sequence[int] | None = None
    task_kind: TargetOnlyTaskKind | str | None = None
    target_profile: TargetProfile | str | None = None

    def __post_init__(self) -> None:
        root = Path(self.root).expanduser().resolve()
        task = validate_dataset_component(self.task, field="dataset binding task")
        camera = validate_dataset_component(self.camera, field="dataset binding camera")
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "task", task)
        object.__setattr__(self, "camera", camera)

        manifest_data: dict[str, Any] | None
        if self.manifest_data is None:
            manifest_data = None
        elif not isinstance(self.manifest_data, Mapping):
            raise ConfigError("dataset binding manifest_data must be a mapping")
        else:
            manifest_data = copy.deepcopy(dict(self.manifest_data))
            object.__setattr__(self, "manifest_data", manifest_data)

        if manifest_data is not None:
            _validate_binding_manifest_identity(
                manifest_data,
                task=self.task,
                camera=self.camera,
            )

        declared_target_profile = (
            None if manifest_data is None else _manifest_target_profile(manifest_data)
        )
        binding_target_profile = (
            None
            if self.target_profile is None
            else _normalise_target_profile(self.target_profile)
        )
        if binding_target_profile is not None:
            try:
                binding_target_profile_enum = TargetProfile(binding_target_profile)
            except (TypeError, ValueError) as exc:
                choices = ", ".join(profile.value for profile in TargetProfile)
                raise ConfigError(
                    f"dataset binding target_profile must be one of: {choices}"
                ) from exc
            object.__setattr__(self, "target_profile", binding_target_profile_enum)
        if (
            declared_target_profile is not None
            and binding_target_profile is not None
            and declared_target_profile.value != binding_target_profile
        ):
            raise ConfigError(
                "dataset binding target_profile differs from manifest target_profile"
            )
        if declared_target_profile is not None and binding_target_profile is None:
            object.__setattr__(self, "target_profile", declared_target_profile)

        raw_manifest_task_kind = (
            manifest_data.get("task_kind") if manifest_data is not None else None
        )
        if self.task_kind is not None:
            try:
                task_kind = TargetOnlyTaskKind(self.task_kind)
            except (TypeError, ValueError) as exc:
                choices = ", ".join(item.value for item in TargetOnlyTaskKind)
                raise ConfigError(
                    f"dataset binding task_kind must be one of: {choices}"
                ) from exc
            object.__setattr__(self, "task_kind", task_kind)
            if raw_manifest_task_kind is not None:
                try:
                    manifest_task_kind = TargetOnlyTaskKind(raw_manifest_task_kind)
                except (TypeError, ValueError) as exc:
                    choices = ", ".join(item.value for item in TargetOnlyTaskKind)
                    raise ConfigError(
                        f"dataset binding manifest task_kind must be one of: {choices}"
                    ) from exc
                if manifest_task_kind is not task_kind:
                    raise ConfigError(
                        "dataset binding task_kind differs from manifest task_kind"
                    )
        elif raw_manifest_task_kind is not None:
            try:
                object.__setattr__(
                    self,
                    "task_kind",
                    TargetOnlyTaskKind(raw_manifest_task_kind),
                )
            except (TypeError, ValueError) as exc:
                choices = ", ".join(item.value for item in TargetOnlyTaskKind)
                raise ConfigError(
                    f"dataset binding manifest task_kind must be one of: {choices}"
                ) from exc

        manifest_regression = (
            None
            if manifest_data is None
            else _manifest_episode_selection(manifest_data)
        )
        alias = _binding_episode_ids(self.episode_ids, field="episode_ids")
        smoke = _binding_episode_ids(self.smoke_episode_ids, field="smoke_episode_ids")
        regression = _binding_episode_ids(
            self.regression_episode_ids,
            field="regression_episode_ids",
        )
        explicit_regression = regression is not None or alias is not None
        if regression is None:
            regression = alias
        elif alias is not None and regression != alias:
            raise ConfigError(
                "dataset binding episode_ids and regression_episode_ids differ"
            )
        if regression is None:
            regression = manifest_regression
        if smoke is None and not explicit_regression and manifest_data is not None:
            smoke_raw = manifest_data.get("smoke_episode_ids")
            smoke = (
                None
                if smoke_raw is None
                else _binding_episode_ids(
                    smoke_raw,
                    field="manifest.smoke_episode_ids",
                )
            )
        # A binding is also used as an intermediate DTO by path discovery.  It
        # is therefore valid to omit episode selections here; ``bind_dataset``
        # performs the final resolution (from the manifest, when available)
        # and emits a useful error if no selection can be established.
        if regression is not None and smoke is None and regression:
            smoke = (regression[0],)
        if (
            smoke is not None
            and regression is not None
            and smoke
            and not set(smoke).issubset(regression)
        ):
            raise ConfigError(
                "dataset binding smoke episodes must be non-empty and included "
                "in regression episodes"
            )
        if smoke is not None and regression is not None and not smoke and regression:
            # Preserve an explicitly empty smoke selection.  The binder will
            # reject it only when constructing a runnable PipelineConfig.
            pass
        object.__setattr__(self, "smoke_episode_ids", smoke)
        object.__setattr__(self, "regression_episode_ids", regression)
        object.__setattr__(
            self,
            "episode_ids",
            regression if regression is not None else alias,
        )

        if self.manifest_path is not None:
            manifest_path = Path(self.manifest_path).expanduser()
            if not manifest_path.is_absolute():
                manifest_path = root / manifest_path
            object.__setattr__(
                self,
                "manifest_path",
                manifest_path.resolve(),
            )


def _binding_episode_ids(
    value: Sequence[int] | None,
    *,
    field: str,
) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        raise ConfigError(f"{field} must be a sequence of integers")
    try:
        values = tuple(value)
    except TypeError as exc:
        raise ConfigError(f"{field} must be a sequence of integers") from exc
    if any(isinstance(item, bool) or not isinstance(item, int) for item in values):
        raise ConfigError(f"{field} must be a sequence of integers")
    if any(item < 0 for item in values):
        raise ConfigError(f"{field} must contain non-negative episode ids")
    if len(set(values)) != len(values):
        raise ConfigError(f"{field} must not contain duplicate episode ids")
    return values


def _manifest_episode_values(manifest: Mapping[str, Any], key: str) -> Any:
    """Read a canonical episode selection after validating all aliases.

    ``episode_indices`` is the historical spelling, while
    ``regression_episode_ids`` and ``episode_ids`` are newer compatibility
    names.  A manifest may contain more than one spelling, but every
    non-null declaration must describe the exact same ordered sequence.  The
    helper remains for callers that need the raw value; new code should use
    :func:`_manifest_episode_selection` directly.
    """

    if key == "regression_episode_ids":
        selection = _manifest_episode_selection(manifest)
        return None if selection is None else list(selection)
    return manifest.get(key)


_MANIFEST_REGRESSION_ALIASES = (
    "episode_indices",
    "regression_episode_ids",
    "episode_ids",
)


def _manifest_episode_selection(
    manifest: Mapping[str, Any],
    *,
    field_prefix: str = "manifest",
) -> tuple[int, ...] | None:
    """Return one validated regression selection from a manifest.

    All non-null aliases are parsed independently so malformed or conflicting
    declarations cannot be hidden by precedence.  Equality is intentionally
    sequence-based (including order), making the serialized contract
    deterministic and fail-closed.
    """

    declared: list[tuple[str, tuple[int, ...]]] = []
    for key in _MANIFEST_REGRESSION_ALIASES:
        raw = manifest.get(key)
        if raw is None:
            continue
        parsed = _binding_episode_ids(raw, field=f"{field_prefix}.{key}")
        if parsed is None:  # pragma: no cover - raw was checked above
            raise ConfigError(f"{field_prefix}.{key} must not be null")
        declared.append((key, parsed))
    smoke_raw = manifest.get("smoke_episode_ids")
    smoke: tuple[int, ...] | None = None
    if smoke_raw is not None:
        smoke = _binding_episode_ids(
            smoke_raw,
            field=f"{field_prefix}.smoke_episode_ids",
        )
        if not smoke:
            raise ConfigError(
                f"{field_prefix}.smoke_episode_ids must be non-empty"
            )
    if not declared:
        return None
    first_key, selection = declared[0]
    for key, candidate in declared[1:]:
        if candidate != selection:
            raise ConfigError(
                f"{field_prefix}.{key} conflicts with {field_prefix}.{first_key}"
            )
    if smoke is not None and not set(smoke).issubset(selection):
        raise ConfigError(
            f"{field_prefix}.smoke_episode_ids must be included in "
            f"{field_prefix}.{first_key}"
        )
    return selection


def _validate_binding_manifest_identity(
    manifest: Mapping[str, Any],
    *,
    task: str,
    camera: str,
) -> None:
    """Reject a binding whose manifest names a different task or camera.

    ``dataset_root`` is deliberately allowed to be stale because datasets can
    be moved between machines.  Task and camera, however, are semantic
    identity and changing either would make episode paths and prompts refer to
    a different dataset.  Keep this check at the binding boundary so direct
    callers get the same fail-closed behavior as the path resolver.
    """

    for field, expected in (("task", task), ("camera", camera)):
        declared = manifest.get(field)
        if declared is not None and declared != expected:
            raise ConfigError(
                f"dataset binding {field} differs from manifest {field}: "
                f"{expected!r} != {declared!r}"
            )


def _deep_merge(
    base: Mapping[str, Any],
    overlay: Mapping[str, Any],
    *,
    field: str = "profile",
) -> dict[str, Any]:
    """Recursively merge profile mappings without mutating YAML input."""

    if not isinstance(base, Mapping) or not isinstance(overlay, Mapping):
        raise ConfigError(f"{field} must be a mapping")
    result: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in overlay.items():
        if not isinstance(key, str):
            raise ConfigError(f"{field} keys must be strings")
        if key in result and isinstance(result[key], Mapping) and isinstance(value, Mapping):
            result[key] = _deep_merge(result[key], value, field=f"{field}.{key}")
        else:
            result[key] = copy.deepcopy(value)
    return result


def _find_dataset_block(value: Any, *, path: str = "config") -> str | None:
    """Return the first nested ``dataset`` key in a profile document.

    A reusable profile must not hide task identity in an unselected mode.  A
    recursive check keeps that invariant true for every overlay, rather than
    only for the mode selected by the current command.
    """

    if isinstance(value, Mapping):
        for key, nested in value.items():
            if key == "dataset":
                return f"{path}.dataset"
            found = _find_dataset_block(nested, path=f"{path}.{key}")
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            found = _find_dataset_block(nested, path=f"{path}[{index}]")
            if found is not None:
                return found
    return None


def _normalise_selector(value: AnnotationMode | str | None, *, field: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, AnnotationMode):
        return value.value
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field} must be a non-empty string")
    return value.strip()


def _normalise_target_profile(value: TargetProfile | str | None) -> str | None:
    """Normalize public profile aliases to canonical domain values.

    ``origin`` is retained as a concise configuration spelling for the
    historical grasp-manipulation target-only profile.  Internally we keep the
    descriptive ``grasp_manipulation`` enum value to avoid conflating profile
    selection with dataset provenance.
    """

    if isinstance(value, TargetProfile):
        selected: str | None = value.value
    else:
        selected = _normalise_selector(value, field="target_profile")
    if selected is None:
        return None
    normalized = selected.lower().replace("-", "_")
    aliases = {
        "origin": TargetProfile.GRASP_MANIPULATION.value,
        "graspmanipulation": TargetProfile.GRASP_MANIPULATION.value,
        "contactpress": TargetProfile.CONTACT_PRESS.value,
        "dooropen": TargetProfile.DOOR_OPEN.value,
    }
    return aliases.get(normalized, normalized)


# ``target_only`` is the timeline; these selectors choose its semantic
# prompt bundle.  Keep the mapping next to the resolver so a semantic profile
# can never silently fall back to the generic target-only overlay.
_TARGET_PROFILE_MODE_KEYS = {
    TargetProfile.GRASP_MANIPULATION.value: "origin",
    TargetProfile.CONTACT_PRESS.value: "contact_press",
    TargetProfile.DOOR_OPEN.value: "door_open",
}
_TARGET_PROFILE_SELECTORS = {
    "origin": TargetProfile.GRASP_MANIPULATION.value,
    "contact_press": TargetProfile.CONTACT_PRESS.value,
    "door_open": TargetProfile.DOOR_OPEN.value,
}


def _overlay_target_profile(candidate: Mapping[str, Any]) -> str | None:
    annotation = candidate.get("annotation", {})
    if not isinstance(annotation, Mapping):
        raise ConfigError("mode overlay annotation must be a mapping")
    return _normalise_target_profile(annotation.get("profile"))


def _normalise_annotation_profile(value: Any) -> str | None:
    """Normalize the historical manifest ``profile`` workflow aliases."""

    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigError("dataset manifest profile must be a non-empty string")
    normalized = value.strip().lower().replace("-", "_")
    aliases = {
        "pickplace": AnnotationMode.PICK_PLACE.value,
        "targetonly": AnnotationMode.TARGET_ONLY.value,
        # Older task-bound manifests used the semantic profile as the
        # workflow selector.  Both variants still execute the target-only
        # timeline and must therefore compare equal at the binding boundary.
        "contact_press": AnnotationMode.TARGET_ONLY.value,
        "contactpress": AnnotationMode.TARGET_ONLY.value,
        "door_open": AnnotationMode.TARGET_ONLY.value,
        "dooropen": AnnotationMode.TARGET_ONLY.value,
        "origin": AnnotationMode.TARGET_ONLY.value,
        "grasp_manipulation": AnnotationMode.TARGET_ONLY.value,
        "graspmanipulation": AnnotationMode.TARGET_ONLY.value,
    }
    return aliases.get(normalized, normalized)


def _manifest_target_profile(manifest: Mapping[str, Any]) -> TargetProfile | None:
    """Read an explicit semantic profile without confusing workflow aliases."""

    def parse(raw: Any, *, field: str) -> TargetProfile:
        normalized = _normalise_target_profile(raw)
        if normalized is None:
            raise ConfigError(f"dataset manifest {field} must be a non-empty string")
        try:
            return TargetProfile(normalized)
        except (TypeError, ValueError) as exc:
            choices = ", ".join(profile.value for profile in TargetProfile)
            raise ConfigError(
                f"dataset manifest {field} must be one of: {choices}"
            ) from exc

    explicit_raw = manifest.get("target_profile")
    explicit = (
        None if explicit_raw is None else parse(explicit_raw, field="target_profile")
    )
    legacy_raw = manifest.get("profile")
    legacy: TargetProfile | None = None
    if legacy_raw is not None:
        if not isinstance(legacy_raw, str) or not legacy_raw.strip():
            raise ConfigError("dataset manifest profile must be a non-empty string")
        normalized = legacy_raw.strip().lower().replace("-", "_")
        if normalized in {
            "pick_place",
            "pickplace",
            "target_only",
            "targetonly",
        }:
            legacy = None
        else:
            legacy = parse(legacy_raw, field="profile")
    if explicit is not None and legacy is not None and explicit is not legacy:
        raise ConfigError(
            "dataset manifest target_profile conflicts with semantic profile alias"
        )
    return explicit or legacy


def _resolve_profile_document(
    raw: Mapping[str, Any],
    *,
    mode: AnnotationMode | str | None = None,
    target_profile: TargetProfile | str | None = None,
    require_semantic_overlay: bool = True,
) -> dict[str, Any]:
    """Resolve ``common``/``defaults`` plus an optional mode overlay.

    The resolver is intentionally independent of the concrete dataclasses so
    that the same document can be inspected by tooling before validation.
    Lists are replaced (rather than concatenated), while nested mappings are
    merged recursively.
    """

    if not isinstance(raw, Mapping):
        raise ConfigError("top-level config must be a mapping")

    requested_mode = _normalise_selector(mode, field="mode")
    declared_mode = raw.get("mode")
    if declared_mode is not None:
        declared_mode = _normalise_selector(declared_mode, field="mode")
        if requested_mode is not None and declared_mode != requested_mode:
            raise ConfigError(
                f"profile mode conflict: argument selects {requested_mode!r}, "
                f"YAML selects {declared_mode!r}"
            )
        requested_mode = declared_mode

    declared_target_profile = raw.get("target_profile")
    requested_target_profile = _normalise_target_profile(target_profile)
    if declared_target_profile is not None:
        declared_target_profile = _normalise_target_profile(declared_target_profile)
        if (
            requested_target_profile is not None
            and declared_target_profile != requested_target_profile
        ):
            raise ConfigError(
                f"target profile conflict: argument selects {requested_target_profile!r}, "
                f"YAML selects {declared_target_profile!r}"
            )
        requested_target_profile = declared_target_profile

    # A semantic mode selector is shorthand for one target-only prompt
    # bundle.  Resolve it to the canonical profile up front, and reject a
    # contradictory explicit profile before any overlay is merged.
    mode_target_profile = _TARGET_PROFILE_SELECTORS.get(requested_mode or "")
    if mode_target_profile is not None:
        if (
            requested_target_profile is not None
            and requested_target_profile != mode_target_profile
        ):
            raise ConfigError(
                f"profile mode {requested_mode!r} conflicts with target profile "
                f"{requested_target_profile!r}"
            )
        requested_target_profile = mode_target_profile

    reserved = {"common", "defaults", "modes", "mode", "target_profile", "version"}
    direct = {key: value for key, value in raw.items() if key not in reserved}
    common_value = raw.get("common")
    defaults_value = raw.get("defaults")
    if common_value is not None and not isinstance(common_value, Mapping):
        raise ConfigError("common must be a mapping")
    if defaults_value is not None and not isinstance(defaults_value, Mapping):
        raise ConfigError("defaults must be a mapping")
    resolved = _deep_merge({}, {})
    # ``defaults`` is retained as a migration-friendly spelling used by the
    # checked-in shared process profile; canonical documents should use
    # ``common``.  Precedence is defaults < common < direct top-level fields <
    # selected mode overlay, which lets a one-off explicit section override
    # reusable defaults without mutating the source document.
    if defaults_value is not None:
        resolved = _deep_merge(resolved, defaults_value, field="defaults")
    if common_value is not None:
        resolved = _deep_merge(resolved, common_value, field="common")
    resolved = _deep_merge(resolved, direct, field="config")

    # A profile declared in the reusable base sections is semantic metadata,
    # too.  Carry it into the same routing path as the explicit
    # ``target_profile`` argument so a generic target-only overlay cannot
    # silently inherit contact/door prompts (or an origin profile) by name
    # alone.
    base_annotation = resolved.get("annotation")
    if base_annotation is not None:
        if not isinstance(base_annotation, Mapping):
            raise ConfigError("annotation must be a mapping")
        base_profile = _normalise_target_profile(base_annotation.get("profile"))
        if base_profile is not None:
            if (
                requested_target_profile is not None
                and requested_target_profile != base_profile
            ):
                raise ConfigError(
                    f"target profile conflict: argument selects {requested_target_profile!r}, "
                    f"YAML selects {base_profile!r}"
                )
            requested_target_profile = base_profile

    modes_value = raw.get("modes")
    if modes_value is not None and not isinstance(modes_value, Mapping):
        raise ConfigError("modes must be a mapping")
    mode_map = {} if modes_value is None else dict(modes_value)
    if mode_map:
        # ``mode=target_only`` plus a semantic profile must select that
        # profile's dedicated overlay.  Merely changing annotation.profile
        # after loading the generic target-only prompts is unsafe.
        if requested_mode == AnnotationMode.TARGET_ONLY.value and requested_target_profile:
            semantic_mode = _TARGET_PROFILE_MODE_KEYS.get(requested_target_profile)
            if semantic_mode is not None and semantic_mode in mode_map:
                requested_mode = semantic_mode
            elif requested_target_profile in _TARGET_PROFILE_MODE_KEYS:
                raise ConfigError(
                    f"target profile {requested_target_profile!r} requires an explicit "
                    f"modes.{semantic_mode} profile overlay"
                )
        if requested_mode is None and requested_target_profile is not None:
            profile_matches: list[str] = []
            for key, candidate in mode_map.items():
                if not isinstance(candidate, Mapping):
                    continue
                candidate_profile = _overlay_target_profile(candidate)
                if str(key) == requested_target_profile or candidate_profile == requested_target_profile:
                    profile_matches.append(str(key))
            if len(profile_matches) == 1:
                requested_mode = profile_matches[0]
            elif len(profile_matches) > 1:
                choices = ", ".join(profile_matches)
                raise ConfigError(
                    f"target profile {requested_target_profile!r} matches multiple modes; "
                    f"choose one explicitly: {choices}"
                )
        if requested_mode is None:
            if len(mode_map) == 1:
                requested_mode = str(next(iter(mode_map)))
            else:
                choices = ", ".join(str(key) for key in mode_map)
                raise ConfigError(
                    f"profile mode is required when multiple modes are defined; choose: {choices}"
                )
        # The mode may have been inferred from a single-entry map only after
        # the initial selector pass above.  Re-run the semantic routing check
        # here so a base annotation.profile=contact_press/door_open cannot
        # silently inherit a generic target_only overlay in custom profiles.
        if requested_mode == AnnotationMode.TARGET_ONLY.value and requested_target_profile:
            semantic_mode = _TARGET_PROFILE_MODE_KEYS.get(requested_target_profile)
            if semantic_mode is not None and semantic_mode in mode_map:
                requested_mode = semantic_mode
            elif requested_target_profile in _TARGET_PROFILE_MODE_KEYS:
                raise ConfigError(
                    f"target profile {requested_target_profile!r} requires an explicit "
                    f"modes.{semantic_mode} profile overlay"
                )
        if requested_mode not in mode_map:
            if requested_mode in _TARGET_PROFILE_SELECTORS:
                raise ConfigError(
                    f"{requested_mode} requires an explicit "
                    f"modes.{requested_mode} profile overlay"
                )
            choices = ", ".join(str(key) for key in mode_map)
            raise ConfigError(
                f"unknown profile mode {requested_mode!r}; choose one of: {choices}"
            )
        selected = mode_map[requested_mode]
        if not isinstance(selected, Mapping):
            raise ConfigError(f"modes.{requested_mode} must be a mapping")
        resolved = _deep_merge(resolved, selected, field=f"modes.{requested_mode}")

        # A mode key is allowed to carry the mode declaration tersely.  The
        # target-only variants are profile selectors layered on one workflow.
        annotation = resolved.get("annotation")
        if annotation is None:
            annotation = {}
        if not isinstance(annotation, Mapping):
            raise ConfigError("annotation must be a mapping")
        annotation = dict(annotation)
        if "mode" not in annotation:
            if requested_mode in _TARGET_PROFILE_SELECTORS:
                annotation["mode"] = AnnotationMode.TARGET_ONLY.value
            elif requested_mode in {item.value for item in AnnotationMode}:
                annotation["mode"] = requested_mode
            else:
                raise ConfigError(
                    f"modes.{requested_mode} must declare annotation.mode"
                )
        resolved["annotation"] = annotation

        # A custom ``target_only`` overlay may carry a semantic profile of its
        # own.  Do not let that declaration inherit whichever generic prompt
        # fields happened to be placed in the overlay: semantic profiles have
        # dedicated mode keys and must be selected through those keys.  This
        # check runs after the merge because the profile may be declared only
        # by the selected overlay (and therefore was unknown during the first
        # routing pass above).
        overlay_profile = _normalise_target_profile(annotation.get("profile"))
        if overlay_profile is not None:
            if (
                requested_target_profile is not None
                and overlay_profile != requested_target_profile
            ):
                raise ConfigError(
                    f"target profile conflict: selected overlay declares "
                    f"{overlay_profile!r}, expected {requested_target_profile!r}"
                )
            requested_target_profile = overlay_profile
            semantic_mode = _TARGET_PROFILE_MODE_KEYS.get(overlay_profile)
            if (
                semantic_mode is not None
                and overlay_profile != TargetProfile.GRASP_MANIPULATION.value
                and requested_mode != semantic_mode
            ):
                raise ConfigError(
                    f"modes.{requested_mode} declares target profile "
                    f"{overlay_profile!r}; select modes.{semantic_mode} "
                    "to use its dedicated prompt overlay"
                )
    elif requested_mode is not None:
        if requested_mode in _TARGET_PROFILE_SELECTORS:
            # Semantic prompts are supplied by dedicated mode overlays.
            # Falling back to a mode-less document would silently run generic
            # prompts while advertising a different target profile.
            raise ConfigError(
                f"{requested_mode} requires an explicit "
                f"modes.{requested_mode} profile overlay"
            )
        annotation = resolved.get("annotation", {})
        if not isinstance(annotation, Mapping):
            raise ConfigError("annotation must be a mapping")
        annotation = dict(annotation)
        # Mode-less profiles can still use a terse annotation mode selector.
        annotation.setdefault(
            "mode",
            requested_mode,
        )
        resolved["annotation"] = annotation
    elif require_semantic_overlay and requested_target_profile in _TARGET_PROFILE_MODE_KEYS:
        # A semantic target profile without a mode map has no safe way to
        # obtain its dedicated prompt/QC bundle.
        semantic_mode = _TARGET_PROFILE_MODE_KEYS[requested_target_profile]
        raise ConfigError(
            f"target profile {requested_target_profile!r} requires an explicit "
            f"modes.{semantic_mode} profile overlay"
        )

    annotation_value = resolved.get("annotation")
    if annotation_value is not None and not isinstance(annotation_value, Mapping):
        raise ConfigError("annotation must be a mapping")
    if requested_mode is not None and isinstance(annotation_value, Mapping):
        actual_mode = annotation_value.get("mode")
        # ``contact_press`` is a mode-map key, not an AnnotationMode value.
        expected_annotation_mode = (
            AnnotationMode.TARGET_ONLY.value
            if requested_mode in _TARGET_PROFILE_SELECTORS
            else requested_mode
        )
        if actual_mode is not None and actual_mode != expected_annotation_mode:
            raise ConfigError(
                f"profile mode {requested_mode!r} conflicts with annotation.mode "
                f"{actual_mode!r}"
            )

    if requested_target_profile is not None:
        annotation = dict(resolved.get("annotation", {}))
        actual_profile = annotation.get("profile")
        actual_profile = _normalise_target_profile(actual_profile)
        if actual_profile is not None and actual_profile != requested_target_profile:
            raise ConfigError(
                f"target profile {requested_target_profile!r} conflicts with "
                f"annotation.profile {actual_profile!r}"
            )
        selected_mode_profile = _TARGET_PROFILE_SELECTORS.get(requested_mode or "")
        if selected_mode_profile is not None and actual_profile != selected_mode_profile:
            raise ConfigError(
                f"modes.{requested_mode} must declare profile "
                f"{selected_mode_profile!r}"
            )
        annotation["profile"] = requested_target_profile
        resolved["annotation"] = annotation
    return resolved


@dataclass(frozen=True)
class QwenConfig:
    endpoint: str
    model: str
    prompt_template: Path
    runtime: str = "local"
    api_key_env: str | None = None
    probe: str = "health"
    temperature: float = 0.0
    enable_thinking: bool = False
    timeout_seconds: float = 180.0
    max_tokens: int = 800
    query_selection: str = "first_recommended"
    allow_query_fallback: bool = False

    def __post_init__(self) -> None:
        if self.runtime not in {"api", "local"}:
            raise ConfigError("qwen.runtime must be api or local")
        if not self.endpoint.strip() or not self.model.strip():
            raise ConfigError("qwen.endpoint and qwen.model must be non-empty")
        expected_probe = "models" if self.runtime == "api" else "health"
        if self.probe != expected_probe:
            raise ConfigError(f"qwen.runtime={self.runtime} requires qwen.probe={expected_probe}")
        if self.runtime == "api":
            if not self.api_key_env:
                raise ConfigError("qwen.api_key_env is required for API runtime")
            if urlsplit(self.endpoint).scheme != "https":
                raise ConfigError("qwen.runtime=api requires an HTTPS endpoint")
        elif self.api_key_env is not None:
            raise ConfigError("qwen.api_key_env is only supported for API runtime")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ConfigError("qwen.temperature must be a finite non-negative number")
        if not isinstance(self.enable_thinking, bool):
            raise ConfigError("qwen.enable_thinking must be a boolean")
        if self.query_selection != "first_recommended":
            raise ConfigError("only query_selection=first_recommended is supported")
        if self.allow_query_fallback:
            raise ConfigError("automatic query fallback is disabled in this experiment")
        if self.timeout_seconds <= 0 or self.max_tokens < 1:
            raise ConfigError("Qwen timeout and max_tokens must be positive")


@dataclass(frozen=True)
class Sam3Config:
    checkpoint: Path
    gpus: tuple[int, ...] = (0,)

    def __post_init__(self) -> None:
        if len(self.gpus) != 1:
            raise ConfigError("sam3.gpus must contain exactly one GPU")
        if any(gpu < 0 for gpu in self.gpus):
            raise ConfigError("sam3.gpus must contain non-negative integers")


@dataclass(frozen=True)
class ParallelConfig:
    """Opt-in process parallelism; an empty GPU tuple preserves serial execution."""

    sam_worker_gpus: tuple[int, ...] = ()
    qwen_max_in_flight: int = 1

    def __post_init__(self) -> None:
        _validate_gpu_list(self.sam_worker_gpus, field="parallel.sam_worker_gpus")
        _integer(
            self.qwen_max_in_flight,
            field="parallel.qwen_max_in_flight",
            minimum=1,
        )

    @property
    def enabled(self) -> bool:
        return bool(self.sam_worker_gpus)


@dataclass(frozen=True)
class MaskConfig:
    target_envelope_padding_px: int = 4
    receiver_envelope_padding_px: int = 4
    temporal_qc_min_adjacent_iou_p05: float = 0.5
    temporal_qc_max_centroid_jump_p95_px: float = 5.0
    temporal_qc_max_area_ratio_jump_p95: float = 0.4
    temporal_qc_quarantine_signal_count: int = 2
    qc_enabled: bool = False
    qc_prompt_template: Path | None = None
    qc_max_candidates: int = 3
    qc_query_fallback_enabled: bool = False
    qc_seed_fallback_enabled: bool = False
    qc_bbox_fallback_enabled: bool = False
    qc_bbox_prompt_template: Path | None = None
    qc_bbox_max_tokens: int = 180
    qc_max_tokens: int = 160
    qc_max_attempts: int = 2
    qc_min_confidence: float = 0.70
    qc_min_area_fraction: float = 0.0001
    qc_max_area_fraction: float = 0.85
    qc_duplicate_iou_threshold: float = 0.98

    def __post_init__(self) -> None:
        if self.target_envelope_padding_px < 0 or self.receiver_envelope_padding_px < 0:
            raise ConfigError("mask envelope padding must be non-negative")
        if not 0.0 <= self.temporal_qc_min_adjacent_iou_p05 <= 1.0:
            raise ConfigError("temporal QC minimum adjacent IoU must be in [0, 1]")
        if self.temporal_qc_max_centroid_jump_p95_px <= 0:
            raise ConfigError("temporal QC maximum centroid jump must be positive")
        if self.temporal_qc_max_area_ratio_jump_p95 <= 0:
            raise ConfigError("temporal QC maximum area-ratio jump must be positive")
        if not 1 <= self.temporal_qc_quarantine_signal_count <= 3:
            raise ConfigError("temporal QC quarantine signal count must be in [1, 3]")
        if self.qc_enabled and self.qc_prompt_template is None:
            raise ConfigError("mask.qc_prompt_template is required when QC is enabled")
        if self.qc_query_fallback_enabled and not self.qc_enabled:
            raise ConfigError("mask.qc_query_fallback_enabled requires mask QC")
        if self.qc_seed_fallback_enabled and not self.qc_enabled:
            raise ConfigError("mask.qc_seed_fallback_enabled requires mask QC")
        if self.qc_bbox_fallback_enabled and not self.qc_enabled:
            raise ConfigError("mask.qc_bbox_fallback_enabled requires mask QC")
        if self.qc_bbox_fallback_enabled and self.qc_bbox_prompt_template is None:
            raise ConfigError(
                "mask.qc_bbox_prompt_template is required when bbox fallback is enabled"
            )
        if self.qc_max_candidates < 1:
            raise ConfigError("mask.qc_max_candidates must be positive")
        if self.qc_bbox_max_tokens < 1:
            raise ConfigError("mask.qc_bbox_max_tokens must be positive")
        if self.qc_max_tokens < 1:
            raise ConfigError("mask.qc_max_tokens must be positive")
        if self.qc_max_attempts < 1:
            raise ConfigError("mask.qc_max_attempts must be positive")
        if not 0.0 <= self.qc_min_confidence <= 1.0:
            raise ConfigError("mask.qc_min_confidence must be between 0 and 1")
        if not 0.0 < self.qc_min_area_fraction <= self.qc_max_area_fraction <= 1.0:
            raise ConfigError("mask QC area fractions must satisfy 0 < min <= max <= 1")
        if not 0.0 <= self.qc_duplicate_iou_threshold <= 1.0:
            raise ConfigError("mask.qc_duplicate_iou_threshold must be between 0 and 1")


def _validate_target_profile_resolution_contract(
    annotation: AnnotationConfig,
    mask: MaskConfig,
) -> None:
    """Keep the door-open object-mask resolver on the complete S1-S3 path."""

    if annotation.profile is not TargetProfile.DOOR_OPEN:
        return
    required = {
        "mask.qc_enabled": mask.qc_enabled,
        "mask.qc_query_fallback_enabled": mask.qc_query_fallback_enabled,
        "mask.qc_seed_fallback_enabled": mask.qc_seed_fallback_enabled,
        "mask.qc_bbox_fallback_enabled": mask.qc_bbox_fallback_enabled,
    }
    disabled = [name for name, enabled in required.items() if not enabled]
    if disabled:
        raise ConfigError(
            "annotation.profile=door_open requires the complete S1-S3 object-mask "
            f"resolution contract; disabled setting(s): {', '.join(disabled)}"
        )


@dataclass(frozen=True)
class GripperRoiConfig:
    """Fixed prompt and final-crop geometry for gripper mask generation."""

    prompt_axial_back_m: float
    prompt_axial_front_m: float
    hard_axial_back_m: float
    hard_axial_front_m: float
    fixed_half_width_m: float

    def __post_init__(self) -> None:
        values = {
            "prompt_axial_back_m": self.prompt_axial_back_m,
            "prompt_axial_front_m": self.prompt_axial_front_m,
            "hard_axial_back_m": self.hard_axial_back_m,
            "hard_axial_front_m": self.hard_axial_front_m,
            "fixed_half_width_m": self.fixed_half_width_m,
        }
        for name, value in values.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ConfigError(f"gripper_roi.{name} must be a finite number greater than zero")


@dataclass(frozen=True)
class AnnotationConfig:
    """Task-declared semantic mode and its resolved pipeline behavior."""

    mode: AnnotationMode
    profile: TargetProfile = TargetProfile.GRASP_MANIPULATION

    def __post_init__(self) -> None:
        if self.profile in {
            TargetProfile.CONTACT_PRESS,
            TargetProfile.DOOR_OPEN,
        } and self.mode is not AnnotationMode.TARGET_ONLY:
            raise ConfigError(
                f"annotation.profile={self.profile.value} requires target_only mode"
            )

    @property
    def spec(self) -> AnnotationSpec:
        return annotation_spec(self.mode)


@dataclass(frozen=True)
class PipelineConfig:
    config_path: Path
    dataset: DatasetConfig
    qwen: QwenConfig
    sam3: Sam3Config
    mask: MaskConfig
    gripper_roi: GripperRoiConfig
    output_root: Path
    annotation: AnnotationConfig = AnnotationConfig(AnnotationMode.PICK_PLACE)
    parallel: ParallelConfig = ParallelConfig()

    def __post_init__(self) -> None:
        if self.parallel.enabled and self.qwen.runtime != "api":
            raise ConfigError("parallel SAM workers require qwen.runtime=api")
        _validate_target_profile_resolution_contract(self.annotation, self.mask)


@dataclass(frozen=True)
class PipelineProfile:
    """Reusable algorithm/runtime configuration without dataset identity.

    A profile can be shared by any number of tasks.  Bind a concrete dataset
    at the CLI/application boundary with :func:`bind_dataset` before passing
    it to existing workflows that consume :class:`PipelineConfig`.
    """

    config_path: Path
    qwen: QwenConfig
    sam3: Sam3Config
    mask: MaskConfig
    gripper_roi: GripperRoiConfig
    output_root: Path
    annotation: AnnotationConfig = AnnotationConfig(AnnotationMode.PICK_PLACE)
    parallel: ParallelConfig = ParallelConfig()

    def __post_init__(self) -> None:
        if self.parallel.enabled and self.qwen.runtime != "api":
            raise ConfigError("parallel SAM workers require qwen.runtime=api")
        _validate_target_profile_resolution_contract(self.annotation, self.mask)

    def bind_dataset(self, binding: DatasetBinding) -> PipelineConfig:
        """Return a legacy-compatible, dataset-bound pipeline configuration."""

        return bind_dataset(self, binding)


def _load_config_impl(
    path: Path,
    *,
    raw_override: dict[str, Any] | None = None,
    require_dataset: bool = True,
    parse_dataset: bool | None = None,
) -> PipelineConfig | PipelineProfile:
    """Parse one config mapping into a bound config or reusable profile."""

    config_path = path.expanduser().resolve()
    if not config_path.is_file():
        raise ConfigError(f"config file does not exist: {config_path}")
    if raw_override is None:
        with config_path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
    else:
        raw = raw_override
    if not isinstance(raw, dict):
        raise ConfigError("top-level config must be a mapping")
    base_dir = config_path.parent

    if parse_dataset is None:
        parse_dataset = require_dataset
    dataset_value = raw.get("dataset") if parse_dataset else None
    if dataset_value is None and require_dataset:
        dataset_value = _required(raw, "dataset")
    dataset_raw = {} if dataset_value is None else dataset_value
    qwen_raw = _required(raw, "qwen")
    sam3_raw = _required(raw, "sam3")
    mask_raw = raw.get("mask", {})
    parallel_raw = raw.get("parallel", {})
    gripper_roi_raw = _required(raw, "gripper_roi")
    output_raw = raw.get("output", {})
    annotation_raw = raw.get("annotation", {"mode": AnnotationMode.PICK_PLACE.value})
    sections = (
        dataset_raw,
        qwen_raw,
        sam3_raw,
        mask_raw,
        parallel_raw,
        gripper_roi_raw,
        output_raw,
        annotation_raw,
    )
    if not all(isinstance(item, dict) for item in sections):
        raise ConfigError(
            "dataset, qwen, sam3, mask, parallel, gripper_roi, output and annotation "
            "must be mappings"
        )

    raw_mode = _string(
        annotation_raw.get("mode", AnnotationMode.PICK_PLACE.value),
        field="annotation.mode",
    )
    raw_profile = annotation_raw.get(
        "profile",
        TargetProfile.GRASP_MANIPULATION.value,
    )
    raw_profile = _normalise_target_profile(raw_profile)
    if raw_profile is None:
        raise ConfigError("annotation.profile must be a non-empty string")
    try:
        annotation_mode = AnnotationMode(raw_mode)
    except (TypeError, ValueError) as exc:
        choices = ", ".join(mode.value for mode in AnnotationMode)
        raise ConfigError(f"annotation.mode must be one of: {choices}") from exc
    try:
        target_profile = TargetProfile(raw_profile)
    except (TypeError, ValueError) as exc:
        choices = ", ".join(profile.value for profile in TargetProfile)
        raise ConfigError(f"annotation.profile must be one of: {choices}") from exc
    annotation = AnnotationConfig(annotation_mode, target_profile)

    prompt_roi_raw = _required(gripper_roi_raw, "prompt", section="gripper_roi")
    hard_roi_raw = _required(gripper_roi_raw, "hard", section="gripper_roi")
    if not isinstance(prompt_roi_raw, dict) or not isinstance(hard_roi_raw, dict):
        raise ConfigError("gripper_roi.prompt and gripper_roi.hard must be mappings")

    dataset: DatasetConfig | None = None
    if dataset_value is not None:
        # Validate every regression-selection alias before choosing one.  This
        # keeps legacy ``episode_indices`` and newer names interchangeable
        # without allowing contradictory values to be hidden by precedence.
        regression = _manifest_episode_selection(
            dataset_raw,
            field_prefix="dataset",
        )
        if regression is None:
            regression = _binding_episode_ids(
                _required(dataset_raw, "regression_episode_ids", section="dataset"),
                field="dataset.regression_episode_ids",
            )
        smoke = _binding_episode_ids(
            _required(dataset_raw, "smoke_episode_ids", section="dataset"),
            field="dataset.smoke_episode_ids",
        )
        if not smoke or not regression or not set(smoke).issubset(regression):
            raise ConfigError("smoke episodes must be non-empty and included in regression episodes")

        dataset = DatasetConfig(
            root=_path(
                _required(dataset_raw, "root", section="dataset"),
                base_dir=base_dir,
                field="dataset.root",
            ),
            manifest=_path(
                _required(dataset_raw, "manifest", section="dataset"),
                base_dir=base_dir,
                field="dataset.manifest",
            ),
            task=validate_dataset_component(
                _required(dataset_raw, "task", section="dataset"),
                field="dataset.task",
            ),
            camera=validate_dataset_component(
                _required(dataset_raw, "camera", section="dataset"),
                field="dataset.camera",
            ),
            smoke_episode_ids=smoke,
            regression_episode_ids=regression,
        )
    qwen_runtime = _string(qwen_raw.get("runtime", "local"), field="qwen.runtime")
    api_key_env_raw = qwen_raw.get("api_key_env")
    api_key_env = (
        _string(api_key_env_raw, field="qwen.api_key_env")
        if api_key_env_raw is not None
        else None
    )
    enable_thinking = _strict_bool(
        qwen_raw.get("enable_thinking", False),
        field="qwen.enable_thinking",
    )
    qwen = QwenConfig(
        endpoint=_string(
            _required(qwen_raw, "endpoint", section="qwen"),
            field="qwen.endpoint",
        ),
        model=_string(
            _required(qwen_raw, "model", section="qwen"),
            field="qwen.model",
        ),
        prompt_template=_path(
            _required(qwen_raw, "prompt_template", section="qwen"),
            base_dir=base_dir,
            field="qwen.prompt_template",
        ),
        runtime=qwen_runtime,
        api_key_env=api_key_env,
        probe=_string(
            qwen_raw.get("probe", "models" if qwen_runtime == "api" else "health"),
            field="qwen.probe",
        ),
        temperature=_strict_float(
            qwen_raw.get("temperature", 0.0),
            field="qwen.temperature",
        ),
        enable_thinking=enable_thinking,
        timeout_seconds=_strict_float(
            qwen_raw.get("timeout_seconds", 180.0),
            field="qwen.timeout_seconds",
        ),
        max_tokens=_integer(
            qwen_raw.get("max_tokens", 800),
            field="qwen.max_tokens",
            minimum=1,
        ),
        query_selection=_string(
            qwen_raw.get("query_selection", "first_recommended"),
            field="qwen.query_selection",
        ),
        allow_query_fallback=_strict_bool(
            qwen_raw.get("allow_query_fallback", False),
            field="qwen.allow_query_fallback",
        ),
    )
    sam3 = Sam3Config(
        checkpoint=_path(
            _required(sam3_raw, "checkpoint", section="sam3"),
            base_dir=base_dir,
            field="sam3.checkpoint",
        ),
        gpus=_integers(sam3_raw.get("gpus", [0]), field="sam3.gpus"),
    )
    parallel = ParallelConfig(
        sam_worker_gpus=_gpu_ids(
            parallel_raw.get("sam_worker_gpus", []),
            field="parallel.sam_worker_gpus",
        ),
        qwen_max_in_flight=_integer(
            parallel_raw.get("qwen_max_in_flight", 1),
            field="parallel.qwen_max_in_flight",
            minimum=1,
        ),
    )
    qc_enabled = _strict_bool(
        mask_raw.get("qc_enabled", False),
        field="mask.qc_enabled",
    )
    removed_s4_fields = sorted(_REMOVED_S4_MASK_FIELDS & mask_raw.keys())
    if removed_s4_fields:
        raise ConfigError(f"removed S4 mask fields are not supported: {removed_s4_fields}")
    qc_template_value = mask_raw.get(
        "qc_prompt_template",
        "prompts/mask_candidate_qc.txt" if qc_enabled else None,
    )
    qc_template = (
        _path(
            qc_template_value,
            base_dir=base_dir,
            field="mask.qc_prompt_template",
        )
        if qc_template_value is not None
        else None
    )
    qc_bbox_enabled = _strict_bool(
        mask_raw.get("qc_bbox_fallback_enabled", False),
        field="mask.qc_bbox_fallback_enabled",
    )
    qc_bbox_template_value = mask_raw.get("qc_bbox_prompt_template")
    qc_bbox_template = (
        _path(
            qc_bbox_template_value,
            base_dir=base_dir,
            field="mask.qc_bbox_prompt_template",
        )
        if qc_bbox_template_value is not None
        else None
    )
    mask = MaskConfig(
        target_envelope_padding_px=_integer(
            mask_raw.get("target_envelope_padding_px", 4),
            field="mask.target_envelope_padding_px",
            minimum=0,
        ),
        receiver_envelope_padding_px=_integer(
            mask_raw.get("receiver_envelope_padding_px", 4),
            field="mask.receiver_envelope_padding_px",
            minimum=0,
        ),
        temporal_qc_min_adjacent_iou_p05=_strict_float(
            mask_raw.get("temporal_qc_min_adjacent_iou_p05", 0.5),
            field="mask.temporal_qc_min_adjacent_iou_p05",
        ),
        temporal_qc_max_centroid_jump_p95_px=_strict_float(
            mask_raw.get("temporal_qc_max_centroid_jump_p95_px", 5.0),
            field="mask.temporal_qc_max_centroid_jump_p95_px",
        ),
        temporal_qc_max_area_ratio_jump_p95=_strict_float(
            mask_raw.get("temporal_qc_max_area_ratio_jump_p95", 0.4),
            field="mask.temporal_qc_max_area_ratio_jump_p95",
        ),
        temporal_qc_quarantine_signal_count=_integer(
            mask_raw.get("temporal_qc_quarantine_signal_count", 2),
            field="mask.temporal_qc_quarantine_signal_count",
            minimum=1,
        ),
        qc_enabled=qc_enabled,
        qc_prompt_template=qc_template,
        qc_max_candidates=_integer(
            mask_raw.get("qc_max_candidates", 3),
            field="mask.qc_max_candidates",
            minimum=1,
        ),
        qc_query_fallback_enabled=_strict_bool(
            mask_raw.get("qc_query_fallback_enabled", False),
            field="mask.qc_query_fallback_enabled",
        ),
        qc_seed_fallback_enabled=_strict_bool(
            mask_raw.get("qc_seed_fallback_enabled", False),
            field="mask.qc_seed_fallback_enabled",
        ),
        qc_bbox_fallback_enabled=qc_bbox_enabled,
        qc_bbox_prompt_template=qc_bbox_template,
        qc_bbox_max_tokens=_integer(
            mask_raw.get("qc_bbox_max_tokens", 180),
            field="mask.qc_bbox_max_tokens",
            minimum=1,
        ),
        qc_max_tokens=_integer(
            mask_raw.get("qc_max_tokens", 160),
            field="mask.qc_max_tokens",
            minimum=1,
        ),
        qc_max_attempts=_integer(
            mask_raw.get("qc_max_attempts", 2),
            field="mask.qc_max_attempts",
            minimum=1,
        ),
        qc_min_confidence=_strict_float(
            mask_raw.get("qc_min_confidence", 0.70),
            field="mask.qc_min_confidence",
        ),
        qc_min_area_fraction=_strict_float(
            mask_raw.get("qc_min_area_fraction", 0.0001),
            field="mask.qc_min_area_fraction",
        ),
        qc_max_area_fraction=_strict_float(
            mask_raw.get("qc_max_area_fraction", 0.85),
            field="mask.qc_max_area_fraction",
        ),
        qc_duplicate_iou_threshold=_strict_float(
            mask_raw.get("qc_duplicate_iou_threshold", 0.98),
            field="mask.qc_duplicate_iou_threshold",
        ),
    )
    gripper_roi = GripperRoiConfig(
        prompt_axial_back_m=_positive_float(
            _required(prompt_roi_raw, "axial_back_m", section="gripper_roi.prompt"),
            field="gripper_roi.prompt.axial_back_m",
        ),
        prompt_axial_front_m=_positive_float(
            _required(prompt_roi_raw, "axial_front_m", section="gripper_roi.prompt"),
            field="gripper_roi.prompt.axial_front_m",
        ),
        hard_axial_back_m=_positive_float(
            _required(hard_roi_raw, "axial_back_m", section="gripper_roi.hard"),
            field="gripper_roi.hard.axial_back_m",
        ),
        hard_axial_front_m=_positive_float(
            _required(hard_roi_raw, "axial_front_m", section="gripper_roi.hard"),
            field="gripper_roi.hard.axial_front_m",
        ),
        fixed_half_width_m=_positive_float(
            _required(gripper_roi_raw, "fixed_half_width_m", section="gripper_roi"),
            field="gripper_roi.fixed_half_width_m",
        ),
    )
    output_root = _path(
        output_raw.get("root", "../artifacts/runs"),
        base_dir=base_dir,
        field="output.root",
    )
    profile = PipelineProfile(
        config_path=config_path,
        qwen=qwen,
        sam3=sam3,
        mask=mask,
        gripper_roi=gripper_roi,
        output_root=output_root,
        annotation=annotation,
        parallel=parallel,
    )
    if dataset is None:
        return profile
    return PipelineConfig(
        config_path=config_path,
        dataset=dataset,
        qwen=profile.qwen,
        sam3=profile.sam3,
        mask=profile.mask,
        gripper_roi=profile.gripper_roi,
        output_root=profile.output_root,
        annotation=profile.annotation,
        parallel=profile.parallel,
    )


def _read_yaml_document(path: Path | str) -> dict[str, Any]:
    """Read one YAML document and normalize the top-level mapping contract."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigError(f"config file does not exist: {config_path}")
    try:
        with config_path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
    except OSError as exc:
        raise ConfigError(f"cannot read config file: {config_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"cannot parse config file: {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError("top-level config must be a mapping")
    return raw


def has_dataset_block(path: Path | str) -> bool:
    """Return whether a YAML document has a top-level legacy ``dataset`` block.

    The path-oriented CLI uses this predicate to decide whether it is safe to
    try the compatibility ``load_config`` parser after ``load_profile`` fails.
    A malformed or unreadable document returns ``False`` so the original
    profile error remains visible instead of being masked by a second parser
    attempt.
    """

    try:
        return "dataset" in _read_yaml_document(path)
    except ConfigError:
        return False


def load_profile(
    path: Path | str,
    *,
    mode: AnnotationMode | str | None = None,
    target_profile: TargetProfile | str | None = None,
) -> PipelineProfile:
    """Load a reusable profile without requiring task-specific dataset fields.

    A profile document may use ``defaults`` (or ``common``) plus a ``modes``
    mapping.  The selected mode is resolved in memory; no task JSON/YAML is
    generated.  ``contact_press`` is a valid mode key and resolves to the
    target-only annotation mode with its dedicated prompts.
    """

    config_path = Path(path).expanduser().resolve()
    raw = _read_yaml_document(config_path)
    # Keep the profile API honest: a profile is reusable algorithm state and
    # must not silently absorb task identity.  Without this guard a legacy
    # task-bound YAML would be accepted as a profile (the parser intentionally
    # ignores its dataset block), making it too easy to reintroduce one config
    # file per task.  ``load_config`` remains the explicit compatibility path
    # for those files.
    dataset_block = _find_dataset_block(raw)
    if dataset_block is not None:
        raise ConfigError(
            "profile config must not contain a dataset block "
            f"({dataset_block}); use load_config() for legacy task-bound configs"
        )
    resolved = _resolve_profile_document(
        raw,
        mode=mode,
        target_profile=target_profile,
    )
    # A nested common/defaults/mode section can carry the same accidental
    # coupling even when the top-level document does not.  Reject it after
    # overlay resolution as well, before the generic parser drops it.
    dataset_block = _find_dataset_block(resolved)
    if dataset_block is not None:
        raise ConfigError(
            "profile config must not contain a dataset block "
            f"({dataset_block}); move dataset identity to DatasetBinding"
        )
    parsed = _load_config_impl(
        config_path,
        raw_override=resolved,
        require_dataset=False,
        parse_dataset=False,
    )
    if not isinstance(parsed, PipelineProfile):
        raise ConfigError("profile parser unexpectedly returned a bound config")
    return parsed


def load_config(
    path: Path | str,
    *,
    mode: AnnotationMode | str | None = None,
    target_profile: TargetProfile | str | None = None,
) -> PipelineConfig:
    """Load a legacy dataset-bound config.

    Existing task YAML files remain supported.  For the shared profile-only
    document, callers should use :func:`load_profile` and bind a dataset at
    runtime; accepting it here would re-introduce the per-task configuration
    coupling this API is intended to remove.
    """

    config_path = Path(path).expanduser().resolve()
    raw = _read_yaml_document(config_path)
    resolved = _resolve_profile_document(
        raw,
        mode=mode,
        target_profile=target_profile,
        # Legacy task-bound files may carry a direct semantic annotation block
        # without a reusable ``modes`` map.  Their prompts are already
        # task-specific, so retain that compatibility path; load_profile()
        # keeps the stricter dedicated-overlay contract.
        require_semantic_overlay=False,
    )
    parsed = _load_config_impl(
        config_path,
        raw_override=resolved,
        require_dataset=True,
        parse_dataset=True,
    )
    if not isinstance(parsed, PipelineConfig):
        raise ConfigError(
            "config has no dataset block; load it with load_profile() and call "
            "bind_dataset()"
        )
    return parsed


def bind_dataset(
    profile: PipelineProfile,
    binding: DatasetBinding,
) -> PipelineConfig:
    """Attach runtime dataset identity and episode selection to a profile.

    ``binding`` is intentionally small and can be produced from a manifest,
    path discovery, or a caller-owned selection.  Its manifest path/data are
    retained for provenance, while the bound root is authoritative for file
    access.  The returned object is the existing ``PipelineConfig`` shape, so
    downstream workflows need no second task-specific configuration file.
    """

    if not isinstance(profile, PipelineProfile):
        raise TypeError("profile must be a PipelineProfile")
    if not isinstance(binding, DatasetBinding):
        raise TypeError("binding must be a DatasetBinding")

    regression = binding.regression_episode_ids
    if regression is None:
        regression = binding.episode_ids

    manifest_path = binding.manifest_path
    if manifest_path is None:
        manifest_path = binding.root / "EXTRACT_MANIFEST.json"

    # Prefer an explicitly supplied in-memory manifest.  If one is not
    # available, read an existing extract manifest so stale absolute roots can
    # be normalized before the adapter sees it.  A missing file remains valid
    # for native directories whose metadata is discovered at runtime.
    manifest_data = (
        None
        if binding.manifest_data is None
        else copy.deepcopy(dict(binding.manifest_data))
    )
    if manifest_data is None and manifest_path.is_file():
        try:
            with manifest_path.open(encoding="utf-8") as handle:
                loaded_manifest = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigError(f"cannot read dataset manifest: {manifest_path}: {exc}") from exc
        if not isinstance(loaded_manifest, dict):
            raise ConfigError(f"dataset manifest must be a mapping: {manifest_path}")
        manifest_data = copy.deepcopy(loaded_manifest)
    if manifest_data is not None:
        _validate_binding_manifest_identity(
            manifest_data,
            task=binding.task,
            camera=binding.camera,
        )
        manifest_regression = _manifest_episode_selection(manifest_data)
        declared_profile = _normalise_annotation_profile(manifest_data.get("profile"))
        if declared_profile is not None and declared_profile != profile.annotation.mode.value:
            raise ConfigError(
                "dataset binding profile differs from annotation mode: "
                f"{declared_profile!r} != {profile.annotation.mode.value!r}"
            )
        declared_target_profile = _manifest_target_profile(manifest_data)
    else:
        manifest_regression = None
        declared_target_profile = None

    binding_target_profile = None
    if binding.target_profile is not None:
        normalized_binding_profile = _normalise_target_profile(binding.target_profile)
        if normalized_binding_profile is None:  # pragma: no cover - guarded above
            raise ConfigError("dataset binding target_profile must be a non-empty string")
        try:
            binding_target_profile = TargetProfile(normalized_binding_profile)
        except (TypeError, ValueError) as exc:
            choices = ", ".join(profile.value for profile in TargetProfile)
            raise ConfigError(
                f"dataset binding target_profile must be one of: {choices}"
            ) from exc
    if (
        declared_target_profile is not None
        and binding_target_profile is not None
        and declared_target_profile is not binding_target_profile
    ):
        raise ConfigError(
            "dataset binding target_profile differs from manifest target_profile"
        )
    selected_target_profile = binding_target_profile or declared_target_profile
    if regression is None:
        regression = manifest_regression
    if regression is None or not regression:
        raise ConfigError(
            "dataset binding requires at least one regression episode; provide "
            "episode_ids or manifest episode_indices"
        )

    smoke = binding.smoke_episode_ids
    if smoke is None and manifest_data is not None:
        raw_smoke = manifest_data.get("smoke_episode_ids")
        if raw_smoke is not None:
            smoke = _binding_episode_ids(raw_smoke, field="manifest.smoke_episode_ids")
    if smoke is None:
        smoke = (regression[0],)
    if not smoke:
        raise ConfigError("dataset binding requires at least one smoke episode")
    if not set(smoke).issubset(regression):
        raise ConfigError(
            "dataset binding smoke episodes must be included in regression episodes"
        )

    # Native RoboTwin directories may legitimately have no extract manifest.
    # Downstream adapters still consume the small manifest contract (most
    # importantly ``regression_episode_ids``), so provide it in memory after
    # episode selection is resolved.  Keeping this synthetic record on the
    # bound config avoids writing a task-specific JSON file while preserving
    # the same adapter API as converted extracts.  Content-derived fields such
    # as frame shape and video surplus are intentionally left absent; the
    # adapter owns their lazy inference during preflight/frame access.
    if manifest_data is None:
        manifest_data = {
            "format_version": "robotwin_dataset_manifest_bound_v1",
            "profile": profile.annotation.mode.value,
            "task": binding.task,
            "camera": binding.camera,
            "dataset_root": str(binding.root),
            "episode_indices": list(regression),
            "smoke_episode_ids": list(smoke),
            "regression_episode_ids": list(regression),
        }

    # ``DatasetBinding`` accepts the string form at its public boundary, but
    # normalize it here as well so the remainder of the binder has one typed
    # enum representation (and cannot accidentally call ``.value`` on a raw
    # string supplied by a lightweight caller).
    task_kind: TargetOnlyTaskKind | None = (
        None
        if binding.task_kind is None
        else TargetOnlyTaskKind(binding.task_kind)
    )
    declared_task_kind: TargetOnlyTaskKind | None = None
    if manifest_data is not None and manifest_data.get("task_kind") is not None:
        try:
            declared_task_kind = TargetOnlyTaskKind(manifest_data["task_kind"])
        except (TypeError, ValueError) as exc:
            choices = ", ".join(item.value for item in TargetOnlyTaskKind)
            raise ConfigError(
                f"dataset binding task_kind must be one of: {choices}"
            ) from exc
    if task_kind is None:
        task_kind = declared_task_kind
    elif declared_task_kind is not None and task_kind is not declared_task_kind:
        raise ConfigError(
            "dataset binding task_kind differs from manifest task_kind: "
            f"{task_kind.value} != {declared_task_kind.value}"
        )
    if task_kind is not None:
        required_profile = target_profile_for_task_kind(task_kind)
        if (
            profile.annotation.mode is not AnnotationMode.TARGET_ONLY
        ):
            raise ConfigError(
                f"{task_kind.value} task_kind requires target_only mode"
            )
        if (
            selected_target_profile is not None
            and selected_target_profile is not required_profile
        ):
            raise ConfigError(
                "dataset binding target_profile differs from task_kind-derived profile: "
                f"{selected_target_profile.value} != {required_profile.value}"
            )
        if profile.annotation.profile is not required_profile:
            raise ConfigError(
                f"{task_kind.value} requires a {required_profile.value} profile; load the "
                f"{required_profile.value} mode before binding"
            )
        selected_target_profile = required_profile
    if (
        selected_target_profile is not None
        and profile.annotation.profile is not selected_target_profile
    ):
        raise ConfigError(
            "dataset binding target_profile differs from annotation profile: "
            f"{selected_target_profile.value} != {profile.annotation.profile.value}"
        )
    # Runtime binding is authoritative for identity.  Keep all other manifest
    # provenance fields, but overwrite values consumed by RoboTwinDataset.
    if manifest_data is not None:
        manifest_data["dataset_root"] = str(binding.root)
        manifest_data["task"] = binding.task
        manifest_data["camera"] = binding.camera
        manifest_data["smoke_episode_ids"] = list(smoke)
        manifest_data["regression_episode_ids"] = list(regression)
        # Keep every historical spelling synchronized when an explicit
        # runtime selection narrows a manifest.  Leaving ``episode_ids`` (or
        # another alias) stale would let a downstream compatibility reader
        # resurrect the original, broader episode universe.
        for key in ("episode_indices", "regression_episode_ids", "episode_ids"):
            if key in manifest_data:
                manifest_data[key] = list(regression)
        if task_kind is not None:
            manifest_data["task_kind"] = (
                task_kind.value if isinstance(task_kind, TargetOnlyTaskKind) else str(task_kind)
            )
        # Do not add a new key to ordinary pick-place manifests.  For target
        # semantic variants (and explicit profile metadata), retain the
        # canonical profile so downstream routing and resume checks are
        # independent of the original alias spelling.
        if selected_target_profile is not None and (
            selected_target_profile is not TargetProfile.GRASP_MANIPULATION
            or manifest_data.get("target_profile") is not None
            or manifest_data.get("profile")
            in {"origin", "grasp_manipulation", "grasp-manipulation"}
            or task_kind is not None
        ):
            manifest_data["target_profile"] = selected_target_profile.value
    dataset = DatasetConfig(
        root=binding.root,
        manifest=manifest_path,
        task=binding.task,
        camera=binding.camera,
        smoke_episode_ids=tuple(smoke),
        regression_episode_ids=tuple(regression),
        manifest_data=manifest_data,
    )
    return PipelineConfig(
        config_path=profile.config_path,
        dataset=dataset,
        qwen=profile.qwen,
        sam3=profile.sam3,
        mask=profile.mask,
        gripper_roi=profile.gripper_roi,
        output_root=profile.output_root,
        annotation=profile.annotation,
        parallel=profile.parallel,
    )


__all__ = [
    "AnnotationConfig",
    "ConfigError",
    "DatasetBinding",
    "DatasetConfig",
    "GripperRoiConfig",
    "MaskConfig",
    "ParallelConfig",
    "PipelineConfig",
    "PipelineProfile",
    "QwenConfig",
    "Sam3Config",
    "bind_dataset",
    "has_dataset_block",
    "load_config",
    "load_profile",
    "parse_gpu_list",
    "validate_dataset_component",
]
