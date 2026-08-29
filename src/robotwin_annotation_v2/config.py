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
    if not isinstance(value, list) or any(isinstance(item, bool) for item in value):
        raise ConfigError(f"{field} must be a list of integers")
    try:
        return tuple(int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field} must be a list of integers") from exc


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


def _positive_float(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise ConfigError(f"{field} must be a finite number greater than zero")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
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
        if regression is None and manifest_data is not None:
            regression = _binding_episode_ids(
                _manifest_episode_values(manifest_data, "regression_episode_ids"),
                field="manifest.regression_episode_ids",
            )
        if smoke is None and not explicit_regression and manifest_data is not None:
            smoke = _binding_episode_ids(
                _manifest_episode_values(manifest_data, "smoke_episode_ids"),
                field="manifest.smoke_episode_ids",
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
    """Read an episode selection from either manifest spelling."""

    value = manifest.get(key)
    if value is None and key == "regression_episode_ids":
        value = manifest.get("episode_indices")
    if value is None and key == "regression_episode_ids":
        value = manifest.get("episode_ids")
    return value


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


def _resolve_profile_document(
    raw: Mapping[str, Any],
    *,
    mode: AnnotationMode | str | None = None,
    target_profile: TargetProfile | str | None = None,
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
    requested_target_profile = _normalise_selector(
        target_profile,
        field="target_profile",
    )
    if declared_target_profile is not None:
        declared_target_profile = _normalise_selector(
            declared_target_profile,
            field="target_profile",
        )
        if (
            requested_target_profile is not None
            and declared_target_profile != requested_target_profile
        ):
            raise ConfigError(
                f"target profile conflict: argument selects {requested_target_profile!r}, "
                f"YAML selects {declared_target_profile!r}"
            )
        requested_target_profile = declared_target_profile

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

    modes_value = raw.get("modes")
    if modes_value is not None and not isinstance(modes_value, Mapping):
        raise ConfigError("modes must be a mapping")
    mode_map = {} if modes_value is None else dict(modes_value)
    if mode_map:
        if requested_mode is None and requested_target_profile is not None:
            profile_matches: list[str] = []
            for key, candidate in mode_map.items():
                if not isinstance(candidate, Mapping):
                    continue
                candidate_annotation = candidate.get("annotation", {})
                candidate_profile = (
                    candidate_annotation.get("profile")
                    if isinstance(candidate_annotation, Mapping)
                    else None
                )
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
        if requested_mode not in mode_map:
            choices = ", ".join(str(key) for key in mode_map)
            raise ConfigError(
                f"unknown profile mode {requested_mode!r}; choose one of: {choices}"
            )
        selected = mode_map[requested_mode]
        if not isinstance(selected, Mapping):
            raise ConfigError(f"modes.{requested_mode} must be a mapping")
        resolved = _deep_merge(resolved, selected, field=f"modes.{requested_mode}")

        # A mode key is allowed to carry the mode declaration tersely.  The
        # contact-press key is a profile selector layered on target-only.
        annotation = resolved.get("annotation")
        if annotation is None:
            annotation = {}
        if not isinstance(annotation, Mapping):
            raise ConfigError("annotation must be a mapping")
        annotation = dict(annotation)
        if "mode" not in annotation:
            if requested_mode == "contact_press":
                annotation["mode"] = AnnotationMode.TARGET_ONLY.value
            elif requested_mode in {item.value for item in AnnotationMode}:
                annotation["mode"] = requested_mode
            else:
                raise ConfigError(
                    f"modes.{requested_mode} must declare annotation.mode"
                )
        resolved["annotation"] = annotation
    elif requested_mode is not None:
        if requested_mode == "contact_press":
            # Contact prompts are supplied by the dedicated mode overlay.
            # Falling back to a mode-less target-only document would silently
            # run the generic prompts while advertising the contact profile.
            raise ConfigError(
                "contact_press requires an explicit modes.contact_press profile overlay"
            )
        annotation = resolved.get("annotation", {})
        if not isinstance(annotation, Mapping):
            raise ConfigError("annotation must be a mapping")
        annotation = dict(annotation)
        # Mode-less profiles can still use a terse annotation mode selector.
        # Contact-press is intentionally excluded above because it needs its
        # dedicated prompt/mask overlay.
        annotation.setdefault(
            "mode",
            requested_mode,
        )
        resolved["annotation"] = annotation

    annotation_value = resolved.get("annotation")
    if annotation_value is not None and not isinstance(annotation_value, Mapping):
        raise ConfigError("annotation must be a mapping")
    if requested_mode is not None and isinstance(annotation_value, Mapping):
        actual_mode = annotation_value.get("mode")
        # ``contact_press`` is a mode-map key, not an AnnotationMode value.
        expected_annotation_mode = (
            AnnotationMode.TARGET_ONLY.value
            if requested_mode == "contact_press"
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
        if actual_profile is not None and actual_profile != requested_target_profile:
            raise ConfigError(
                f"target profile {requested_target_profile!r} conflicts with "
                f"annotation.profile {actual_profile!r}"
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
        if (
            self.profile is TargetProfile.CONTACT_PRESS
            and self.mode is not AnnotationMode.TARGET_ONLY
        ):
            raise ConfigError("annotation.profile=contact_press requires target_only mode")

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

    raw_mode = annotation_raw.get("mode", AnnotationMode.PICK_PLACE.value)
    raw_profile = annotation_raw.get(
        "profile",
        TargetProfile.GRASP_MANIPULATION.value,
    )
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
        smoke = _integers(
            _required(dataset_raw, "smoke_episode_ids", section="dataset"),
            field="dataset.smoke_episode_ids",
        )
        regression = _integers(
            _required(dataset_raw, "regression_episode_ids", section="dataset"),
            field="dataset.regression_episode_ids",
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
    qwen_runtime = str(qwen_raw.get("runtime", "local"))
    api_key_env_raw = qwen_raw.get("api_key_env")
    if api_key_env_raw is not None and not isinstance(api_key_env_raw, str):
        raise ConfigError("qwen.api_key_env must be a string or null")
    enable_thinking = qwen_raw.get("enable_thinking", False)
    if not isinstance(enable_thinking, bool):
        raise ConfigError("qwen.enable_thinking must be a boolean")
    qwen = QwenConfig(
        endpoint=str(_required(qwen_raw, "endpoint", section="qwen")),
        model=str(_required(qwen_raw, "model", section="qwen")),
        prompt_template=_path(
            _required(qwen_raw, "prompt_template", section="qwen"),
            base_dir=base_dir,
            field="qwen.prompt_template",
        ),
        runtime=qwen_runtime,
        api_key_env=api_key_env_raw,
        probe=str(qwen_raw.get("probe", "models" if qwen_runtime == "api" else "health")),
        temperature=float(qwen_raw.get("temperature", 0.0)),
        enable_thinking=enable_thinking,
        timeout_seconds=float(qwen_raw.get("timeout_seconds", 180.0)),
        max_tokens=int(qwen_raw.get("max_tokens", 800)),
        query_selection=str(qwen_raw.get("query_selection", "first_recommended")),
        allow_query_fallback=bool(qwen_raw.get("allow_query_fallback", False)),
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
    qc_enabled = bool(mask_raw.get("qc_enabled", False))
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
    qc_bbox_enabled = bool(mask_raw.get("qc_bbox_fallback_enabled", False))
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
        target_envelope_padding_px=int(mask_raw.get("target_envelope_padding_px", 4)),
        receiver_envelope_padding_px=int(mask_raw.get("receiver_envelope_padding_px", 4)),
        temporal_qc_min_adjacent_iou_p05=float(
            mask_raw.get("temporal_qc_min_adjacent_iou_p05", 0.5)
        ),
        temporal_qc_max_centroid_jump_p95_px=float(
            mask_raw.get("temporal_qc_max_centroid_jump_p95_px", 5.0)
        ),
        temporal_qc_max_area_ratio_jump_p95=float(
            mask_raw.get("temporal_qc_max_area_ratio_jump_p95", 0.4)
        ),
        temporal_qc_quarantine_signal_count=int(
            mask_raw.get("temporal_qc_quarantine_signal_count", 2)
        ),
        qc_enabled=qc_enabled,
        qc_prompt_template=qc_template,
        qc_max_candidates=int(mask_raw.get("qc_max_candidates", 3)),
        qc_query_fallback_enabled=bool(mask_raw.get("qc_query_fallback_enabled", False)),
        qc_seed_fallback_enabled=bool(mask_raw.get("qc_seed_fallback_enabled", False)),
        qc_bbox_fallback_enabled=qc_bbox_enabled,
        qc_bbox_prompt_template=qc_bbox_template,
        qc_bbox_max_tokens=int(mask_raw.get("qc_bbox_max_tokens", 180)),
        qc_max_tokens=int(mask_raw.get("qc_max_tokens", 160)),
        qc_max_attempts=int(mask_raw.get("qc_max_attempts", 2)),
        qc_min_confidence=float(mask_raw.get("qc_min_confidence", 0.70)),
        qc_min_area_fraction=float(mask_raw.get("qc_min_area_fraction", 0.0001)),
        qc_max_area_fraction=float(mask_raw.get("qc_max_area_fraction", 0.85)),
        qc_duplicate_iou_threshold=float(mask_raw.get("qc_duplicate_iou_threshold", 0.98)),
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
        declared_profile = manifest_data.get("profile")
        if (
            declared_profile is not None
            and declared_profile != profile.annotation.mode.value
        ):
            raise ConfigError(
                "dataset binding profile differs from annotation mode: "
                f"{declared_profile!r} != {profile.annotation.mode.value!r}"
            )
    if regression is None and manifest_data is not None:
        raw_ids = _manifest_episode_values(manifest_data, "regression_episode_ids")
        if raw_ids is not None:
            regression = _binding_episode_ids(raw_ids, field="manifest.regression_episode_ids")
    if regression is None or not regression:
        raise ConfigError(
            "dataset binding requires at least one regression episode; provide "
            "episode_ids or manifest episode_indices"
        )

    smoke = binding.smoke_episode_ids
    if smoke is None and manifest_data is not None:
        raw_smoke = _manifest_episode_values(manifest_data, "smoke_episode_ids")
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
    if (
        task_kind is TargetOnlyTaskKind.CONTACT_ACTION_SITE
        and profile.annotation.profile is not TargetProfile.CONTACT_PRESS
    ):
        raise ConfigError(
            "contact_action_site requires a contact_press profile; load the "
            "contact_press mode before binding"
        )
    # Runtime binding is authoritative for identity.  Keep all other manifest
    # provenance fields, but overwrite values consumed by RoboTwinDataset.
    if manifest_data is not None:
        manifest_data["dataset_root"] = str(binding.root)
        manifest_data["task"] = binding.task
        manifest_data["camera"] = binding.camera
        manifest_data["smoke_episode_ids"] = list(smoke)
        manifest_data["regression_episode_ids"] = list(regression)
        if "episode_indices" in manifest_data:
            manifest_data["episode_indices"] = list(regression)
        if task_kind is not None:
            manifest_data["task_kind"] = (
                task_kind.value if isinstance(task_kind, TargetOnlyTaskKind) else str(task_kind)
            )
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
