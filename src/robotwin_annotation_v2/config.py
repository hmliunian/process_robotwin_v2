"""Configuration loading for the small target/receiver experiment."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Configuration is missing or violates the pipeline contract."""


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


@dataclass(frozen=True)
class QwenConfig:
    endpoint: str
    model: str
    prompt_template: Path
    timeout_seconds: float = 180.0
    max_tokens: int = 800
    query_selection: str = "first_recommended"
    allow_query_fallback: bool = False

    def __post_init__(self) -> None:
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
        if self.qc_max_candidates < 1:
            raise ConfigError("mask.qc_max_candidates must be positive")
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
                raise ConfigError(
                    f"gripper_roi.{name} must be a finite number greater than zero"
                )


@dataclass(frozen=True)
class PipelineConfig:
    config_path: Path
    dataset: DatasetConfig
    qwen: QwenConfig
    sam3: Sam3Config
    mask: MaskConfig
    gripper_roi: GripperRoiConfig
    output_root: Path


def load_config(path: Path) -> PipelineConfig:
    """Load and validate one YAML config, resolving paths relative to it."""

    config_path = path.expanduser().resolve()
    if not config_path.is_file():
        raise ConfigError(f"config file does not exist: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ConfigError("top-level config must be a mapping")
    base_dir = config_path.parent

    dataset_raw = _required(raw, "dataset")
    qwen_raw = _required(raw, "qwen")
    sam3_raw = _required(raw, "sam3")
    mask_raw = raw.get("mask", {})
    gripper_roi_raw = _required(raw, "gripper_roi")
    output_raw = raw.get("output", {})
    sections = (dataset_raw, qwen_raw, sam3_raw, mask_raw, gripper_roi_raw, output_raw)
    if not all(isinstance(item, dict) for item in sections):
        raise ConfigError(
            "dataset, qwen, sam3, mask, gripper_roi and output must be mappings"
        )

    prompt_roi_raw = _required(gripper_roi_raw, "prompt", section="gripper_roi")
    hard_roi_raw = _required(gripper_roi_raw, "hard", section="gripper_roi")
    if not isinstance(prompt_roi_raw, dict) or not isinstance(hard_roi_raw, dict):
        raise ConfigError("gripper_roi.prompt and gripper_roi.hard must be mappings")

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
        task=str(_required(dataset_raw, "task", section="dataset")),
        camera=str(_required(dataset_raw, "camera", section="dataset")),
        smoke_episode_ids=smoke,
        regression_episode_ids=regression,
    )
    qwen = QwenConfig(
        endpoint=str(_required(qwen_raw, "endpoint", section="qwen")),
        model=str(_required(qwen_raw, "model", section="qwen")),
        prompt_template=_path(
            _required(qwen_raw, "prompt_template", section="qwen"),
            base_dir=base_dir,
            field="qwen.prompt_template",
        ),
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
    qc_enabled = bool(mask_raw.get("qc_enabled", False))
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
        qc_max_tokens=int(mask_raw.get("qc_max_tokens", 160)),
        qc_max_attempts=int(mask_raw.get("qc_max_attempts", 2)),
        qc_min_confidence=float(mask_raw.get("qc_min_confidence", 0.70)),
        qc_min_area_fraction=float(mask_raw.get("qc_min_area_fraction", 0.0001)),
        qc_max_area_fraction=float(mask_raw.get("qc_max_area_fraction", 0.85)),
        qc_duplicate_iou_threshold=float(
            mask_raw.get("qc_duplicate_iou_threshold", 0.98)
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
    return PipelineConfig(
        config_path=config_path,
        dataset=dataset,
        qwen=qwen,
        sam3=sam3,
        mask=mask,
        gripper_roi=gripper_roi,
        output_root=output_root,
    )
