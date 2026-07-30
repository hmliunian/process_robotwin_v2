"""Configuration loading for the small target/receiver experiment."""

from __future__ import annotations

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
    same_frame_text: bool = True

    def __post_init__(self) -> None:
        if len(self.gpus) != 1:
            raise ConfigError("sam3.gpus must contain exactly one GPU")
        if any(gpu < 0 for gpu in self.gpus):
            raise ConfigError("sam3.gpus must contain non-negative integers")
        if not self.same_frame_text:
            raise ConfigError("same-frame text observation is required")


@dataclass(frozen=True)
class MaskConfig:
    target_envelope_padding_px: int = 4
    receiver_envelope_padding_px: int = 4

    def __post_init__(self) -> None:
        if self.target_envelope_padding_px < 0 or self.receiver_envelope_padding_px < 0:
            raise ConfigError("mask envelope padding must be non-negative")


@dataclass(frozen=True)
class PipelineConfig:
    config_path: Path
    dataset: DatasetConfig
    qwen: QwenConfig
    sam3: Sam3Config
    mask: MaskConfig
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
    output_raw = raw.get("output", {})
    sections = (dataset_raw, qwen_raw, sam3_raw, mask_raw, output_raw)
    if not all(isinstance(item, dict) for item in sections):
        raise ConfigError("dataset, qwen, sam3, mask and output must be mappings")

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
        same_frame_text=bool(sam3_raw.get("same_frame_text", True)),
    )
    mask = MaskConfig(
        target_envelope_padding_px=int(mask_raw.get("target_envelope_padding_px", 4)),
        receiver_envelope_padding_px=int(mask_raw.get("receiver_envelope_padding_px", 4)),
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
        output_root=output_root,
    )
