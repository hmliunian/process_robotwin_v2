"""Resolve task datasets and collections from extract manifests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robotwin_annotation_v2.domain import AnnotationMode


@dataclass(frozen=True)
class DatasetTarget:
    root: Path
    task: str
    camera: str
    episode_ids: tuple[int, ...]


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


def _task_target(root: Path, mode: AnnotationMode) -> DatasetTarget:
    manifest = _read_manifest(root)
    if manifest.get("profile") != mode.value:
        raise ValueError(
            f"dataset profile {manifest.get('profile')!r} does not match "
            f"--{mode.value.replace('_', '-')}"
        )
    task = manifest.get("task")
    camera = manifest.get("camera")
    episode_ids = manifest.get("episode_indices")
    if not isinstance(task, str) or not task:
        raise ValueError(f"dataset manifest does not define a task: {root}")
    if not isinstance(camera, str) or not camera:
        raise ValueError(f"dataset manifest does not define a camera: {root}")
    if not isinstance(episode_ids, list) or not episode_ids:
        raise ValueError(f"dataset manifest has no episode_indices: {root}")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in episode_ids):
        raise ValueError(f"dataset manifest episode_indices must be integers: {root}")
    unique_ids = tuple(dict.fromkeys(episode_ids))
    if len(unique_ids) != len(episode_ids):
        raise ValueError(f"dataset manifest contains duplicate episode_indices: {root}")
    for directory in ("data", "videos", "sidecars", "meta"):
        if not (root / directory).is_dir():
            raise ValueError(f"task dataset is missing {directory}/: {root}")
    return DatasetTarget(root, task, camera, unique_ids)


def resolve_dataset_input(
    path: Path,
    *,
    mode: AnnotationMode,
    task: str | None = None,
) -> DatasetInput:
    """Resolve and validate one task dataset or a task collection."""

    root = path.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"dataset path is not a directory: {root}")
    manifest = _read_manifest(root)
    datasets = manifest.get("datasets")
    if datasets is None:
        target = _task_target(root, mode)
        if task is not None and task != target.task:
            raise ValueError(f"requested task {task!r} does not match dataset task {target.task!r}")
        return DatasetInput(root, (target,), False)

    if manifest.get("profile") != mode.value:
        raise ValueError(
            f"dataset profile {manifest.get('profile')!r} does not match "
            f"--{mode.value.replace('_', '-')}"
        )
    if not isinstance(datasets, list) or not datasets:
        raise ValueError(f"collection manifest has no datasets: {root}")
    names: list[str] = []
    for record in datasets:
        name = record.get("task") if isinstance(record, dict) else None
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError(f"collection manifest contains an invalid task record: {root}")
        names.append(name)
    if len(set(names)) != len(names):
        raise ValueError(f"collection manifest contains duplicate tasks: {root}")
    selected = names if task is None else [task]
    unknown = sorted(set(selected) - set(names))
    if unknown:
        raise ValueError(f"requested collection task is absent: {unknown}")
    targets = tuple(_task_target(root / name, mode) for name in sorted(selected))
    return DatasetInput(root, targets, True)


__all__ = ["DatasetInput", "DatasetTarget", "resolve_dataset_input"]
