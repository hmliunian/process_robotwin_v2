"""Pure helpers for binding a discovered dataset to a reusable profile.

The profile configuration contains algorithm settings only.  Dataset identity
and episode selection enter at the application boundary through
``DatasetBinding``.  This module intentionally performs no manifest or output
file writes; callers may pass an in-memory manifest when one is already
available from discovery.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from ..config import DatasetBinding
from ..domain import TargetOnlyTaskKind
from .dataset_input import DatasetTarget


def _episode_ids(
    values: Sequence[int],
    *,
    field: str,
) -> tuple[int, ...]:
    """Validate and normalize episode IDs while preserving first-seen order."""

    if isinstance(values, (str, bytes)):
        raise ValueError(  # noqa: TRY004 - preserve the binding validation contract
            f"{field} must be a sequence of integers"
        )
    normalized: list[int] = []
    seen: set[int] = set()
    try:
        iterator = iter(values)
    except TypeError as exc:
        raise ValueError(f"{field} must be a sequence of integers") from exc
    for value in iterator:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(  # noqa: TRY004 - preserve the binding validation contract
                f"{field} must contain only integers"
            )
        if value < 0:
            raise ValueError(f"{field} must contain non-negative integers")
        if value not in seen:
            seen.add(value)
            normalized.append(value)
    return tuple(normalized)


def dataset_binding_from_target(
    target: DatasetTarget,
    *,
    episode_ids: Sequence[int] | None = None,
    manifest_data: Mapping[str, Any] | None = None,
    manifest_path: Path | None = None,
) -> DatasetBinding:
    """Create an immutable runtime binding from a resolved dataset target.

    ``episode_ids`` is an explicit selection when supplied, including an
    intentional empty selection.  Otherwise the IDs declared by ``target``
    are used.  Manifest data is copied before it crosses the boundary so the
    caller's mapping cannot be changed by downstream code.
    """

    # Keep the application port friendly to lightweight resolver doubles used
    # by callers/tests while still rejecting arbitrary objects early.  The
    # required attributes are the same small contract as ``DatasetTarget``;
    # the concrete dataclass is not needed for this pure binding operation.
    if not isinstance(target, DatasetTarget) and not all(
        hasattr(target, attribute)
        for attribute in ("root", "task", "camera", "episode_ids")
    ):
        raise TypeError("target must provide root, task, camera and episode_ids")
    selected = target.episode_ids if episode_ids is None else episode_ids
    normalized_ids = _episode_ids(selected, field="episode_ids")
    source_manifest = (
        getattr(target, "manifest_data", None)
        if manifest_data is None
        else manifest_data
    )
    copied_manifest = (
        None if source_manifest is None else deepcopy(dict(source_manifest))
    )
    source_manifest_path = (
        getattr(target, "manifest_path", None)
        if manifest_path is None
        else manifest_path
    )
    task_kind = getattr(target, "task_kind", None)
    if task_kind is None and copied_manifest is not None:
        raw_task_kind = copied_manifest.get("task_kind")
        if raw_task_kind is not None:
            try:
                task_kind = TargetOnlyTaskKind(raw_task_kind)
            except (TypeError, ValueError) as exc:
                choices = ", ".join(item.value for item in TargetOnlyTaskKind)
                raise ValueError(
                    f"manifest task_kind must be one of: {choices}"
                ) from exc
    kwargs: dict[str, Any] = {
        "root": target.root,
        "task": target.task,
        "camera": target.camera,
        "manifest_path": source_manifest_path,
        "manifest_data": copied_manifest,
    }
    # A target's manifest is authoritative when the caller did not request a
    # new episode selection.  Passing the target IDs through the compatibility
    # alias would otherwise mark them as explicit and silently replace a
    # manifest's dedicated smoke episode.
    manifest_declares_selection = bool(
        source_manifest
        and any(
            key in source_manifest
            for key in ("regression_episode_ids", "episode_indices", "episode_ids")
        )
    )
    if episode_ids is not None or not manifest_declares_selection:
        kwargs["episode_ids"] = normalized_ids
    if task_kind is not None:
        kwargs["task_kind"] = TargetOnlyTaskKind(task_kind)
    target_profile = getattr(target, "target_profile", None)
    if target_profile is None:
        # ``profile`` is the public compatibility property on older target
        # doubles.  Preserve it when it is an actual semantic profile, while
        # leaving pick-place targets at the historical default.
        target_profile = getattr(target, "profile", None)
    if target_profile is not None:
        kwargs["target_profile"] = target_profile
    return DatasetBinding(**kwargs)


__all__ = ["DatasetBinding", "dataset_binding_from_target"]
