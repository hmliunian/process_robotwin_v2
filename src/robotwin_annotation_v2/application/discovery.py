"""Dataset input discovery and dynamic manifest construction."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import numpy as np
import pandas as pd

from robotwin_annotation_v2.domain import (
    TargetOnlyTaskKind,
    TargetProfile,
    target_profile_for_task_kind,
)

from .provenance import (
    attach_profile_provenance,
    prompt_bundle_from_manifest,
    target_profile_from_manifest,
    validate_profile_provenance,
)

CHUNK_PATTERN = re.compile(r"chunk-(\d{3})$")
EPISODE_FILE_PATTERN = re.compile(r"episode_(\d+)\.parquet$")

EpisodeMeasurement = tuple[int, tuple[int, int], int]


@dataclass(frozen=True)
class DiscoveredEpisode:
    episode_id: int
    parquet: Path
    video: Path
    sidecar: Path


@dataclass(frozen=True)
class DiscoveryResult:
    episodes: tuple[DiscoveredEpisode, ...]
    skipped: tuple[dict[str, Any], ...]

    @property
    def episode_ids(self) -> tuple[int, ...]:
        return tuple(episode.episode_id for episode in self.episodes)


def _episode_video_path(root: Path, camera: str, episode_id: int) -> Path:
    chunk = f"chunk-{episode_id // 1000:03d}"
    return (
        root
        / "videos"
        / chunk
        / f"observation.images.{camera}"
        / f"episode_{episode_id:06d}.mp4"
    )


def _episode_depth_path(root: Path, camera: str, episode_id: int) -> Path:
    chunk = f"chunk-{episode_id // 1000:03d}"
    return (
        root
        / "sidecars"
        / "videos"
        / chunk
        / f"observation.depths.{camera}"
        / f"episode_{episode_id:06d}.mkv"
    )


def discover_episodes(
    root: Path,
    *,
    camera: str,
    require_depth: bool = False,
) -> DiscoveryResult:
    """Discover complete dataset inputs by episode id."""

    dataset_root = root.expanduser().resolve()
    data_root = dataset_root / "data"
    if not data_root.is_dir():
        return DiscoveryResult((), ())
    discovered: dict[int, DiscoveredEpisode] = {}
    skipped: list[dict[str, Any]] = []
    for chunk_dir in sorted(data_root.iterdir()):
        if not chunk_dir.is_dir():
            continue
        parquet_files = sorted(chunk_dir.glob("episode_*.parquet"))
        if not parquet_files:
            continue
        match = CHUNK_PATTERN.fullmatch(chunk_dir.name)
        if match is None:
            raise ValueError(f"invalid chunk directory name: {chunk_dir.name}")
        for parquet in parquet_files:
            file_match = EPISODE_FILE_PATTERN.fullmatch(parquet.name)
            if file_match is None:
                raise ValueError(f"invalid episode parquet name: {parquet}")
            episode_id = int(file_match.group(1))
            expected_chunk = f"chunk-{episode_id // 1000:03d}"
            if chunk_dir.name != expected_chunk:
                raise ValueError(
                    f"episode {episode_id} is in {chunk_dir.name}, expected {expected_chunk}"
                )
            if episode_id in discovered:
                raise ValueError(f"duplicate episode id discovered: {episode_id}")
            video = _episode_video_path(dataset_root, camera, episode_id)
            sidecar = dataset_root / "sidecars" / f"episode_{episode_id:06d}.hdf5"
            required_paths = [
                ("video", video),
                ("sidecar", sidecar),
            ]
            if require_depth:
                required_paths.append(
                    (
                        "depth_video",
                        _episode_depth_path(dataset_root, camera, episode_id),
                    )
                )
            missing = [
                name
                for name, path in required_paths
                if not path.is_file()
            ]
            if missing:
                skipped.append(
                    {
                        "episode": episode_id,
                        "status": "discovery_skipped",
                        "missing": missing,
                        "parquet": str(parquet),
                    }
                )
                continue
            discovered[episode_id] = DiscoveredEpisode(
                episode_id=episode_id,
                parquet=parquet,
                video=video,
                sidecar=sidecar,
            )
    return DiscoveryResult(
        tuple(discovered[key] for key in sorted(discovered)),
        tuple(skipped),
    )


def select_manifest_episodes(
    episodes: Sequence[DiscoveredEpisode],
    *,
    manifest: Mapping[str, Any] | None,
    episode_ids: Sequence[int],
) -> tuple[DiscoveredEpisode, ...]:
    """Apply a bound manifest's episode scope while preserving discovery order.

    A runtime request may process only a subset while its dynamic manifest
    continues to describe the complete discovery result.  Narrow the manifest
    only when dataset binding has explicitly carried an episode selection.
    """

    selection_keys = (
        "regression_episode_ids",
        "episode_indices",
        "episode_ids",
    )
    if not isinstance(manifest, Mapping) or not any(
        manifest.get(key) is not None for key in selection_keys
    ):
        return tuple(episodes)
    selected = set(episode_ids)
    return tuple(episode for episode in episodes if episode.episode_id in selected)


def _parquet_frame_count(parquet: Path) -> int:
    frame = pd.read_parquet(parquet, columns=["frame_index"])
    if frame.empty:
        raise ValueError(f"episode parquet is empty: {parquet}")
    frame_indices = frame["frame_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(frame_indices, np.arange(len(frame_indices))):
        raise ValueError(f"episode frame_index is not contiguous: {parquet}")
    return len(frame_indices)


def _measure_episode(episode: DiscoveredEpisode) -> EpisodeMeasurement:
    frame_count = _parquet_frame_count(episode.parquet)
    raw_count = 0
    shape: tuple[int, int] | None = None
    with av.open(str(episode.video)) as container:
        for video_frame in container.decode(video=0):
            raw_count += 1
            if shape is None:
                shape = (int(video_frame.height), int(video_frame.width))
    if shape is None:
        raise ValueError(f"episode video contains no frames: {episode.video}")
    return frame_count, shape, raw_count - frame_count


def build_dynamic_manifest(
    root: Path,
    *,
    task: str,
    camera: str,
    episodes: Sequence[DiscoveredEpisode],
    measure_episode_fn: Callable[[DiscoveredEpisode], EpisodeMeasurement] | None = None,
    task_kind: TargetOnlyTaskKind | str | None = None,
    target_profile: str | None = None,
    prompt_bundle: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the manifest contract expected by RoboTwinDataset in memory."""

    if not episodes:
        raise ValueError("cannot build a manifest without discovered episodes")
    measure_episode = _measure_episode if measure_episode_fn is None else measure_episode_fn
    frame_count, shape, surplus = measure_episode(episodes[0])
    if frame_count < 1:
        raise ValueError("first discovered episode has no usable frames")
    manifest: dict[str, Any] = {
        "format_version": "robotwin_dataset_manifest_dynamic_v1",
        "task": task,
        "camera": camera,
        "frame_shape_hw": list(shape),
        "raw_video_frame_surplus": surplus,
        "usable_frame_count_source": "parquet",
        "dataset_root": str(root.expanduser().resolve()),
        "smoke_episode_ids": [episodes[0].episode_id],
        "regression_episode_ids": [episode.episode_id for episode in episodes],
        "required_relative_files": [
            "data/chunk-*/episode_{episode_id}.parquet",
            "videos/chunk-*/observation.images.{camera}/episode_{episode_id}.mp4",
            "sidecars/episode_{episode_id}.hdf5",
        ],
    }
    declared_kind: TargetOnlyTaskKind | None = None
    if task_kind is not None:
        try:
            declared_kind = TargetOnlyTaskKind(task_kind)
        except (TypeError, ValueError) as exc:
            choices = ", ".join(item.value for item in TargetOnlyTaskKind)
            raise ValueError(
                f"unsupported dynamic-manifest task_kind {task_kind!r}; choose {choices}"
            ) from exc
        manifest["task_kind"] = declared_kind.value
    # Preserve extract identity when discovery is run directly (the public
    # DatasetPipeline path does not go through the legacy dataset_runtime
    # wrapper).  The file is optional for native task directories and malformed
    # provenance must not make otherwise discoverable inputs unusable.
    resolved_root = root.expanduser().resolve()
    extract_manifest = resolved_root / "EXTRACT_MANIFEST.json"
    source_manifest: Mapping[str, Any] | None = None
    if extract_manifest.is_file():
        try:
            import json

            raw_source = json.loads(extract_manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            raw_source = None
        if isinstance(raw_source, Mapping):
            source_manifest = raw_source
            manifest["source_manifest_path"] = str(extract_manifest)
            source_root = raw_source.get("dataset_root")
            if isinstance(source_root, str) and source_root.strip():
                manifest["source_dataset_root"] = source_root
            source_bundle = prompt_bundle_from_manifest(raw_source, strict=True)
            if source_bundle is not None and prompt_bundle is None:
                prompt_bundle = source_bundle
            source_profile = target_profile_from_manifest(raw_source)
            if source_profile is not None:
                if target_profile is not None:
                    explicit_profile = target_profile_from_manifest(
                        {"target_profile": target_profile}
                    )
                    if explicit_profile != source_profile:
                        raise ValueError(
                            "dynamic manifest target_profile conflicts with source metadata"
                        )
                else:
                    target_profile = source_profile
            if "task_kind" not in manifest and raw_source.get("task_kind") is not None:
                try:
                    source_kind = TargetOnlyTaskKind(raw_source["task_kind"])
                except (TypeError, ValueError) as exc:
                    choices = ", ".join(item.value for item in TargetOnlyTaskKind)
                    raise ValueError(
                        "unsupported source task_kind "
                        f"{raw_source['task_kind']!r}; choose {choices}"
                    ) from exc
                manifest["task_kind"] = source_kind.value
                declared_kind = source_kind
            elif raw_source.get("task_kind") is not None:
                try:
                    source_kind = TargetOnlyTaskKind(raw_source["task_kind"])
                except (TypeError, ValueError) as exc:
                    choices = ", ".join(item.value for item in TargetOnlyTaskKind)
                    raise ValueError(
                        "unsupported source task_kind "
                        f"{raw_source['task_kind']!r}; choose {choices}"
                    ) from exc
                if declared_kind is not None and source_kind is not declared_kind:
                    raise ValueError(
                        "dynamic manifest task_kind conflicts with source metadata"
                    )
    manifest.setdefault("source_dataset_root", str(resolved_root))
    if source_manifest is not None:
        manifest.setdefault("source_manifest_path", str(extract_manifest))
    # A declared task kind is the authoritative profile selector.  Explicit
    # profile metadata is retained for native/converted manifests that predate
    # task_kind.
    kind_profile = (
        None
        if declared_kind is None
        else target_profile_for_task_kind(declared_kind).value
    )
    if manifest.get("task_kind") is not None and declared_kind is None:
        try:
            declared_kind = TargetOnlyTaskKind(manifest["task_kind"])
        except (TypeError, ValueError) as exc:
            choices = ", ".join(item.value for item in TargetOnlyTaskKind)
            raise ValueError(
                "unsupported dynamic-manifest task_kind "
                f"{manifest['task_kind']!r}; choose {choices}"
            ) from exc
        kind_profile = target_profile_for_task_kind(declared_kind).value
    if target_profile is None:
        target_profile = kind_profile or TargetProfile.GRASP_MANIPULATION.value
    else:
        target_profile = target_profile_from_manifest(
            {"target_profile": target_profile}
        )
    if kind_profile is not None and target_profile != kind_profile:
        raise ValueError(
            "dynamic manifest target_profile conflicts with task_kind: "
            f"{target_profile!r} != {kind_profile!r}"
        )
    if prompt_bundle is not None:
        bundle_profile = target_profile_from_manifest(prompt_bundle)
        if bundle_profile is not None and bundle_profile != target_profile:
            raise ValueError(
                "dynamic manifest prompt_bundle target_profile conflicts with "
                "task profile"
            )
        try:
            validate_profile_provenance(
                {
                    "target_profile": target_profile,
                    "prompt_bundle": prompt_bundle,
                },
                expected_profile=target_profile,
                expected_prompt_bundle=prompt_bundle,
                strict=True,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid dynamic manifest prompt_bundle: {exc}") from exc
    attach_profile_provenance(
        manifest,
        target_profile=target_profile,
        prompt_bundle=prompt_bundle,
    )
    return manifest


__all__ = [
    "CHUNK_PATTERN",
    "EPISODE_FILE_PATTERN",
    "DiscoveredEpisode",
    "DiscoveryResult",
    "build_dynamic_manifest",
    "discover_episodes",
    "select_manifest_episodes",
]
