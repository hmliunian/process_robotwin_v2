#!/usr/bin/env python3
"""Process every complete episode in a RoboTwin-format directory."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import av
import numpy as np
import pandas as pd

from robotwin_annotation_v2.adapters.artifact_store import ArtifactStore
from robotwin_annotation_v2.adapters.robotwin_dataset import RoboTwinDataset
from robotwin_annotation_v2.config import PipelineConfig, load_config
from robotwin_annotation_v2.models import EpisodeRef
from robotwin_annotation_v2.urdf_gripper_publisher import (
    UrdfGripperPublishError,
    validate_derivation_source_episode,
)

CHUNK_PATTERN = re.compile(r"chunk-(\d{3})$")
EPISODE_FILE_PATTERN = re.compile(r"episode_(\d+)\.parquet$")
GRIPPER_BACKENDS = ("sam", "urdf")
DEFAULT_URDF_DEPTH_TOLERANCE_MM = 8.0
DEFAULT_URDF_MINIMUM_ELIGIBLE_NONEMPTY_FRACTION = 0.90
PROCESS_SUMMARY_FORMAT_VERSION = "robotwin_process_dataset_summary_v1"


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


@dataclass(frozen=True)
class UrdfSourceSelection:
    """Frozen, QC-passed source masks selected for URDF replacement."""

    episode_ids: tuple[int, ...]
    excluded: tuple[dict[str, Any], ...]
    source_summary: Mapping[str, Any]
    source_lineages: Mapping[int, Mapping[str, Any]]


@dataclass(frozen=True)
class SamRuntime:
    """SAM-only dependencies loaded after the backend has been selected."""

    qwen_client_factory: Callable[..., Any]
    backend_factory: Callable[..., Any]
    execution_errors: tuple[type[BaseException], ...]
    emit_gripper_result: Callable[..., Any]
    emit_sam_result: Callable[..., Any]
    execute_gripper_episode: Callable[..., Any]
    execute_sam_episode: Callable[..., Any]
    fatal_cuda_error: Callable[..., Any]
    gripper_episode_complete: Callable[..., Any]
    run_qwen: Callable[..., Any]


def _load_sam_runtime() -> SamRuntime:
    """Load SAM/Qwen/OpenCV code only when the SAM backend is executed."""

    from robotwin_annotation_v2.adapters.qwen_client import (
        OpenAICompatibleQwenClient,
    )
    from robotwin_annotation_v2.adapters.sam3_adapter import Sam3Adapter

    try:
        runtime = importlib.import_module("run_target_receiver")
    except ModuleNotFoundError as exc:
        if exc.name != "run_target_receiver":
            raise
        runtime = importlib.import_module("scripts.run_target_receiver")
    return SamRuntime(
        qwen_client_factory=OpenAICompatibleQwenClient,
        backend_factory=Sam3Adapter,
        execution_errors=tuple(runtime.SAM_EXECUTION_ERRORS),
        emit_gripper_result=runtime._emit_gripper_result,
        emit_sam_result=runtime._emit_sam_result,
        execute_gripper_episode=runtime._execute_gripper_episode,
        execute_sam_episode=runtime._execute_sam_episode,
        fatal_cuda_error=runtime._fatal_cuda_error,
        gripper_episode_complete=runtime._gripper_episode_complete,
        run_qwen=runtime.run_qwen,
    )


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


def _read_json_object(path: Path, *, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{description} is missing: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {description}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must contain one JSON object: {path}")
    return payload


def _validate_run_id(run_id: str) -> str:
    """Validate a public run id before it participates in path construction."""

    if (
        not isinstance(run_id, str)
        or not run_id
        or run_id != run_id.strip()
        or run_id in {".", ".."}
        or "/" in run_id
        or "\\" in run_id
        or ".." in run_id
    ):
        raise ValueError("run_id must be a simple non-empty directory name")
    return run_id


def _summary_gripper_backend(summary: Mapping[str, Any]) -> str | None:
    top_level = summary.get("gripper_backend")
    backend_record = summary.get("backend")
    nested = backend_record.get("type") if isinstance(backend_record, Mapping) else None
    if top_level is not None and nested is not None and top_level != nested:
        raise ValueError("existing process summary has conflicting backend ownership")
    value = top_level if top_level is not None else nested
    return None if value is None else str(value)


def _validate_sam_run_ownership(run_dir: Path, *, run_id: str) -> None:
    """Keep legacy SAM resume, but never adopt an URDF-owned public run."""

    if (run_dir / "_backend" / "urdf").exists():
        raise ValueError(f"existing run is owned by the URDF backend: {run_dir}")
    summary_path = run_dir / "process_summary.json"
    if not summary_path.exists():
        return
    summary = _read_json_object(summary_path, description="existing process summary")
    if summary.get("run_id") != run_id:
        raise ValueError("existing process summary run_id does not match its directory")
    backend = _summary_gripper_backend(summary)
    # Summaries written before the backend discriminator was introduced are SAM runs.
    if backend not in {None, "sam"}:
        raise ValueError(f"existing run is owned by backend {backend!r}, not SAM")


def _validate_urdf_run_ownership(
    run_dir: Path,
    *,
    run_id: str,
    resume: bool,
) -> None:
    """Require a fresh public run, or a positively identified URDF resume run."""

    if not resume:
        if run_dir.exists():
            raise FileExistsError(f"canonical output run already exists: {run_dir}")
        return
    if not run_dir.is_dir():
        raise FileNotFoundError(f"canonical resume run directory is missing: {run_dir}")
    backend_manifest = run_dir / "_backend" / "urdf" / "manifest.json"
    if not backend_manifest.is_file():
        raise ValueError(
            "canonical resume run is not owned by the URDF backend: "
            f"{backend_manifest} is missing"
        )
    summary_path = run_dir / "process_summary.json"
    if not summary_path.exists():
        return
    summary = _read_json_object(summary_path, description="existing process summary")
    if summary.get("format_version") != PROCESS_SUMMARY_FORMAT_VERSION:
        raise ValueError("existing process summary format is not resumable")
    if summary.get("run_id") != run_id:
        raise ValueError("existing process summary run_id does not match its directory")
    if _summary_gripper_backend(summary) != "urdf":
        raise ValueError("canonical resume run is not owned by the URDF backend")


def select_urdf_source_episodes(
    source_run_dir: Path,
    *,
    dataset_root: Path,
    task: str,
    camera: str,
    discovered_episode_ids: tuple[int, ...],
    requested_episode_ids: tuple[int, ...] | None = None,
    expected_frame_counts: Mapping[int, int] | None = None,
) -> UrdfSourceSelection:
    """Select only completed source episodes with valid target/receiver masks."""

    source_root = source_run_dir.expanduser().resolve()
    summary = _read_json_object(
        source_root / "process_summary.json",
        description="source process summary",
    )
    if summary.get("format_version") != "robotwin_process_dataset_summary_v1":
        raise ValueError(
            "source process summary format must be robotwin_process_dataset_summary_v1"
        )
    if summary.get("run_id") != source_root.name:
        raise ValueError("source process summary run_id does not match its directory")
    if summary.get("task") != task or summary.get("camera") != camera:
        raise ValueError(
            "source process summary task/camera does not match the requested dataset"
        )
    summary_dataset = Path(str(summary.get("dataset_root", ""))).expanduser().resolve()
    if summary_dataset != dataset_root.expanduser().resolve():
        raise ValueError(
            "source process summary dataset_root does not match --dataset-root"
        )

    discovered = set(discovered_episode_ids)
    selected = (
        discovered_episode_ids
        if requested_episode_ids is None
        else tuple(dict.fromkeys(int(value) for value in requested_episode_ids))
    )
    unknown = sorted(set(selected) - discovered)
    if unknown:
        raise ValueError(f"requested episodes were not discovered: {unknown}")
    if expected_frame_counts is not None:
        missing_frame_counts = sorted(set(selected) - set(expected_frame_counts))
        if missing_frame_counts:
            raise ValueError(
                "expected frame counts are missing for episodes: "
                f"{missing_frame_counts}"
            )
    accepted: list[int] = []
    excluded: list[dict[str, Any]] = []
    source_lineages: dict[int, Mapping[str, Any]] = {}
    for episode_id in selected:
        episode_dir = (
            source_root / task / f"episode_{episode_id:06d}" / camera
        )
        try:
            validated = validate_derivation_source_episode(
                episode_dir,
                task=task,
                camera=camera,
                episode_index=episode_id,
                expected_frame_count=(
                    None
                    if expected_frame_counts is None
                    else expected_frame_counts.get(episode_id)
                ),
                expected_dataset_root=dataset_root,
            )
        except (FileNotFoundError, UrdfGripperPublishError) as exc:
            excluded.append(
                {
                    "episode": episode_id,
                    "status": "source_excluded",
                    "reason": f"source_contract_error:{type(exc).__name__}",
                    "error": str(exc),
                }
            )
            continue
        accepted.append(episode_id)
        source_lineages[episode_id] = validated.lineage

    if requested_episode_ids is not None and excluded:
        rendered = ", ".join(
            f"{record['episode']} ({record['reason']})" for record in excluded
        )
        raise ValueError(f"requested URDF source episodes are not publishable: {rendered}")
    if not accepted:
        raise ValueError("source run contains no publishable target/receiver episodes")
    return UrdfSourceSelection(
        episode_ids=tuple(accepted),
        excluded=tuple(excluded),
        source_summary=summary,
        source_lineages=source_lineages,
    )


def _parquet_frame_count(parquet: Path) -> int:
    frame = pd.read_parquet(parquet, columns=["frame_index"])
    if frame.empty:
        raise ValueError(f"episode parquet is empty: {parquet}")
    frame_indices = frame["frame_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(frame_indices, np.arange(len(frame_indices))):
        raise ValueError(f"episode frame_index is not contiguous: {parquet}")
    return len(frame_indices)


def _measure_episode(episode: DiscoveredEpisode) -> tuple[int, tuple[int, int], int]:
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
) -> dict[str, Any]:
    """Build the manifest contract expected by RoboTwinDataset in memory."""

    if not episodes:
        raise ValueError("cannot build a manifest without discovered episodes")
    frame_count, shape, surplus = _measure_episode(episodes[0])
    if frame_count < 1:
        raise ValueError("first discovered episode has no usable frames")
    return {
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


def _dynamic_config(
    config: PipelineConfig,
    *,
    root: Path,
    task: str,
    camera: str,
    manifest: dict[str, Any],
    output_root: Path,
) -> PipelineConfig:
    dataset = replace(
        config.dataset,
        root=root.expanduser().resolve(),
        task=task,
        camera=camera,
        smoke_episode_ids=tuple(manifest["smoke_episode_ids"]),
        regression_episode_ids=tuple(manifest["regression_episode_ids"]),
        manifest_data=manifest,
    )
    return replace(config, dataset=dataset, output_root=output_root.expanduser().resolve())


def _render_processed(
    config: PipelineConfig,
    *,
    run_id: str,
    episode_ids: tuple[int, ...],
    output_dir: Path,
) -> dict[str, Any]:
    try:
        import render_coverage20_videos as render
    except ModuleNotFoundError:
        from scripts import render_coverage20_videos as render

    dataset = RoboTwinDataset(
        config.dataset.root,
        task=config.dataset.task,
        camera=config.dataset.camera,
        manifest_path=config.dataset.manifest,
        manifest_data=config.dataset.manifest_data,
    )
    selected = render.select_best_masks(
        config.output_root,
        task=config.dataset.task,
        camera=config.dataset.camera,
        episode_ids=episode_ids,
        run_id=run_id,
    )
    video_dir = output_dir / "rendered_videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for episode_id in episode_ids:
        candidate = selected[episode_id]
        artifact = render._load_masks(candidate.path)
        ref = EpisodeRef(
            config.dataset.task,
            episode_id,
            config.dataset.camera,
        )
        video_path = dataset.paths(ref).video
        task_text = dataset.task_text(episode_id)
        output_path = video_dir / render._output_video_name(
            episode_id=episode_id,
            camera=config.dataset.camera,
            task_text=task_text,
            filename_mode="episode",
        )
        video = render.render_video(
            video_path,
            artifact,
            output_path,
            alpha=render.DEFAULT_FILL_ALPHA,
            outline_radius=render.DEFAULT_OUTLINE_RADIUS,
            halo_radius=render.DEFAULT_HALO_RADIUS,
            crf=18,
            preset="medium",
            overwrite=True,
        )
        records.append(
            {
                "episode_index": episode_id,
                "task_text": task_text,
                "run_id": candidate.run_id,
                "source_video": str(video_path),
                "source_masks": str(candidate.path),
                "mask_sha256": render._sha256(candidate.path),
                "mask_format": artifact.format_version,
                "annotation_status": dict(
                    zip(artifact.instance_names, artifact.annotation_status, strict=True)
                ),
                "qc_status": dict(
                    zip(artifact.instance_names, artifact.qc_status, strict=True)
                ),
                "output_video": output_path.name,
                "output_sha256": render._sha256(output_path),
                "output_bytes": output_path.stat().st_size,
                **video,
            }
        )
    manifest = {
        "format": "robotwin_coverage20_overlay_videos_v3",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "requested_run_id": run_id,
        "config": str(config.config_path),
        "dataset_root": str(config.dataset.root),
        "runs_root": str(config.output_root),
        "task": config.dataset.task,
        "camera": config.dataset.camera,
        "filename_mode": "episode",
        "episode_count": len(records),
        "rendered_roles": ["target", "receiver", "gripper"],
        "alpha": render.DEFAULT_FILL_ALPHA,
        "colors_rgb": {key: list(value) for key, value in render.ROLE_COLORS.items()},
        "episodes": records,
        "review_sheets": [],
    }
    manifest_path = ArtifactStore.write_json(video_dir / "manifest.json", manifest)
    sheets = render.build_sheets(manifest_path, video_dir / "review_sheets")
    manifest["review_sheets"] = [
        str(path.relative_to(video_dir)) for path in sheets
    ]
    ArtifactStore.write_json(manifest_path, manifest)
    return {
        "manifest": str(manifest_path),
        "episode_count": len(records),
        "review_sheets": [str(path) for path in sheets],
    }


def process_urdf_source_run(
    *,
    pipeline_config: PipelineConfig | None = None,
    dataset_root: Path,
    source_run_dir: Path,
    task: str,
    camera: str,
    output_root: Path,
    urdf_path: Path,
    mesh_root: Path | None = None,
    run_id: str | None = None,
    episode_ids: tuple[int, ...] | None = None,
    skip_render: bool = False,
    dry_run: bool = False,
    resume: bool = False,
    depth_tolerance_mm: float = DEFAULT_URDF_DEPTH_TOLERANCE_MM,
    minimum_eligible_nonempty_fraction: float = (
        DEFAULT_URDF_MINIMUM_ELIGIBLE_NONEMPTY_FRACTION
    ),
    fit_config_json: Path | None = None,
    allow_partial_source: bool = False,
    experiment_runner: Callable[..., Mapping[str, Any]] | None = None,
    episode_publisher: Callable[..., Mapping[str, Any]] | None = None,
    episode_validator: Callable[..., Mapping[str, Any]] | None = None,
    render_builder: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Replace gripper masks in a frozen process run with canonical URDF output."""

    if dry_run and resume:
        raise ValueError("--dry-run and --resume cannot be used together")
    if run_id is not None:
        _validate_run_id(run_id)

    try:
        import render_urdf_gripper_masks as urdf_runner
    except ModuleNotFoundError:
        from scripts import render_urdf_gripper_masks as urdf_runner
    from robotwin_annotation_v2.urdf_gripper_publisher import (
        publish_urdf_episode,
        validate_published_urdf_episode,
    )

    public_discovery = discover_episodes(dataset_root, camera=camera)
    discovery = discover_episodes(dataset_root, camera=camera, require_depth=True)
    dataset_excluded: list[dict[str, Any]] = [
        {
            "episode": int(record["episode"]),
            "status": "dataset_excluded",
            "reason": "dataset_inputs_missing",
            "missing": list(record["missing"]),
            "parquet": str(record["parquet"]),
        }
        for record in discovery.skipped
    ]
    expected_frame_counts: dict[int, int] = {}
    for discovered_episode in discovery.episodes:
        try:
            expected_frame_counts[discovered_episode.episode_id] = (
                _parquet_frame_count(discovered_episode.parquet)
            )
        except (KeyError, OSError, TypeError, ValueError) as exc:
            dataset_excluded.append(
                {
                    "episode": discovered_episode.episode_id,
                    "status": "dataset_excluded",
                    "reason": "dataset_parquet_invalid",
                    "error": f"{type(exc).__name__}: {exc}",
                    "parquet": str(discovered_episode.parquet),
                }
            )
    eligible_dataset_ids = tuple(sorted(expected_frame_counts))
    relevant_dataset_excluded = dataset_excluded
    if episode_ids is not None:
        requested = set(episode_ids)
        relevant_dataset_excluded = [
            record for record in dataset_excluded if record["episode"] in requested
        ]
        if relevant_dataset_excluded:
            rendered = ", ".join(
                f"{record['episode']} ({record['reason']})"
                for record in relevant_dataset_excluded
            )
            raise ValueError(
                "requested URDF episodes do not satisfy the dataset contract: "
                f"{rendered}"
            )
    if not eligible_dataset_ids:
        raise ValueError(f"no complete URDF episodes found under {dataset_root}")
    selection = select_urdf_source_episodes(
        source_run_dir,
        dataset_root=dataset_root,
        task=task,
        camera=camera,
        discovered_episode_ids=eligible_dataset_ids,
        requested_episode_ids=episode_ids,
        expected_frame_counts=expected_frame_counts,
    )
    all_excluded = sorted(
        [*relevant_dataset_excluded, *selection.excluded],
        key=lambda record: int(record["episode"]),
    )
    if all_excluded and not allow_partial_source:
        examples = ", ".join(
            f"{record['episode']} ({record['reason']})"
            for record in all_excluded[:10]
        )
        suffix = "" if len(all_excluded) <= 10 else ", ..."
        raise ValueError(
            f"dataset/source contracts exclude {len(all_excluded)} episodes: "
            f"{examples}{suffix}; pass --allow-partial-source to process only the "
            "fully eligible subset"
        )
    if not math.isfinite(depth_tolerance_mm) or depth_tolerance_mm < 0:
        raise ValueError("URDF depth tolerance must be finite and non-negative")
    if not math.isfinite(minimum_eligible_nonempty_fraction) or not (
        0.0 <= minimum_eligible_nonempty_fraction <= 1.0
    ):
        raise ValueError(
            "URDF minimum eligible nonempty fraction must be finite and in [0, 1]"
        )
    selected_run_id = _validate_run_id(run_id or urdf_runner.new_run_id())
    resolved_output_root = output_root.expanduser().resolve()
    canonical_run_dir = resolved_output_root / selected_run_id
    _validate_urdf_run_ownership(
        canonical_run_dir,
        run_id=selected_run_id,
        resume=resume,
    )
    backend_output_root = canonical_run_dir / "_backend"
    config = urdf_runner.RunConfig(
        dataset_root=dataset_root.expanduser().resolve(),
        source_run_dir=source_run_dir.expanduser().resolve(),
        output_root=backend_output_root,
        run_id="urdf",
        urdf_path=urdf_path.expanduser().resolve(),
        mesh_root=None if mesh_root is None else mesh_root.expanduser().resolve(),
        episode_ids=selection.episode_ids,
        task=task,
        camera=camera,
        depth_tolerance_mm=depth_tolerance_mm,
        minimum_eligible_nonempty_fraction=minimum_eligible_nonempty_fraction,
        fit_config_json=(
            None if fit_config_json is None else fit_config_json.expanduser().resolve()
        ),
        # Public overlays are generated from the canonical masks by the shared renderer.
        skip_overlay=True,
        dry_run=dry_run,
        resume=resume,
    )
    runner = urdf_runner.run_experiment if experiment_runner is None else experiment_runner
    result: Mapping[str, Any]
    batch_error: str | None = None
    try:
        result = runner(config)
    except urdf_runner.UrdfBatchIncompleteError as exc:
        result = exc.result
        batch_error = f"{type(exc).__name__}: {exc}"

    source_dynamic_manifest = selection.source_summary.get("dynamic_manifest")
    if not isinstance(source_dynamic_manifest, Mapping):
        raise ValueError("source process summary has no dynamic_manifest object")
    dynamic_manifest = dict(source_dynamic_manifest)
    records: list[dict[str, Any]] = list(public_discovery.skipped)
    recorded_episode_ids = {
        int(record["episode"])
        for record in records
        if "episode" in record
    }
    for excluded_record in all_excluded:
        episode_id = int(excluded_record["episode"])
        if episode_id not in recorded_episode_ids:
            records.append(dict(excluded_record))
            recorded_episode_ids.add(episode_id)
    render_report: dict[str, Any] | None = None
    renderable_ids: list[int] = []
    published_source_lineages: dict[int, Mapping[str, Any]] = {}
    published_contexts: dict[int, dict[str, Any]] = {}

    if dry_run:
        records.extend(
            {
                "episode": episode_id,
                "status": "planned",
                "gripper_backend": "urdf",
                "source_lineage_sha256": selection.source_lineages[episode_id][
                    "lineage_sha256"
                ],
            }
            for episode_id in selection.episode_ids
        )
    else:
        raw_episodes = result.get("episodes")
        backend_records = raw_episodes if isinstance(raw_episodes, list) else []
        backend_by_episode: dict[int, Mapping[str, Any]] = {}
        for raw_record in backend_records:
            if not isinstance(raw_record, Mapping) or "episode_index" not in raw_record:
                continue
            backend_by_episode[int(raw_record["episode_index"])] = raw_record

        publisher = publish_urdf_episode if episode_publisher is None else episode_publisher
        for episode_id in selection.episode_ids:
            backend_record = backend_by_episode.get(episode_id)
            if backend_record is None:
                records.append(
                    {
                        "episode": episode_id,
                        "status": "failed",
                        "gripper_backend": "urdf",
                        "error": "URDF backend manifest has no episode record",
                    }
                )
                continue
            if backend_record.get("status") != "complete":
                backend_error = str(
                    backend_record.get("error", "URDF backend episode is incomplete")
                )
                records.append(
                    {
                        "episode": episode_id,
                        "status": (
                            "gripper_incomplete"
                            if "eligible nonempty fraction" in backend_error
                            else "failed"
                        ),
                        "gripper_backend": "urdf",
                        "error": backend_error,
                        "backend_status": backend_record.get("status"),
                    }
                )
                continue
            frozen_lineage = selection.source_lineages[episode_id]
            if backend_record.get("source_lineage") != frozen_lineage:
                records.append(
                    {
                        "episode": episode_id,
                        "status": "failed",
                        "gripper_backend": "urdf",
                        "error": (
                            "URDF backend source lineage differs from the "
                            "preflight source contract"
                        ),
                    }
                )
                continue
            source_episode_dir = (
                config.source_run_dir
                / task
                / f"episode_{episode_id:06d}"
                / camera
            )
            backend_episode_dir = config.run_dir / str(
                backend_record.get("output_dir", f"episode_{episode_id:06d}")
            )
            destination_dir = (
                canonical_run_dir
                / task
                / f"episode_{episode_id:06d}"
                / camera
            )
            try:
                published = publisher(
                    source_episode_dir=source_episode_dir,
                    backend_episode_dir=backend_episode_dir,
                    destination_dir=destination_dir,
                    run_id=selected_run_id,
                    task=task,
                    camera=camera,
                    backend_episode_record=backend_record,
                    resume=resume,
                )
            except Exception as exc:
                records.append(
                    {
                        "episode": episode_id,
                        "status": "failed",
                        "gripper_backend": "urdf",
                        "error": f"canonical publish failed: {type(exc).__name__}: {exc}",
                    }
                )
                continue
            publish_status = str(published.get("status", "published"))
            records.append(
                {
                    "episode": episode_id,
                    "status": (
                        "skipped_complete"
                        if publish_status in {"validated_skip", "skipped_complete"}
                        else "completed"
                    ),
                    "gripper_backend": "urdf",
                    "active_arm": backend_record.get("active_arm"),
                    "artifact_dir": str(destination_dir),
                    "backend_output_dir": str(backend_episode_dir),
                    "publish_status": publish_status,
                    "source_lineage_sha256": frozen_lineage["lineage_sha256"],
                }
            )
            renderable_ids.append(episode_id)
            published_source_lineages[episode_id] = frozen_lineage
            published_contexts[episode_id] = {
                "source_episode_dir": source_episode_dir,
                "backend_episode_dir": backend_episode_dir,
                "destination_dir": destination_dir,
                "backend_episode_record": backend_record,
            }

        if not skip_render and renderable_ids:
            pre_render_failures: list[dict[str, Any]] = []
            canonical_validator = (
                validate_published_urdf_episode
                if episode_validator is None
                else episode_validator
            )
            for episode_id in renderable_ids:
                context = published_contexts[episode_id]
                try:
                    current_source = validate_derivation_source_episode(
                        context["source_episode_dir"],
                        task=task,
                        camera=camera,
                        episode_index=episode_id,
                        expected_frame_count=expected_frame_counts[episode_id],
                        expected_dataset_root=config.dataset_root,
                    )
                    if current_source.lineage != published_source_lineages[episode_id]:
                        raise UrdfGripperPublishError(
                            "source lineage differs from the published episode"
                        )
                except (FileNotFoundError, UrdfGripperPublishError) as exc:
                    pre_render_failures.append(
                        {
                            "episode": episode_id,
                            "status": "source_lineage_changed",
                            "gripper_backend": "urdf",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    continue
                try:
                    canonical_validator(
                        **context,
                        run_id=selected_run_id,
                        task=task,
                        camera=camera,
                    )
                except Exception as exc:
                    pre_render_failures.append(
                        {
                            "episode": episode_id,
                            "status": "canonical_validation_failed",
                            "gripper_backend": "urdf",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
            if pre_render_failures:
                records.extend(pre_render_failures)
            else:
                try:
                    if pipeline_config is None:
                        raise ValueError(
                            "pipeline_config is required to render canonical URDF output"
                        )
                    dynamic = _dynamic_config(
                        pipeline_config,
                        root=config.dataset_root,
                        task=task,
                        camera=camera,
                        manifest=dynamic_manifest,
                        output_root=resolved_output_root,
                    )
                    builder = _render_processed if render_builder is None else render_builder
                    render_report = builder(
                        dynamic,
                        run_id=selected_run_id,
                        episode_ids=tuple(renderable_ids),
                        output_dir=canonical_run_dir,
                    )
                except Exception as exc:
                    records.append(
                        {
                            "status": "render_failed",
                            "gripper_backend": "urdf",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )

    failure_statuses = {
        "failed",
        "gripper_incomplete",
        "render_failed",
        "source_lineage_changed",
        "canonical_validation_failed",
    }
    summary = {
        "format_version": "robotwin_process_dataset_summary_v1",
        "gripper_backend": "urdf",
        "run_id": selected_run_id,
        "dataset_root": str(config.dataset_root),
        "task": task,
        "camera": camera,
        "discovered_episode_ids": list(public_discovery.episode_ids),
        "requested_episode_ids": (
            list(public_discovery.episode_ids)
            if episode_ids is None
            else list(episode_ids)
        ),
        "dynamic_manifest": dynamic_manifest,
        "qwen_health": None,
        "records": records,
        "render": render_report,
        "fatal_error": batch_error,
        "backend": {
            "type": "urdf",
            "source_run_dir": str(config.source_run_dir),
            "source_run_id": selection.source_summary["run_id"],
            "source_lineage_sha256_by_episode": {
                str(episode_id): lineage["lineage_sha256"]
                for episode_id, lineage in selection.source_lineages.items()
            },
            "selected_episode_ids": list(selection.episode_ids),
            "dataset_excluded": relevant_dataset_excluded,
            "source_excluded": list(selection.excluded),
            "source_selection_complete": not all_excluded,
            "allow_partial_source": allow_partial_source,
            "run_dir": str(config.run_dir),
            "manifest": None if dry_run else str(config.run_dir / "manifest.json"),
            "status": result.get("status"),
            "error": batch_error,
        },
        "passed": (
            batch_error is None
            and not any(
                record.get("status") in failure_statuses for record in records
            )
        ),
    }
    if dry_run:
        summary["plan"] = result
        return summary
    summary_path = ArtifactStore.write_json(
        canonical_run_dir / "process_summary.json",
        summary,
    )
    summary["artifact"] = str(summary_path)
    return summary


def process_dataset(
    config: PipelineConfig,
    *,
    dataset_root: Path,
    task: str,
    camera: str,
    output_root: Path,
    run_id: str | None = None,
    episode_ids: tuple[int, ...] | None = None,
    force: bool = False,
    skip_render: bool = False,
    backend_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    if run_id is not None:
        _validate_run_id(run_id)
    runtime = _load_sam_runtime()
    discovery = discover_episodes(dataset_root, camera=camera)
    if not discovery.episodes:
        raise ValueError(f"no complete episodes found under {dataset_root}")
    manifest = build_dynamic_manifest(
        dataset_root,
        task=task,
        camera=camera,
        episodes=discovery.episodes,
    )
    discovered_ids = set(discovery.episode_ids)
    selected_ids = (
        discovery.episode_ids
        if episode_ids is None
        else tuple(dict.fromkeys(int(value) for value in episode_ids))
    )
    if not selected_ids:
        raise ValueError("process_dataset requires at least one selected episode")
    unknown = sorted(set(selected_ids) - discovered_ids)
    if unknown:
        raise ValueError(f"requested episodes were not discovered: {unknown}")
    dynamic = _dynamic_config(
        config,
        root=dataset_root,
        task=task,
        camera=camera,
        manifest=manifest,
        output_root=output_root,
    )
    store = ArtifactStore(dynamic.output_root)
    selected_run_id = _validate_run_id(run_id or store.new_run_id())
    canonical_run_dir = store.run_dir(selected_run_id)
    _validate_sam_run_ownership(canonical_run_dir, run_id=selected_run_id)
    qwen = runtime.qwen_client_factory(
        endpoint=dynamic.qwen.endpoint,
        model=dynamic.qwen.model,
        timeout_seconds=dynamic.qwen.timeout_seconds,
    )
    health = qwen.health()
    records: list[dict[str, Any]] = list(discovery.skipped)
    pending: list[int] = []
    for episode_id in selected_ids:
        ref = EpisodeRef(
            task,
            episode_id,
            camera,
        )
        if (
            not force
            and runtime.gripper_episode_complete(dynamic, store, selected_run_id, ref)
        ):
            records.append({"episode": episode_id, "status": "skipped_complete"})
        else:
            pending.append(episode_id)

    factory = runtime.backend_factory if backend_factory is None else backend_factory
    backend: Any | None = None
    fatal_error: BaseException | None = None
    try:
        if pending:
            backend = factory(
                checkpoint_path=dynamic.sam3.checkpoint,
                gpus=dynamic.sam3.gpus,
            )
            for episode_id in pending:
                try:
                    runtime.run_qwen(dynamic, episode_id, selected_run_id)
                    sam_execution = runtime.execute_sam_episode(
                        dynamic,
                        episode_id,
                        selected_run_id,
                        backend,
                    )
                    if not runtime.emit_sam_result(selected_run_id, sam_execution):
                        records.append(
                            {"episode": episode_id, "status": "sam_incomplete"}
                        )
                        continue
                    gripper_execution = runtime.execute_gripper_episode(
                        dynamic,
                        episode_id,
                        selected_run_id,
                        backend,
                    )
                    records.append(
                        {
                            "episode": episode_id,
                            "status": (
                                "completed"
                                if runtime.emit_gripper_result(
                                    selected_run_id,
                                    gripper_execution,
                                )
                                else "gripper_incomplete"
                            ),
                        }
                    )
                except SystemExit as exc:
                    records.append(
                        {
                            "episode": episode_id,
                            "status": "failed",
                            "error": f"stage exited with code {exc.code}",
                        }
                    )
                except runtime.execution_errors as exc:
                    records.append(
                        {
                            "episode": episode_id,
                            "status": "failed",
                            "error": str(exc),
                        }
                    )
                    if runtime.fatal_cuda_error(exc):
                        fatal_error = exc
                        break
    finally:
        if backend is not None:
            backend.shutdown()

    if fatal_error is not None:
        recorded_ids = {
            int(record["episode"])
            for record in records
            if "episode" in record and str(record["episode"]).isdigit()
        }
        records.extend(
            {"episode": episode_id, "status": "not_run_after_fatal_cuda"}
            for episode_id in selected_ids
            if episode_id not in recorded_ids
        )

    render_report: dict[str, Any] | None = None
    renderable_ids = tuple(
        int(record["episode"])
        for record in records
        if record.get("status") in {"completed", "skipped_complete"}
    )
    if not skip_render and fatal_error is None and renderable_ids:
        try:
            render_report = _render_processed(
                dynamic,
                run_id=selected_run_id,
                episode_ids=renderable_ids,
                output_dir=canonical_run_dir,
            )
        except Exception as exc:
            records.append({"status": "render_failed", "error": str(exc)})

    summary = {
        "format_version": "robotwin_process_dataset_summary_v1",
        "gripper_backend": "sam",
        "run_id": selected_run_id,
        "dataset_root": str(dataset_root.expanduser().resolve()),
        "task": task,
        "camera": camera,
        "discovered_episode_ids": list(discovery.episode_ids),
        "requested_episode_ids": list(selected_ids),
        "dynamic_manifest": manifest,
        "qwen_health": health,
        "records": records,
        "render": render_report,
        "fatal_error": None if fatal_error is None else str(fatal_error),
        "backend": {"type": "sam"},
    }
    failure_statuses = {
        "failed",
        "sam_incomplete",
        "gripper_incomplete",
        "not_run_after_fatal_cuda",
        "render_failed",
    }
    summary["passed"] = (
        fatal_error is None
        and not any(record.get("status") in failure_statuses for record in records)
    )
    summary_path = store.write_json(
        canonical_run_dir / "process_summary.json",
        summary,
    )
    summary["artifact"] = str(summary_path)
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pilot_move_pillbottle_pad.yaml"),
    )
    parser.add_argument("--task")
    parser.add_argument("--camera")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--run-id")
    parser.add_argument("--episode-ids", type=int, nargs="*")
    parser.add_argument(
        "--gripper-backend",
        choices=GRIPPER_BACKENDS,
        default="sam",
        help="Use the existing SAM gripper stage or replace it from a frozen run with URDF",
    )
    parser.add_argument(
        "--source-run-dir",
        help="Frozen just-process run containing QC-passed target/receiver masks",
    )
    parser.add_argument(
        "--urdf-path",
        help="RoboTwin Aloha URDF; required for --gripper-backend urdf",
    )
    parser.add_argument("--urdf-mesh-root", type=Path)
    parser.add_argument("--urdf-depth-tolerance-mm", type=float)
    parser.add_argument(
        "--urdf-minimum-eligible-nonempty-fraction",
        type=float,
    )
    parser.add_argument("--urdf-fit-config-json", type=Path)
    parser.add_argument(
        "--allow-partial-source",
        action="store_true",
        help=(
            "During automatic episode discovery, process only source episodes whose "
            "target and receiver passed QC; explicit --episode-ids remain fail-closed"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
    return parser.parse_args(argv)


def _optional_cli_path(value: str | None) -> Path | None:
    if value is None or value.strip() in {"", "-"}:
        return None
    return Path(value)


def main() -> None:
    args = _parse_args()
    if args.run_id is not None:
        _validate_run_id(args.run_id)
    config = load_config(args.config)
    source_run_dir = _optional_cli_path(args.source_run_dir)
    urdf_path = _optional_cli_path(args.urdf_path)
    if args.gripper_backend == "sam":
        if (
            source_run_dir is not None
            or urdf_path is not None
            or args.urdf_mesh_root is not None
            or args.urdf_depth_tolerance_mm is not None
            or args.urdf_minimum_eligible_nonempty_fraction is not None
            or args.urdf_fit_config_json is not None
            or args.allow_partial_source
        ):
            raise ValueError("URDF-only options require --gripper-backend urdf")
        if args.dry_run or args.resume:
            raise ValueError("--dry-run/--resume are only supported by the URDF backend")
        dataset_root = (
            config.dataset.root if args.dataset_root is None else args.dataset_root
        )
        task = config.dataset.task if args.task is None else args.task
        camera = config.dataset.camera if args.camera is None else args.camera
        summary = process_dataset(
            config,
            dataset_root=dataset_root,
            task=task,
            camera=camera,
            output_root=args.output_dir,
            run_id=args.run_id,
            episode_ids=None if args.episode_ids is None else tuple(args.episode_ids),
            force=args.force,
            skip_render=args.skip_render,
        )
    else:
        if source_run_dir is None:
            raise ValueError("--source-run-dir is required for the URDF backend")
        if urdf_path is None:
            raise ValueError("--urdf-path is required for the URDF backend")
        if args.force:
            raise ValueError(
                "--force is not supported by the immutable URDF backend; use a new run id"
            )
        if args.resume and not args.run_id:
            raise ValueError("--resume requires an explicit --run-id")
        if args.dry_run and args.resume:
            raise ValueError("--dry-run and --resume cannot be used together")
        source_summary = _read_json_object(
            source_run_dir.expanduser().resolve() / "process_summary.json",
            description="source process summary",
        )
        dataset_root = (
            Path(str(source_summary.get("dataset_root", "")))
            if args.dataset_root is None
            else args.dataset_root
        )
        task = str(source_summary.get("task", "")) if args.task is None else args.task
        camera = (
            str(source_summary.get("camera", ""))
            if args.camera is None
            else args.camera
        )
        if not task or not camera:
            raise ValueError("source process summary does not define task/camera")
        summary = process_urdf_source_run(
            pipeline_config=config,
            dataset_root=dataset_root,
            source_run_dir=source_run_dir,
            task=task,
            camera=camera,
            output_root=args.output_dir,
            urdf_path=urdf_path,
            mesh_root=args.urdf_mesh_root,
            run_id=args.run_id,
            episode_ids=(
                None if args.episode_ids is None else tuple(args.episode_ids)
            ),
            skip_render=args.skip_render,
            dry_run=args.dry_run,
            resume=args.resume,
            depth_tolerance_mm=(
                DEFAULT_URDF_DEPTH_TOLERANCE_MM
                if args.urdf_depth_tolerance_mm is None
                else args.urdf_depth_tolerance_mm
            ),
            minimum_eligible_nonempty_fraction=(
                DEFAULT_URDF_MINIMUM_ELIGIBLE_NONEMPTY_FRACTION
                if args.urdf_minimum_eligible_nonempty_fraction is None
                else args.urdf_minimum_eligible_nonempty_fraction
            ),
            fit_config_json=args.urdf_fit_config_json,
            allow_partial_source=args.allow_partial_source,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
