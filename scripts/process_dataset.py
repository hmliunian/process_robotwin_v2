#!/usr/bin/env python3
"""Process every complete episode in a RoboTwin-format directory."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import av
import numpy as np
import pandas as pd

from robotwin_annotation_v2.adapters import (
    ArtifactStore,
    OpenAICompatibleQwenClient,
    RoboTwinDataset,
    Sam3Adapter,
)
from robotwin_annotation_v2.config import PipelineConfig, load_config
from robotwin_annotation_v2.models import EpisodeRef

try:
    from run_target_receiver import (
        SAM_EXECUTION_ERRORS,
        _emit_gripper_result,
        _emit_sam_result,
        _execute_gripper_episode,
        _execute_sam_episode,
        _fatal_cuda_error,
        _gripper_episode_complete,
        run_qwen,
    )
except ModuleNotFoundError:
    from scripts.run_target_receiver import (
        SAM_EXECUTION_ERRORS,
        _emit_gripper_result,
        _emit_sam_result,
        _execute_gripper_episode,
        _execute_sam_episode,
        _fatal_cuda_error,
        _gripper_episode_complete,
        run_qwen,
    )


CHUNK_PATTERN = re.compile(r"chunk-(\d{3})$")
EPISODE_FILE_PATTERN = re.compile(r"episode_(\d+)\.parquet$")


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


def discover_episodes(root: Path, *, camera: str) -> DiscoveryResult:
    """Discover complete parquet/video/sidecar triplets by episode id."""

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
            missing = [
                name
                for name, path in (
                    ("video", video),
                    ("sidecar", sidecar),
                )
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


def _measure_episode(episode: DiscoveredEpisode) -> tuple[int, tuple[int, int], int]:
    frame = pd.read_parquet(episode.parquet, columns=["frame_index"])
    if frame.empty:
        raise ValueError(f"episode parquet is empty: {episode.parquet}")
    frame_indices = frame["frame_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(frame_indices, np.arange(len(frame_indices))):
        raise ValueError(f"episode frame_index is not contiguous: {episode.parquet}")
    frame_count = len(frame_indices)
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
    backend_factory: Callable[..., Sam3Adapter] | None = None,
) -> dict[str, Any]:
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
    selected_run_id = run_id or store.new_run_id()
    qwen = OpenAICompatibleQwenClient(
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
        if not force and _gripper_episode_complete(dynamic, store, selected_run_id, ref):
            records.append({"episode": episode_id, "status": "skipped_complete"})
        else:
            pending.append(episode_id)

    factory = Sam3Adapter if backend_factory is None else backend_factory
    backend: Sam3Adapter | None = None
    fatal_error: BaseException | None = None
    try:
        if pending:
            backend = factory(
                checkpoint_path=dynamic.sam3.checkpoint,
                gpus=dynamic.sam3.gpus,
            )
            for episode_id in pending:
                try:
                    run_qwen(dynamic, episode_id, selected_run_id)
                    sam_execution = _execute_sam_episode(
                        dynamic,
                        episode_id,
                        selected_run_id,
                        backend,
                    )
                    if not _emit_sam_result(selected_run_id, sam_execution):
                        records.append(
                            {"episode": episode_id, "status": "sam_incomplete"}
                        )
                        continue
                    gripper_execution = _execute_gripper_episode(
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
                                if _emit_gripper_result(selected_run_id, gripper_execution)
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
                except SAM_EXECUTION_ERRORS as exc:
                    records.append(
                        {
                            "episode": episode_id,
                            "status": "failed",
                            "error": str(exc),
                        }
                    )
                    if _fatal_cuda_error(exc):
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
                output_dir=dynamic.output_root / selected_run_id,
            )
        except Exception as exc:
            records.append({"status": "render_failed", "error": str(exc)})

    summary = {
        "format_version": "robotwin_process_dataset_summary_v1",
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
        store.run_dir(selected_run_id) / "process_summary.json",
        summary,
    )
    summary["artifact"] = str(summary_path)
    return summary


def _parse_args() -> argparse.Namespace:
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
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
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
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
