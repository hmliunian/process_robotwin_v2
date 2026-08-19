"""Frozen-source URDF dataset workflow.

This module owns source selection, backend execution, canonical publication,
revalidation, rendering, and summary policy.  OpenGL/renderer code remains
behind the lazy runtime hook.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from robotwin_annotation_v2.adapters.artifact_store import ArtifactStore
from robotwin_annotation_v2.config import PipelineConfig
from robotwin_annotation_v2.domain import AnnotationMode, ObjectRole
from robotwin_annotation_v2.models import EpisodeRecord, ProcessSummary
from robotwin_annotation_v2.terminal_ui import ProcessUI
from robotwin_annotation_v2.urdf_gripper_publisher import (
    DerivationSourceEpisode,
    UrdfGripperPublishError,
)

from .discovery import DiscoveryResult

DEFAULT_URDF_DEPTH_TOLERANCE_MM = 8.0
DEFAULT_URDF_MINIMUM_ELIGIBLE_NONEMPTY_FRACTION = 0.90


class UrdfRunConfig(Protocol):
    """Backend configuration attributes consumed after construction."""

    dataset_root: Path
    source_run_dir: Path

    @property
    def run_dir(self) -> Path: ...


@dataclass(frozen=True)
class UrdfSourceSelection:
    """Frozen, QC-passed source masks selected for URDF replacement."""

    episode_ids: tuple[int, ...]
    excluded: tuple[dict[str, Any], ...]
    source_summary: Mapping[str, Any]
    source_lineages: Mapping[int, Mapping[str, Any]]
    annotation_mode: AnnotationMode
    required_object_roles: tuple[ObjectRole, ...]


@dataclass(frozen=True)
class UrdfWorkflowRuntime:
    """Lazy backend and publisher operations used by the workflow."""

    new_run_id: Callable[[], str]
    run_config_factory: Callable[..., UrdfRunConfig]
    run_experiment: Callable[[UrdfRunConfig], Mapping[str, Any]]
    batch_incomplete_errors: tuple[type[BaseException], ...]
    incomplete_result: Callable[[BaseException], Mapping[str, Any]]
    publish_episode: Callable[..., Mapping[str, Any]]
    validate_episode: Callable[..., Mapping[str, Any]]


@dataclass(frozen=True)
class UrdfWorkflowHooks:
    """Compatibility seams resolved from legacy runtime globals per call."""

    runtime_loader: Callable[[], UrdfWorkflowRuntime]
    discover_episodes: Callable[..., DiscoveryResult]
    parquet_frame_count: Callable[[Path], int]
    select_source_episodes: Callable[..., UrdfSourceSelection]
    validate_run_id: Callable[[str], str]
    validate_run_ownership: Callable[..., None]
    capture_progress: Callable[[ProcessUI | None], AbstractContextManager[None]]
    validate_source_episode: Callable[..., DerivationSourceEpisode]
    build_dynamic_config: Callable[..., PipelineConfig]
    render_processed: Callable[..., dict[str, Any]]
    summary_format_version: str


@dataclass(frozen=True)
class UrdfWorkflow:
    """Coordinate a complete frozen-source URDF derivation run."""

    hooks: UrdfWorkflowHooks

    def run(
        self,
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
        source_mode: str = "frozen_run",
        source_release: Mapping[str, Any] | None = None,
        experiment_runner: Callable[..., Mapping[str, Any]] | None = None,
        episode_publisher: Callable[..., Mapping[str, Any]] | None = None,
        episode_validator: Callable[..., Mapping[str, Any]] | None = None,
        render_builder: Callable[..., dict[str, Any]] | None = None,
        prepared_backend_result: Mapping[str, Any] | None = None,
        prepared_backend_error: str | None = None,
        egl_device_id: int | None = None,
        report_lifecycle: bool = True,
        pipeline_episode_ids: tuple[int, ...] | None = None,
        reporter: ProcessUI | None = None,
    ) -> dict[str, Any]:
        """Replace gripper masks in a frozen process run with canonical URDF output."""

        if dry_run and resume:
            raise ValueError("--dry-run and --resume cannot be used together")
        if run_id is not None:
            self.hooks.validate_run_id(run_id)
        if reporter is not None and report_lifecycle:
            reporter.run_started(
                backend="urdf",
                dataset_root=str(dataset_root.expanduser().resolve()),
                task=task,
                camera=camera,
            )
            reporter.phase_started("dataset_contract")

        runtime = self.hooks.runtime_loader()

        public_discovery = self.hooks.discover_episodes(dataset_root, camera=camera)
        discovery = self.hooks.discover_episodes(dataset_root, camera=camera, require_depth=True)
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
                    self.hooks.parquet_frame_count(discovered_episode.parquet)
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
        if reporter is not None:
            reporter.phase_finished(
                "dataset_contract",
                detail=(
                    f"public={len(public_discovery.episodes)} "
                    f"depth_eligible={len(expected_frame_counts)} "
                    f"excluded={len(dataset_excluded)}"
                ),
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
                    f"requested URDF episodes do not satisfy the dataset contract: {rendered}"
                )
        if not eligible_dataset_ids:
            raise ValueError(f"no complete URDF episodes found under {dataset_root}")
        if reporter is not None:
            reporter.phase_started("source_selection", total=len(eligible_dataset_ids))
        selection = self.hooks.select_source_episodes(
            source_run_dir,
            dataset_root=dataset_root,
            task=task,
            camera=camera,
            discovered_episode_ids=eligible_dataset_ids,
            requested_episode_ids=episode_ids,
            expected_frame_counts=expected_frame_counts,
        )
        if (
            pipeline_config is not None
            and pipeline_config.annotation.mode is not selection.annotation_mode
        ):
            raise ValueError(
                "pipeline config annotation mode differs from the frozen source run: "
                f"{pipeline_config.annotation.mode.value} != {selection.annotation_mode.value}"
            )
        all_excluded = sorted(
            [*relevant_dataset_excluded, *selection.excluded],
            key=lambda record: int(record["episode"]),
        )
        if reporter is not None:
            reporter.phase_finished(
                "source_selection",
                detail=(f"selected={len(selection.episode_ids)} excluded={len(all_excluded)}"),
            )
        if all_excluded and not allow_partial_source:
            examples = ", ".join(
                f"{record['episode']} ({record['reason']})" for record in all_excluded[:10]
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
            raise ValueError("URDF minimum eligible nonempty fraction must be finite and in [0, 1]")
        selected_run_id = self.hooks.validate_run_id(run_id or runtime.new_run_id())
        resolved_output_root = output_root.expanduser().resolve()
        canonical_run_dir = resolved_output_root / selected_run_id
        self.hooks.validate_run_ownership(
            canonical_run_dir,
            run_id=selected_run_id,
            resume=resume or prepared_backend_result is not None,
        )
        if reporter is not None and report_lifecycle:
            reporter.run_ready(run_id=selected_run_id, episode_ids=selection.episode_ids)
            if all_excluded:
                reporter.note(
                    f"processing partial source: excluded={len(all_excluded)}",
                    level="warning",
                )
        backend_output_root = canonical_run_dir / "_backend"
        config = runtime.run_config_factory(
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
            egl_device_id=egl_device_id,
        )
        runner = runtime.run_experiment if experiment_runner is None else experiment_runner
        result: Mapping[str, Any]
        batch_error: str | None = prepared_backend_error
        if reporter is not None:
            reporter.phase_started("urdf_backend", total=len(selection.episode_ids))
        if prepared_backend_result is not None:
            result = prepared_backend_result
        else:
            try:
                with self.hooks.capture_progress(reporter):
                    result = runner(config)
            except runtime.batch_incomplete_errors as exc:
                result = runtime.incomplete_result(exc)
                batch_error = f"{type(exc).__name__}: {exc}"
        if reporter is not None:
            reporter.phase_finished(
                "urdf_backend",
                status="completed" if batch_error is None else "failed",
                detail=batch_error,
            )
            if not report_lifecycle:
                reporter.lane_progress(
                    "urdf",
                    len(selection.episode_ids),
                    len(selection.episode_ids),
                    status="completed" if batch_error is None else "failed",
                    detail=batch_error,
                )
                reporter.lane_finished(
                    "urdf",
                    status="completed" if batch_error is None else "failed",
                    detail=batch_error,
                )

        source_dynamic_manifest = selection.source_summary.get("dynamic_manifest")
        if not isinstance(source_dynamic_manifest, Mapping):
            raise ValueError(  # noqa: TRY004 - preserve the public error contract
                "source process summary has no dynamic_manifest object"
            )
        dynamic_manifest = dict(source_dynamic_manifest)
        source_annotation_mode = selection.annotation_mode.value
        source_required_roles = [role.value for role in selection.required_object_roles]
        records: list[dict[str, Any]] = list(public_discovery.skipped)
        recorded_episode_ids = {int(record["episode"]) for record in records if "episode" in record}
        for excluded_record in all_excluded:
            episode_id = int(excluded_record["episode"])
            if episode_id not in recorded_episode_ids:
                records.append(dict(excluded_record))
                recorded_episode_ids.add(episode_id)
        render_report: dict[str, Any] | None = None
        renderable_ids: list[int] = []
        published_source_lineages: dict[int, Mapping[str, Any]] = {}
        published_contexts: dict[int, dict[str, Any]] = {}
        published_episode_statuses: dict[int, str] = {}
        report_pipeline_lanes = reporter is not None and not report_lifecycle
        lane_episode_ids = (
            selection.episode_ids if pipeline_episode_ids is None else pipeline_episode_ids
        )
        lane_positions = {
            episode_id: position for position, episode_id in enumerate(lane_episode_ids, start=1)
        }
        lane_total = len(lane_episode_ids)

        def publish_progress(
            position: int,
            episode_id: int,
            status: str,
            detail: str | None = None,
        ) -> None:
            if report_pipeline_lanes and reporter is not None:
                reporter.lane_progress(
                    "publish",
                    lane_positions.get(episode_id, position),
                    lane_total,
                    episode_id,
                    status,
                    detail,
                )

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
            if reporter is not None:
                for episode_id in selection.episode_ids:
                    reporter.episode_finished(episode_id, status="planned")
        else:
            raw_episodes = result.get("episodes")
            backend_records = raw_episodes if isinstance(raw_episodes, list) else []
            backend_by_episode: dict[int, Mapping[str, Any]] = {}
            for raw_record in backend_records:
                if not isinstance(raw_record, Mapping) or "episode_index" not in raw_record:
                    continue
                backend_by_episode[int(raw_record["episode_index"])] = raw_record

            publisher = runtime.publish_episode if episode_publisher is None else episode_publisher
            for position, episode_id in enumerate(selection.episode_ids, start=1):
                if reporter is not None:
                    reporter.episode_started(
                        episode_id,
                        position=position,
                        total=len(selection.episode_ids),
                    )
                backend_record = backend_by_episode.get(episode_id)
                if backend_record is None:
                    error = "URDF backend manifest has no episode record"
                    records.append(
                        {
                            "episode": episode_id,
                            "status": "failed",
                            "gripper_backend": "urdf",
                            "error": error,
                        }
                    )
                    if reporter is not None:
                        reporter.episode_finished(
                            episode_id,
                            status="failed",
                            detail=error,
                        )
                    publish_progress(position, episode_id, "failed", error)
                    continue
                if backend_record.get("status") != "complete":
                    backend_error = str(
                        backend_record.get("error", "URDF backend episode is incomplete")
                    )
                    episode_status = (
                        "gripper_incomplete"
                        if "eligible nonempty fraction" in backend_error
                        else "failed"
                    )
                    records.append(
                        {
                            "episode": episode_id,
                            "status": episode_status,
                            "gripper_backend": "urdf",
                            "error": backend_error,
                            "backend_status": backend_record.get("status"),
                        }
                    )
                    if reporter is not None:
                        reporter.episode_finished(
                            episode_id,
                            status=episode_status,
                            detail=backend_error,
                        )
                    publish_progress(position, episode_id, episode_status, backend_error)
                    continue
                frozen_lineage = selection.source_lineages[episode_id]
                if backend_record.get("source_lineage") != frozen_lineage:
                    error = "URDF backend source lineage differs from the preflight source contract"
                    records.append(
                        {
                            "episode": episode_id,
                            "status": "failed",
                            "gripper_backend": "urdf",
                            "error": error,
                        }
                    )
                    if reporter is not None:
                        reporter.episode_finished(
                            episode_id,
                            status="failed",
                            detail=error,
                        )
                    publish_progress(position, episode_id, "failed", error)
                    continue
                source_episode_dir = (
                    config.source_run_dir / task / f"episode_{episode_id:06d}" / camera
                )
                backend_episode_dir = config.run_dir / str(
                    backend_record.get("output_dir", f"episode_{episode_id:06d}")
                )
                destination_dir = canonical_run_dir / task / f"episode_{episode_id:06d}" / camera
                if reporter is not None:
                    reporter.stage_started(episode_id, "canonical_publish")
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
                except Exception as exc:  # noqa: BLE001 - publish errors are per-episode
                    error = f"canonical publish failed: {type(exc).__name__}: {exc}"
                    records.append(
                        {
                            "episode": episode_id,
                            "status": "failed",
                            "gripper_backend": "urdf",
                            "error": error,
                        }
                    )
                    if reporter is not None:
                        reporter.stage_finished(
                            episode_id,
                            "canonical_publish",
                            status="failed",
                            detail=error,
                        )
                        reporter.episode_finished(
                            episode_id,
                            status="failed",
                            detail=error,
                        )
                    publish_progress(position, episode_id, "failed", error)
                    continue
                publish_status = str(published.get("status", "published"))
                episode_status = (
                    "skipped_complete"
                    if publish_status in {"validated_skip", "skipped_complete"}
                    else "completed"
                )
                records.append(
                    {
                        "episode": episode_id,
                        "status": episode_status,
                        "gripper_backend": "urdf",
                        "active_arm": backend_record.get("active_arm"),
                        "artifact_dir": str(destination_dir),
                        "backend_output_dir": str(backend_episode_dir),
                        "publish_status": publish_status,
                        "source_lineage_sha256": frozen_lineage["lineage_sha256"],
                    }
                )
                if reporter is not None:
                    reporter.stage_finished(
                        episode_id,
                        "canonical_publish",
                        status=episode_status,
                    )
                    if skip_render:
                        reporter.episode_finished(episode_id, status=episode_status)
                publish_progress(position, episode_id, episode_status)
                renderable_ids.append(episode_id)
                published_source_lineages[episode_id] = frozen_lineage
                published_episode_statuses[episode_id] = episode_status
                published_contexts[episode_id] = {
                    "source_episode_dir": source_episode_dir,
                    "backend_episode_dir": backend_episode_dir,
                    "destination_dir": destination_dir,
                    "backend_episode_record": backend_record,
                }

            if report_pipeline_lanes and reporter is not None:
                reporter.lane_progress(
                    "publish",
                    lane_total,
                    lane_total,
                    status="completed",
                )
                reporter.lane_finished("publish")

            if not skip_render and renderable_ids:
                pre_render_failures: list[dict[str, Any]] = []
                canonical_validator = (
                    runtime.validate_episode if episode_validator is None else episode_validator
                )
                if reporter is not None:
                    reporter.phase_started(
                        "canonical_validation",
                        total=len(renderable_ids),
                    )
                selection_positions = lane_positions
                for phase_position, episode_id in enumerate(renderable_ids, start=1):
                    position = selection_positions[episode_id]
                    context = published_contexts[episode_id]
                    try:
                        current_source = self.hooks.validate_source_episode(
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
                        if reporter is not None:
                            reporter.phase_progress(
                                phase_position,
                                total=len(renderable_ids),
                                episode_id=episode_id,
                                status="source_lineage_changed",
                            )
                            reporter.episode_finished(
                                episode_id,
                                status="source_lineage_changed",
                                detail=f"{type(exc).__name__}: {exc}",
                            )
                            if report_pipeline_lanes:
                                reporter.lane_progress(
                                    "validation",
                                    position,
                                    lane_total,
                                    episode_id,
                                    "source_lineage_changed",
                                    str(exc),
                                )
                        continue
                    try:
                        canonical_validator(
                            **context,
                            run_id=selected_run_id,
                            task=task,
                            camera=camera,
                        )
                    except Exception as exc:  # noqa: BLE001 - validation is fail-closed
                        pre_render_failures.append(
                            {
                                "episode": episode_id,
                                "status": "canonical_validation_failed",
                                "gripper_backend": "urdf",
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                        if reporter is not None:
                            reporter.phase_progress(
                                phase_position,
                                total=len(renderable_ids),
                                episode_id=episode_id,
                                status="canonical_validation_failed",
                            )
                            reporter.episode_finished(
                                episode_id,
                                status="canonical_validation_failed",
                                detail=f"{type(exc).__name__}: {exc}",
                            )
                            if report_pipeline_lanes:
                                reporter.lane_progress(
                                    "validation",
                                    position,
                                    lane_total,
                                    episode_id,
                                    "canonical_validation_failed",
                                    str(exc),
                                )
                        continue
                    if reporter is not None:
                        reporter.phase_progress(
                            phase_position,
                            total=len(renderable_ids),
                            episode_id=episode_id,
                            status="validated",
                        )
                        if report_pipeline_lanes:
                            reporter.lane_progress(
                                "validation",
                                position,
                                lane_total,
                                episode_id,
                                "validated",
                            )
                        reporter.episode_finished(
                            episode_id,
                            status=published_episode_statuses[episode_id],
                        )
                if report_pipeline_lanes and reporter is not None:
                    reporter.lane_progress(
                        "validation",
                        lane_total,
                        lane_total,
                        status="failed" if pre_render_failures else "completed",
                    )
                    reporter.lane_finished(
                        "validation",
                        status="failed" if pre_render_failures else "completed",
                        detail=(
                            f"failures={len(pre_render_failures)}" if pre_render_failures else None
                        ),
                    )
                if pre_render_failures:
                    records.extend(pre_render_failures)
                    if reporter is not None:
                        reporter.phase_finished(
                            "canonical_validation",
                            status="failed",
                            detail=f"failures={len(pre_render_failures)}; render blocked",
                        )
                        if report_pipeline_lanes:
                            reporter.lane_progress(
                                "render",
                                lane_total,
                                lane_total,
                                status="skipped",
                                detail="blocked by canonical validation failures",
                            )
                            reporter.lane_finished(
                                "render",
                                status="skipped",
                                detail="blocked by canonical validation failures",
                            )
                else:
                    if reporter is not None:
                        reporter.phase_finished("canonical_validation")
                        reporter.phase_started(
                            "canonical_render",
                            total=len(renderable_ids),
                        )
                    try:
                        if pipeline_config is None:
                            raise ValueError(
                                "pipeline_config is required to render canonical URDF output"
                            )
                        dynamic = self.hooks.build_dynamic_config(
                            pipeline_config,
                            root=config.dataset_root,
                            task=task,
                            camera=camera,
                            manifest=dynamic_manifest,
                            output_root=resolved_output_root,
                        )
                        if render_builder is None:
                            render_report = self.hooks.render_processed(
                                dynamic,
                                run_id=selected_run_id,
                                episode_ids=tuple(renderable_ids),
                                output_dir=canonical_run_dir,
                                reporter=reporter,
                            )
                        else:
                            render_report = render_builder(
                                dynamic,
                                run_id=selected_run_id,
                                episode_ids=tuple(renderable_ids),
                                output_dir=canonical_run_dir,
                            )
                        if reporter is not None:
                            reporter.phase_finished("canonical_render")
                            if report_pipeline_lanes:
                                reporter.lane_progress(
                                    "render",
                                    lane_total,
                                    lane_total,
                                    status="completed",
                                )
                                reporter.lane_finished("render")
                    except Exception as exc:  # noqa: BLE001 - render failure is summarized
                        error = f"{type(exc).__name__}: {exc}"
                        records.append(
                            {
                                "status": "render_failed",
                                "gripper_backend": "urdf",
                                "error": error,
                            }
                        )
                        if reporter is not None:
                            reporter.phase_finished(
                                "canonical_render",
                                status="render_failed",
                                detail=error,
                            )
                            if report_pipeline_lanes:
                                reporter.lane_progress(
                                    "render",
                                    lane_total,
                                    lane_total,
                                    status="failed",
                                    detail=error,
                                )
                                reporter.lane_finished("render", status="failed", detail=error)
            elif reporter is not None:
                reason = "disabled by --skip-render" if skip_render else "no publishable episodes"
                reporter.note(f"canonical_render skipped: {reason}", level="warning")
                if report_pipeline_lanes and not skip_render:
                    reporter.lane_progress(
                        "validation",
                        lane_total,
                        lane_total,
                        status="skipped",
                        detail=reason,
                    )
                    reporter.lane_finished("validation", status="skipped", detail=reason)
                    reporter.lane_progress(
                        "render",
                        lane_total,
                        lane_total,
                        status="skipped",
                        detail=reason,
                    )
                    reporter.lane_finished("render", status="skipped", detail=reason)

        failure_statuses = {
            "failed",
            "gripper_incomplete",
            "render_failed",
            "source_lineage_changed",
            "canonical_validation_failed",
        }
        qwen_health = (
            selection.source_summary.get("qwen_health")
            if source_mode in {"live_object_source_stage", "live_target_receiver_stage"}
            else None
        )
        passed = batch_error is None and not any(
            record.get("status") in failure_statuses for record in records
        )
        summary_model = ProcessSummary(
            format_version=self.hooks.summary_format_version,
            annotation_mode=source_annotation_mode,
            required_object_roles=tuple(source_required_roles),
            gripper_backend="urdf",
            run_id=selected_run_id,
            dataset_root=str(config.dataset_root),
            task=task,
            camera=camera,
            discovered_episode_ids=tuple(public_discovery.episode_ids),
            requested_episode_ids=(
                tuple(public_discovery.episode_ids) if episode_ids is None else tuple(episode_ids)
            ),
            dynamic_manifest=dynamic_manifest,
            qwen_health=qwen_health,
            records=tuple(EpisodeRecord.from_payload(record) for record in records),
            render=render_report,
            fatal_error=batch_error,
            backend={
                "type": "urdf",
                "source_mode": source_mode,
                "source_release": (None if source_release is None else dict(source_release)),
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
            passed=passed,
            plan=result if dry_run else None,
        )
        if dry_run:
            return summary_model.to_json()
        summary = summary_model.to_json()
        summary_path = ArtifactStore.write_json(
            canonical_run_dir / "process_summary.json",
            summary,
        )
        return summary_model.with_artifact(str(summary_path)).to_json()


__all__ = [
    "DEFAULT_URDF_DEPTH_TOLERANCE_MM",
    "DEFAULT_URDF_MINIMUM_ELIGIBLE_NONEMPTY_FRACTION",
    "UrdfRunConfig",
    "UrdfSourceSelection",
    "UrdfWorkflow",
    "UrdfWorkflowHooks",
    "UrdfWorkflowRuntime",
]
