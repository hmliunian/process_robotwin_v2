"""Dataset-level lifecycle for Qwen, object SAM, and optional SAM gripper masks.

The workflow owns sequencing and failure policy.  Heavy SAM/Qwen dependencies
remain behind ``runtime_loader`` so importing this module is safe for CPU-only
and frozen-source URDF paths.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar

from robotwin_annotation_v2.adapters.artifact_store import ArtifactStore
from robotwin_annotation_v2.config import PipelineConfig
from robotwin_annotation_v2.domain import AnnotationMode
from robotwin_annotation_v2.models import EpisodeRecord, EpisodeRef, ProcessSummary
from robotwin_annotation_v2.terminal_ui import ProcessUI

from .discovery import DiscoveryResult

PROCESS_SUMMARY_FORMAT_VERSION = "robotwin_process_dataset_summary_v1"

BackendT_co = TypeVar("BackendT_co", covariant=True)


class QwenHealthClient(Protocol):
    """Narrow client surface needed before dataset execution."""

    def health(self) -> dict[str, Any]: ...


class QwenClientFactory(Protocol):
    def __call__(
        self,
        *,
        endpoint: str,
        model: str,
        timeout_seconds: float,
    ) -> QwenHealthClient: ...


class SamBackend(Protocol):
    def shutdown(self) -> None: ...


class SamBackendFactory(Protocol[BackendT_co]):
    def __call__(
        self,
        *,
        checkpoint_path: Path,
        gpus: Sequence[int],
    ) -> BackendT_co: ...


type EpisodeExecutor[BackendT, ExecutionT] = Callable[
    [PipelineConfig, int, str, BackendT], ExecutionT
]
type ResultEmitter[ExecutionT] = Callable[[str, ExecutionT], bool]
type EpisodeCompletionCheck = Callable[
    [PipelineConfig, ArtifactStore, str, EpisodeRef], bool
]
type RunQwenStage = Callable[[PipelineConfig, int, str | None], None]


@dataclass(frozen=True)
class SamRuntime[BackendT: SamBackend, SamExecutionT, GripperExecutionT]:
    """Lazy-loaded model runtime with linked backend and execution result types."""

    qwen_client_factory: QwenClientFactory
    backend_factory: SamBackendFactory[BackendT]
    execution_errors: tuple[type[BaseException], ...]
    emit_gripper_result: ResultEmitter[GripperExecutionT]
    emit_sam_result: ResultEmitter[SamExecutionT]
    execute_gripper_episode: EpisodeExecutor[BackendT, GripperExecutionT]
    execute_sam_episode: EpisodeExecutor[BackendT, SamExecutionT]
    fatal_cuda_error: Callable[[BaseException], bool]
    gripper_episode_complete: EpisodeCompletionCheck
    sam_episode_complete: EpisodeCompletionCheck
    run_qwen: RunQwenStage


type DiscoverEpisodes = Callable[..., DiscoveryResult]
type BuildDynamicManifest = Callable[..., dict[str, Any]]
type BuildDynamicConfig = Callable[..., PipelineConfig]
type CaptureStageOutput = Callable[[ProcessUI | None], AbstractContextManager[None]]
type RenderProcessed = Callable[..., dict[str, Any]]
type ValidateSamRunOwnership = Callable[..., None]
type WriteSourceRunContract = Callable[..., Mapping[str, Any]]
type WriteSourceEpisodeReceipt = Callable[..., Mapping[str, Any]]


@dataclass(frozen=True)
class SamWorkflowHooks[BackendT: SamBackend, SamExecutionT, GripperExecutionT]:
    """Explicit seams retained while the legacy runtime is being strangled."""

    runtime_loader: Callable[
        [], SamRuntime[BackendT, SamExecutionT, GripperExecutionT]
    ]
    discover_episodes: DiscoverEpisodes
    build_dynamic_manifest: BuildDynamicManifest
    build_dynamic_config: BuildDynamicConfig
    capture_stage_output: CaptureStageOutput
    render_processed: RenderProcessed
    validate_run_id: Callable[[str], str]
    validate_run_ownership: ValidateSamRunOwnership
    write_source_run_contract: WriteSourceRunContract
    write_source_episode_receipt: WriteSourceEpisodeReceipt


@dataclass(frozen=True)
class SamWorkflow[BackendT: SamBackend, SamExecutionT, GripperExecutionT]:
    """Coordinate one complete SAM-backed dataset run."""

    config: PipelineConfig
    hooks: SamWorkflowHooks[BackendT, SamExecutionT, GripperExecutionT]

    def run(
        self,
        *,
        dataset_root: Path,
        task: str,
        camera: str,
        output_root: Path,
        run_id: str | None = None,
        episode_ids: tuple[int, ...] | None = None,
        force: bool = False,
        skip_render: bool = False,
        object_source_only: bool | None = None,
        target_receiver_only: bool = False,
        report_lifecycle: bool = True,
        incremental_source: bool = False,
        episode_terminal_callback: Callable[[int, str], None] | None = None,
        backend_factory: SamBackendFactory[BackendT] | None = None,
        reporter: ProcessUI | None = None,
    ) -> dict[str, Any]:
        """Run Qwen and object SAM, optionally followed by SAM gripper masks."""

        if object_source_only is not None and target_receiver_only:
            raise ValueError(
                "object_source_only and deprecated target_receiver_only cannot both be set"
            )
        source_only = (
            target_receiver_only
            if object_source_only is None
            else bool(object_source_only)
        )
        if self.config.annotation.mode is AnnotationMode.TARGET_ONLY and not source_only:
            raise ValueError(
                "target_only does not support --gripper-backend sam; "
                "use the default URDF backend"
            )
        if incremental_source and not source_only:
            raise ValueError("incremental source receipts require object_source_only mode")
        if run_id is not None:
            self.hooks.validate_run_id(run_id)
        if reporter is not None:
            if report_lifecycle:
                reporter.run_started(
                    backend="sam",
                    dataset_root=str(dataset_root.expanduser().resolve()),
                    task=task,
                    camera=camera,
                )
            reporter.phase_started("dataset_discovery")

        runtime = self.hooks.runtime_loader()
        discovery = self.hooks.discover_episodes(dataset_root, camera=camera)
        if not discovery.episodes:
            raise ValueError(f"no complete episodes found under {dataset_root}")
        manifest = self.hooks.build_dynamic_manifest(
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
        if reporter is not None:
            reporter.phase_finished(
                "dataset_discovery",
                detail=(
                    f"discovered={len(discovery.episodes)} "
                    f"selected={len(selected_ids)} "
                    f"skipped_inputs={len(discovery.skipped)}"
                ),
            )

        dynamic = self.hooks.build_dynamic_config(
            self.config,
            root=dataset_root,
            task=task,
            camera=camera,
            manifest=manifest,
            output_root=output_root,
        )
        store = ArtifactStore(dynamic.output_root)
        selected_run_id = self.hooks.validate_run_id(run_id or store.new_run_id())
        canonical_run_dir = store.run_dir(selected_run_id)
        self.hooks.validate_run_ownership(
            canonical_run_dir,
            run_id=selected_run_id,
        )
        if incremental_source:
            self.hooks.write_source_run_contract(
                canonical_run_dir,
                run_id=selected_run_id,
                dataset_root=dataset_root,
                task=task,
                camera=camera,
                dynamic_manifest=manifest,
                requested_episode_ids=selected_ids,
                annotation_mode=self.config.annotation.mode.value,
                required_object_roles=self.config.annotation.spec.required_role_names,
            )

        def episode_terminal(episode_id: int, status: str) -> None:
            if incremental_source and status in {"completed", "skipped_complete"}:
                ref = EpisodeRef(task, episode_id, camera)
                self.hooks.write_source_episode_receipt(
                    store.episode_dir(selected_run_id, ref),
                    task=task,
                    camera=camera,
                    episode_index=episode_id,
                    status=status,
                    expected_dataset_root=dataset_root,
                )
            if episode_terminal_callback is not None:
                episode_terminal_callback(episode_id, status)

        if reporter is not None and report_lifecycle:
            reporter.run_ready(run_id=selected_run_id, episode_ids=selected_ids)
        if reporter is not None:
            reporter.phase_started("qwen_health")
        qwen = runtime.qwen_client_factory(
            endpoint=dynamic.qwen.endpoint,
            model=dynamic.qwen.model,
            timeout_seconds=dynamic.qwen.timeout_seconds,
        )
        health = qwen.health()
        if reporter is not None:
            reporter.phase_finished("qwen_health")
            reporter.phase_started("resume_scan", total=len(selected_ids))

        records: list[dict[str, Any]] = list(discovery.skipped)
        pending: list[int] = []
        completion_check = (
            runtime.sam_episode_complete
            if source_only
            else runtime.gripper_episode_complete
        )
        for position, episode_id in enumerate(selected_ids, start=1):
            ref = EpisodeRef(task, episode_id, camera)
            if not force and completion_check(dynamic, store, selected_run_id, ref):
                records.append({"episode": episode_id, "status": "skipped_complete"})
                episode_terminal(episode_id, "skipped_complete")
                if reporter is not None and report_lifecycle:
                    reporter.episode_finished(episode_id, status="skipped_complete")
            else:
                pending.append(episode_id)
            if reporter is not None:
                reporter.phase_progress(
                    position,
                    total=len(selected_ids),
                    episode_id=episode_id,
                    status=(
                        "pending" if episode_id in pending else "skipped_complete"
                    ),
                )
        if reporter is not None:
            reporter.phase_finished(
                "resume_scan",
                detail=(
                    f"pending={len(pending)} "
                    f"skipped={len(selected_ids) - len(pending)}"
                ),
            )

        factory = runtime.backend_factory if backend_factory is None else backend_factory
        backend: BackendT | None = None
        fatal_error: BaseException | None = None
        try:
            if pending:
                if reporter is not None:
                    reporter.phase_started("sam_backend_load")
                try:
                    backend = factory(
                        checkpoint_path=dynamic.sam3.checkpoint,
                        gpus=dynamic.sam3.gpus,
                    )
                except Exception as exc:
                    if reporter is not None:
                        reporter.phase_finished(
                            "sam_backend_load",
                            status="failed",
                            detail=f"{type(exc).__name__}: {exc}",
                        )
                    raise
                if reporter is not None:
                    reporter.phase_finished("sam_backend_load")
                for episode_id in pending:
                    position = selected_ids.index(episode_id) + 1
                    current_stage: str | None = None
                    if reporter is not None and report_lifecycle:
                        reporter.episode_started(
                            episode_id,
                            position=position,
                            total=len(selected_ids),
                        )
                    try:
                        current_stage = "qwen"
                        if reporter is not None:
                            reporter.stage_started(episode_id, current_stage)
                        with self.hooks.capture_stage_output(reporter):
                            runtime.run_qwen(dynamic, episode_id, selected_run_id)
                        if reporter is not None:
                            reporter.stage_finished(episode_id, current_stage)

                        current_stage = "object_sam"
                        if reporter is not None:
                            reporter.stage_started(episode_id, current_stage)
                        sam_execution = runtime.execute_sam_episode(
                            dynamic,
                            episode_id,
                            selected_run_id,
                            backend,
                        )
                        with self.hooks.capture_stage_output(reporter):
                            sam_complete = runtime.emit_sam_result(
                                selected_run_id,
                                sam_execution,
                            )
                        if not sam_complete:
                            records.append(
                                {"episode": episode_id, "status": "sam_incomplete"}
                            )
                            episode_terminal(episode_id, "sam_incomplete")
                            if reporter is not None:
                                reporter.stage_finished(
                                    episode_id,
                                    current_stage,
                                    status="sam_incomplete",
                                )
                                if report_lifecycle:
                                    reporter.episode_finished(
                                        episode_id,
                                        status="sam_incomplete",
                                    )
                            current_stage = None
                            continue
                        if reporter is not None:
                            reporter.stage_finished(episode_id, current_stage)
                        if source_only:
                            records.append(
                                {"episode": episode_id, "status": "completed"}
                            )
                            episode_terminal(episode_id, "completed")
                            if reporter is not None and report_lifecycle:
                                reporter.episode_finished(
                                    episode_id,
                                    status="completed",
                                )
                            current_stage = None
                            continue

                        current_stage = "gripper_sam"
                        if reporter is not None:
                            reporter.stage_started(episode_id, current_stage)
                        gripper_execution = runtime.execute_gripper_episode(
                            dynamic,
                            episode_id,
                            selected_run_id,
                            backend,
                        )
                        with self.hooks.capture_stage_output(reporter):
                            gripper_complete = runtime.emit_gripper_result(
                                selected_run_id,
                                gripper_execution,
                            )
                        episode_status = (
                            "completed" if gripper_complete else "gripper_incomplete"
                        )
                        records.append(
                            {"episode": episode_id, "status": episode_status}
                        )
                        episode_terminal(episode_id, episode_status)
                        if reporter is not None:
                            reporter.stage_finished(
                                episode_id,
                                current_stage,
                                status=episode_status,
                            )
                            if report_lifecycle:
                                reporter.episode_finished(
                                    episode_id,
                                    status=episode_status,
                                )
                        current_stage = None
                    except SystemExit as exc:
                        error = f"stage exited with code {exc.code}"
                        records.append(
                            {
                                "episode": episode_id,
                                "status": "failed",
                                "error": error,
                            }
                        )
                        episode_terminal(episode_id, "failed")
                        if reporter is not None:
                            if current_stage is not None:
                                reporter.stage_finished(
                                    episode_id,
                                    current_stage,
                                    status="failed",
                                    detail=error,
                                )
                            if report_lifecycle:
                                reporter.episode_finished(
                                    episode_id,
                                    status="failed",
                                    detail=error,
                                )
                    except runtime.execution_errors as exc:
                        error = str(exc)
                        records.append(
                            {
                                "episode": episode_id,
                                "status": "failed",
                                "error": error,
                            }
                        )
                        episode_terminal(episode_id, "failed")
                        if reporter is not None:
                            if current_stage is not None:
                                reporter.stage_finished(
                                    episode_id,
                                    current_stage,
                                    status="failed",
                                    detail=error,
                                )
                            if report_lifecycle:
                                reporter.episode_finished(
                                    episode_id,
                                    status="failed",
                                    detail=error,
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
            for episode_id in selected_ids:
                if episode_id not in recorded_ids:
                    episode_terminal(episode_id, "not_run_after_fatal_cuda")
            if reporter is not None and report_lifecycle:
                for episode_id in selected_ids:
                    if episode_id not in recorded_ids:
                        reporter.episode_finished(
                            episode_id,
                            status="not_run_after_fatal_cuda",
                            detail=str(fatal_error),
                        )

        render_report: dict[str, Any] | None = None
        renderable_ids = tuple(
            int(record["episode"])
            for record in records
            if record.get("status") in {"completed", "skipped_complete"}
        )
        if (
            not source_only
            and not skip_render
            and fatal_error is None
            and renderable_ids
        ):
            if reporter is not None:
                reporter.phase_started("canonical_render", total=len(renderable_ids))
            try:
                render_report = self.hooks.render_processed(
                    dynamic,
                    run_id=selected_run_id,
                    episode_ids=renderable_ids,
                    output_dir=canonical_run_dir,
                    reporter=reporter,
                )
                if reporter is not None:
                    reporter.phase_finished("canonical_render")
            except Exception as exc:  # noqa: BLE001 - render failure is a run record
                records.append({"status": "render_failed", "error": str(exc)})
                if reporter is not None:
                    reporter.phase_finished(
                        "canonical_render",
                        status="render_failed",
                        detail=f"{type(exc).__name__}: {exc}",
                    )
        elif reporter is not None:
            reason = (
                "object source stage"
                if source_only
                else "disabled by --skip-render"
                if skip_render
                else "blocked by fatal CUDA error"
                if fatal_error is not None
                else "no renderable episodes"
            )
            reporter.note(f"canonical_render skipped: {reason}", level="warning")

        failure_statuses = {
            "failed",
            "sam_incomplete",
            "gripper_incomplete",
            "not_run_after_fatal_cuda",
            "render_failed",
        }
        passed = fatal_error is None and not any(
            record.get("status") in failure_statuses for record in records
        )
        summary_model = ProcessSummary(
            format_version=PROCESS_SUMMARY_FORMAT_VERSION,
            annotation_mode=self.config.annotation.mode.value,
            required_object_roles=tuple(
                self.config.annotation.spec.required_role_names
            ),
            gripper_backend=None if source_only else "sam",
            run_id=selected_run_id,
            dataset_root=str(dataset_root.expanduser().resolve()),
            task=task,
            camera=camera,
            discovered_episode_ids=tuple(discovery.episode_ids),
            requested_episode_ids=tuple(selected_ids),
            dynamic_manifest=manifest,
            qwen_health=health,
            records=tuple(EpisodeRecord.from_payload(record) for record in records),
            render=render_report,
            fatal_error=None if fatal_error is None else str(fatal_error),
            backend={
                "object_masks": "sam",
                "gripper": None if source_only else "sam",
            },
            passed=passed,
            stage_mode="object_source_only" if source_only else "full_sam",
        )
        persisted_summary = summary_model.to_json()
        summary_path = store.write_json(
            canonical_run_dir / "process_summary.json",
            persisted_summary,
        )
        return summary_model.with_artifact(str(summary_path)).to_json()


__all__ = [
    "PROCESS_SUMMARY_FORMAT_VERSION",
    "SamBackend",
    "SamBackendFactory",
    "SamRuntime",
    "SamWorkflow",
    "SamWorkflowHooks",
]
