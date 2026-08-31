"""Dataset-level lifecycle for Qwen, object SAM, and optional SAM gripper masks.

The workflow owns sequencing and failure policy.  Heavy SAM/Qwen dependencies
remain behind ``runtime_loader`` so importing this module is safe for CPU-only
and frozen-source URDF paths.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, TypeVar

from robotwin_annotation_v2.adapters.artifact_store import ArtifactStore
from robotwin_annotation_v2.config import PipelineConfig, QwenConfig
from robotwin_annotation_v2.domain import AnnotationMode
from robotwin_annotation_v2.models import EpisodeRecord, EpisodeRef, ProcessSummary
from robotwin_annotation_v2.terminal_ui import ProcessUI

from .discovery import (
    DiscoveryResult,
    build_dynamic_manifest,
    discover_episodes,
    select_manifest_episodes,
)
from .provenance import (
    attach_profile_provenance,
    object_resolution_strategy,
    prompt_bundle_for_config,
    prompt_bundle_from_manifest,
    target_profile_from_config,
    target_profile_from_manifest,
    validate_profile_provenance_pair,
)

PROCESS_SUMMARY_FORMAT_VERSION = "robotwin_process_dataset_summary_v1"

BackendT_co = TypeVar("BackendT_co", covariant=True)


class QwenHealthClient(Protocol):
    """Narrow client surface needed before dataset execution."""

    def health(self) -> dict[str, Any]: ...


class QwenClientFactory(Protocol):
    def __call__(
        self,
        *,
        config: QwenConfig,
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
class SamEpisodeResult:
    """Small, process-safe result of one complete SAM-backed episode."""

    episode_id: int
    status: str
    error: str | None = None
    fatal_cuda: bool = False

    def to_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {"episode": self.episode_id, "status": self.status}
        if self.error is not None:
            record["error"] = self.error
        return record


def execute_sam_episode[
    BackendT: SamBackend,
    SamExecutionT,
    GripperExecutionT,
](
    runtime: SamRuntime[BackendT, SamExecutionT, GripperExecutionT],
    *,
    capture_stage_output: CaptureStageOutput,
    dynamic: PipelineConfig,
    episode_id: int,
    run_id: str,
    backend: BackendT,
    source_only: bool,
    position: int,
    total: int,
    report_lifecycle: bool,
    reporter: ProcessUI | None,
) -> SamEpisodeResult:
    """Execute Qwen and SAM stages for one episode using one resident backend."""

    current_stage: str | None = None
    if reporter is not None and report_lifecycle:
        reporter.episode_started(episode_id, position=position, total=total)
    try:
        current_stage = "qwen"
        if reporter is not None:
            reporter.stage_started(episode_id, current_stage)
        with capture_stage_output(reporter):
            runtime.run_qwen(dynamic, episode_id, run_id)
        if reporter is not None:
            reporter.stage_finished(episode_id, current_stage)

        current_stage = "object_sam"
        if reporter is not None:
            reporter.stage_started(episode_id, current_stage)
        sam_execution = runtime.execute_sam_episode(dynamic, episode_id, run_id, backend)
        with capture_stage_output(reporter):
            sam_complete = runtime.emit_sam_result(run_id, sam_execution)
        if not sam_complete:
            status = "sam_incomplete"
        elif source_only:
            status = "completed"
        else:
            if reporter is not None:
                reporter.stage_finished(episode_id, current_stage)
            current_stage = "gripper_sam"
            if reporter is not None:
                reporter.stage_started(episode_id, current_stage)
            gripper_execution = runtime.execute_gripper_episode(
                dynamic,
                episode_id,
                run_id,
                backend,
            )
            with capture_stage_output(reporter):
                gripper_complete = runtime.emit_gripper_result(run_id, gripper_execution)
            status = "completed" if gripper_complete else "gripper_incomplete"
        if reporter is not None:
            reporter.stage_finished(episode_id, current_stage, status=status)
            if report_lifecycle:
                reporter.episode_finished(episode_id, status=status)
        return SamEpisodeResult(episode_id, status)
    except SystemExit as exc:
        error = f"stage exited with code {exc.code}"
    except runtime.execution_errors as exc:
        error = str(exc)
        fatal_cuda = runtime.fatal_cuda_error(exc)
        if reporter is not None:
            if current_stage is not None:
                reporter.stage_finished(
                    episode_id,
                    current_stage,
                    status="failed",
                    detail=error,
                )
            if report_lifecycle:
                reporter.episode_finished(episode_id, status="failed", detail=error)
        return SamEpisodeResult(episode_id, "failed", error, fatal_cuda)

    if reporter is not None:
        if current_stage is not None:
            reporter.stage_finished(
                episode_id,
                current_stage,
                status="failed",
                detail=error,
            )
        if report_lifecycle:
            reporter.episode_finished(episode_id, status="failed", detail=error)
    return SamEpisodeResult(episode_id, "failed", error)


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
        object_source_only: bool = False,
        report_lifecycle: bool = True,
        incremental_source: bool = False,
        episode_terminal_callback: Callable[[int, str], None] | None = None,
        backend_factory: SamBackendFactory[BackendT] | None = None,
        reporter: ProcessUI | None = None,
    ) -> dict[str, Any]:
        """Run Qwen and object SAM, optionally followed by SAM gripper masks."""

        source_only = bool(object_source_only)
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
        manifest_episodes = select_manifest_episodes(
            discovery.episodes,
            manifest=self.config.dataset.manifest_data,
            episode_ids=self.config.dataset.regression_episode_ids,
        )
        if not manifest_episodes:
            raise ValueError("bound dataset manifest contains no discovered episodes")
        bound_manifest = self.config.dataset.manifest_data
        has_bound_selection = isinstance(bound_manifest, Mapping) and any(
            bound_manifest.get(key) is not None
            for key in ("regression_episode_ids", "episode_indices", "episode_ids")
        )
        manifest_scope_ids = {episode.episode_id for episode in manifest_episodes}
        manifest = self.hooks.build_dynamic_manifest(
            dataset_root,
            task=task,
            camera=camera,
            episodes=manifest_episodes,
        )
        # Hooks are kept as a compatibility seam, so enrich their result here
        # rather than requiring every lightweight hook double to learn the new
        # keyword arguments.  This is the canonical point where algorithm
        # profile identity meets a concrete dataset selection.
        manifest = dict(manifest)
        manifest_profile = target_profile_from_manifest(manifest)
        config_profile = target_profile_from_config(self.config)
        if (
            manifest_profile is not None
            and config_profile is not None
            and manifest_profile != config_profile
        ):
            raise ValueError(
                "dynamic manifest target_profile differs from config: "
                f"{manifest_profile!r} != {config_profile!r}"
            )
        selected_profile = manifest_profile or config_profile
        prompt_bundle = prompt_bundle_for_config(
            self.config,
            target_profile=selected_profile,
        )
        manifest_bundle = prompt_bundle_from_manifest(manifest, strict=True)
        if manifest_bundle is not None and prompt_bundle is not None:
            validate_profile_provenance_pair(
                {
                    "target_profile": selected_profile,
                    "prompt_bundle": manifest_bundle,
                },
                {
                    "target_profile": selected_profile,
                    "prompt_bundle": prompt_bundle,
                },
                label="dynamic manifest and configured prompt provenance",
            )
        elif manifest_bundle is not None:
            # Lightweight/legacy config doubles may not expose prompt files;
            # retain the immutable bundle supplied by the bound manifest so
            # all run-level and episode-level artifacts still agree.
            prompt_bundle = manifest_bundle
        attach_profile_provenance(
            manifest,
            target_profile=selected_profile,
            prompt_bundle=prompt_bundle,
        )
        manifest.setdefault(
            "runtime_dataset_root",
            str(dataset_root.expanduser().resolve()),
        )
        discovered_ids = set(discovery.episode_ids)
        # ``manifest_episodes`` is already narrowed to a bound dataset
        # selection when one exists.  Use it as the implicit request so a
        # caller that omits ``episode_ids`` cannot accidentally expand a task
        # slice back to every episode in a mixed root.  For an unbound legacy
        # input it is the complete discovery result, preserving old behavior.
        selected_ids = (
            tuple(episode.episode_id for episode in manifest_episodes)
            if episode_ids is None
            else tuple(dict.fromkeys(int(value) for value in episode_ids))
        )
        if episode_ids is not None:
            outside_scope = sorted(set(selected_ids) - manifest_scope_ids)
            if outside_scope and manifest_scope_ids != set(discovery.episode_ids):
                raise ValueError(
                    "requested episodes fall outside the bound dataset manifest: "
                    f"{outside_scope}"
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
                    status="completed",
                    expected_dataset_root=dataset_root,
                )
            if episode_terminal_callback is not None:
                episode_terminal_callback(episode_id, status)

        if reporter is not None and report_lifecycle:
            reporter.run_ready(run_id=selected_run_id, episode_ids=selected_ids)
        if reporter is not None:
            reporter.phase_started("qwen_health")
        qwen = runtime.qwen_client_factory(config=dynamic.qwen)
        health = qwen.health()
        if reporter is not None:
            reporter.phase_finished("qwen_health")
            reporter.phase_started("resume_scan", total=len(selected_ids))

        records: list[dict[str, Any]] = [
            record
            for record in discovery.skipped
            if not has_bound_selection
            or (
                record.get("episode") is not None
                and int(record["episode"]) in manifest_scope_ids
            )
        ]
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
        fatal_error: str | None = None
        parallel_used = bool(
            pending and dynamic.parallel.enabled and backend_factory is None
        )
        if parallel_used:
            from .parallel_sam_runtime import run_parallel_sam_episodes

            if reporter is not None:
                reporter.phase_started(
                    "sam_worker_pool",
                    total=len(dynamic.parallel.sam_worker_gpus),
                )
            try:
                parallel_outcomes = run_parallel_sam_episodes(
                    dynamic,
                    run_id=selected_run_id,
                    episode_ids=tuple(pending),
                    source_only=source_only,
                    qwen_max_in_flight=dynamic.parallel.qwen_max_in_flight,
                    outcome_callback=lambda result: episode_terminal(
                        result.episode_id,
                        result.status,
                    ),
                    reporter=reporter,
                    report_lifecycle=report_lifecycle,
                )
            except BaseException as exc:
                if reporter is not None:
                    reporter.phase_finished(
                        "sam_worker_pool",
                        status="failed",
                        detail=f"{type(exc).__name__}: {exc}",
                    )
                raise
            if reporter is not None:
                reporter.phase_finished(
                    "sam_worker_pool",
                    detail=(
                        f"workers={len(dynamic.parallel.sam_worker_gpus)} "
                        f"episodes={len(parallel_outcomes)}"
                    ),
                )
            for outcome in parallel_outcomes:
                record: dict[str, Any] = {
                    "episode": outcome.episode_id,
                    "status": outcome.status,
                    "attempt": outcome.attempt,
                }
                if outcome.worker_id is not None:
                    record["worker_id"] = outcome.worker_id
                if outcome.gpu_id is not None:
                    record["gpu"] = outcome.gpu_id
                if outcome.error is not None:
                    record["error"] = outcome.error
                if outcome.fatal_cuda:
                    record["fatal_cuda"] = True
                records.append(record)
        else:
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
                        result = execute_sam_episode(
                            runtime,
                            capture_stage_output=self.hooks.capture_stage_output,
                            dynamic=dynamic,
                            episode_id=episode_id,
                            run_id=selected_run_id,
                            backend=backend,
                            source_only=source_only,
                            position=selected_ids.index(episode_id) + 1,
                            total=len(selected_ids),
                            report_lifecycle=report_lifecycle,
                            reporter=reporter,
                        )
                        records.append(result.to_record())
                        episode_terminal(episode_id, result.status)
                        if result.fatal_cuda:
                            fatal_error = result.error or "fatal CUDA error"
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
                            detail=fatal_error,
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
            "not_run_no_healthy_sam_worker",
            "render_failed",
        }
        passed = fatal_error is None and not any(
            record.get("status") in failure_statuses for record in records
        )
        backend_record: dict[str, Any] = {
            "object_masks": "sam",
            "gripper": None if source_only else "sam",
        }
        if dynamic.parallel.enabled:
            backend_record["parallel_sam"] = {
                "worker_gpus": list(dynamic.parallel.sam_worker_gpus),
                "worker_count": len(dynamic.parallel.sam_worker_gpus),
                "qwen_max_in_flight": dynamic.parallel.qwen_max_in_flight,
                "used": parallel_used,
            }
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
            discovered_episode_ids=tuple(
                episode.episode_id for episode in manifest_episodes
            ),
            requested_episode_ids=tuple(selected_ids),
            dynamic_manifest=manifest,
            qwen_health=health,
            records=tuple(EpisodeRecord.from_payload(record) for record in records),
            render=render_report,
            fatal_error=fatal_error,
            backend=backend_record,
            passed=passed,
            stage_mode="object_source_only" if source_only else "full_sam",
            # Persist the complete semantic contract for every fresh run,
            # including ordinary pick/place.  Frozen and incremental URDF
            # validation can then compare summaries, dynamic manifests and
            # per-episode artifacts without a mode-dependent omission.
            target_profile=selected_profile,
            prompt_bundle=prompt_bundle,
            resolution_strategy=object_resolution_strategy(
                target_profile=selected_profile,
                prompt_bundle=prompt_bundle,
            ),
        )
        persisted_summary = summary_model.to_json()
        summary_path = store.write_json(
            canonical_run_dir / "process_summary.json",
            persisted_summary,
        )
        return summary_model.with_artifact(str(summary_path)).to_json()


@contextlib.contextmanager
def capture_sam_stage_output(reporter: ProcessUI | None) -> Iterator[None]:
    """Hide embedded SAM-stage JSON while retaining it for verbose UI output."""

    if reporter is None:
        yield
        return
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured):
            yield
    finally:
        reporter.detail(captured.getvalue().rstrip())


def load_sam_runtime() -> SamRuntime[SamBackend, Any, Any]:
    """Resolve production model integrations only when a SAM workflow runs."""

    from robotwin_annotation_v2.adapters.qwen_client import (
        OpenAICompatibleQwenClient,
    )
    from robotwin_annotation_v2.adapters.sam3_adapter import Sam3Adapter

    episode_runtime = importlib.import_module(
        "robotwin_annotation_v2.application.episode_pipeline"
    )

    def run_qwen_without_health(
        config: PipelineConfig,
        episode_id: int,
        run_id: str | None,
    ) -> None:
        episode_runtime.run_qwen(
            config,
            episode_id,
            run_id,
            check_health=False,
        )

    return SamRuntime(
        qwen_client_factory=OpenAICompatibleQwenClient.from_config,
        backend_factory=Sam3Adapter,
        execution_errors=tuple(episode_runtime.SAM_EXECUTION_ERRORS),
        emit_gripper_result=episode_runtime._emit_gripper_result,
        emit_sam_result=episode_runtime._emit_sam_result,
        execute_gripper_episode=episode_runtime._execute_gripper_episode,
        execute_sam_episode=episode_runtime._execute_sam_episode,
        fatal_cuda_error=episode_runtime._fatal_cuda_error,
        gripper_episode_complete=episode_runtime._gripper_episode_complete,
        sam_episode_complete=episode_runtime._sam_episode_complete,
        run_qwen=run_qwen_without_health,
    )


def build_sam_dynamic_config(
    config: PipelineConfig,
    *,
    root: Path,
    task: str,
    camera: str,
    manifest: dict[str, Any],
    output_root: Path,
) -> PipelineConfig:
    """Bind one discovered dataset to a SAM workflow configuration."""

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


def validate_sam_run_id(run_id: str) -> str:
    """Validate and return a run identifier safe for use as one directory name."""

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


def read_json_object(path: Path, *, description: str) -> dict[str, Any]:
    """Read one JSON object with errors that identify the owning artifact."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{description} is missing: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {description}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(  # noqa: TRY004 - preserve the JSON input error contract
            f"{description} must contain one JSON object: {path}"
        )
    return payload


def summary_gripper_backend(summary: Mapping[str, Any]) -> str | None:
    """Return the gripper backend recorded by either supported summary layout."""

    top_level = summary.get("gripper_backend")
    backend_record = summary.get("backend")
    nested = None
    if isinstance(backend_record, Mapping):
        nested = backend_record.get("gripper", backend_record.get("type"))
    if top_level is not None and nested is not None and top_level != nested:
        raise ValueError("existing process summary has conflicting backend ownership")
    value = top_level if top_level is not None else nested
    return None if value is None else str(value)


def validate_sam_run_ownership(run_dir: Path, *, run_id: str) -> None:
    """Reject attempts to publish SAM artifacts into a run owned elsewhere."""

    if (run_dir / "_backend" / "urdf").exists():
        raise ValueError(f"existing run is owned by the URDF backend: {run_dir}")
    summary_path = run_dir / "process_summary.json"
    if not summary_path.exists():
        return
    summary = read_json_object(
        summary_path,
        description="existing process summary",
    )
    if summary.get("run_id") != run_id:
        raise ValueError("existing process summary run_id does not match its directory")
    backend = summary_gripper_backend(summary)
    if backend not in {None, "sam"}:
        raise ValueError(f"existing run is owned by backend {backend!r}, not SAM")


def render_sam_processed(
    config: PipelineConfig,
    *,
    run_id: str,
    episode_ids: tuple[int, ...],
    output_dir: Path,
    reporter: ProcessUI | None = None,
    dataset_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Render canonical masks without coupling the workflow to renderer imports."""

    from robotwin_annotation_v2.adapters.robotwin_dataset import RoboTwinDataset

    render = importlib.import_module("robotwin_annotation_v2.adapters.rendering")
    factory = RoboTwinDataset if dataset_factory is None else dataset_factory
    dataset = factory(
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
    for position, episode_id in enumerate(episode_ids, start=1):
        candidate = selected[episode_id]
        artifact = render.load_masks(candidate.path)
        ref = EpisodeRef(config.dataset.task, episode_id, config.dataset.camera)
        video_path = dataset.paths(ref).video
        task_text = dataset.task_text(episode_id)
        output_path = video_dir / render.output_video_name(
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
                "mask_sha256": render.file_sha256(candidate.path),
                "mask_format": artifact.format_version,
                "annotation_status": dict(
                    zip(artifact.instance_names, artifact.annotation_status, strict=True)
                ),
                "qc_status": dict(
                    zip(artifact.instance_names, artifact.qc_status, strict=True)
                ),
                "output_video": output_path.name,
                "output_sha256": render.file_sha256(output_path),
                "output_bytes": output_path.stat().st_size,
                **video,
            }
        )
        if reporter is not None:
            reporter.phase_progress(
                position,
                total=len(episode_ids),
                episode_id=episode_id,
                status="rendered",
            )
    manifest = {
        "format": "robotwin_coverage20_overlay_videos_v3",
        "created_at": datetime.now(UTC).isoformat(),
        "requested_run_id": run_id,
        "config": str(config.config_path),
        "dataset_root": str(config.dataset.root),
        "runs_root": str(config.output_root),
        "task": config.dataset.task,
        "camera": config.dataset.camera,
        "filename_mode": "episode",
        "episode_count": len(records),
        "rendered_roles": [*config.annotation.spec.required_role_names, "gripper"],
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


def default_sam_workflow_hooks() -> SamWorkflowHooks[SamBackend, Any, Any]:
    """Build production hooks without routing through the legacy runtime."""

    from robotwin_annotation_v2.urdf_gripper_publisher import (
        write_source_episode_completion_receipt,
        write_source_run_contract,
    )

    return SamWorkflowHooks(
        runtime_loader=load_sam_runtime,
        discover_episodes=discover_episodes,
        build_dynamic_manifest=build_dynamic_manifest,
        build_dynamic_config=build_sam_dynamic_config,
        capture_stage_output=capture_sam_stage_output,
        render_processed=render_sam_processed,
        validate_run_id=validate_sam_run_id,
        validate_run_ownership=validate_sam_run_ownership,
        write_source_run_contract=write_source_run_contract,
        write_source_episode_receipt=write_source_episode_completion_receipt,
    )


__all__ = [
    "PROCESS_SUMMARY_FORMAT_VERSION",
    "SamBackend",
    "SamBackendFactory",
    "SamRuntime",
    "SamWorkflow",
    "SamWorkflowHooks",
    "build_sam_dynamic_config",
    "capture_sam_stage_output",
    "default_sam_workflow_hooks",
    "load_sam_runtime",
    "read_json_object",
    "render_sam_processed",
    "summary_gripper_backend",
    "validate_sam_run_id",
    "validate_sam_run_ownership",
]
