"""Dataset execution primitives used by the readable application pipeline."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import robotwin_annotation_v2.application.discovery as _discovery
import robotwin_annotation_v2.application.urdf_runtime as _urdf_runtime
from robotwin_annotation_v2.adapters.artifact_store import ArtifactStore
from robotwin_annotation_v2.adapters.robotwin_dataset import RoboTwinDataset
from robotwin_annotation_v2.application.dataset_input import resolve_dataset_input
from robotwin_annotation_v2.application.dataset_pipeline import (
    DatasetBackendRunner,
    DatasetPipeline,
)
from robotwin_annotation_v2.application.sam_workflow import (
    PROCESS_SUMMARY_FORMAT_VERSION,
    SamWorkflow,
    SamWorkflowHooks,
    build_sam_dynamic_config,
    capture_sam_stage_output,
    load_sam_runtime,
    read_json_object,
    render_sam_processed,
    summary_gripper_backend,
    validate_sam_run_id,
    validate_sam_run_ownership,
)
from robotwin_annotation_v2.application.urdf_workflow import (
    DEFAULT_URDF_DEPTH_TOLERANCE_MM,
    DEFAULT_URDF_MINIMUM_ELIGIBLE_NONEMPTY_FRACTION,
    DEFAULT_URDF_PIPELINE_BUFFER_SIZE,
    UrdfWorkflow,
    UrdfWorkflowHooks,
)
from robotwin_annotation_v2.config import PipelineConfig, load_config
from robotwin_annotation_v2.domain import (
    AnnotationMode,
    GripperBackend,
)
from robotwin_annotation_v2.models import ProcessRequest
from robotwin_annotation_v2.terminal_ui import UI_MODES, ProcessUI, create_process_ui
from robotwin_annotation_v2.urdf_gripper_publisher import (
    validate_derivation_source_episode,
    write_source_episode_completion_receipt,
    write_source_run_contract,
)

GRIPPER_BACKENDS = ("sam", "urdf")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BUNDLED_URDF_PATH = (
    PROJECT_ROOT
    / "configs"
    / "assets"
    / "aloha-agilex"
    / "arx5_description_isaac_gripper.urdf"
)
PATH_MODE_CONFIGS = {
    AnnotationMode.PICK_PLACE: PROJECT_ROOT / "configs" / "pilot_move_pillbottle_pad.yaml",
    AnnotationMode.TARGET_ONLY: PROJECT_ROOT / "configs" / "pilot_adjust_bottle_target_only.yaml",
}
CHUNK_PATTERN = _discovery.CHUNK_PATTERN
EPISODE_FILE_PATTERN = _discovery.EPISODE_FILE_PATTERN
DiscoveredEpisode = _discovery.DiscoveredEpisode
DiscoveryResult = _discovery.DiscoveryResult
_episode_video_path = _discovery._episode_video_path
_episode_depth_path = _discovery._episode_depth_path
_parquet_frame_count = _discovery._parquet_frame_count
_measure_episode = _discovery._measure_episode


_captured_stage_output = capture_sam_stage_output


_JsonProgressWriter = _urdf_runtime.JsonProgressWriter
_captured_json_progress = _urdf_runtime.capture_urdf_json_progress
_load_sam_runtime = load_sam_runtime
_load_urdf_runner = _urdf_runtime.load_urdf_runner
_load_urdf_workflow_runtime = _urdf_runtime.load_urdf_workflow_runtime

_release_sam_cuda_cache = _urdf_runtime.release_sam_cuda_cache


_select_urdf_egl_device = _urdf_runtime.select_urdf_egl_device


_ProcessEventSender = _urdf_runtime.ProcessEventSender


_object_source_process_entry = _urdf_runtime.object_source_process_entry


_incremental_urdf_process_entry = _urdf_runtime.incremental_urdf_process_entry


_run_streaming_source_urdf_workers = _urdf_runtime.run_streaming_source_urdf_workers


_run_object_source_process = _urdf_runtime.run_object_source_process
_validate_urdf_run_ownership = _urdf_runtime.validate_urdf_run_ownership
select_urdf_source_episodes = _urdf_runtime.select_urdf_source_episodes
mp = _urdf_runtime.mp
Full = _urdf_runtime.Full


def discover_episodes(
    root: Path,
    *,
    camera: str,
    require_depth: bool = False,
) -> DiscoveryResult:
    """Compatibility delegate for the canonical discovery module."""

    return _discovery.discover_episodes(
        root,
        camera=camera,
        require_depth=require_depth,
    )


_read_json_object = read_json_object
_validate_run_id = validate_sam_run_id
_summary_gripper_backend = summary_gripper_backend
_validate_sam_run_ownership = validate_sam_run_ownership


def build_dynamic_manifest(
    root: Path,
    *,
    task: str,
    camera: str,
    episodes: Sequence[DiscoveredEpisode],
) -> dict[str, Any]:
    """Compatibility delegate for the canonical discovery module."""

    return _discovery.build_dynamic_manifest(
        root,
        task=task,
        camera=camera,
        episodes=episodes,
        measure_episode_fn=_measure_episode,
    )


def _dynamic_config(
    config: PipelineConfig,
    *,
    root: Path,
    task: str,
    camera: str,
    manifest: dict[str, Any],
    output_root: Path,
) -> PipelineConfig:
    """Compatibility delegate for the canonical SAM config builder."""

    return build_sam_dynamic_config(
        config,
        root=root,
        task=task,
        camera=camera,
        manifest=manifest,
        output_root=output_root,
    )


def _render_processed(
    config: PipelineConfig,
    *,
    run_id: str,
    episode_ids: tuple[int, ...],
    output_dir: Path,
    reporter: ProcessUI | None = None,
) -> dict[str, Any]:
    """Compatibility delegate for the canonical SAM render coordinator."""

    return render_sam_processed(
        config,
        run_id=run_id,
        episode_ids=episode_ids,
        output_dir=output_dir,
        reporter=reporter,
        dataset_factory=RoboTwinDataset,
    )


def _urdf_workflow() -> UrdfWorkflow:
    """Build workflow hooks from current globals to preserve monkeypatch seams."""

    return UrdfWorkflow(
        UrdfWorkflowHooks(
            runtime_loader=_load_urdf_workflow_runtime,
            discover_episodes=discover_episodes,
            parquet_frame_count=_parquet_frame_count,
            select_source_episodes=select_urdf_source_episodes,
            validate_run_id=_validate_run_id,
            validate_run_ownership=_validate_urdf_run_ownership,
            capture_progress=_captured_json_progress,
            validate_source_episode=validate_derivation_source_episode,
            build_dynamic_config=_dynamic_config,
            render_processed=_render_processed,
            select_egl_device=_select_urdf_egl_device,
            run_streaming_workers=_run_streaming_source_urdf_workers,
            run_object_source_process=_run_object_source_process,
            process_dataset=process_dataset,
            release_sam_cuda_cache=_release_sam_cuda_cache,
            process_frozen_source=process_urdf_source_run,
            summary_format_version=PROCESS_SUMMARY_FORMAT_VERSION,
        )
    )


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
    """Compatibility entry point for the canonical frozen-source workflow."""

    return _urdf_workflow().run(
        pipeline_config=pipeline_config,
        dataset_root=dataset_root,
        source_run_dir=source_run_dir,
        task=task,
        camera=camera,
        output_root=output_root,
        urdf_path=urdf_path,
        mesh_root=mesh_root,
        run_id=run_id,
        episode_ids=episode_ids,
        skip_render=skip_render,
        dry_run=dry_run,
        resume=resume,
        depth_tolerance_mm=depth_tolerance_mm,
        minimum_eligible_nonempty_fraction=minimum_eligible_nonempty_fraction,
        fit_config_json=fit_config_json,
        allow_partial_source=allow_partial_source,
        source_mode=source_mode,
        source_release=source_release,
        experiment_runner=experiment_runner,
        episode_publisher=episode_publisher,
        episode_validator=episode_validator,
        render_builder=render_builder,
        prepared_backend_result=prepared_backend_result,
        prepared_backend_error=prepared_backend_error,
        egl_device_id=egl_device_id,
        report_lifecycle=report_lifecycle,
        pipeline_episode_ids=pipeline_episode_ids,
        reporter=reporter,
    )


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
    object_source_only: bool | None = None,
    # Deprecated compatibility alias.  New callers must use object_source_only.
    target_receiver_only: bool = False,
    report_lifecycle: bool = True,
    incremental_source: bool = False,
    episode_terminal_callback: Callable[[int, str], None] | None = None,
    backend_factory: Callable[..., Any] | None = None,
    reporter: ProcessUI | None = None,
) -> dict[str, Any]:
    """Compatibility entry point for the canonical :class:`DatasetPipeline`."""

    if object_source_only is not None and target_receiver_only:
        raise ValueError(
            "object_source_only and deprecated target_receiver_only cannot both be set"
        )
    source_only = (
        target_receiver_only
        if object_source_only is None
        else bool(object_source_only)
    )
    hooks = SamWorkflowHooks(
        runtime_loader=_load_sam_runtime,
        discover_episodes=discover_episodes,
        build_dynamic_manifest=build_dynamic_manifest,
        build_dynamic_config=_dynamic_config,
        capture_stage_output=_captured_stage_output,
        render_processed=_render_processed,
        validate_run_id=_validate_run_id,
        validate_run_ownership=_validate_sam_run_ownership,
        write_source_run_contract=write_source_run_contract,
        write_source_episode_receipt=write_source_episode_completion_receipt,
    )
    workflow = SamWorkflow(config, hooks)

    def run_sam_backend(
        request: ProcessRequest,
        *,
        reporter: ProcessUI | None = None,
    ) -> dict[str, Any]:
        return workflow.run(
            dataset_root=request.dataset_root,
            task=request.task,
            camera=request.camera,
            output_root=request.output_root,
            run_id=request.run_id,
            episode_ids=request.episode_ids,
            force=force,
            skip_render=request.skip_render,
            object_source_only=source_only,
            report_lifecycle=report_lifecycle,
            incremental_source=incremental_source,
            episode_terminal_callback=episode_terminal_callback,
            backend_factory=backend_factory,
            reporter=reporter,
        )

    request = ProcessRequest(
        dataset_root=dataset_root,
        output_root=output_root,
        task=task,
        camera=camera,
        run_id=run_id,
        episode_ids=episode_ids,
        skip_render=skip_render,
    )
    return DatasetPipeline(config, sam_runner=run_sam_backend).run(
        request,
        backend=GripperBackend.SAM,
        reporter=reporter,
    )


def process_live_urdf_pipeline(
    *,
    pipeline_config: PipelineConfig,
    dataset_root: Path,
    task: str,
    camera: str,
    output_root: Path,
    urdf_path: Path,
    mesh_root: Path | None = None,
    run_id: str | None = None,
    episode_ids: tuple[int, ...] | None = None,
    skip_render: bool = False,
    depth_tolerance_mm: float = DEFAULT_URDF_DEPTH_TOLERANCE_MM,
    minimum_eligible_nonempty_fraction: float = (
        DEFAULT_URDF_MINIMUM_ELIGIBLE_NONEMPTY_FRACTION
    ),
    fit_config_json: Path | None = None,
    allow_partial_source: bool = False,
    urdf_pipeline: bool = True,
    urdf_pipeline_buffer_size: int = DEFAULT_URDF_PIPELINE_BUFFER_SIZE,
    urdf_egl_device_id: int | None = None,
    backend_factory: Callable[..., Any] | None = None,
    reporter: ProcessUI | None = None,
) -> dict[str, Any]:
    """Compatibility entry point for live source-to-URDF orchestration."""

    return _urdf_workflow().run_live(
        pipeline_config=pipeline_config,
        dataset_root=dataset_root,
        task=task,
        camera=camera,
        output_root=output_root,
        urdf_path=urdf_path,
        mesh_root=mesh_root,
        run_id=run_id,
        episode_ids=episode_ids,
        skip_render=skip_render,
        depth_tolerance_mm=depth_tolerance_mm,
        minimum_eligible_nonempty_fraction=minimum_eligible_nonempty_fraction,
        fit_config_json=fit_config_json,
        allow_partial_source=allow_partial_source,
        urdf_pipeline=urdf_pipeline,
        urdf_pipeline_buffer_size=urdf_pipeline_buffer_size,
        urdf_egl_device_id=urdf_egl_device_id,
        backend_factory=backend_factory,
        reporter=reporter,
    )

def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument(
        "--data-path",
        "--data_path",
        type=Path,
        help="Single-task dataset or collection root",
    )
    path_mode = parser.add_mutually_exclusive_group()
    path_mode.add_argument(
        "--target-only",
        "--target_only",
        dest="path_mode",
        action="store_const",
        const=AnnotationMode.TARGET_ONLY.value,
    )
    path_mode.add_argument(
        "--pick-place",
        "--pick_place",
        dest="path_mode",
        action="store_const",
        const=AnnotationMode.PICK_PLACE.value,
    )
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
        default="urdf",
        help="Use the SAM gripper stage or generate it from URDF after target/receiver",
    )
    parser.add_argument(
        "--source-run-dir",
        help=(
            "Optional frozen run containing QC-passed target/receiver masks; when "
            "omitted, the URDF pipeline generates a fresh internal source stage"
        ),
    )
    parser.add_argument(
        "--urdf-path",
        help=(
            "RoboTwin Aloha URDF; defaults to the bundled render asset for the "
            "URDF backend"
        ),
    )
    parser.add_argument("--urdf-mesh-root", type=Path)
    parser.add_argument("--urdf-depth-tolerance-mm", type=float)
    parser.add_argument(
        "--urdf-minimum-eligible-nonempty-fraction",
        type=float,
    )
    parser.add_argument("--urdf-fit-config-json", type=Path)
    parser.add_argument(
        "--urdf-egl-device-id",
        type=int,
        help=(
            "Physical GPU for EGL rendering; live URDF mode otherwise selects the "
            "freest GPU not used by SAM"
        ),
    )
    parser.add_argument(
        "--urdf-pipeline-buffer-size",
        type=int,
        default=DEFAULT_URDF_PIPELINE_BUFFER_SIZE,
        help="Maximum source-ready episodes queued ahead of the URDF worker",
    )
    parser.add_argument(
        "--no-urdf-pipeline",
        action="store_true",
        help="Disable Source-to-URDF overlap and use the legacy serial execution path",
    )
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
    parser.add_argument(
        "--ui",
        "--output-format",
        dest="ui",
        choices=UI_MODES,
        default="auto",
        help="Terminal output mode; auto uses Rich on a TTY and plain logs otherwise",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def _optional_cli_path(value: str | None) -> Path | None:
    if value is None or value.strip() in {"", "-"}:
        return None
    return Path(value)


def _path_target_args(
    args: argparse.Namespace,
    *,
    config: Path,
    dataset_root: Path,
    task: str,
    camera: str,
    run_id: str | None,
) -> argparse.Namespace:
    values = vars(args) | {
        "config": config,
        "data_path": None,
        "dataset_root": dataset_root,
        "task": task,
        "camera": camera,
        "path_mode": None,
        "run_id": run_id,
    }
    return argparse.Namespace(**values)


def _run_path_input(args: argparse.Namespace, reporter: ProcessUI) -> dict[str, Any]:
    if args.dataset_root is not None:
        raise ValueError("--data-path and --dataset-root cannot be used together")
    if args.path_mode is None:
        raise ValueError("--data-path requires exactly one of --target-only/--pick-place")
    if _optional_cli_path(args.source_run_dir) is not None:
        raise ValueError("--source-run-dir is not supported with --data-path")
    mode = AnnotationMode(args.path_mode)
    resolved = resolve_dataset_input(args.data_path, mode=mode, task=args.task)
    if args.camera is not None and any(target.camera != args.camera for target in resolved.targets):
        raise ValueError("--camera does not match the dataset extract manifest")
    if resolved.is_collection and args.episode_ids is not None and len(resolved.targets) != 1:
        raise ValueError("collection --episode-ids requires selecting one --task")

    config = PATH_MODE_CONFIGS[mode]
    if not resolved.is_collection:
        target = resolved.targets[0]
        return _run_from_args(
            _path_target_args(
                args,
                config=config,
                dataset_root=target.root,
                task=target.task,
                camera=target.camera,
                run_id=args.run_id,
            ),
            reporter,
        )

    collection_run_id = _validate_run_id(args.run_id or ArtifactStore.new_run_id())
    records: list[dict[str, Any]] = []
    for target in resolved.targets:
        task_run_id = _validate_run_id(f"{collection_run_id}-{target.task}")
        try:
            summary = _run_from_args(
                _path_target_args(
                    args,
                    config=config,
                    dataset_root=target.root,
                    task=target.task,
                    camera=target.camera,
                    run_id=task_run_id,
                ),
                reporter,
            )
        except Exception as exc:  # noqa: BLE001 - one task must not stop a collection
            reporter.note(
                f"collection task {target.task} failed: {type(exc).__name__}: {exc}",
                level="error",
            )
            records.append(
                {
                    "task": target.task,
                    "run_id": task_run_id,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        else:
            records.append(
                {
                    "task": target.task,
                    "run_id": task_run_id,
                    "status": "completed" if summary["passed"] else "failed",
                    "artifact": summary.get("artifact"),
                }
            )
    result: dict[str, Any] = {
        "format_version": "robotwin_process_collection_summary_v1",
        "run_id": collection_run_id,
        "dataset_root": str(resolved.root),
        "annotation_mode": mode.value,
        "records": records,
        "passed": all(record["status"] == "completed" for record in records),
    }
    artifact = ArtifactStore.write_json(
        args.output_dir.expanduser().resolve() / f"{collection_run_id}-collection-summary.json",
        result,
    )
    result["artifact"] = str(artifact)
    return result


def _run_from_args(
    args: argparse.Namespace,
    reporter: ProcessUI,
) -> dict[str, Any]:
    if args.run_id is not None:
        _validate_run_id(args.run_id)
    if args.data_path is not None:
        return _run_path_input(args, reporter)
    if args.path_mode is not None:
        raise ValueError("--target-only/--pick-place require --data-path")
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
            or args.urdf_egl_device_id is not None
            or args.urdf_pipeline_buffer_size != DEFAULT_URDF_PIPELINE_BUFFER_SIZE
            or args.no_urdf_pipeline
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
        request = ProcessRequest(
            dataset_root=dataset_root,
            output_root=args.output_dir,
            task=task,
            camera=camera,
            run_id=args.run_id,
            episode_ids=None if args.episode_ids is None else tuple(args.episode_ids),
            skip_render=args.skip_render,
        )

        def run_sam(
            selected: ProcessRequest,
            *,
            reporter: ProcessUI | None = None,
        ) -> dict[str, Any]:
            return process_dataset(
                config,
                dataset_root=selected.dataset_root,
                task=selected.task,
                camera=selected.camera,
                output_root=selected.output_root,
                run_id=selected.run_id,
                episode_ids=selected.episode_ids,
                force=args.force,
                skip_render=selected.skip_render,
                reporter=reporter,
            )

        return DatasetPipeline(
            config,
            sam_runner=cast(DatasetBackendRunner, run_sam),
        ).run(
            request,
            backend=GripperBackend.SAM,
            reporter=reporter,
        )
    else:
        if args.force:
            raise ValueError(
                "--force is not supported by the immutable URDF backend; use a new run id"
            )
        if args.dry_run and args.resume:
            raise ValueError("--dry-run and --resume cannot be used together")
        resolved_urdf_path = (
            DEFAULT_BUNDLED_URDF_PATH if urdf_path is None else urdf_path
        )
        selected_episode_ids = (
            None if args.episode_ids is None else tuple(args.episode_ids)
        )
        depth_tolerance_mm = (
            DEFAULT_URDF_DEPTH_TOLERANCE_MM
            if args.urdf_depth_tolerance_mm is None
            else args.urdf_depth_tolerance_mm
        )
        minimum_eligible_nonempty_fraction = (
            DEFAULT_URDF_MINIMUM_ELIGIBLE_NONEMPTY_FRACTION
            if args.urdf_minimum_eligible_nonempty_fraction is None
            else args.urdf_minimum_eligible_nonempty_fraction
        )
        if source_run_dir is None:
            if args.dry_run or args.resume:
                raise ValueError(
                    "live URDF mode is fresh-only; --dry-run/--resume require "
                    "--source-run-dir"
                )
            dataset_root = (
                config.dataset.root if args.dataset_root is None else args.dataset_root
            )
            task = config.dataset.task if args.task is None else args.task
            camera = config.dataset.camera if args.camera is None else args.camera
            request = ProcessRequest(
                dataset_root=dataset_root,
                output_root=args.output_dir,
                task=task,
                camera=camera,
                run_id=args.run_id,
                episode_ids=selected_episode_ids,
                skip_render=args.skip_render,
            )

            def run_live_urdf(
                selected: ProcessRequest,
                *,
                reporter: ProcessUI | None = None,
            ) -> dict[str, Any]:
                return process_live_urdf_pipeline(
                    pipeline_config=config,
                    dataset_root=selected.dataset_root,
                    task=selected.task,
                    camera=selected.camera,
                    output_root=selected.output_root,
                    urdf_path=resolved_urdf_path,
                    mesh_root=args.urdf_mesh_root,
                    run_id=selected.run_id,
                    episode_ids=selected.episode_ids,
                    skip_render=selected.skip_render,
                    depth_tolerance_mm=depth_tolerance_mm,
                    minimum_eligible_nonempty_fraction=(
                        minimum_eligible_nonempty_fraction
                    ),
                    fit_config_json=args.urdf_fit_config_json,
                    allow_partial_source=args.allow_partial_source,
                    urdf_pipeline=not args.no_urdf_pipeline,
                    urdf_pipeline_buffer_size=args.urdf_pipeline_buffer_size,
                    urdf_egl_device_id=args.urdf_egl_device_id,
                    reporter=reporter,
                )

            return DatasetPipeline(
                config,
                urdf_runner=cast(DatasetBackendRunner, run_live_urdf),
            ).run(
                request,
                backend=GripperBackend.URDF,
                reporter=reporter,
            )
        else:
            if args.resume and not args.run_id:
                raise ValueError("--resume requires an explicit --run-id")
            source_summary = _read_json_object(
                source_run_dir.expanduser().resolve() / "process_summary.json",
                description="source process summary",
            )
            dataset_root = (
                Path(str(source_summary.get("dataset_root", "")))
                if args.dataset_root is None
                else args.dataset_root
            )
            task = (
                str(source_summary.get("task", ""))
                if args.task is None
                else args.task
            )
            camera = (
                str(source_summary.get("camera", ""))
                if args.camera is None
                else args.camera
            )
            if not task or not camera:
                raise ValueError("source process summary does not define task/camera")
            request = ProcessRequest(
                dataset_root=dataset_root,
                output_root=args.output_dir,
                task=task,
                camera=camera,
                run_id=args.run_id,
                episode_ids=selected_episode_ids,
                skip_render=args.skip_render,
            )

            def run_frozen_urdf(
                selected: ProcessRequest,
                *,
                reporter: ProcessUI | None = None,
            ) -> dict[str, Any]:
                return process_urdf_source_run(
                    pipeline_config=config,
                    dataset_root=selected.dataset_root,
                    source_run_dir=source_run_dir,
                    task=selected.task,
                    camera=selected.camera,
                    output_root=selected.output_root,
                    urdf_path=resolved_urdf_path,
                    mesh_root=args.urdf_mesh_root,
                    run_id=selected.run_id,
                    episode_ids=selected.episode_ids,
                    skip_render=selected.skip_render,
                    dry_run=args.dry_run,
                    resume=args.resume,
                    depth_tolerance_mm=depth_tolerance_mm,
                    minimum_eligible_nonempty_fraction=(
                        minimum_eligible_nonempty_fraction
                    ),
                    fit_config_json=args.urdf_fit_config_json,
                    allow_partial_source=args.allow_partial_source,
                    egl_device_id=args.urdf_egl_device_id,
                    reporter=reporter,
                )

            return DatasetPipeline(
                config,
                urdf_runner=cast(DatasetBackendRunner, run_frozen_urdf),
            ).run(
                request,
                backend=GripperBackend.URDF,
                reporter=reporter,
            )


def main() -> None:
    args = _parse_args()
    reporter = create_process_ui(args.ui, verbose=args.verbose)
    try:
        summary = _run_from_args(args, reporter)
    except BaseException as exc:
        reporter.failed(exc)
        raise
    else:
        reporter.finish(summary)
        if reporter.emit_json_summary:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        reporter.close()
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
