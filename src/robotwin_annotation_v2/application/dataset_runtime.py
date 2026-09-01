"""Dataset execution primitives used by the readable application pipeline."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import robotwin_annotation_v2.application.discovery as _discovery
import robotwin_annotation_v2.application.urdf_runtime as _urdf_runtime
from robotwin_annotation_v2.adapters.artifact_store import ArtifactStore
from robotwin_annotation_v2.adapters.robotwin_dataset import RoboTwinDataset
from robotwin_annotation_v2.application.dataset_binding import dataset_binding_from_target
from robotwin_annotation_v2.application.dataset_input import (
    DatasetTarget,
    discover_task_episode_ids,
    read_dataset_task_kind,
    resolve_dataset_input,
)
from robotwin_annotation_v2.application.dataset_pipeline import (
    DatasetBackendRunner,
    DatasetPipeline,
)
from robotwin_annotation_v2.application.provenance import (
    prompt_bundle_for_config,
    prompt_bundle_from_manifest,
    target_profile_from_manifest,
    validate_profile_provenance_pair,
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
    DEFAULT_URDF_PIPELINE_BUFFER_SIZE,
    UrdfWorkflow,
    UrdfWorkflowHooks,
)
from robotwin_annotation_v2.config import (
    ConfigError,
    ParallelConfig,
    PipelineConfig,
    PipelineProfile,
    bind_dataset,
    has_dataset_block,
    load_config,
    load_profile,
    parse_gpu_list,
    validate_dataset_component,
)
from robotwin_annotation_v2.domain import (
    AnnotationMode,
    GripperBackend,
    TargetOnlyTaskKind,
    TargetProfile,
    target_profile_for_task_kind,
)
from robotwin_annotation_v2.models import ProcessRequest
from robotwin_annotation_v2.terminal_ui import UI_MODES, ProcessUI, create_process_ui
from robotwin_annotation_v2.urdf_gripper_publisher import (
    validate_derivation_source_episode,
    write_source_episode_completion_receipt,
    write_source_run_contract,
)

GRIPPER_BACKENDS = ("sam", "urdf")
TARGET_ONLY_PROFILE_SELECTORS: dict[str, TargetProfile] = {
    "origin": TargetProfile.GRASP_MANIPULATION,
    "contact_press": TargetProfile.CONTACT_PRESS,
    "door_open": TargetProfile.DOOR_OPEN,
}
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BUNDLED_URDF_PATH = (
    PROJECT_ROOT
    / "configs"
    / "assets"
    / "aloha-agilex"
    / "arx5_description_isaac_gripper.urdf"
)
# ``process.yaml`` is the single reusable profile used by the path-oriented
# CLI.  Keep the old API file available only for explicit legacy task-bound
# configurations; it must not silently become the default again.
DEFAULT_PROFILE_CONFIG = PROJECT_ROOT / "configs" / "process.yaml"
DEFAULT_PROCESS_CONFIG = DEFAULT_PROFILE_CONFIG
LEGACY_DEFAULT_PROCESS_CONFIG = PROJECT_ROOT / "configs" / "process_qwen38_api.yaml"
PATH_MODE_CONFIGS = {
    "local": {
        AnnotationMode.PICK_PLACE: PROJECT_ROOT / "configs" / "pilot_move_pillbottle_pad.yaml",
        AnnotationMode.TARGET_ONLY: PROJECT_ROOT / "configs" / "pilot_adjust_bottle_target_only.yaml",
    },
    "api": {
        AnnotationMode.PICK_PLACE: LEGACY_DEFAULT_PROCESS_CONFIG,
        AnnotationMode.TARGET_ONLY: PROJECT_ROOT / "configs" / "process_target_only_qwen38_api.yaml",
    },
}
CONTACT_PRESS_CONFIGS = {
    "local": PROJECT_ROOT / "configs" / "pilot_contact_press_target_only.yaml",
    "api": PROJECT_ROOT / "configs" / "process_contact_press_qwen38_api.yaml",
}
# ``target_profile`` is orthogonal to the target-only timeline.  Keep the
# overlay names in one place so path-mode dispatch cannot accidentally fall
# back to the generic target-only prompts when a new semantic profile is
# introduced.  ``origin`` is the canonical shared-profile selector for the
# historical grasp-manipulation target-only behavior; ``target_only`` remains
# a backwards-compatible mode key accepted by ``load_profile``.
TARGET_ONLY_PROFILE_OVERLAYS: dict[TargetProfile, str] = {
    TargetProfile.GRASP_MANIPULATION: "origin",
    TargetProfile.CONTACT_PRESS: "contact_press",
    TargetProfile.DOOR_OPEN: "door_open",
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
    require_sidecar: bool = True,
) -> DiscoveryResult:
    """Compatibility delegate for the canonical discovery module."""

    return _discovery.discover_episodes(
        root,
        camera=camera,
        require_depth=require_depth,
        require_sidecar=require_sidecar,
    )


_read_json_object = read_json_object
_validate_run_id = validate_sam_run_id
_summary_gripper_backend = summary_gripper_backend
_validate_sam_run_ownership = validate_sam_run_ownership


def _read_runtime_manifest(root: Path) -> Mapping[str, Any] | None:
    """Read a child extract manifest for semantic conflict checks."""

    path = root.expanduser().resolve() / "EXTRACT_MANIFEST.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # The canonical discovery helper keeps malformed optional provenance
        # compatible with native datasets.  Do the same here; a bound parent
        # contract is still validated and propagated below.
        return None
    return payload if isinstance(payload, Mapping) else None


def _task_kind_from_manifest(
    manifest: Mapping[str, Any] | None,
    *,
    label: str,
) -> TargetOnlyTaskKind | None:
    """Parse optional task-kind metadata at the runtime boundary."""

    if manifest is None or manifest.get("task_kind") is None:
        return None
    try:
        return TargetOnlyTaskKind(manifest["task_kind"])
    except (TypeError, ValueError) as exc:
        choices = ", ".join(item.value for item in TargetOnlyTaskKind)
        raise ValueError(
            f"{label} task_kind must be one of: {choices}"
        ) from exc


def _validate_bound_semantics(
    bound_manifest: Mapping[str, Any],
    child_manifest: Mapping[str, Any] | None,
) -> tuple[TargetOnlyTaskKind | None, str | None, Mapping[str, Any] | None]:
    """Validate child metadata against a collection-level bound contract.

    Collection records are often the only source of semantic metadata because
    native child directories may not contain ``EXTRACT_MANIFEST.json``.  If a
    child does provide metadata, it may narrow identity but cannot silently
    change the target profile, task kind, or immutable prompt bundle.
    """

    parent_kind = _task_kind_from_manifest(bound_manifest, label="bound manifest")
    child_kind = _task_kind_from_manifest(child_manifest, label="child manifest")
    if parent_kind is not None and child_kind is not None and parent_kind is not child_kind:
        raise ValueError(
            "child manifest task_kind differs from bound manifest task_kind: "
            f"{child_kind.value} != {parent_kind.value}"
        )

    parent_profile = target_profile_from_manifest(bound_manifest)
    child_profile = target_profile_from_manifest(child_manifest)
    if parent_profile is not None and child_profile is not None and child_profile != parent_profile:
        raise ValueError(
            "child manifest target_profile differs from bound manifest target_profile: "
            f"{child_profile!r} != {parent_profile!r}"
        )

    # A task kind is itself a semantic declaration.  Check it even when the
    # corresponding target_profile field was omitted by an older manifest.
    if parent_profile is not None and child_kind is not None:
        child_kind_profile = target_profile_for_task_kind(child_kind).value
        if child_kind_profile != parent_profile:
            raise ValueError(
                "child manifest task_kind conflicts with bound target_profile: "
                f"{child_kind_profile!r} != {parent_profile!r}"
            )
    if child_profile is not None and parent_kind is not None:
        parent_kind_profile = target_profile_for_task_kind(parent_kind).value
        if parent_kind_profile != child_profile:
            raise ValueError(
                "child manifest target_profile conflicts with bound task_kind: "
                f"{child_profile!r} != {parent_kind_profile!r}"
            )

    parent_bundle = prompt_bundle_from_manifest(bound_manifest, strict=True)
    child_bundle = prompt_bundle_from_manifest(child_manifest, strict=True)
    if parent_bundle is not None and child_bundle is not None:
        # Reuse the content-addressed provenance contract so prompt paths may
        # differ after relocation while template bytes remain authoritative.
        validate_profile_provenance_pair(
            {
                "target_profile": parent_profile,
                "prompt_bundle": parent_bundle,
            },
            {
                "target_profile": child_profile or parent_profile,
                "prompt_bundle": child_bundle,
            },
            label="bound/child prompt provenance",
        )

    # ``profile`` historically doubled as a semantic alias.  If the parent
    # uses that legacy spelling, a child explicitly replacing it with the
    # generic workflow name is ambiguous and must not fall back to origin.
    parent_raw_profile = bound_manifest.get("profile")
    child_raw_profile = None if child_manifest is None else child_manifest.get("profile")
    semantic_aliases = {
        "origin",
        "grasp_manipulation",
        "graspmanipulation",
        "contact_press",
        "contactpress",
        "door_open",
        "dooropen",
    }
    if (
        isinstance(parent_raw_profile, str)
        and parent_raw_profile.strip().lower().replace("-", "_") in semantic_aliases
        and isinstance(child_raw_profile, str)
        and child_raw_profile.strip().lower().replace("-", "_")
        in {AnnotationMode.PICK_PLACE.value, AnnotationMode.TARGET_ONLY.value}
        and child_raw_profile.strip().lower().replace("-", "_")
        != parent_raw_profile.strip().lower().replace("-", "_")
    ):
        raise ValueError(
            "child manifest workflow profile overrides bound semantic profile"
        )

    effective_kind = parent_kind or child_kind
    effective_profile = parent_profile or child_profile
    effective_bundle = parent_bundle or child_bundle
    return effective_kind, effective_profile, effective_bundle


def build_dynamic_manifest(
    root: Path,
    *,
    task: str,
    camera: str,
    episodes: Sequence[DiscoveredEpisode],
    bound_manifest_data: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a runtime manifest while retaining bound semantic metadata."""

    if bound_manifest_data is not None and not isinstance(bound_manifest_data, Mapping):
        raise TypeError("bound_manifest_data must be a mapping")
    child_manifest = _read_runtime_manifest(root)
    bound_manifest = {} if bound_manifest_data is None else dict(bound_manifest_data)
    effective_kind, effective_profile, effective_bundle = _validate_bound_semantics(
        bound_manifest,
        child_manifest,
    )
    # Without a bound collection record, preserve the historical local-manifest
    # discovery path.  With one, parent metadata is authoritative and fills in
    # children that have no local extract manifest.
    local_kind = read_dataset_task_kind(root)
    if effective_kind is None:
        effective_kind = local_kind
    elif local_kind is not None and local_kind is not effective_kind:
        raise ValueError(
            "local manifest task_kind differs from bound task_kind: "
            f"{local_kind.value} != {effective_kind.value}"
        )
    manifest = _discovery.build_dynamic_manifest(
        root,
        task=task,
        camera=camera,
        episodes=episodes,
        measure_episode_fn=_measure_episode,
        task_kind=effective_kind,
        target_profile=effective_profile,
        prompt_bundle=effective_bundle,
    )
    # Keep extract provenance alongside the runtime-normalized identity.  The
    # downstream binder rewrites ``dataset_root`` to the bound path; retaining
    # the original value prevents dynamic manifests from becoming detached
    # from the source extract (especially for materialized/relocated datasets).
    resolved_root = root.expanduser().resolve()
    manifest.setdefault("source_dataset_root", str(resolved_root))
    manifest.setdefault(
        "target_profile",
        target_profile_for_task_kind(manifest.get("task_kind")).value,
    )
    extract_manifest = resolved_root / "EXTRACT_MANIFEST.json"
    if extract_manifest.is_file():
        manifest.setdefault("source_manifest_path", str(extract_manifest))
        try:
            payload = json.loads(extract_manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, Mapping):
            source_root = payload.get("dataset_root")
            if isinstance(source_root, str) and source_root.strip():
                manifest["source_dataset_root"] = source_root
            # Dynamic discovery is authoritative for task/camera/episodes, but
            # preserve task-kind metadata when the resolver was not able to
            # infer it (for example an older manifest shape).
            if "task_kind" not in manifest and payload.get("task_kind") is not None:
                manifest["task_kind"] = payload["task_kind"]
                manifest["target_profile"] = target_profile_for_task_kind(
                    manifest["task_kind"]
                ).value
    # Carry collection-level provenance into a child runtime manifest when no
    # local file supplied it.  ``setdefault`` keeps a real child source path
    # authoritative while preserving the parent contract for native children.
    for key in ("source_dataset_root", "source_manifest_path"):
        if key in bound_manifest:
            manifest.setdefault(key, deepcopy(bound_manifest[key]))
    if effective_kind is not None:
        manifest["task_kind"] = effective_kind.value
    if effective_profile is not None:
        actual_profile = target_profile_from_manifest(manifest)
        if actual_profile != effective_profile:
            raise ValueError(
                "dynamic manifest target_profile differs from bound target_profile: "
                f"{actual_profile!r} != {effective_profile!r}"
            )
        manifest["target_profile"] = effective_profile
    if effective_bundle is not None:
        actual_bundle = prompt_bundle_from_manifest(manifest, strict=True)
        if actual_bundle is None:
            manifest["prompt_bundle"] = deepcopy(dict(effective_bundle))
        else:
            validate_profile_provenance_pair(
                {
                    "target_profile": effective_profile,
                    "prompt_bundle": effective_bundle,
                },
                {
                    "target_profile": target_profile_from_manifest(manifest),
                    "prompt_bundle": actual_bundle,
                },
                label="bound/dynamic prompt provenance",
            )
    return manifest


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
    def build_bound_dynamic_manifest(
        root: Path,
        *,
        task: str,
        camera: str,
        episodes: Sequence[DiscoveredEpisode],
    ) -> dict[str, Any]:
        """Pass bound collection metadata into the legacy manifest hook."""

        return build_dynamic_manifest(
            root,
            task=task,
            camera=camera,
            episodes=episodes,
            bound_manifest_data=config.dataset.manifest_data,
        )

    hooks = SamWorkflowHooks(
        runtime_loader=_load_sam_runtime,
        discover_episodes=discover_episodes,
        build_dynamic_manifest=build_bound_dynamic_manifest,
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
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help=(
            "Legacy task-bound dataset root; with the shared process profile, "
            "use --data-path instead"
        ),
    )
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
    path_mode.add_argument(
        "--mode",
        dest="path_mode",
        choices=(
            AnnotationMode.PICK_PLACE.value,
            AnnotationMode.TARGET_ONLY.value,
            *TARGET_ONLY_PROFILE_SELECTORS,
        ),
        help=(
            "Workflow/profile for --data-path (pick_place, target_only, origin, "
            "contact_press, or door_open)"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_PROCESS_CONFIG,
        help=f"Pipeline YAML (default: {DEFAULT_PROCESS_CONFIG})",
    )
    parser.add_argument("--task")
    parser.add_argument("--camera")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override output.root from the selected config",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--episode-ids", type=_nonnegative_cli_integer, nargs="*")
    parser.add_argument(
        "--all-episodes",
        action="store_true",
        help=(
            "Ignore episode IDs recorded in an extract manifest and process every "
            "complete episode discovered under the selected task"
        ),
    )
    parser.add_argument(
        "--sam-worker-gpus",
        type=parse_gpu_list,
        help=(
            "Comma-separated physical GPU ids for one persistent SAM worker each; "
            "an empty value disables a YAML-configured pool"
        ),
    )
    parser.add_argument(
        "--qwen-max-in-flight",
        type=_positive_cli_integer,
        help="Maximum concurrent remote Qwen HTTP requests across SAM workers",
    )
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
    parser.add_argument("--urdf-fit-config-json", type=Path)
    parser.add_argument(
        "--urdf-egl-device-id",
        type=int,
        help=(
            "Physical GPU for EGL rendering; live URDF mode otherwise selects the "
            "first GPU outside the SAM worker pool"
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
            "Retained for compatibility; automatic discovery already processes eligible "
            "episodes independently, while explicit --episode-ids remain strict"
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


def _positive_cli_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _nonnegative_cli_integer(value: str) -> int:
    """Parse an episode id while rejecting path-invalid negative values."""

    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a non-negative integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def _runtime_dataset_component(value: str, *, field: str) -> str:
    """Apply the shared path-component contract at CLI/workflow boundaries."""

    # ``validate_dataset_component`` raises ConfigError (a ValueError), which
    # is intentionally allowed to propagate as a normal CLI validation error.
    return validate_dataset_component(value, field=field)


def _apply_parallel_cli_overrides(
    config: PipelineConfig,
    args: argparse.Namespace,
) -> PipelineConfig:
    """Apply explicit CLI parallelism without replacing YAML defaults."""

    if args.sam_worker_gpus is None and args.qwen_max_in_flight is None:
        return config
    parallel = replace(
        config.parallel,
        sam_worker_gpus=(
            config.parallel.sam_worker_gpus
            if args.sam_worker_gpus is None
            else args.sam_worker_gpus
        ),
        qwen_max_in_flight=(
            config.parallel.qwen_max_in_flight
            if args.qwen_max_in_flight is None
            else args.qwen_max_in_flight
        ),
    )
    return replace(config, parallel=parallel)


def _optional_cli_path(value: str | None) -> Path | None:
    if value is None or value.strip() in {"", "-"}:
        return None
    return Path(value)


def _semantic_path_selector(payload: Mapping[str, Any]) -> str | None:
    """Return a target-only selector implied by semantic manifest fields.

    ``profile`` is overloaded in historical manifests, while newer records
    may carry only ``target_profile`` or ``task_kind``.  Keep this inference
    deliberately narrow: an unknown value returns ``None`` and is left for
    the canonical dataset resolver to report.  Generic target-only kinds use
    the workflow selector itself; contact/door kinds retain their dedicated
    selectors when the whole input is homogeneous.
    """

    def normalize(value: Any) -> str | None:
        if not isinstance(value, str) or not value.strip():
            return None
        return value.strip().lower().replace("-", "_")

    raw_profile = normalize(payload.get("profile"))
    if raw_profile in {AnnotationMode.PICK_PLACE.value, "pickplace"}:
        return AnnotationMode.PICK_PLACE.value
    if raw_profile in {AnnotationMode.TARGET_ONLY.value, "targetonly"}:
        return AnnotationMode.TARGET_ONLY.value
    if raw_profile in {"contact_press", "contactpress"}:
        return "contact_press"
    if raw_profile in {"door_open", "dooropen"}:
        return "door_open"
    if raw_profile in {"origin", "grasp_manipulation", "graspmanipulation"}:
        return "origin"

    raw_target_profile = normalize(payload.get("target_profile"))
    if raw_target_profile in {"contact_press", "contactpress"}:
        return "contact_press"
    if raw_target_profile in {"door_open", "dooropen"}:
        return "door_open"
    if raw_target_profile in {"origin", "grasp_manipulation", "graspmanipulation"}:
        return "origin"

    raw_task_kind = normalize(payload.get("task_kind"))
    if raw_task_kind is None:
        return None
    try:
        task_kind = TargetOnlyTaskKind(raw_task_kind)
    except ValueError:
        return None
    if task_kind is TargetOnlyTaskKind.CONTACT_ACTION_SITE:
        return "contact_press"
    if task_kind is TargetOnlyTaskKind.DOOR_OPEN_ACTION_SITE:
        return "door_open"
    return AnnotationMode.TARGET_ONLY.value


def _infer_path_mode(path: Path) -> str:
    """Infer the annotation mode from an extract manifest when available."""

    manifest_path = path.expanduser().resolve() / "EXTRACT_MANIFEST.json"
    if manifest_path.is_file():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot infer annotation mode from {manifest_path}: {exc}") from exc
        if isinstance(payload, dict):
            direct_selector = _semantic_path_selector(payload)
            if direct_selector is not None:
                return direct_selector
            # A few collection manifests omit the top-level profile while
            # retaining workflow/semantic metadata on every task record.  If
            # all records are target-only (even when their semantic profiles
            # differ), infer the shared target-only workflow.  A homogeneous
            # contact/door collection keeps its dedicated selector; mixed
            # semantic profiles must stay on ``target_only`` so each target
            # can be routed independently by ``_target_profile_overlay_key``.
            datasets = payload.get("datasets")
            if isinstance(datasets, list) and datasets:
                record_selectors = [
                    _semantic_path_selector(record)
                    for record in datasets
                    if isinstance(record, Mapping)
                ]
                if len(record_selectors) == len(datasets) and all(
                    selector is not None for selector in record_selectors
                ):
                    selectors = {selector for selector in record_selectors if selector is not None}
                    if selectors == {AnnotationMode.PICK_PLACE.value}:
                        return AnnotationMode.PICK_PLACE.value
                    if AnnotationMode.PICK_PLACE.value not in selectors:
                        if len(selectors) == 1:
                            return next(iter(selectors))
                        return AnnotationMode.TARGET_ONLY.value
    # Native RoboTwin directories have historically been pick-place inputs;
    # callers can select target-only explicitly with --mode/--target-only.
    return AnnotationMode.PICK_PLACE.value


def _path_target_args(
    args: argparse.Namespace,
    *,
    config: Path,
    dataset_root: Path,
    task: str,
    camera: str,
    run_id: str | None,
    parallel_defaults: ParallelConfig | None,
    bound_config: PipelineConfig | None = None,
    episode_ids: tuple[int, ...] | None = None,
) -> argparse.Namespace:
    values = vars(args) | {
        "config": config,
        "data_path": None,
        "dataset_root": dataset_root,
        "task": task,
        "camera": camera,
        "path_mode": None,
        "run_id": run_id,
        "path_parallel_defaults": parallel_defaults,
        "bound_config": bound_config,
        "episode_ids": episode_ids,
    }
    return argparse.Namespace(**values)


def _target_profile_of(target: DatasetTarget) -> TargetProfile:
    """Return a validated target profile from a resolver target.

    ``DatasetTarget.profile`` is an enum in the canonical resolver, but the
    path runtime is also used with small adapter/test doubles.  Normalizing at
    this boundary keeps routing fail-closed for malformed metadata instead of
    silently selecting the generic target-only prompt bundle.
    """

    raw_profile = getattr(target, "profile", TargetProfile.GRASP_MANIPULATION)
    try:
        return TargetProfile(raw_profile)
    except (TypeError, ValueError) as exc:
        choices = ", ".join(profile.value for profile in TargetProfile)
        raise ValueError(
            f"dataset target {getattr(target, 'task', '<unknown>')!r} has unsupported "
            f"target profile {raw_profile!r}; choose {choices}"
        ) from exc


def _target_profile_overlay_key(
    target: DatasetTarget,
    *,
    requested_path_mode: str,
    mode: AnnotationMode,
) -> str:
    """Resolve the shared-profile overlay for one target.

    ``contact_press`` is retained as a CLI compatibility selector, but all
    three semantic variants run under the target-only timeline.  Pick-place
    inputs are intentionally restricted to the grasp profile so a malformed
    manifest cannot receive a target-only prompt bundle by accident.
    """

    target_profile = _target_profile_of(target)
    if requested_path_mode in TARGET_ONLY_PROFILE_SELECTORS and requested_path_mode != "target_only":
        expected_profile = TARGET_ONLY_PROFILE_SELECTORS[requested_path_mode]
        if target_profile is not expected_profile:
            requirement = (
                "contact_action_site task metadata"
                if requested_path_mode == "contact_press"
                else f"{expected_profile.value} task metadata"
            )
            raise ValueError(
                f"--mode {requested_path_mode} requires {requirement}; "
                f"{target.task!r} declares {target_profile.value}"
            )
        return TARGET_ONLY_PROFILE_OVERLAYS[expected_profile]

    if mode is AnnotationMode.PICK_PLACE:
        if target_profile is not TargetProfile.GRASP_MANIPULATION:
            raise ValueError(
                f"{target_profile.value} profile requires target_only mode; "
                f"{target.task!r} was selected with --pick-place"
            )
        return AnnotationMode.PICK_PLACE.value

    if mode is AnnotationMode.TARGET_ONLY:
        try:
            return TARGET_ONLY_PROFILE_OVERLAYS[target_profile]
        except KeyError as exc:  # pragma: no cover - enum exhaustiveness guard
            raise ValueError(
                f"no target-only overlay is registered for {target_profile.value}"
            ) from exc

    raise ValueError(f"unsupported annotation mode for path routing: {mode.value}")


def _path_profile_config(
    profile: PipelineConfig,
    target: DatasetTarget,
) -> Path:
    """Resolve a legacy task-bound config for a target profile.

    The reusable ``configs/process.yaml`` path is handled by
    :func:`_target_profile_overlay_key`.  This compatibility helper is only
    used when callers explicitly provide an older task-bound YAML.  There is
    no checked-in legacy door-open config yet, so refusing that combination is
    safer than silently loading ordinary target-only prompts.
    """

    mode = profile.annotation.mode
    target_profile = _target_profile_of(target)
    if target_profile is profile.annotation.profile:
        return profile.config_path
    # A legacy config that explicitly carries a semantic target profile is
    # not a safe generic fallback for a different target.  Without this guard
    # a contact/door config could silently route an ordinary target-only task
    # through the origin prompts (or vice versa).
    if profile.annotation.profile is not TargetProfile.GRASP_MANIPULATION:
        raise ValueError(
            f"legacy config {profile.config_path} declares "
            f"{profile.annotation.profile.value}, but target {target.task!r} "
            f"declares {target_profile.value}"
        )
    if mode is AnnotationMode.PICK_PLACE:
        if target_profile is not TargetProfile.GRASP_MANIPULATION:
            raise ValueError(
                f"{target_profile.value} profile requires target_only mode; "
                f"legacy config {profile.config_path} declares pick_place"
            )
        return PATH_MODE_CONFIGS[profile.qwen.runtime][mode]
    if target_profile is TargetProfile.CONTACT_PRESS:
        if mode is not AnnotationMode.TARGET_ONLY:
            raise ValueError("contact_press profile requires target_only mode")
        return CONTACT_PRESS_CONFIGS[profile.qwen.runtime]
    if target_profile is TargetProfile.DOOR_OPEN:
        if mode is not AnnotationMode.TARGET_ONLY:
            raise ValueError("door_open profile requires target_only mode")
        raise ValueError(
            "door_open target profile requires the shared process profile with a "
            "modes.door_open overlay; no legacy task-bound door_open config is available"
        )
    return PATH_MODE_CONFIGS[profile.qwen.runtime][mode]


def _load_path_config(
    path: Path,
    *,
    mode: AnnotationMode | str,
) -> PipelineProfile | PipelineConfig:
    """Load the shared profile, with an explicit legacy-config fallback.

    A malformed shared profile must not be reported as a missing ``dataset``
    block merely because the compatibility loader was tried afterward.  The
    fallback is only useful when the profile loader rejects a legacy,
    task-bound document; if neither parser succeeds, retain the original
    profile error and chain the legacy error for debugging.
    """

    selector = mode.value if isinstance(mode, AnnotationMode) else str(mode)
    expected_annotation_mode = (
        AnnotationMode.TARGET_ONLY
        if selector in TARGET_ONLY_PROFILE_SELECTORS
        else AnnotationMode(selector)
    )
    try:
        return load_profile(path, mode=selector)
    except ConfigError as profile_error:
        # Only task-bound YAML documents belong to the compatibility parser.
        # In particular, do not turn a malformed shared profile into a
        # misleading "dataset is required" error by trying ``load_config``
        # unconditionally.
        # A missing path is kept on the compatibility seam as well: callers
        # may provide a loader double (and the legacy loader gives the same
        # actionable missing-file error in production).  For an existing
        # document, require an explicit top-level dataset block before trying
        # the legacy parser.
        if path.expanduser().resolve().is_file() and not has_dataset_block(path):
            raise
        try:
            # Keep the compatibility seam callable by older loader doubles
            # and task-bound files.  Legacy documents carry one concrete
            # annotation mode; validate it immediately below instead of
            # asking the loader to resolve shared-profile overlays.
            legacy_profile = load_config(path)
        except ConfigError as legacy_error:
            # A strict profile rejection is the expected first step for a
            # valid legacy file.  If that file is malformed as well, expose
            # the legacy parser's field-level error instead of the generic
            # profile/legacy distinction.
            if "dataset block" in str(profile_error):
                raise legacy_error from profile_error
            raise profile_error from legacy_error
        if legacy_profile.annotation.mode is not expected_annotation_mode:
            raise ValueError(
                f"--{selector.replace('_', '-')} requires "
                f"annotation.mode={expected_annotation_mode.value}, "
                f"but {legacy_profile.config_path} declares "
                f"{legacy_profile.annotation.mode.value}"
            )
        return legacy_profile


def _is_unbound_profile_error(path: Path, error: ConfigError) -> bool:
    """Identify the actionable shared-profile/no-dataset failure mode.

    The legacy execution branch calls :func:`load_config`, which intentionally
    rejects a reusable profile because it has no dataset block (and may have
    several mode overlays).  Restrict the migration hint to the two errors
    emitted for an otherwise valid profile; malformed YAML and legacy config
    validation errors should retain their original diagnostics.
    """

    resolved = path.expanduser().resolve()
    if not resolved.is_file() or has_dataset_block(resolved):
        return False
    message = str(error)
    return message.startswith("profile mode is required when multiple modes are defined") or (
        message == "config.dataset is required"
    )


def _run_path_input(args: argparse.Namespace, reporter: ProcessUI) -> dict[str, Any]:
    if args.dataset_root is not None:
        raise ValueError("--data-path and --dataset-root cannot be used together")
    requested_task = (
        None
        if args.task is None
        else _runtime_dataset_component(args.task, field="--task")
    )
    requested_camera = (
        None
        if args.camera is None
        else _runtime_dataset_component(args.camera, field="--camera")
    )
    requested_path_mode = (
        str(args.path_mode)
        if args.path_mode is not None
        else _infer_path_mode(args.data_path)
    )
    source_run_dir = _optional_cli_path(args.source_run_dir)
    mode = (
        AnnotationMode.TARGET_ONLY
        if requested_path_mode in TARGET_ONLY_PROFILE_SELECTORS
        else AnnotationMode(requested_path_mode)
    )
    resolved = resolve_dataset_input(
        args.data_path,
        mode=mode,
        task=requested_task,
        camera=requested_camera,
    )
    if source_run_dir is not None and resolved.is_collection:
        raise ValueError(
            "--source-run-dir with --data-path requires selecting one task; "
            "collection runs do not have a per-task frozen source mapping"
        )
    if resolved.is_collection and args.episode_ids is not None and len(resolved.targets) != 1:
        raise ValueError("collection --episode-ids requires selecting one --task")

    # Try the single profile document first.  Legacy task-bound YAML remains a
    # compatibility path and keeps the old task-kind profile selection intact.
    base_profile = _load_path_config(args.config, mode=requested_path_mode)

    def selected_episode_ids(target: DatasetTarget) -> tuple[int, ...]:
        if getattr(args, "all_episodes", False):
            selected = discover_task_episode_ids(
                target.root,
                task=target.task,
                camera=target.camera,
                require_depth=args.gripper_backend == "urdf",
            )
        elif args.episode_ids is not None:
            selected = tuple(args.episode_ids)
            if getattr(target, "discovered_all_episodes", False):
                # Native targets already carry the resolver's complete,
                # task-scoped universe.  Use it for membership only; the CLI
                # selection itself remains authoritative.
                task_episode_ids = set(target.episode_ids)
                foreign_ids = tuple(
                    episode_id for episode_id in selected if episode_id not in task_episode_ids
                )
                if foreign_ids:
                    rendered = ", ".join(str(value) for value in foreign_ids)
                    raise ValueError(
                        f"episode ids do not belong to task {target.task!r}: {rendered}"
                    )
        elif getattr(target, "discovered_all_episodes", False):
            selected = discover_task_episode_ids(
                target.root,
                task=target.task,
                camera=target.camera,
                require_depth=args.gripper_backend == "urdf",
            )
        else:
            selected = target.episode_ids
        selected = tuple(dict.fromkeys(int(value) for value in selected))
        if not selected:
            raise ValueError(f"no episodes selected for task {target.task!r}")
        return selected

    def target_profile_key(target: DatasetTarget) -> str:
        return _target_profile_overlay_key(
            target,
            requested_path_mode=requested_path_mode,
            mode=mode,
        )

    profile_cache: dict[str, PipelineProfile] = {}
    if isinstance(base_profile, PipelineProfile):
        profile_cache[requested_path_mode] = base_profile

    def bound_target_config(target: DatasetTarget) -> tuple[PipelineConfig, tuple[int, ...]]:
        _runtime_dataset_component(target.task, field="dataset.task")
        _runtime_dataset_component(target.camera, field="dataset.camera")
        selected = selected_episode_ids(target)
        if isinstance(base_profile, PipelineProfile):
            key = target_profile_key(target)
            if key not in profile_cache:
                profile_cache[key] = load_profile(args.config, mode=key)
            binding = dataset_binding_from_target(
                target,
                # Let a manifest's declared smoke episode survive the normal
                # (no CLI override) path.  ``selected`` is explicit only for
                # ``--episode-ids`` or ``--all-episodes``; in the default case
                # the target already carries the manifest selection.
                episode_ids=(
                    selected
                    if getattr(args, "all_episodes", False) or args.episode_ids is not None
                    else None
                ),
                manifest_data=getattr(target, "manifest_data", None),
                manifest_path=getattr(target, "manifest_path", None),
            )
            return bind_dataset(profile_cache[key], binding), selected
        target_config_path = _path_profile_config(base_profile, target)
        # ``base_profile`` is a legacy PipelineConfig.  Preserve its previous
        # task-specific override behavior while forwarding manifest-selected
        # episode IDs into the request.
        # Keep the legacy loader call per target for compatibility with tools
        # that monkeypatch or audit task-specific YAML selection.  The shared
        # profile path never enters this branch.
        if requested_path_mode in TARGET_ONLY_PROFILE_SELECTORS:
            # Keep explicit semantic selectors fail-closed for legacy
            # task-bound configs too.  Otherwise a task could be silently
            # routed through a generic target-only YAML.
            target_profile_key(target)
        target_config = load_config(target_config_path)
        return target_config, selected

    if not resolved.is_collection:
        target = resolved.targets[0]
        target_config, selected = bound_target_config(target)
        return _run_from_args(
            _path_target_args(
                args,
                config=target_config.config_path,
                dataset_root=target.root,
                task=target.task,
                camera=target.camera,
                run_id=args.run_id,
                parallel_defaults=(
                    None
                    if isinstance(base_profile, PipelineProfile)
                    or target_config.config_path == base_profile.config_path
                    else base_profile.parallel
                ),
                bound_config=target_config,
                episode_ids=selected,
            ),
            reporter,
        )

    collection_run_id = _validate_run_id(args.run_id or ArtifactStore.new_run_id())
    records: list[dict[str, Any]] = []
    for target in resolved.targets:
        task_run_id = _validate_run_id(f"{collection_run_id}-{target.task}")
        try:
            target_config, selected = bound_target_config(target)
            summary = _run_from_args(
                _path_target_args(
                    args,
                    config=target_config.config_path,
                    dataset_root=target.root,
                    task=target.task,
                    camera=target.camera,
                    run_id=task_run_id,
                    parallel_defaults=(
                        None
                        if isinstance(base_profile, PipelineProfile)
                        or target_config.config_path == base_profile.config_path
                        else base_profile.parallel
                    ),
                    bound_config=target_config,
                    episode_ids=selected,
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
                    "target_profile": _target_profile_of(target).value,
                }
            )
        else:
            record: dict[str, Any] = {
                "task": target.task,
                "run_id": task_run_id,
                "status": "completed" if summary["passed"] else "failed",
                "artifact": summary.get("artifact"),
                "target_profile": target_config.annotation.profile.value,
            }
            bundle = summary.get("prompt_bundle")
            if isinstance(bundle, Mapping):
                record["prompt_bundle"] = dict(bundle)
            else:
                derived_bundle = prompt_bundle_for_config(target_config)
                if derived_bundle is not None:
                    record["prompt_bundle"] = derived_bundle
            records.append(record)
    result: dict[str, Any] = {
        "format_version": "robotwin_process_collection_summary_v1",
        "run_id": collection_run_id,
        "dataset_root": str(resolved.root),
        "annotation_mode": mode.value,
        "profile": (
            requested_path_mode
            if requested_path_mode in TARGET_ONLY_PROFILE_SELECTORS
            else "shared"
        ),
        "records": records,
        "passed": all(record["status"] == "completed" for record in records),
    }
    artifact = ArtifactStore.write_json(
        (
            base_profile.output_root
            if args.output_dir is None
            else args.output_dir
        ).expanduser().resolve()
        / f"{collection_run_id}-collection-summary.json",
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
        raise ValueError("--mode/--target-only/--pick-place require --data-path")
    bound_config = getattr(args, "bound_config", None)
    if bound_config is not None:
        config = bound_config
    else:
        try:
            config = load_config(args.config)
        except ConfigError as exc:
            if _is_unbound_profile_error(args.config, exc):
                raise ValueError(
                    "the shared pipeline profile requires --data-path; use "
                    "--data-path DATASET_OR_COLLECTION with --mode/--target-only/"
                    "--pick-place, or pass a legacy task-bound --config containing "
                    "a dataset block when using --dataset-root"
                ) from exc
            raise
    parallel_defaults = getattr(args, "path_parallel_defaults", None)
    if parallel_defaults is not None:
        config = replace(config, parallel=parallel_defaults)
    config = _apply_parallel_cli_overrides(config, args)
    output_root = config.output_root if args.output_dir is None else args.output_dir
    source_run_dir = _optional_cli_path(args.source_run_dir)
    urdf_path = _optional_cli_path(args.urdf_path)
    if args.gripper_backend == "sam":
        if (
            source_run_dir is not None
            or urdf_path is not None
            or args.urdf_mesh_root is not None
            or args.urdf_depth_tolerance_mm is not None
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
        task = _runtime_dataset_component(
            config.dataset.task if args.task is None else args.task,
            field="--task",
        )
        camera = _runtime_dataset_component(
            config.dataset.camera if args.camera is None else args.camera,
            field="--camera",
        )
        request = ProcessRequest(
            dataset_root=dataset_root,
            output_root=output_root,
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
        if source_run_dir is None:
            if args.dry_run or args.resume:
                raise ValueError(
                    "live URDF mode is fresh-only; --dry-run/--resume require "
                    "--source-run-dir"
                )
            dataset_root = (
                config.dataset.root if args.dataset_root is None else args.dataset_root
            )
            task = _runtime_dataset_component(
                config.dataset.task if args.task is None else args.task,
                field="--task",
            )
            camera = _runtime_dataset_component(
                config.dataset.camera if args.camera is None else args.camera,
                field="--camera",
            )
            request = ProcessRequest(
                dataset_root=dataset_root,
                output_root=output_root,
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
            task = _runtime_dataset_component(task, field="--task")
            camera = _runtime_dataset_component(camera, field="--camera")
            request = ProcessRequest(
                dataset_root=dataset_root,
                output_root=output_root,
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
