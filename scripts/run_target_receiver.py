#!/usr/bin/env python3
"""Run or inspect the target/receiver plus gripper annotation pipeline."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from robotwin_annotation_v2.adapters import (
    ArtifactStore,
    OpenAICompatibleQwenClient,
    RoboTwinDataset,
    Sam3Adapter,
    Sam3Error,
    sam3_video_resource,
)
from robotwin_annotation_v2.config import PipelineConfig, load_config
from robotwin_annotation_v2.domain import AnnotationMode, ObjectRole, annotation_spec
from robotwin_annotation_v2.models import (
    EpisodeRef,
    FrameWindow,
    LoopContext,
    MaskQCStatus,
    MaskRun,
    MaskStatus,
    SemanticPlan,
)
from robotwin_annotation_v2.pipeline import (
    GripperSeedQualityGateConfig,
    GripperStageError,
    MaskQCError,
    QwenStageError,
    RoleMaskData,
    SamStageError,
    SamStageResult,
    build_loop_context,
    compose_visible_mask,
    evaluate_temporal_mask,
    parse_semantic_plan,
    run_gripper_stage,
    run_mask_qc_stage,
    run_qwen_stage,
    run_sam_stage,
    save_mask_qc_artifacts,
    save_sam_artifacts,
)

SAM_EXECUTION_ERRORS = (
    GripperStageError,
    MaskQCError,
    Sam3Error,
    SamStageError,
    RuntimeError,
    ValueError,
    OSError,
)


@dataclass(frozen=True)
class SamEpisodeExecution:
    mask_run: MaskRun
    qc_path: Path | None
    annotation_mode: AnnotationMode = AnnotationMode.PICK_PLACE


@dataclass(frozen=True)
class GripperEpisodeExecution:
    mask_run: MaskRun
    active_arm: str
    gripper_status: str
    selected_candidate: str | None
    seed_qc_path: Path | None


def _dataset(config: PipelineConfig) -> RoboTwinDataset:
    return RoboTwinDataset(
        config.dataset.root,
        task=config.dataset.task,
        camera=config.dataset.camera,
        manifest_path=config.dataset.manifest,
        manifest_data=config.dataset.manifest_data,
    )


def _print(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _episode_ref(config: PipelineConfig, episode_index: int) -> EpisodeRef:
    if episode_index not in config.dataset.regression_episode_ids:
        raise ValueError(f"episode {episode_index} is not in the configured regression manifest")
    return EpisodeRef(config.dataset.task, episode_index, config.dataset.camera)


def _annotation_mode(config: PipelineConfig) -> AnnotationMode:
    """Resolve mode while retaining compatibility with legacy test configs."""

    annotation = getattr(config, "annotation", None)
    return AnnotationMode(getattr(annotation, "mode", AnnotationMode.PICK_PLACE))


def _build_context(
    config: PipelineConfig,
    dataset: RoboTwinDataset,
    ref: EpisodeRef,
) -> LoopContext:
    """Build Stage-1 context with the task-declared annotation contract."""

    return build_loop_context(dataset, ref, _annotation_mode(config))


def run_preflight(config: PipelineConfig) -> None:
    dataset = _dataset(config)
    manifest_ids = tuple(int(value) for value in dataset.manifest["regression_episode_ids"])
    if manifest_ids != config.dataset.regression_episode_ids:
        raise ValueError("config regression_episode_ids differ from the dataset manifest")
    report = dataset.preflight(config.dataset.regression_episode_ids)
    _print(report)


def run_loop(config: PipelineConfig, episode_index: int, run_id: str | None) -> None:
    dataset = _dataset(config)
    ref = _episode_ref(config, episode_index)
    context = _build_context(config, dataset, ref)
    store = ArtifactStore(config.output_root)
    selected_run_id = run_id or store.new_run_id()
    path = store.save_loop(selected_run_id, ref, context.to_json())
    _print(
        {
            "run_id": selected_run_id,
            "stage": "loop",
            "artifact": str(path),
            "events": context.events.to_json(),
            "semantic_frame_ids": [frame.frame_id for frame in context.semantic_frames],
        }
    )


def run_qwen(config: PipelineConfig, episode_index: int, run_id: str | None) -> None:
    dataset = _dataset(config)
    ref = _episode_ref(config, episode_index)
    context = _build_context(config, dataset, ref)
    frames = dataset.read_frames(
        ref,
        (frame.frame_id for frame in context.semantic_frames),
    )
    client = OpenAICompatibleQwenClient(
        endpoint=config.qwen.endpoint,
        model=config.qwen.model,
        timeout_seconds=config.qwen.timeout_seconds,
    )
    store = ArtifactStore(config.output_root)
    selected_run_id = run_id or store.new_run_id()
    loop_path = store.save_loop(selected_run_id, ref, context.to_json())
    try:
        result = run_qwen_stage(context, frames, config.qwen, client)
    except QwenStageError as exc:
        paths = store.save_qwen_failure(
            selected_run_id,
            ref,
            error=str(exc),
            rendered_prompt=exc.rendered_prompt,
            raw_response=exc.raw_response,
        )
        _print(
            {
                "run_id": selected_run_id,
                "stage": "qwen_failed",
                "loop_artifact": str(loop_path),
                "artifacts": {name: str(path) for name, path in paths.items()},
                "error": str(exc),
            }
        )
        raise SystemExit(2) from exc

    plan = result.semantic_plan
    paths = store.save_semantic_plan(
        selected_run_id,
        ref,
        plan.to_json(),
        rendered_prompt=result.rendered_prompt,
        raw_response=plan.raw_response,
    )
    _print(
        {
            "run_id": selected_run_id,
            "stage": "qwen",
            "loop_artifact": str(loop_path),
            "artifacts": {name: str(path) for name, path in paths.items()},
            "server": result.health,
            "annotation_mode": context.annotation_mode.value,
            "roles": {
                role_plan.role: {
                    "seed_frame_id": role_plan.seed_frame_id,
                    "primary_query": role_plan.primary_query,
                }
                for role_plan in plan.role_plans
            },
        }
    )


def _load_saved_semantic_plan(
    store: ArtifactStore,
    run_id: str,
    context: LoopContext,
) -> SemanticPlan:
    episode_dir = store.episode_dir(run_id, context.episode)
    loop_path = episode_dir / "loop.json"
    semantic_path = episode_dir / "semantic_plan.json"
    prompt_path = episode_dir / "qwen_rendered_prompt.txt"
    raw_path = episode_dir / "qwen_raw_response.txt"
    for path in (loop_path, semantic_path, prompt_path, raw_path):
        if not path.is_file():
            raise ValueError(f"required Stage-2 artifact is missing: {path}")
    saved_loop = json.loads(loop_path.read_text(encoding="utf-8"))
    if saved_loop != context.to_json():
        raise ValueError("saved loop.json differs from the current Stage-1 result")
    saved_plan = json.loads(semantic_path.read_text(encoding="utf-8"))
    rendered_prompt = prompt_path.read_text(encoding="utf-8")
    raw_response = raw_path.read_text(encoding="utf-8")
    plan = parse_semantic_plan(
        raw_response,
        context=context,
        model=str(saved_plan.get("model", "")),
        rendered_prompt=rendered_prompt,
    )
    if saved_plan != plan.to_json():
        raise ValueError("saved semantic_plan.json fails provenance validation")
    return plan


def _default_gripper_qc_prompt(config: PipelineConfig) -> Path:
    return config.config_path.parent / "prompts" / "gripper_seed_candidate_qc.txt"


def _load_bool_png(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8) != 0


def _load_npz_masks(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if "masks" not in archive.files:
            raise ValueError(f"mask archive has no masks array: {path}")
        return np.asarray(archive["masks"], dtype=bool)


def _safe_episode_relative(episode_dir: Path, relative_path: Any, *, label: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise ValueError(f"{label} path is missing from run_manifest.json")
    root = episode_dir.resolve()
    path = (episode_dir / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} path escapes episode directory: {relative_path}") from exc
    return path


def _role_window(record: dict[str, Any]) -> FrameWindow:
    value = record.get("output_window")
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) for item in value)
    ):
        raise ValueError(f"{record.get('role')} output_window is invalid")
    return FrameWindow(int(value[0]), int(value[1]))


def _load_completed_sam_stage(
    config: PipelineConfig,
    store: ArtifactStore,
    run_id: str,
    context: LoopContext,
    frame_shape: tuple[int, int],
) -> SamStageResult:
    episode_dir = store.episode_dir(run_id, context.episode)
    manifest_path = episode_dir / "run_manifest.json"
    masks_path = episode_dir / "masks.npz"
    if not manifest_path.is_file() or not masks_path.is_file():
        raise ValueError(
            f"required SAM artifacts are missing for gripper stage: {episode_dir}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("episode") != context.episode.to_json():
        raise ValueError("saved run_manifest.json episode differs from current context")
    if manifest.get("frame_count") != context.frame_count:
        raise ValueError("saved run_manifest.json frame_count differs from current context")
    role_records: dict[str, dict[str, Any]] = {
        item.get("role"): item
        for item in manifest.get("roles", [])
        if item.get("role") in {"target", "receiver"}
    }
    if set(role_records) != {"target", "receiver"}:
        raise ValueError("gripper stage requires both canonical object role records")
    _validate_manifest_mode(manifest, config)
    required_names = set(annotation_spec(_annotation_mode(config)).required_role_names)
    for role, record in role_records.items():
        if role in required_names:
            if record.get("status") != "ok":
                raise ValueError(f"gripper stage requires {role} status=ok")
            if config.mask.qc_enabled and record.get("qc_status") != "passed":
                raise ValueError(f"gripper stage requires {role} qc_status=passed")
        elif (
            record.get("status") != "not_applicable"
            or record.get("qc_status") != "not_applicable"
        ):
            raise ValueError(
                f"gripper stage requires non-applicable {role} to remain empty"
            )

    def load_role(role: str) -> RoleMaskData:
        record = role_records[role]
        seed_mask_path = _safe_episode_relative(
            episode_dir,
            record.get("seed_mask_path"),
            label=f"{role} seed_mask",
        )
        native_path = _safe_episode_relative(
            episode_dir,
            record.get("native_track_path"),
            label=f"{role} native_track",
        )
        seed_mask = _load_bool_png(seed_mask_path)
        native = _load_npz_masks(native_path)
        expected_shape = (context.frame_count, *frame_shape)
        if seed_mask.shape != frame_shape:
            raise ValueError(f"{role} seed mask has shape {seed_mask.shape}")
        if native.shape != expected_shape:
            raise ValueError(f"{role} native track has shape {native.shape}")
        output_window = _role_window(record)
        visible = compose_visible_mask(native, output_window)
        temporal_qc = evaluate_temporal_mask(
            visible,
            output_window,
            config.mask,
            reference_mask=seed_mask,
        )
        envelope = None
        envelope_path = record.get("canonical_envelope_path")
        if isinstance(envelope_path, str) and envelope_path.strip():
            envelope = _load_bool_png(
                _safe_episode_relative(
                    episode_dir,
                    envelope_path,
                    label=f"{role} canonical_envelope",
                )
            )
        return RoleMaskData(
            role=role,  # type: ignore[arg-type]
            status=MaskStatus.OK,
            seed_frame_id=record.get("seed_frame_id"),
            primary_query=record.get("primary_query"),
            output_window=output_window,
            seed_mask=seed_mask,
            canonical_envelope=envelope,
            native_track=native,
            visible_mask=visible,
            temporal_qc=temporal_qc,
            failure=None,
            qc_status=MaskQCStatus(record.get("qc_status", "not_run")),
            qc_selected_candidate=record.get("qc_selected_candidate"),
            qc_reason=record.get("qc_reason"),
        )

    return SamStageResult(
        frame_count=context.frame_count,
        frame_shape=frame_shape,
        role_masks=tuple(
            load_role(role.value)
            for role in context.annotation_spec.required_object_roles
        ),
    )


def _execute_gripper_episode(
    config: PipelineConfig,
    episode_index: int,
    run_id: str,
    backend: Sam3Adapter,
) -> GripperEpisodeExecution:
    dataset = _dataset(config)
    ref = _episode_ref(config, episode_index)
    context = _build_context(config, dataset, ref)
    store = ArtifactStore(config.output_root)
    shape_values = tuple(int(value) for value in dataset.manifest["frame_shape_hw"])
    if len(shape_values) != 2:
        raise ValueError(f"invalid dataset frame shape: {shape_values}")
    frame_shape = (shape_values[0], shape_values[1])
    sam_result = _load_completed_sam_stage(
        config,
        store,
        run_id,
        context,
        frame_shape,
    )
    seed_frame_ids = {
        data.seed_frame_id for data in sam_result.role_masks if data.seed_frame_id is not None
    }
    seed_images = dataset.read_frames(ref, seed_frame_ids)
    qc_client = OpenAICompatibleQwenClient(
        endpoint=config.qwen.endpoint,
        model=config.qwen.model,
        timeout_seconds=config.qwen.timeout_seconds,
    )
    prompt_path = _default_gripper_qc_prompt(config)
    with sam3_video_resource(
        Path(context.video_source),
        minimum_frame_count=context.frame_count,
    ) as resource_path:
        gripper_result = run_gripper_stage(
            context,
            backend=backend,
            resource_path=resource_path,
            frame_shape=frame_shape,
            gripper_roi_config=config.gripper_roi,
            object_tracks={
                ObjectRole(data.role): data.native_track for data in sam_result.role_masks
            },
            qc_client=qc_client,
            qc_prompt_template=prompt_path,
            qc_max_tokens=220,
            qc_max_attempts=2,
            qc_min_confidence=0.70,
            seed_quality_gate=GripperSeedQualityGateConfig(
                duplicate_iou_threshold=config.mask.qc_duplicate_iou_threshold,
            ),
        )
    mask_run = save_sam_artifacts(
        store,
        run_id,
        context,
        _load_saved_semantic_plan(store, run_id, context),
        sam_result,
        seed_images=seed_images,
        gripper_result=gripper_result,
    )
    episode_dir = store.episode_dir(run_id, ref)
    (episode_dir / "gripper_failure.json").unlink(missing_ok=True)
    seed_qc_path = episode_dir / gripper_result.instance_name / "gripper_seed_qc.json"
    return GripperEpisodeExecution(
        mask_run=mask_run,
        active_arm=gripper_result.active_arm,
        gripper_status=gripper_result.status,
        selected_candidate=gripper_result.selected_candidate,
        seed_qc_path=seed_qc_path if seed_qc_path.is_file() else None,
    )


def _execute_sam_episode(
    config: PipelineConfig,
    episode_index: int,
    run_id: str,
    backend: Sam3Adapter,
) -> SamEpisodeExecution:
    dataset = _dataset(config)
    ref = _episode_ref(config, episode_index)
    context = _build_context(config, dataset, ref)
    store = ArtifactStore(config.output_root)
    plan = _load_saved_semantic_plan(store, run_id, context)
    shape_values = tuple(int(value) for value in dataset.manifest["frame_shape_hw"])
    if len(shape_values) != 2:
        raise ValueError(f"invalid dataset frame shape: {shape_values}")
    frame_shape = (shape_values[0], shape_values[1])
    semantic_frame_ids = {frame.frame_id for frame in context.semantic_frames}
    stage_images = dataset.read_frames(ref, semantic_frame_ids)
    seed_frame_ids = {
        role_plan.seed_frame_id
        for role_plan in plan.role_plans
        if role_plan.seed_frame_id is not None
    }
    seed_images = {frame_id: stage_images[frame_id] for frame_id in seed_frame_ids}

    qc_path: Path | None = None
    with sam3_video_resource(
        Path(context.video_source),
        minimum_frame_count=context.frame_count,
    ) as resource_path:
        mask_qc = None
        if config.mask.qc_enabled:
            qc_client = OpenAICompatibleQwenClient(
                endpoint=config.qwen.endpoint,
                model=config.qwen.model,
                timeout_seconds=config.qwen.timeout_seconds,
            )
            mask_qc = run_mask_qc_stage(
                context,
                plan,
                backend,
                resource_path,
                seed_images=seed_images,
                context_images=stage_images,
                frame_shape=frame_shape,
                mask_config=config.mask,
                client=qc_client,
            )
            qc_path = save_mask_qc_artifacts(store, run_id, context, mask_qc)
        result = run_sam_stage(
            context,
            plan,
            backend,
            resource_path,
            frame_shape=frame_shape,
            mask_config=config.mask,
            mask_qc=mask_qc,
        )
    mask_run = save_sam_artifacts(
        store,
        run_id,
        context,
        plan,
        result,
        seed_images=seed_images,
    )
    (store.episode_dir(run_id, ref) / "sam_failure.json").unlink(missing_ok=True)
    return SamEpisodeExecution(mask_run, qc_path, context.annotation_mode)


def _emit_sam_result(run_id: str, execution: SamEpisodeExecution) -> bool:
    mask_run = execution.mask_run
    required_names = set(
        annotation_spec(execution.annotation_mode).required_role_names
    )
    object_roles = dict(zip(("target", "receiver"), mask_run.roles[:2], strict=True))
    complete = set(object_roles) == {"target", "receiver"} and all(
        role.status
        is (MaskStatus.OK if name in required_names else MaskStatus.NOT_APPLICABLE)
        for name, role in object_roles.items()
    )
    _print(
        {
            "run_id": run_id,
            "stage": "sam" if complete else "sam_incomplete",
            "artifact": str(Path(mask_run.artifact_dir) / "run_manifest.json"),
            "mask_qc_artifact": (
                None if execution.qc_path is None else str(execution.qc_path)
            ),
            "roles": [role.to_json() for role in mask_run.roles],
        }
    )
    return complete


def run_sam(config: PipelineConfig, episode_index: int, run_id: str) -> None:
    ref = _episode_ref(config, episode_index)
    store = ArtifactStore(config.output_root)
    backend: Sam3Adapter | None = None
    try:
        backend = Sam3Adapter(
            checkpoint_path=config.sam3.checkpoint,
            gpus=config.sam3.gpus,
        )
        execution = _execute_sam_episode(config, episode_index, run_id, backend)
    except SAM_EXECUTION_ERRORS as exc:
        failure_path = store.save_sam_failure(run_id, ref, error=str(exc))
        _print(
            {
                "run_id": run_id,
                "stage": "sam_failed",
                "artifact": str(failure_path),
                "error": str(exc),
            }
        )
        raise SystemExit(3) from exc
    finally:
        if backend is not None:
            backend.shutdown()
    if not _emit_sam_result(run_id, execution):
        raise SystemExit(4)


def _emit_gripper_result(run_id: str, execution: GripperEpisodeExecution) -> bool:
    complete = execution.gripper_status == "ok"
    _print(
        {
            "run_id": run_id,
            "stage": "gripper" if complete else "gripper_incomplete",
            "artifact": str(Path(execution.mask_run.artifact_dir) / "run_manifest.json"),
            "seed_qc_artifact": (
                None if execution.seed_qc_path is None else str(execution.seed_qc_path)
            ),
            "active_arm": execution.active_arm,
            "selected_candidate": execution.selected_candidate,
            "roles": [role.to_json() for role in execution.mask_run.roles],
        }
    )
    return complete


def run_gripper(config: PipelineConfig, episode_index: int, run_id: str) -> None:
    ref = _episode_ref(config, episode_index)
    store = ArtifactStore(config.output_root)
    backend: Sam3Adapter | None = None
    try:
        backend = Sam3Adapter(
            checkpoint_path=config.sam3.checkpoint,
            gpus=config.sam3.gpus,
        )
        execution = _execute_gripper_episode(config, episode_index, run_id, backend)
    except SAM_EXECUTION_ERRORS as exc:
        failure_path = store.write_json(
            store.episode_dir(run_id, ref) / "gripper_failure.json",
            {
                "format_version": "robotwin_gripper_failure_v1",
                "error": str(exc),
            },
        )
        _print(
            {
                "run_id": run_id,
                "stage": "gripper_failed",
                "artifact": str(failure_path),
                "error": str(exc),
            }
        )
        raise SystemExit(5) from exc
    finally:
        if backend is not None:
            backend.shutdown()
    if not _emit_gripper_result(run_id, execution):
        raise SystemExit(6)


def _validate_manifest_mode(
    manifest: dict[str, Any],
    config: PipelineConfig,
) -> None:
    """Validate mode metadata, accepting only legacy pick/place omissions."""

    mode = _annotation_mode(config)
    raw_mode = manifest.get("annotation_mode")
    if raw_mode is None:
        if mode is not AnnotationMode.PICK_PLACE:
            raise ValueError("target_only artifacts must declare annotation_mode")
    elif raw_mode != mode.value:
        raise ValueError("saved run_manifest.json annotation_mode differs from config")

    expected_roles = list(annotation_spec(mode).required_role_names)
    raw_roles = manifest.get("required_object_roles")
    if raw_roles is None:
        if mode is not AnnotationMode.PICK_PLACE:
            raise ValueError("target_only artifacts must declare required_object_roles")
    elif raw_roles != expected_roles:
        raise ValueError(
            "saved run_manifest.json required_object_roles differ from config"
        )


def _sam_episode_complete(
    config: PipelineConfig,
    store: ArtifactStore,
    run_id: str,
    ref: EpisodeRef,
) -> bool:
    episode_dir = store.episode_dir(run_id, ref)
    masks_path = episode_dir / "masks.npz"
    manifest_path = episode_dir / "run_manifest.json"
    if not masks_path.is_file() or not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _validate_manifest_mode(manifest, config)
        required_names = set(
            annotation_spec(_annotation_mode(config)).required_role_names
        )
        roles: dict[str, dict[str, Any]] = {
            item["role"]: item
            for item in manifest["roles"]
            if item.get("role") in {"target", "receiver"}
        }
        if set(roles) != {"target", "receiver"}:
            return False
        for role, record in roles.items():
            expected = "ok" if role in required_names else "not_applicable"
            if record.get("status") != expected:
                return False
        if config.mask.qc_enabled:
            qc_path = episode_dir / "mask_qc.json"
            if not qc_path.is_file():
                return False
            qc = json.loads(qc_path.read_text(encoding="utf-8"))
            qc_roles = qc["roles"]
            if set(qc_roles) != required_names:
                return False
            if any(
                qc_roles[role].get("status") != "passed"
                for role in required_names
            ):
                return False
            if any(
                roles[role].get("qc_status") != "passed"
                for role in required_names
            ):
                return False
            if any(
                roles[role].get("qc_status") != "not_applicable"
                for role in set(roles) - required_names
            ):
                return False
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
        return False
    return True


def _gripper_episode_complete(
    config: PipelineConfig,
    store: ArtifactStore,
    run_id: str,
    ref: EpisodeRef,
) -> bool:
    episode_dir = store.episode_dir(run_id, ref)
    masks_path = episode_dir / "masks.npz"
    manifest_path = episode_dir / "run_manifest.json"
    if not masks_path.is_file() or not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _validate_manifest_mode(manifest, config)
        required_names = set(
            annotation_spec(_annotation_mode(config)).required_role_names
        )
        algorithm = manifest.get("algorithm")
        if not isinstance(algorithm, dict):
            return False
        gripper_stage = algorithm.get("gripper_stage")
        if not isinstance(gripper_stage, dict):
            return False
        active_arm = gripper_stage.get("active_arm")
        if active_arm not in {"left", "right"}:
            return False
        role_name = f"gripper_{active_arm}"
        roles = {
            item["role"]: item
            for item in manifest["roles"]
            if item.get("role") in {"target", "receiver", role_name}
        }
        if set(roles) != {"target", "receiver", role_name}:
            return False
        for role in {"target", "receiver"}:
            expected = "ok" if role in required_names else "not_applicable"
            if roles[role].get("status") != expected:
                return False
        if roles[role_name].get("status") != "ok":
            return False
        if roles[role_name].get("qc_status") != "passed":
            return False
        channels = manifest.get("channels", {})
        if channels.get(role_name) not in {2, 3}:
            return False
        qc_path = episode_dir / role_name / "gripper_seed_qc.json"
        if not qc_path.is_file():
            return False
        qc = json.loads(qc_path.read_text(encoding="utf-8"))
        if qc.get("status") != "passed":
            return False
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
        return False
    return True


def _fatal_cuda_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "cuda is unavailable",
            "cuda error",
            "cuda-capable device",
            "cuda initialization",
            "device-side assert",
            "unspecified launch failure",
            "failed to load sam3",
        )
    )


def run_sam_batch(
    config: PipelineConfig,
    episode_ids: tuple[int, ...],
    run_id: str,
    *,
    force: bool = False,
    backend_factory: Callable[..., Sam3Adapter] | None = None,
    episode_runner: Callable[
        [PipelineConfig, int, str, Sam3Adapter], SamEpisodeExecution
    ] | None = None,
) -> None:
    """Run Stage 3 serially while keeping one SAM3 predictor resident."""

    if not episode_ids:
        raise ValueError("sam-batch requires at least one episode")
    factory = Sam3Adapter if backend_factory is None else backend_factory
    runner = _execute_sam_episode if episode_runner is None else episode_runner
    store = ArtifactStore(config.output_root)
    records: list[dict[str, Any]] = []
    pending: list[int] = []
    for episode_id in episode_ids:
        ref = _episode_ref(config, episode_id)
        if not force and _sam_episode_complete(config, store, run_id, ref):
            records.append({"episode": episode_id, "status": "skipped_complete"})
        else:
            pending.append(episode_id)

    fatal_error: BaseException | None = None
    backend: Sam3Adapter | None = None
    if pending:
        try:
            backend = factory(
                checkpoint_path=config.sam3.checkpoint,
                gpus=config.sam3.gpus,
            )
            for episode_id in pending:
                ref = _episode_ref(config, episode_id)
                try:
                    execution = runner(config, episode_id, run_id, backend)
                except SAM_EXECUTION_ERRORS as exc:
                    failure_path = store.save_sam_failure(run_id, ref, error=str(exc))
                    records.append(
                        {
                            "episode": episode_id,
                            "status": "failed",
                            "artifact": str(failure_path),
                            "error": str(exc),
                        }
                    )
                    if _fatal_cuda_error(exc):
                        fatal_error = exc
                        break
                    continue
                complete = _emit_sam_result(run_id, execution)
                records.append(
                    {
                        "episode": episode_id,
                        "status": "completed" if complete else "incomplete",
                        "artifact": str(
                            Path(execution.mask_run.artifact_dir) / "run_manifest.json"
                        ),
                    }
                )
        except SAM_EXECUTION_ERRORS as exc:
            fatal_error = exc
        finally:
            if backend is not None:
                backend.shutdown()

    recorded = {int(record["episode"]) for record in records}
    if fatal_error is not None:
        records.extend(
            {"episode": episode_id, "status": "not_run_after_fatal_cuda"}
            for episode_id in episode_ids
            if episode_id not in recorded
        )
    summary = {
        "format_version": "robotwin_sam_batch_summary_v1",
        "run_id": run_id,
        "annotation_mode": _annotation_mode(config).value,
        "required_object_roles": list(
            annotation_spec(_annotation_mode(config)).required_role_names
        ),
        "requested_episode_ids": list(episode_ids),
        "resident_sam3": True,
        "records": records,
        "fatal_error": None if fatal_error is None else str(fatal_error),
    }
    summary_path = store.write_json(store.run_dir(run_id) / "sam_batch_summary.json", summary)
    _print({"stage": "sam_batch", "artifact": str(summary_path), **summary})
    if fatal_error is not None:
        raise SystemExit(3)
    if any(record["status"] in {"failed", "incomplete"} for record in records):
        raise SystemExit(4)


def run_gripper_batch(
    config: PipelineConfig,
    episode_ids: tuple[int, ...],
    run_id: str,
    *,
    force: bool = False,
    backend_factory: Callable[..., Sam3Adapter] | None = None,
    episode_runner: Callable[
        [PipelineConfig, int, str, Sam3Adapter], GripperEpisodeExecution
    ]
    | None = None,
) -> None:
    """Run gripper serially while keeping one SAM3 predictor resident."""

    if not episode_ids:
        raise ValueError("gripper-batch requires at least one episode")
    factory = Sam3Adapter if backend_factory is None else backend_factory
    runner = _execute_gripper_episode if episode_runner is None else episode_runner
    store = ArtifactStore(config.output_root)
    records: list[dict[str, Any]] = []
    pending: list[int] = []
    for episode_id in episode_ids:
        ref = _episode_ref(config, episode_id)
        if not force and _gripper_episode_complete(config, store, run_id, ref):
            records.append({"episode": episode_id, "status": "skipped_complete"})
        else:
            pending.append(episode_id)

    fatal_error: BaseException | None = None
    backend: Sam3Adapter | None = None
    if pending:
        try:
            backend = factory(
                checkpoint_path=config.sam3.checkpoint,
                gpus=config.sam3.gpus,
            )
            for episode_id in pending:
                ref = _episode_ref(config, episode_id)
                try:
                    execution = runner(config, episode_id, run_id, backend)
                except SAM_EXECUTION_ERRORS as exc:
                    failure_path = store.write_json(
                        store.episode_dir(run_id, ref) / "gripper_failure.json",
                        {
                            "format_version": "robotwin_gripper_failure_v1",
                            "error": str(exc),
                        },
                    )
                    records.append(
                        {
                            "episode": episode_id,
                            "status": "failed",
                            "artifact": str(failure_path),
                            "error": str(exc),
                        }
                    )
                    if _fatal_cuda_error(exc):
                        fatal_error = exc
                        break
                    continue
                complete = _emit_gripper_result(run_id, execution)
                records.append(
                    {
                        "episode": episode_id,
                        "status": "completed" if complete else "incomplete",
                        "artifact": str(
                            Path(execution.mask_run.artifact_dir) / "run_manifest.json"
                        ),
                    }
                )
        except SAM_EXECUTION_ERRORS as exc:
            fatal_error = exc
        finally:
            if backend is not None:
                backend.shutdown()

    recorded = {int(record["episode"]) for record in records}
    if fatal_error is not None:
        records.extend(
            {"episode": episode_id, "status": "not_run_after_fatal_cuda"}
            for episode_id in episode_ids
            if episode_id not in recorded
        )
    summary = {
        "format_version": "robotwin_gripper_batch_summary_v1",
        "run_id": run_id,
        "annotation_mode": _annotation_mode(config).value,
        "required_object_roles": list(
            annotation_spec(_annotation_mode(config)).required_role_names
        ),
        "requested_episode_ids": list(episode_ids),
        "resident_sam3": True,
        "records": records,
        "fatal_error": None if fatal_error is None else str(fatal_error),
    }
    summary_path = store.write_json(
        store.run_dir(run_id) / "gripper_batch_summary.json",
        summary,
    )
    _print({"stage": "gripper_batch", "artifact": str(summary_path), **summary})
    if fatal_error is not None:
        raise SystemExit(5)
    if any(record["status"] in {"failed", "incomplete"} for record in records):
        raise SystemExit(6)


def run_pipeline(config: PipelineConfig, episode_index: int, run_id: str | None) -> None:
    selected_run_id = run_id or ArtifactStore.new_run_id()
    run_qwen(config, episode_index, selected_run_id)
    run_sam(config, episode_index, selected_run_id)
    run_gripper(config, episode_index, selected_run_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in (
        "preflight",
        "loop",
        "qwen",
        "sam",
        "sam-batch",
        "gripper",
        "gripper-batch",
        "run",
    ):
        subparser = subparsers.add_parser(command)
        subparser.add_argument(
            "--config",
            type=Path,
            default=Path("configs/pilot_move_pillbottle_pad.yaml"),
        )
        if command in {"loop", "qwen", "sam", "gripper", "run"}:
            subparser.add_argument("--episode", type=int, required=True)
            subparser.add_argument("--run-id", required=command in {"sam", "gripper"})
        elif command in {"sam-batch", "gripper-batch"}:
            subparser.add_argument("--run-id", required=True)
            subparser.add_argument("--episode-ids", type=int, nargs="*")
            subparser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.command == "preflight":
        run_preflight(config)
    elif args.command == "loop":
        run_loop(config, args.episode, args.run_id)
    elif args.command == "qwen":
        run_qwen(config, args.episode, args.run_id)
    elif args.command == "sam":
        run_sam(config, args.episode, args.run_id)
    elif args.command == "sam-batch":
        run_sam_batch(
            config,
            tuple(args.episode_ids or config.dataset.regression_episode_ids),
            args.run_id,
            force=args.force,
        )
    elif args.command == "gripper":
        run_gripper(config, args.episode, args.run_id)
    elif args.command == "gripper-batch":
        run_gripper_batch(
            config,
            tuple(args.episode_ids or config.dataset.regression_episode_ids),
            args.run_id,
            force=args.force,
        )
    elif args.command == "run":
        run_pipeline(config, args.episode, args.run_id)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
