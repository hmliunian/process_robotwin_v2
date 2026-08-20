"""Artifact publication for SAM object and optional gripper mask results."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from ..adapters.artifact_store import ArtifactStore
from ..adapters.canonical_masks import build_canonical_mask_bundle
from ..adapters.canonical_publication import CanonicalMaskPublisher
from ..mask_schema import (
    FRAME_ENCODING_LEGEND,
    MASK_FORMAT_VERSION,
    FrameEncoding,
    build_frame_encoding,
    target_hold_window,
)
from ..models import (
    LoopContext,
    MaskQCStatus,
    MaskRun,
    MaskStatus,
    RoleMaskResult,
    SemanticPlan,
)
from ..pipeline.sam_stage import RoleMaskData, SamStageError, SamStageResult

if TYPE_CHECKING:
    from ..pipeline.gripper.sam.annotator import GripperStageResult


_CANONICAL_MASK_PUBLISHER = CanonicalMaskPublisher()


def _window_coverage(
    mask_stack: np.ndarray[Any, Any],
    window: tuple[int, int],
) -> dict[str, Any]:
    """Summarize report-only presence for one inclusive frame interval."""

    masks = np.asarray(mask_stack, dtype=bool)
    if masks.ndim != 3:
        raise ValueError("coverage mask must have [T,H,W] shape")
    start, end = window
    if start < 0 or end < start or end >= masks.shape[0]:
        raise ValueError(f"coverage window is outside the mask stack: {window}")
    present = masks[start : end + 1].reshape(end - start + 1, -1).any(axis=1)
    return {
        "window": [start, end],
        "window_frames": int(present.size),
        "nonempty_frames": int(present.sum()),
        "coverage": float(present.mean()),
    }


def save_sam_artifacts(
    store: ArtifactStore,
    run_id: str,
    context: LoopContext,
    semantic_plan: SemanticPlan,
    result: SamStageResult,
    *,
    seed_images: Mapping[int, Image.Image],
    gripper_result: GripperStageResult | None = None,
    canonical_mask_publisher: CanonicalMaskPublisher | None = None,
) -> MaskRun:
    """Persist Stage-3 diagnostics, compatible masks, and provenance."""

    episode_dir = store.episode_dir(run_id, context.episode)
    hold = target_hold_window(
        context.events,
        frame_count=context.frame_count,
    )
    target_hold_coverage = (
        None
        if hold is None
        else _window_coverage(result.target.visible_mask, hold)
    )
    role_results: list[RoleMaskResult] = []
    role_data = result.role_masks
    for data in role_data:
        role_name = f"{data.role}_0"
        role_dir = episode_dir / role_name
        seed_rgb_path: str | None = None
        seed_mask_path: str | None = None
        envelope_path: str | None = None
        native_path: str | None = None
        temporal_qc_path: str | None = None
        if data.seed_frame_id is not None and data.seed_mask is not None:
            seed_image = seed_images.get(data.seed_frame_id)
            if seed_image is None:
                raise SamStageError(
                    f"missing seed RGB frame {data.seed_frame_id} for {data.role}"
                )
            seed_rgb_file = store.write_png(
                role_dir / "seed.rgb.png",
                np.asarray(seed_image.convert("RGB")),
                rgb=True,
            )
            seed_rgb_path = str(seed_rgb_file.relative_to(episode_dir))
            seed_mask_file = store.write_png(role_dir / "seed.mask.png", data.seed_mask)
            seed_mask_path = str(seed_mask_file.relative_to(episode_dir))
            if data.canonical_envelope is not None:
                envelope_file = store.write_png(
                    role_dir / "canonical_envelope.png",
                    data.canonical_envelope,
                )
                envelope_path = str(envelope_file.relative_to(episode_dir))
            native_file = store.write_npz(
                role_dir / "native_track.npz",
                masks=data.native_track,
            )
            native_path = str(native_file.relative_to(episode_dir))
            if data.temporal_qc is not None:
                temporal_qc_payload = data.temporal_qc.to_json()
                if data.role == "target" and target_hold_coverage is not None:
                    temporal_qc_payload["target_hold_coverage"] = (
                        target_hold_coverage
                    )
                temporal_qc_file = store.write_json(
                    role_dir / "temporal_qc.json",
                    temporal_qc_payload,
                )
                temporal_qc_path = str(temporal_qc_file.relative_to(episode_dir))
        role_results.append(
            RoleMaskResult(
                role=data.role,
                status=data.status,
                seed_frame_id=data.seed_frame_id,
                primary_query=data.primary_query,
                output_window=data.output_window,
                seed_rgb_path=seed_rgb_path,
                seed_mask_path=seed_mask_path,
                canonical_envelope_path=envelope_path,
                native_track_path=native_path,
                temporal_qc_path=temporal_qc_path,
                nonempty_frames=len(data.nonempty_frame_ids),
                failure=data.failure,
                qc_status=data.qc_status,
                qc_selected_candidate=data.qc_selected_candidate,
                qc_reason=data.qc_reason,
            )
        )

    applicable = {data.role for data in role_data}
    for role in ("target", "receiver"):
        if role in applicable:
            continue
        role_results.append(
            RoleMaskResult(
                role=role,
                status=MaskStatus.NOT_APPLICABLE,
                seed_frame_id=None,
                primary_query=None,
                output_window=None,
                seed_rgb_path=None,
                seed_mask_path=None,
                canonical_envelope_path=None,
                native_track_path=None,
                temporal_qc_path=None,
                nonempty_frames=0,
                qc_status=MaskQCStatus.NOT_APPLICABLE,
            )
        )
    role_order = {"target": 0, "receiver": 1}
    role_results.sort(key=lambda item: role_order.get(item.role, 2))

    gripper_role_name: str | None = None
    gripper_seed_mask_path: str | None = None
    gripper_native_path: str | None = None
    gripper_candidate_path: str | None = None
    gripper_seed_qc_path: str | None = None
    gripper_panel_paths: dict[str, str] = {}
    if gripper_result is not None:
        gripper_role_name = gripper_result.instance_name
        gripper_dir = episode_dir / gripper_role_name
        if gripper_result.seed_mask is not None:
            seed_mask_file = store.write_png(
                gripper_dir / "seed.mask.png",
                gripper_result.seed_mask,
            )
            gripper_seed_mask_path = str(seed_mask_file.relative_to(episode_dir))
        native_file = store.write_npz(
            gripper_dir / "native_track.npz",
            masks=gripper_result.native_track,
        )
        gripper_native_path = str(native_file.relative_to(episode_dir))
        candidate_file = store.write_npz(
            gripper_dir / "diagnostics.npz",
            roi_track=gripper_result.roi_track,
            candidate_track=gripper_result.candidate_track,
            removed_track=gripper_result.removed_track,
            target_removed_track=gripper_result.target_removed_track,
            receiver_removed_track=gripper_result.receiver_removed_track,
        )
        gripper_candidate_path = str(candidate_file.relative_to(episode_dir))
        seed_qc_file = store.write_json(
            gripper_dir / "gripper_seed_qc.json",
            gripper_result.qc_result.to_json(),
        )
        gripper_seed_qc_path = str(seed_qc_file.relative_to(episode_dir))
        for candidate_id, panel in gripper_result.candidate_panels.items():
            panel_file = store.write_png(
                gripper_dir / "seed_candidates" / f"candidate_{candidate_id}.png",
                np.asarray(panel.convert("RGB")),
                rgb=True,
            )
            gripper_panel_paths[candidate_id] = str(panel_file.relative_to(episode_dir))
        gripper_status = (
            MaskStatus.OK if gripper_result.status == "ok" else MaskStatus.FAILED
        )
        role_results.append(
            RoleMaskResult(
                role=gripper_role_name,  # type: ignore[arg-type]
                status=gripper_status,
                seed_frame_id=gripper_result.seed_frame_id,
                primary_query="black robot gripper",
                output_window=gripper_result.active_window,
                seed_rgb_path=None,
                seed_mask_path=gripper_seed_mask_path,
                canonical_envelope_path=None,
                native_track_path=gripper_native_path,
                temporal_qc_path=None,
                nonempty_frames=len(gripper_result.nonempty_frame_ids),
                failure=(
                    None if gripper_status is MaskStatus.OK else gripper_result.failure
                ),
                qc_status=gripper_result.qc_result.status,
                qc_selected_candidate=gripper_result.selected_candidate,
                qc_reason=gripper_result.qc_result.reason,
            )
        )

    def annotation_status(data: RoleMaskData) -> str:
        if data.status is MaskStatus.OK:
            return "valid"
        if data.status is MaskStatus.QUARANTINED:
            return "quarantined"
        return "failed"

    masks = result.masks.copy()
    gripper_annotation = ["not_annotated", "not_annotated"]
    gripper_qc = [MaskQCStatus.NOT_RUN.value, MaskQCStatus.NOT_RUN.value]
    if gripper_result is not None:
        gripper_index = 2 if gripper_result.active_arm == "left" else 3
        masks[gripper_index] = gripper_result.gripper_track
        gripper_local = gripper_index - 2
        gripper_annotation[gripper_local] = (
            "valid" if gripper_result.status == "ok" else "failed"
        )
        gripper_qc[gripper_local] = gripper_result.qc_result.status.value

    annotation_statuses = np.asarray(
        [
            *(
                annotation_status(result.for_role(role))
                if role in applicable
                else "not_applicable"
                for role in ("target", "receiver")
            ),
            *gripper_annotation,
        ]
    )
    qc_status = np.asarray(
        [
            *(
                result.for_role(role).qc_status.value
                if role in applicable
                else "not_applicable"
                for role in ("target", "receiver")
            ),
            *gripper_qc,
        ]
    )
    frame_encoding = build_frame_encoding(masks, context.events)
    masks_path = episode_dir / "masks.npz"
    canonical_bundle = build_canonical_mask_bundle(
        masks_path,
        frame_count=result.frame_count,
        masks=masks,
        annotation_status=annotation_statuses,
        qc_status=qc_status,
        frame_encoding=frame_encoding,
    )
    publisher = canonical_mask_publisher or _CANONICAL_MASK_PUBLISHER
    masks_path = publisher.publish(masks_path, canonical_bundle)
    provenance_channels: dict[str, Any] = {
        "gripper_left": {"status": "not_annotated"},
        "gripper_right": {"status": "not_annotated"},
    }
    for role in ("target", "receiver"):
        channel_name = f"{role}_0"
        if role not in applicable:
            provenance_channels[channel_name] = {
                "status": "not_applicable",
                "qc_status": "not_applicable",
                "reason": f"{role} is not required in {context.annotation_mode.value} mode",
                "nonempty_frame_ids": [],
            }
            continue
        data = result.for_role(role)
        channel_provenance: dict[str, Any] = {
            "status": data.status.value,
            "seed_frame_id": data.seed_frame_id,
            "primary_query": data.primary_query,
            "failure": data.failure,
            "qc_status": data.qc_status.value,
            "qc_selected_candidate": data.qc_selected_candidate,
            "qc_reason": data.qc_reason,
            "output_window": data.output_window.to_json(),
            "nonempty_frame_ids": list(data.nonempty_frame_ids),
            "temporal_qc": (
                None if data.temporal_qc is None else data.temporal_qc.to_json()
            ),
        }
        if role == "target" and target_hold_coverage is not None:
            channel_provenance["target_hold_coverage"] = target_hold_coverage
        provenance_channels[channel_name] = channel_provenance
    encoding_metadata: dict[str, Any] = {
        "npz_key": "frame_encoding",
        "legend": FRAME_ENCODING_LEGEND,
        "target_hold_window": None if hold is None else list(hold),
    }
    provenance: dict[str, Any] = {
        "format_version": "robotwin_frame_provenance_v2",
        "annotation_mode": context.annotation_mode.value,
        "required_object_roles": list(context.annotation_spec.required_role_names),
        "gripper_backend": "sam",
        "composition": "native_track clipped_to role_output_window",
        "frame_encoding": encoding_metadata,
        "channels": provenance_channels,
    }
    if gripper_result is not None and gripper_role_name is not None:
        provenance["composition"] = (
            "target/receiver native_track clipped_to role_output_window; "
            "gripper native_track clipped_to hard pose ROI and known objects"
        )
        provenance["channels"][gripper_role_name] = {
            "status": gripper_result.status,
            "backend": "sam",
            "active_arm": gripper_result.active_arm,
            "seed_frame_id": gripper_result.seed_frame_id,
            "selected_candidate": gripper_result.selected_candidate,
            "failure": gripper_result.failure,
            "qc_status": gripper_result.qc_result.status.value,
            "qc_confidence": gripper_result.qc_result.confidence,
            "qc_reason": gripper_result.qc_result.reason,
            "forced_fallback": gripper_result.qc_result.forced_fallback,
            "active_window": gripper_result.active_window.to_json(),
            "nonempty_frame_ids": list(gripper_result.nonempty_frame_ids),
            "seed_mask_path": gripper_seed_mask_path,
            "native_track_path": gripper_native_path,
            "diagnostics_path": gripper_candidate_path,
            "seed_qc_path": gripper_seed_qc_path,
            "candidate_panels": gripper_panel_paths,
            "provenance": gripper_result.provenance,
        }
    provenance_path = store.write_json(episode_dir / "frame_provenance.json", provenance)
    mask_run = MaskRun(
        run_id=run_id,
        episode=context.episode.to_json(),
        frame_count=context.frame_count,
        roles=tuple(role_results),
        artifact_dir=str(episode_dir),
    )
    manifest = mask_run.to_json()
    candidate_mask_qc = any(
        data.qc_status is not MaskQCStatus.NOT_RUN for data in role_data
    )
    fallback_used = any(
        data.qc_status is MaskQCStatus.PASSED
        and (
            data.seed_frame_id != semantic_plan.for_role(data.role).seed_frame_id
            or data.primary_query != semantic_plan.for_role(data.role).primary_query
        )
        for data in role_data
    )
    if gripper_result is not None and gripper_role_name is not None:
        manifest["channels"][gripper_role_name] = (
            2 if gripper_result.active_arm == "left" else 3
        )
    manifest.update(
        {
            "annotation_mode": context.annotation_mode.value,
            "required_object_roles": list(context.annotation_spec.required_role_names),
            "gripper_backend": "sam",
            "mask_format_version": MASK_FORMAT_VERSION,
            "frame_encoding": encoding_metadata,
            "semantic_prompt_sha256": semantic_plan.prompt_sha256,
            "algorithm": {
                "seed": (
                    "sam3_mask_qc_selected_candidate"
                    if candidate_mask_qc
                    else "sam3_text_only_primary_query"
                ),
                "propagation": "sam3_native_mask_forward_backward",
                "visibility": "native_track clipped_to role_output_window",
                "target_hold_encoding": {
                    "code": FrameEncoding.TARGET_GRASP_HOLD.value,
                    "window": encoding_metadata["target_hold_window"],
                    "ends_before_open_start": True,
                },
                "per_frame_text_observation": False,
                "canonical_envelope_usage": "seed_diagnostic_only",
                "automatic_query_fallback": False,
                "mask_qc_fallback_used": fallback_used,
                "candidate_mask_qc": candidate_mask_qc,
                "gripper_stage": None
                if gripper_result is None
                else {
                    "backend": "sam",
                    "producer": "sam3_pose_roi_qwen_selected_candidate",
                    "seed": "pose_roi_text_box_qwen_selected_candidate",
                    "propagation": "sam3_native_mask_forward_backward",
                    "visibility": (
                        "native_track clipped_to hard_pose_roi and known_objects"
                    ),
                    "active_arm": gripper_result.active_arm,
                    "active_window": gripper_result.active_window.to_json(),
                    "qc_status": gripper_result.qc_result.status.value,
                    "selected_candidate": gripper_result.selected_candidate,
                    "forced_fallback": gripper_result.qc_result.forced_fallback,
                },
                "amodal_completion": False,
            },
            "roi_policy": None if gripper_result is None else gripper_result.roi_policy,
            "gripper_qc": None
            if gripper_result is None
            else {
                "backend": "sam",
                "status": gripper_result.status,
                "qc_status": gripper_result.qc_result.status.value,
                "active_arm": gripper_result.active_arm,
                "selected_candidate": gripper_result.selected_candidate,
                "confidence": gripper_result.qc_result.confidence,
                "reason": gripper_result.qc_result.reason,
                "forced_fallback": gripper_result.qc_result.forced_fallback,
                "nonempty_frames": len(gripper_result.nonempty_frame_ids),
                "quality": None,
            },
            "artifacts": {
                "masks": str(masks_path.relative_to(episode_dir)),
                "frame_provenance": str(provenance_path.relative_to(episode_dir)),
                **(
                    {}
                    if gripper_seed_qc_path is None
                    else {"gripper_seed_qc": gripper_seed_qc_path}
                ),
            },
        }
    )
    store.write_json(episode_dir / "run_manifest.json", manifest)
    return mask_run


__all__ = ["save_sam_artifacts"]
