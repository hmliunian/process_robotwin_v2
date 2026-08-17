"""Stage 3: one SAM3 text seed followed by native video propagation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Protocol

import numpy as np
from PIL import Image

from ..adapters.artifact_store import ArtifactStore
from ..config import MaskConfig
from ..mask_schema import (
    FRAME_ENCODING_LEGEND,
    MASK_FORMAT_VERSION,
    FrameEncoding,
    build_frame_encoding,
    target_hold_window,
)
from ..models import (
    FrameWindow,
    LoopContext,
    MaskQCResult,
    MaskQCStatus,
    MaskRun,
    MaskStatus,
    RoleMaskQC,
    RoleMaskResult,
    RoleSemanticPlan,
    SemanticPlan,
    SemanticStatus,
)
from .gripper_stage import GripperStageResult

INSTANCE_NAMES = ("target_0", "receiver_0", "gripper_left", "gripper_right")
ROLES = ("target", "receiver", "gripper", "gripper")


class SamStageError(RuntimeError):
    """Stage 3 cannot execute the declared SAM3 contract."""


class SamBackend(Protocol):
    def text_mask(
        self,
        resource_path: Path,
        text: str,
        *,
        frame_id: int,
        frame_count: int,
        frame_shape: tuple[int, int],
    ) -> np.ndarray: ...

    def propagate_mask(
        self,
        resource_path: Path,
        seed_mask: np.ndarray,
        *,
        seed_frame: int,
        frame_count: int,
        frame_shape: tuple[int, int],
        tracking_window: tuple[int, int],
        object_id: int = 1,
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class TemporalMaskQc:
    window: FrameWindow
    status: Literal["pass", "review", "quarantine"]
    window_frames: int
    nonempty_frames: int
    coverage: float
    presence_transitions: int
    internal_missing_frames: int
    adjacent_iou_mean: float | None
    adjacent_iou_p05: float | None
    centroid_jump_p95_px: float | None
    area_ratio_jump_p95: float | None
    max_reference_centroid_distance_px: float | None
    issues: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "format_version": "robotwin_temporal_mask_qc_v1",
            "window": self.window.to_json(),
            "status": self.status,
            "window_frames": self.window_frames,
            "nonempty_frames": self.nonempty_frames,
            "coverage": self.coverage,
            "presence_transitions": self.presence_transitions,
            "internal_missing_frames": self.internal_missing_frames,
            "adjacent_iou_mean": self.adjacent_iou_mean,
            "adjacent_iou_p05": self.adjacent_iou_p05,
            "centroid_jump_p95_px": self.centroid_jump_p95_px,
            "area_ratio_jump_p95": self.area_ratio_jump_p95,
            "max_reference_centroid_distance_px": (
                self.max_reference_centroid_distance_px
            ),
            "issues": list(self.issues),
        }


@dataclass(frozen=True)
class RoleMaskData:
    role: Literal["target", "receiver"]
    status: MaskStatus
    seed_frame_id: int | None
    primary_query: str | None
    output_window: FrameWindow
    seed_mask: np.ndarray | None
    canonical_envelope: np.ndarray | None
    native_track: np.ndarray
    visible_mask: np.ndarray
    temporal_qc: TemporalMaskQc | None
    failure: str | None
    qc_status: MaskQCStatus = MaskQCStatus.NOT_RUN
    qc_selected_candidate: str | None = None
    qc_reason: str | None = None

    @property
    def nonempty_frame_ids(self) -> tuple[int, ...]:
        present = self.visible_mask.reshape(self.visible_mask.shape[0], -1).any(axis=1)
        return tuple(int(value) for value in np.flatnonzero(present))


@dataclass(frozen=True)
class SamStageResult:
    frame_count: int
    frame_shape: tuple[int, int]
    role_masks: tuple[RoleMaskData, ...]

    def __post_init__(self) -> None:
        roles = tuple(data.role for data in self.role_masks)
        if not roles or roles[0] != "target" or len(set(roles)) != len(roles):
            raise ValueError("SamStageResult requires unique roles beginning with target")

    def for_role(self, role: Literal["target", "receiver"]) -> RoleMaskData:
        for data in self.role_masks:
            if data.role == role:
                return data
        raise KeyError(f"SAM result has no mask for non-applicable role {role!r}")

    @property
    def target(self) -> RoleMaskData:
        return self.for_role("target")

    @property
    def receiver(self) -> RoleMaskData:
        return self.for_role("receiver")

    @property
    def masks(self) -> np.ndarray:
        output = np.zeros(
            (len(INSTANCE_NAMES), self.frame_count, *self.frame_shape),
            dtype=bool,
        )
        channel_index = {"target": 0, "receiver": 1}
        for data in self.role_masks:
            output[channel_index[data.role]] = data.visible_mask
        return output


def dilate_envelope(mask: np.ndarray, padding: int) -> np.ndarray:
    """Dilate one seed mask without adding an image-processing dependency."""

    value = np.asarray(mask, dtype=bool)
    if value.ndim != 2:
        raise ValueError("canonical envelope seed must be a 2-D mask")
    if padding < 0:
        raise ValueError("canonical envelope padding must be non-negative")
    if padding == 0:
        return value.copy()
    height, width = value.shape
    padded = np.pad(value, padding)
    envelope = np.zeros_like(value)
    radius_squared = padding * padding
    for row_offset in range(-padding, padding + 1):
        for column_offset in range(-padding, padding + 1):
            if row_offset * row_offset + column_offset * column_offset > radius_squared:
                continue
            row_start = padding + row_offset
            column_start = padding + column_offset
            envelope |= padded[
                row_start : row_start + height,
                column_start : column_start + width,
            ]
    return envelope


def compose_visible_mask(
    native_track: np.ndarray,
    output_window: FrameWindow,
) -> np.ndarray:
    """Crop a native identity track to the role's inclusive output window."""

    native = np.asarray(native_track, dtype=bool)
    if native.ndim != 3:
        raise ValueError("native mask must have [T,H,W] shape")
    if output_window.end >= native.shape[0]:
        raise ValueError("output window extends beyond the mask stack")
    visible = native.copy()
    visible[: output_window.start] = False
    visible[output_window.end + 1 :] = False
    return visible


def _centroid(mask: np.ndarray) -> np.ndarray | None:
    rows, columns = np.nonzero(mask)
    if not rows.size:
        return None
    return np.asarray([columns.mean(), rows.mean()], dtype=np.float64)


def _p95(values: list[float]) -> float | None:
    return None if not values else float(np.quantile(values, 0.95))


def evaluate_temporal_mask(
    mask_stack: np.ndarray,
    output_window: FrameWindow,
    mask_config: MaskConfig,
    *,
    reference_mask: np.ndarray | None = None,
) -> TemporalMaskQc:
    """Measure continuity and quarantine tracks with several severe jump signals."""

    masks = np.asarray(mask_stack, dtype=bool)
    if masks.ndim != 3:
        raise ValueError("temporal mask must have [T,H,W] shape")
    if output_window.end >= masks.shape[0]:
        raise ValueError("output window extends beyond the temporal mask")
    if reference_mask is not None and np.asarray(reference_mask).shape != masks.shape[1:]:
        raise ValueError("reference mask must match the temporal mask frame shape")

    window = masks[output_window.start : output_window.end + 1]
    flattened = window.reshape(window.shape[0], -1)
    present = flattened.any(axis=1)
    nonempty_frames = int(present.sum())
    presence_transitions = int(np.count_nonzero(present[1:] != present[:-1]))
    present_indices = np.flatnonzero(present)
    internal_missing_frames = (
        0
        if present_indices.size < 2
        else int((~present[present_indices[0] : present_indices[-1] + 1]).sum())
    )

    adjacent_ious: list[float] = []
    centroid_jumps: list[float] = []
    area_ratio_jumps: list[float] = []
    centroids: list[np.ndarray] = []
    previous_centroid: np.ndarray | None = None
    previous_area: int | None = None
    for frame in window:
        area = int(frame.sum())
        centroid = _centroid(frame)
        if centroid is not None:
            centroids.append(centroid)
        if previous_centroid is not None and centroid is not None:
            centroid_jumps.append(float(np.linalg.norm(centroid - previous_centroid)))
        if previous_area is not None and previous_area > 0 and area > 0:
            area_ratio_jumps.append(abs(area - previous_area) / previous_area)
        previous_centroid = centroid
        previous_area = area
    for left, right in zip(window, window[1:], strict=False):
        if not left.any() or not right.any():
            continue
        union = int((left | right).sum())
        adjacent_ious.append(int((left & right).sum()) / union)

    adjacent_iou_mean = None if not adjacent_ious else float(np.mean(adjacent_ious))
    adjacent_iou_p05 = (
        None if not adjacent_ious else float(np.quantile(adjacent_ious, 0.05))
    )
    centroid_jump_p95_px = _p95(centroid_jumps)
    area_ratio_jump_p95 = _p95(area_ratio_jumps)
    reference_centroid = (
        None if reference_mask is None else _centroid(np.asarray(reference_mask, dtype=bool))
    )
    max_reference_distance = (
        None
        if reference_centroid is None or not centroids
        else max(float(np.linalg.norm(value - reference_centroid)) for value in centroids)
    )

    severe_signals: list[str] = []
    if (
        adjacent_iou_p05 is not None
        and adjacent_iou_p05 < mask_config.temporal_qc_min_adjacent_iou_p05
    ):
        severe_signals.append("low_adjacent_iou_p05")
    if (
        centroid_jump_p95_px is not None
        and centroid_jump_p95_px > mask_config.temporal_qc_max_centroid_jump_p95_px
    ):
        severe_signals.append("large_centroid_jump_p95")
    if (
        area_ratio_jump_p95 is not None
        and area_ratio_jump_p95 > mask_config.temporal_qc_max_area_ratio_jump_p95
    ):
        severe_signals.append("large_area_ratio_jump_p95")

    issues = list(severe_signals)
    if nonempty_frames < len(window):
        issues.append("incomplete_window_coverage")
    if internal_missing_frames:
        issues.append("internal_missing_frames")
    if len(severe_signals) >= mask_config.temporal_qc_quarantine_signal_count:
        status: Literal["pass", "review", "quarantine"] = "quarantine"
    elif issues:
        status = "review"
    else:
        status = "pass"
    return TemporalMaskQc(
        window=output_window,
        status=status,
        window_frames=len(window),
        nonempty_frames=nonempty_frames,
        coverage=float(present.mean()),
        presence_transitions=presence_transitions,
        internal_missing_frames=internal_missing_frames,
        adjacent_iou_mean=adjacent_iou_mean,
        adjacent_iou_p05=adjacent_iou_p05,
        centroid_jump_p95_px=centroid_jump_p95_px,
        area_ratio_jump_p95=area_ratio_jump_p95,
        max_reference_centroid_distance_px=max_reference_distance,
        issues=tuple(issues),
    )


def _window_coverage(
    mask_stack: np.ndarray,
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


def _empty_role(
    role: Literal["target", "receiver"],
    *,
    window: FrameWindow,
    frame_count: int,
    frame_shape: tuple[int, int],
    seed_frame_id: int | None,
    primary_query: str | None,
    failure: str,
    seed_mask: np.ndarray | None = None,
    envelope: np.ndarray | None = None,
    native: np.ndarray | None = None,
    qc_report: RoleMaskQC | None = None,
) -> RoleMaskData:
    empty = np.zeros((frame_count, *frame_shape), dtype=bool)
    return RoleMaskData(
        role=role,
        status=MaskStatus.FAILED,
        seed_frame_id=seed_frame_id,
        primary_query=primary_query,
        output_window=window,
        seed_mask=seed_mask,
        canonical_envelope=envelope,
        native_track=empty if native is None else native,
        visible_mask=empty,
        temporal_qc=None,
        failure=failure,
        qc_status=(MaskQCStatus.NOT_RUN if qc_report is None else qc_report.status),
        qc_selected_candidate=(
            None if qc_report is None else qc_report.selected_candidate
        ),
        qc_reason=None if qc_report is None else qc_report.reason,
    )


def _run_role(
    role: Literal["target", "receiver"],
    *,
    semantic: RoleSemanticPlan,
    output_window: FrameWindow,
    temporal_qc_window: FrameWindow,
    padding: int,
    mask_config: MaskConfig,
    context: LoopContext,
    backend: SamBackend,
    resource_path: Path,
    frame_shape: tuple[int, int],
    qc_report: RoleMaskQC | None = None,
    qc_seed_mask: np.ndarray | None = None,
) -> RoleMaskData:
    if not (
        output_window.start <= temporal_qc_window.start
        and temporal_qc_window.end <= output_window.end
    ):
        raise SamStageError(
            f"{role} temporal QC window must be within its output window"
        )
    if semantic.status is SemanticStatus.NO_CLEAR_SEED:
        return _empty_role(
            role,
            window=output_window,
            frame_count=context.frame_count,
            frame_shape=frame_shape,
            seed_frame_id=None,
            primary_query=None,
            failure="semantic_plan_no_clear_seed",
            qc_report=qc_report,
        )
    seed_frame = semantic.seed_frame_id
    if qc_report is not None:
        if qc_report.role != role:
            raise SamStageError(
                f"{role} received mismatched mask QC report for {qc_report.role}"
            )
        if qc_report.status is not MaskQCStatus.PASSED:
            return _empty_role(
                role,
                window=output_window,
                frame_count=context.frame_count,
                frame_shape=frame_shape,
                seed_frame_id=semantic.seed_frame_id,
                primary_query=None,
                failure=f"mask_qc_{qc_report.status.value}",
                qc_report=qc_report,
            )
        query = qc_report.selected_query
    else:
        query = semantic.primary_query
    if seed_frame is None or query is None:
        raise SamStageError(f"{role} semantic plan has no usable seed/query")
    if seed_frame > output_window.end:
        raise SamStageError(f"{role} seed occurs after its output window")

    if qc_seed_mask is None:
        seed_mask = backend.text_mask(
            resource_path,
            query,
            frame_id=seed_frame,
            frame_count=context.frame_count,
            frame_shape=frame_shape,
        ).astype(bool, copy=False)
    else:
        seed_mask = np.asarray(qc_seed_mask, dtype=bool)
    if seed_mask.shape != frame_shape:
        raise SamStageError(f"{role} seed mask has shape {seed_mask.shape}")
    envelope = dilate_envelope(seed_mask, padding)
    if not seed_mask.any():
        return _empty_role(
            role,
            window=output_window,
            frame_count=context.frame_count,
            frame_shape=frame_shape,
            seed_frame_id=seed_frame,
            primary_query=query,
            seed_mask=seed_mask,
            envelope=envelope,
            failure="empty_text_seed",
            qc_report=qc_report,
        )

    native = backend.propagate_mask(
        resource_path,
        seed_mask,
        seed_frame=seed_frame,
        frame_count=context.frame_count,
        frame_shape=frame_shape,
        tracking_window=(min(seed_frame, output_window.start), output_window.end),
    ).astype(bool, copy=False)
    expected_shape = (context.frame_count, *frame_shape)
    if native.shape != expected_shape:
        raise SamStageError(f"{role} native track has shape {native.shape}")

    visible = compose_visible_mask(native, output_window)
    temporal_qc = evaluate_temporal_mask(
        visible,
        temporal_qc_window,
        mask_config,
        reference_mask=seed_mask,
    )

    native_window = native[
        temporal_qc_window.start : temporal_qc_window.end + 1
    ]
    if not native_window.any():
        failure = "native_track_empty_in_output_window"
        status = MaskStatus.FAILED
    elif temporal_qc.status == "quarantine":
        failure = "temporal_qc_quarantine:" + ",".join(temporal_qc.issues)
        status = MaskStatus.QUARANTINED
        visible[:] = False
    else:
        failure = None
        status = MaskStatus.OK
    return RoleMaskData(
        role=role,
        status=status,
        seed_frame_id=seed_frame,
        primary_query=query,
        output_window=output_window,
        seed_mask=seed_mask,
        canonical_envelope=envelope,
        native_track=native,
        visible_mask=visible,
        temporal_qc=temporal_qc,
        failure=failure,
        qc_status=(MaskQCStatus.NOT_RUN if qc_report is None else qc_report.status),
        qc_selected_candidate=(
            None if qc_report is None else qc_report.selected_candidate
        ),
        qc_reason=None if qc_report is None else qc_report.reason,
    )


def run_sam_stage(
    context: LoopContext,
    semantic_plan: SemanticPlan,
    backend: SamBackend,
    resource_path: Path,
    *,
    frame_shape: tuple[int, int],
    mask_config: MaskConfig,
    mask_qc: MaskQCResult | None = None,
) -> SamStageResult:
    """Execute Stage 3 using either the semantic primary or a QC-approved query."""

    if semantic_plan.episode != context.episode:
        raise SamStageError("SemanticPlan and LoopContext refer to different episodes")
    if semantic_plan.annotation_mode is not context.annotation_mode:
        raise SamStageError("SemanticPlan and LoopContext use different annotation modes")
    if mask_qc is not None and tuple(
        report.role for report in mask_qc.role_reports
    ) != tuple(plan.role for plan in semantic_plan.role_plans):
        raise SamStageError("MaskQCResult roles do not match SemanticPlan")

    role_masks: list[RoleMaskData] = []
    padding_by_role = {
        "target": mask_config.target_envelope_padding_px,
        "receiver": mask_config.receiver_envelope_padding_px,
    }
    for semantic in semantic_plan.role_plans:
        role = semantic.role
        output_window = (
            context.windows.target
            if role == "target"
            else context.windows.receiver
        )
        if output_window is None:
            raise SamStageError(
                f"{role} has no output window in {context.annotation_mode.value} mode"
            )
        role_masks.append(
            _run_role(
                role,
                semantic=semantic,
                output_window=output_window,
                temporal_qc_window=(
                    context.events.target_window
                    if role == "target"
                    else output_window
                ),
                padding=padding_by_role[role],
                mask_config=mask_config,
                context=context,
                backend=backend,
                resource_path=resource_path,
                frame_shape=frame_shape,
                qc_report=None if mask_qc is None else mask_qc.for_role(role),
                qc_seed_mask=(
                    None if mask_qc is None else mask_qc.selected_masks.get(role)
                ),
            )
        )
    return SamStageResult(
        frame_count=context.frame_count,
        frame_shape=frame_shape,
        role_masks=tuple(role_masks),
    )


def save_sam_artifacts(
    store: ArtifactStore,
    run_id: str,
    context: LoopContext,
    semantic_plan: SemanticPlan,
    result: SamStageResult,
    *,
    seed_images: Mapping[int, Image.Image],
    gripper_result: GripperStageResult | None = None,
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
                role=role,  # type: ignore[arg-type]
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
    masks_path = store.write_npz(
        episode_dir / "masks.npz",
        format_version=np.asarray(MASK_FORMAT_VERSION),
        frame_count=np.asarray(result.frame_count, dtype=np.int64),
        masks=masks,
        instance_names=np.asarray(INSTANCE_NAMES),
        roles=np.asarray(ROLES),
        annotation_status=annotation_statuses,
        qc_status=qc_status,
        frame_encoding=frame_encoding,
    )
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
                "seed": "sam3_text_only_primary_query",
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
                "candidate_mask_qc": any(
                    data.qc_status is not MaskQCStatus.NOT_RUN for data in role_data
                ),
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
