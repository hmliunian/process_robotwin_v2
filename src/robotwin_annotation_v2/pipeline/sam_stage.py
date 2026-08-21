"""Stage 3: one SAM3 text seed followed by native video propagation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

from ..config import MaskConfig
from ..models import (
    FrameWindow,
    LoopContext,
    MaskQCResult,
    MaskQCStatus,
    MaskStatus,
    RoleMaskQC,
    RoleSemanticPlan,
    SemanticPlan,
    SemanticStatus,
)
from .object_mask.temporal_qc import (
    TemporalMaskQc,
    compose_visible_mask,
    evaluate_temporal_mask,
)

NDArray = np.ndarray[Any, Any]

INSTANCE_NAMES = ("target_0", "receiver_0", "gripper_left", "gripper_right")


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
    ) -> NDArray: ...

    def propagate_mask(
        self,
        resource_path: Path,
        seed_mask: NDArray,
        *,
        seed_frame: int,
        frame_count: int,
        frame_shape: tuple[int, int],
        tracking_window: tuple[int, int],
        object_id: int = 1,
    ) -> NDArray: ...


@dataclass(frozen=True)
class RoleMaskData:
    role: Literal["target", "receiver"]
    status: MaskStatus
    seed_frame_id: int | None
    primary_query: str | None
    output_window: FrameWindow
    seed_mask: NDArray | None
    canonical_envelope: NDArray | None
    native_track: NDArray
    visible_mask: NDArray
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
    def masks(self) -> NDArray:
        output = np.zeros(
            (len(INSTANCE_NAMES), self.frame_count, *self.frame_shape),
            dtype=bool,
        )
        channel_index = {"target": 0, "receiver": 1}
        for data in self.role_masks:
            output[channel_index[data.role]] = data.visible_mask
        return output


def dilate_envelope(mask: NDArray, padding: int) -> NDArray:
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


def _empty_role(
    role: Literal["target", "receiver"],
    *,
    window: FrameWindow,
    frame_count: int,
    frame_shape: tuple[int, int],
    seed_frame_id: int | None,
    primary_query: str | None,
    failure: str,
    seed_mask: NDArray | None = None,
    envelope: NDArray | None = None,
    native: NDArray | None = None,
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
    qc_seed_mask: NDArray | None = None,
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
        if qc_report.selected_seed_frame_id is not None:
            seed_frame = qc_report.selected_seed_frame_id
    else:
        query = semantic.primary_query
    if seed_frame is None or query is None:
        raise SamStageError(f"{role} semantic plan has no usable seed/query")
    if seed_frame not in context.seed_candidates(role):
        raise SamStageError(f"{role} QC selected an ineligible seed frame {seed_frame}")
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
