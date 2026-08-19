"""SAM gripper stage orchestration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Protocol, cast

import cv2
import numpy as np
from PIL import Image

from ....config import GripperRoiConfig
from ....domain import AnnotationMode, ObjectRole
from ....models.loop_context import LoopContext
from ....models.mask_qc import MaskQCStatus
from ....models.timeline import FrameWindow, PickPlaceEvents
from .candidates import (
    GripperSeedCandidate,
    GripperSeedQualityGateConfig,
    apply_gripper_seed_quality_gate,
    build_gripper_seed_candidate,
    mark_same_frame_duplicates,
)
from .composition import compose_gripper_track
from .geometry import (
    DEFAULT_GRIPPER_ROI_GEOMETRY,
    GripperRoiGeometry,
    ProjectedGripperRoi,
    normalized_roi_box,
    project_gripper_roi,
)
from .qc import (
    GripperQwenClient,
    GripperSeedQCResult,
    render_gripper_candidate_panel,
    run_gripper_seed_qc,
)

NDArray = np.ndarray[Any, Any]


@dataclass(frozen=True)
class GripperStageResult:
    """Stage-level gripper output consumed by artifact persistence."""

    active_arm: str
    active_window: FrameWindow
    frame_count: int
    frame_shape: tuple[int, int]
    seed_frame_id: int | None
    selected_candidate: str | None
    seed_mask: NDArray | None
    native_track: NDArray
    roi_track: NDArray
    candidate_track: NDArray
    gripper_track: NDArray
    removed_track: NDArray
    target_removed_track: NDArray
    receiver_removed_track: NDArray
    prompt_rois: dict[int, ProjectedGripperRoi]
    hard_rois: dict[int, ProjectedGripperRoi]
    qc_result: GripperSeedQCResult
    candidate_panels: dict[str, Image.Image]
    roi_policy: dict[str, Any]
    provenance: dict[str, Any]
    failure: str | None = None

    def __post_init__(self) -> None:
        if self.active_arm not in {"left", "right"}:
            raise ValueError("active_arm must be left or right")
        stacks = (
            self.native_track,
            self.roi_track,
            self.candidate_track,
            self.gripper_track,
            self.removed_track,
            self.target_removed_track,
            self.receiver_removed_track,
        )
        expected = (self.frame_count, *self.frame_shape)
        if any(np.asarray(stack).shape != expected for stack in stacks):
            raise ValueError(f"gripper tracks must have shape {expected}")
        if (self.qc_result.status is MaskQCStatus.PASSED) != (
            self.seed_frame_id is not None and self.seed_mask is not None
        ):
            raise ValueError("passed gripper stage must carry the selected seed mask")
        if self.seed_mask is not None and np.asarray(self.seed_mask).shape != self.frame_shape:
            raise ValueError("gripper seed mask must match frame_shape")

    @property
    def instance_name(self) -> str:
        return f"gripper_{self.active_arm}"

    @property
    def nonempty_frame_ids(self) -> tuple[int, ...]:
        present = self.gripper_track.reshape(self.gripper_track.shape[0], -1).any(axis=1)
        return tuple(int(value) for value in np.flatnonzero(present))

    @property
    def status(self) -> str:
        if self.failure is not None:
            return "failed"
        if self.qc_result.status is MaskQCStatus.PASSED and self.gripper_track.any():
            return "ok"
        return "failed"


class GripperStageError(RuntimeError):
    """The gripper stage cannot execute its pose/seed/propagation contract."""


class GripperSamBackend(Protocol):
    def box_mask(
        self,
        resource_path: Path,
        box_xyxy: Sequence[float],
        *,
        frame_id: int,
        frame_count: int,
        frame_shape: tuple[int, int],
    ) -> NDArray: ...

    def text_box_mask(
        self,
        resource_path: Path,
        text: str,
        box_xyxy: Sequence[float],
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


def gripper_keyframes(
    rois: Mapping[int, ProjectedGripperRoi],
    events: PickPlaceEvents,
    *,
    frame_shape: tuple[int, int],
) -> tuple[int, ...]:
    """Return the same seven state-derived review frames used by the experiment."""

    height, width = frame_shape
    visible = [
        frame_id
        for frame_id, roi in sorted(rois.items())
        if 0 <= roi.tcp_pixel_xy[0] < width and 0 <= roi.tcp_pixel_xy[1] < height
    ]
    first_tcp = visible[0] if visible else events.t_close_start
    values = (
        first_tcp,
        max(first_tcp, events.t_close_start - 1),
        events.t_close_done,
        (events.t_close_done + events.t_open_start) // 2,
        max(events.t_close_done + 1, events.t_open_start - 1),
        events.t_open_start,
        events.t_open_done,
    )
    return tuple(dict.fromkeys(int(value) for value in values))


def _load_state_arrays(context: LoopContext) -> tuple[NDArray, NDArray]:
    """Load the state columns needed for pose ROI projection."""

    try:
        import pandas as pd

        frame = pd.read_parquet(
            Path(context.state_source),
            columns=["frame_index", "episode_index", "observation.state"],
        )
    except Exception as exc:
        raise GripperStageError(
            f"failed to read gripper state source {context.state_source}: {exc}"
        ) from exc
    if frame.empty:
        raise GripperStageError("gripper state source is empty")
    frame_indices: NDArray = frame["frame_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(frame_indices, np.arange(len(frame_indices))):
        raise GripperStageError("gripper state frame_index must be contiguous and zero-based")
    episode_indices: NDArray = frame["episode_index"].to_numpy(dtype=np.int64)
    if not np.all(episode_indices == context.episode.episode_index):
        raise GripperStageError("gripper state episode_index does not match the context")
    state: NDArray = np.stack(frame["observation.state"].to_numpy()).astype(np.float64)
    if state.shape != (context.frame_count, 14):
        raise GripperStageError(
            f"gripper state must have shape {(context.frame_count, 14)}, got {state.shape}"
        )
    if not np.isfinite(state).all():
        raise GripperStageError("gripper state contains non-finite values")
    gripper_states: NDArray = state[:, (6, 13)]
    eef_states: NDArray = np.stack((state[:, 0:6], state[:, 7:13]), axis=1)
    return eef_states, gripper_states


def _roi_geometries(
    config: GripperRoiConfig,
) -> tuple[GripperRoiGeometry, GripperRoiGeometry]:
    width_overrides = {
        "closed_half_width_m": config.fixed_half_width_m,
        "open_half_width_m": config.fixed_half_width_m,
    }
    prompt = replace(
        DEFAULT_GRIPPER_ROI_GEOMETRY,
        axial_back_m=config.prompt_axial_back_m,
        axial_front_m=config.prompt_axial_front_m,
        **width_overrides,
    )
    hard = replace(
        DEFAULT_GRIPPER_ROI_GEOMETRY,
        axial_back_m=config.hard_axial_back_m,
        axial_front_m=config.hard_axial_front_m,
        **width_overrides,
    )
    return prompt, hard


def _roi_policy(config: GripperRoiConfig) -> dict[str, Any]:
    prompt, hard = _roi_geometries(config)
    return {
        "prompt": {
            "geometry": asdict(prompt),
            "usage": "SAM text-box/box-only candidate and selected-seed crop",
        },
        "hard": {
            "geometry": asdict(hard),
            "usage": "propagated native track crop before known-object exclusion",
        },
        "legacy_roi_track_alias": "hard_roi_track",
    }


def _polygon_mask(roi: ProjectedGripperRoi, frame_shape: tuple[int, int]) -> NDArray:
    mask = np.zeros(frame_shape, dtype=np.uint8)
    polygon = np.rint(roi.hull_pixels_xy).astype(np.int32)
    if len(polygon) >= 3:
        cv2.fillConvexPoly(mask, polygon, 1)
    return mask.astype(bool)


def _build_roi_track(
    eef_states: NDArray,
    gripper_states: NDArray,
    events: PickPlaceEvents,
    *,
    frame_shape: tuple[int, int],
    geometry: GripperRoiGeometry,
) -> tuple[NDArray, dict[int, ProjectedGripperRoi]]:
    eef = np.asarray(eef_states, dtype=np.float64)
    gripper = np.asarray(gripper_states, dtype=np.float64)
    if eef.ndim != 3 or eef.shape[1:] != (2, 6):
        raise GripperStageError(f"eef_states must have [T,2,6] shape, got {eef.shape}")
    if gripper.shape != (eef.shape[0], 2):
        raise GripperStageError(
            f"gripper_states must have shape {(eef.shape[0], 2)}, got {gripper.shape}"
        )
    height, width = frame_shape
    if height < 1 or width < 1:
        raise GripperStageError(f"invalid frame_shape: {frame_shape}")
    arm_index = 0 if events.active_arm == "left" else 1
    track = np.zeros((eef.shape[0], height, width), dtype=bool)
    rois: dict[int, ProjectedGripperRoi] = {}
    for frame_id in range(events.t_move_start, events.t_open_done + 1):
        try:
            roi = project_gripper_roi(
                eef[frame_id, arm_index],
                gripper[frame_id, arm_index],
                geometry=geometry,
            )
        except (IndexError, ValueError) as exc:
            raise GripperStageError(
                f"failed to project gripper ROI at frame {frame_id}: {exc}"
            ) from exc
        track[frame_id] = _polygon_mask(roi, frame_shape)
        rois[frame_id] = roi
    return track, rois


def _load_resource_image(resource_path: Path, frame_id: int) -> Image.Image:
    image_path = resource_path / f"{frame_id:06d}.jpg"
    if not image_path.is_file():
        raise GripperStageError(f"decoded SAM3 frame is missing: {image_path}")
    try:
        with Image.open(image_path) as image:
            return image.convert("RGB")
    except OSError as exc:
        raise GripperStageError(f"failed to read decoded frame {image_path}: {exc}") from exc


def _context_frame_ids(
    keyframes: Sequence[int],
    events: PickPlaceEvents,
    frame_count: int,
) -> tuple[int, ...]:
    values = (
        *keyframes,
        events.t_close_done,
        (events.t_close_done + events.t_open_start) // 2,
        events.t_open_done,
    )
    return tuple(
        dict.fromkeys(int(value) for value in values if 0 <= int(value) < frame_count)
    )


def _build_gripper_candidates(
    backend: GripperSamBackend,
    resource_path: Path,
    *,
    frame_images: Mapping[int, Image.Image],
    keyframes: Sequence[int],
    prompt_rois: Mapping[int, ProjectedGripperRoi],
    prompt_roi_track: NDArray,
    target_track: NDArray,
    receiver_track: NDArray,
    events: PickPlaceEvents,
    frame_count: int,
    frame_shape: tuple[int, int],
    gripper_text: str,
    gate: GripperSeedQualityGateConfig,
) -> tuple[GripperSeedCandidate, ...]:
    def build_bank(
        prompt_mode: str,
        prompt_text: str | None,
        *,
        first_index: int,
    ) -> list[GripperSeedCandidate]:
        bank: list[GripperSeedCandidate] = []
        for frame_id in keyframes:
            roi = prompt_rois.get(frame_id)
            if roi is None:
                continue
            box = normalized_roi_box(roi, frame_shape)
            if box is None:
                continue
            if prompt_text is None:
                raw = backend.box_mask(
                    resource_path,
                    box,
                    frame_id=frame_id,
                    frame_count=frame_count,
                    frame_shape=frame_shape,
                )
            else:
                raw = backend.text_box_mask(
                    resource_path,
                    prompt_text,
                    box,
                    frame_id=frame_id,
                    frame_count=frame_count,
                    frame_shape=frame_shape,
                )
            candidate = build_gripper_seed_candidate(
                candidate_id=chr(ord("A") + first_index + len(bank)),
                frame_id=frame_id,
                events=events,
                prompt_mode=prompt_mode,
                prompt_text=prompt_text,
                raw_mask=raw,
                roi_mask=prompt_roi_track[frame_id],
                target_mask=target_track[frame_id],
                receiver_mask=receiver_track[frame_id],
                rgb=np.asarray(frame_images[frame_id], dtype=np.uint8),
                tcp_pixel_xy=roi.tcp_pixel_xy,
                minimum_pixels=gate.minimum_pixels,
            )
            bank.append(
                apply_gripper_seed_quality_gate(
                    candidate,
                    minimum_dark_fraction=gate.minimum_dark_fraction,
                    maximum_components=gate.maximum_components,
                    minimum_largest_component_fraction=(
                        gate.minimum_largest_component_fraction
                    ),
                    maximum_tcp_distance_px=gate.maximum_tcp_distance_px,
                )
            )
        return bank

    text_candidates = build_bank("text_box", gripper_text, first_index=0)
    candidates = list(text_candidates)
    if not any(candidate.basic_valid for candidate in text_candidates):
        candidates.extend(
            build_bank(
                "box_only",
                None,
                first_index=len(text_candidates),
            )
        )
    return mark_same_frame_duplicates(
        candidates,
        iou_threshold=gate.duplicate_iou_threshold,
    )


def _track_summary(track: NDArray, window: FrameWindow) -> dict[str, Any]:
    value = np.asarray(track, dtype=bool)[window.start : window.end + 1]
    areas = value.reshape(value.shape[0], -1).sum(axis=1)
    present = areas > 0
    return {
        "window_inclusive": window.to_json(),
        "window_frames": len(value),
        "nonempty_frames": int(present.sum()),
        "coverage": float(present.mean()),
        "pixels_min_nonempty": None if not present.any() else int(areas[present].min()),
        "pixels_median_nonempty": (
            None if not present.any() else float(np.median(areas[present]))
        ),
        "pixels_max": int(areas.max()) if areas.size else 0,
    }


def run_gripper_stage(
    context: LoopContext,
    *,
    backend: GripperSamBackend,
    resource_path: Path,
    frame_shape: tuple[int, int],
    gripper_roi_config: GripperRoiConfig,
    object_tracks: Mapping[ObjectRole, NDArray],
    qc_client: GripperQwenClient,
    qc_prompt_template: Path,
    qc_max_tokens: int = 220,
    qc_max_attempts: int = 2,
    qc_min_confidence: float = 0.70,
    seed_quality_gate: GripperSeedQualityGateConfig | None = None,
    gripper_text: str = "black robot gripper",
) -> GripperStageResult:
    """Run pose-ROI candidate selection and one native gripper propagation."""

    if context.annotation_mode is AnnotationMode.TARGET_ONLY:
        raise GripperStageError("target_only does not support the SAM gripper backend; use URDF")
    events = cast(PickPlaceEvents, context.events)
    gate = seed_quality_gate or GripperSeedQualityGateConfig()
    expected_shape = (context.frame_count, *frame_shape)
    expected_roles = context.annotation_spec.required_object_roles
    if set(object_tracks) != set(expected_roles):
        raise GripperStageError(
            "object_tracks roles must exactly match annotation mode: "
            f"expected={[role.value for role in expected_roles]}, "
            f"actual={[role.value for role in object_tracks]}"
        )
    normalized_tracks = {
        role: np.asarray(track, dtype=bool) for role, track in object_tracks.items()
    }
    if any(track.shape != expected_shape for track in normalized_tracks.values()):
        raise GripperStageError(f"every object track must have shape {expected_shape}")
    target = normalized_tracks[ObjectRole.TARGET]
    receiver = normalized_tracks.get(
        ObjectRole.RECEIVER,
        np.zeros(expected_shape, dtype=bool),
    )
    if not resource_path.is_dir():
        raise GripperStageError(f"SAM3 resource directory does not exist: {resource_path}")

    eef_states, gripper_states = _load_state_arrays(context)
    prompt_geometry, hard_geometry = _roi_geometries(gripper_roi_config)
    prompt_track, prompt_rois = _build_roi_track(
        eef_states,
        gripper_states,
        events,
        frame_shape=frame_shape,
        geometry=prompt_geometry,
    )
    hard_track, hard_rois = _build_roi_track(
        eef_states,
        gripper_states,
        events,
        frame_shape=frame_shape,
        geometry=hard_geometry,
    )
    keyframes = gripper_keyframes(prompt_rois, events, frame_shape=frame_shape)
    context_ids = _context_frame_ids(keyframes, events, context.frame_count)
    frame_images = {
        frame_id: _load_resource_image(resource_path, frame_id) for frame_id in context_ids
    }
    candidates = _build_gripper_candidates(
        backend,
        resource_path,
        frame_images=frame_images,
        keyframes=keyframes,
        prompt_rois=prompt_rois,
        prompt_roi_track=prompt_track,
        target_track=target,
        receiver_track=receiver,
        events=events,
        frame_count=context.frame_count,
        frame_shape=frame_shape,
        gripper_text=gripper_text,
        gate=gate,
    )
    panels = {
        candidate.candidate_id: render_gripper_candidate_panel(
            frame_images[candidate.frame_id],
            candidate,
            prompt_rois[candidate.frame_id],
        )
        for candidate in candidates
    }
    context_images = {
        frame_id: frame_images[frame_id]
        for frame_id in _context_frame_ids(keyframes, events, context.frame_count)
    }
    qc = run_gripper_seed_qc(
        context,
        candidates,
        panels,
        context_images,
        prompt_template_path=qc_prompt_template,
        client=qc_client,
        max_tokens=qc_max_tokens,
        max_attempts=qc_max_attempts,
        minimum_confidence=qc_min_confidence,
    )
    active_window = FrameWindow(events.t_move_start, events.t_open_done)
    zeros = np.zeros(expected_shape, dtype=bool)
    policy = _roi_policy(gripper_roi_config)
    provenance = {
        "keyframes": list(keyframes),
        "context_frame_ids": list(context_images),
        "candidate_count": len(candidates),
        "known_object_tracks": "saved_sam_native_track",
        "target_track": _track_summary(target, active_window),
        "receiver_track": _track_summary(receiver, active_window),
        "quality_gate": asdict(gate),
    }
    selected = qc.selected
    if qc.status is not MaskQCStatus.PASSED or selected is None:
        return GripperStageResult(
            active_arm=events.active_arm,
            active_window=active_window,
            frame_count=context.frame_count,
            frame_shape=frame_shape,
            seed_frame_id=None,
            selected_candidate=None,
            seed_mask=None,
            native_track=zeros,
            roi_track=hard_track,
            candidate_track=zeros,
            gripper_track=zeros,
            removed_track=zeros,
            target_removed_track=zeros,
            receiver_removed_track=zeros,
            prompt_rois=dict(prompt_rois),
            hard_rois=dict(hard_rois),
            qc_result=qc,
            candidate_panels=panels,
            roi_policy=policy,
            provenance=provenance,
            failure=f"gripper_seed_qc_{qc.status.value}:{qc.reason}",
        )
    if not selected.clean_mask.any():
        return GripperStageResult(
            active_arm=events.active_arm,
            active_window=active_window,
            frame_count=context.frame_count,
            frame_shape=frame_shape,
            seed_frame_id=selected.frame_id,
            selected_candidate=selected.candidate_id,
            seed_mask=selected.clean_mask,
            native_track=zeros,
            roi_track=hard_track,
            candidate_track=zeros,
            gripper_track=zeros,
            removed_track=zeros,
            target_removed_track=zeros,
            receiver_removed_track=zeros,
            prompt_rois=dict(prompt_rois),
            hard_rois=dict(hard_rois),
            qc_result=qc,
            candidate_panels=panels,
            roi_policy=policy,
            provenance=provenance,
            failure="selected_gripper_seed_is_empty",
        )

    native = np.asarray(
        backend.propagate_mask(
            resource_path,
            selected.clean_mask,
            seed_frame=selected.frame_id,
            frame_count=context.frame_count,
            frame_shape=frame_shape,
            tracking_window=(active_window.start, active_window.end),
        ),
        dtype=bool,
    )
    if native.shape != expected_shape:
        raise GripperStageError(
            f"gripper native track has shape {native.shape}, expected {expected_shape}"
        )
    composed = compose_gripper_track(
        native,
        hard_track,
        target,
        receiver,
        active_window=(active_window.start, active_window.end),
    )
    failure = None if composed.gripper_mask.any() else "gripper_track_empty_after_constraints"
    provenance.update(
        {
            "selected_candidate": selected.candidate_id,
            "seed_frame_id": selected.frame_id,
            "native_track": _track_summary(native, active_window),
            "final_track": _track_summary(composed.gripper_mask, active_window),
        }
    )
    return GripperStageResult(
        active_arm=events.active_arm,
        active_window=active_window,
        frame_count=context.frame_count,
        frame_shape=frame_shape,
        seed_frame_id=selected.frame_id,
        selected_candidate=selected.candidate_id,
        seed_mask=selected.clean_mask,
        native_track=native,
        roi_track=composed.roi_mask,
        candidate_track=composed.candidate_mask,
        gripper_track=composed.gripper_mask,
        removed_track=composed.removed_mask,
        target_removed_track=composed.target_removed,
        receiver_removed_track=composed.receiver_removed,
        prompt_rois=dict(prompt_rois),
        hard_rois=dict(hard_rois),
        qc_result=qc,
        candidate_panels=panels,
        roi_policy=policy,
        provenance=provenance,
        failure=failure,
    )


__all__ = [
    "GripperSamBackend",
    "GripperStageError",
    "GripperStageResult",
    "gripper_keyframes",
    "run_gripper_stage",
]
