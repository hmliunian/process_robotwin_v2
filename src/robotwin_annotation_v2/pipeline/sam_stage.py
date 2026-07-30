"""Stage 3: SAM3 text seed, native propagation, and visible composition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Protocol

import numpy as np
from PIL import Image

from ..adapters.artifact_store import ArtifactStore
from ..config import MaskConfig
from ..models import (
    FrameWindow,
    LoopContext,
    MaskRun,
    MaskStatus,
    RoleMaskResult,
    RoleSemanticPlan,
    SemanticPlan,
    SemanticStatus,
)


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

    def text_masks(
        self,
        resource_path: Path,
        text: str,
        *,
        frame_ids: tuple[int, ...],
        frame_count: int,
        frame_shape: tuple[int, int],
    ) -> dict[int, np.ndarray]: ...

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
class RoleMaskData:
    role: Literal["target", "receiver"]
    status: MaskStatus
    seed_frame_id: int | None
    primary_query: str | None
    output_window: FrameWindow
    seed_mask: np.ndarray | None
    canonical_envelope: np.ndarray | None
    native_track: np.ndarray
    text_observations: np.ndarray
    visible_mask: np.ndarray
    failure: str | None

    @property
    def nonempty_frame_ids(self) -> tuple[int, ...]:
        present = self.visible_mask.reshape(self.visible_mask.shape[0], -1).any(axis=1)
        return tuple(int(value) for value in np.flatnonzero(present))


@dataclass(frozen=True)
class SamStageResult:
    frame_count: int
    frame_shape: tuple[int, int]
    target: RoleMaskData
    receiver: RoleMaskData

    @property
    def masks(self) -> np.ndarray:
        output = np.zeros(
            (len(INSTANCE_NAMES), self.frame_count, *self.frame_shape),
            dtype=bool,
        )
        output[0] = self.target.visible_mask
        output[1] = self.receiver.visible_mask
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
    text_observations: np.ndarray,
    canonical_envelope: np.ndarray,
    output_window: FrameWindow,
) -> np.ndarray:
    """Keep pixels supported by identity, same-frame visibility, and seed geometry."""

    native = np.asarray(native_track, dtype=bool)
    observed = np.asarray(text_observations, dtype=bool)
    envelope = np.asarray(canonical_envelope, dtype=bool)
    if native.shape != observed.shape or native.ndim != 3:
        raise ValueError("native and text masks must have the same [T,H,W] shape")
    if envelope.shape != native.shape[1:]:
        raise ValueError("canonical envelope must match the video frame shape")
    if output_window.end >= native.shape[0]:
        raise ValueError("output window extends beyond the mask stack")
    visible = native & observed & envelope[None, :, :]
    visible[: output_window.start] = False
    visible[output_window.end + 1 :] = False
    return visible


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
    text: np.ndarray | None = None,
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
        text_observations=empty if text is None else text,
        visible_mask=empty,
        failure=failure,
    )


def _run_role(
    role: Literal["target", "receiver"],
    *,
    semantic: RoleSemanticPlan,
    output_window: FrameWindow,
    padding: int,
    context: LoopContext,
    backend: SamBackend,
    resource_path: Path,
    frame_shape: tuple[int, int],
) -> RoleMaskData:
    if semantic.status is SemanticStatus.NO_CLEAR_SEED:
        return _empty_role(
            role,
            window=output_window,
            frame_count=context.frame_count,
            frame_shape=frame_shape,
            seed_frame_id=None,
            primary_query=None,
            failure="semantic_plan_no_clear_seed",
        )
    seed_frame = semantic.seed_frame_id
    query = semantic.primary_query
    if seed_frame is None or query is None:
        raise SamStageError(f"{role} semantic plan has no usable seed/query")
    if seed_frame > output_window.end:
        raise SamStageError(f"{role} seed occurs after its output window")

    seed_mask = backend.text_mask(
        resource_path,
        query,
        frame_id=seed_frame,
        frame_count=context.frame_count,
        frame_shape=frame_shape,
    ).astype(bool, copy=False)
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

    frame_ids = tuple(range(output_window.start, output_window.end + 1))
    measurements = backend.text_masks(
        resource_path,
        query,
        frame_ids=frame_ids,
        frame_count=context.frame_count,
        frame_shape=frame_shape,
    )
    text = np.zeros(expected_shape, dtype=bool)
    for frame_id in frame_ids:
        measurement = np.asarray(
            measurements.get(frame_id, np.zeros(frame_shape, dtype=bool)),
            dtype=bool,
        )
        if measurement.shape != frame_shape:
            raise SamStageError(
                f"{role} text observation frame {frame_id} has shape {measurement.shape}"
            )
        text[frame_id] = measurement
    visible = compose_visible_mask(native, text, envelope, output_window)

    native_window = native[output_window.start : output_window.end + 1]
    text_window = text[output_window.start : output_window.end + 1]
    if not native_window.any():
        failure = "native_track_empty_in_output_window"
    elif not text_window.any():
        failure = "same_frame_text_empty_in_output_window"
    elif not visible.any():
        failure = "visible_intersection_empty"
    else:
        failure = None
    return RoleMaskData(
        role=role,
        status=MaskStatus.OK if failure is None else MaskStatus.FAILED,
        seed_frame_id=seed_frame,
        primary_query=query,
        output_window=output_window,
        seed_mask=seed_mask,
        canonical_envelope=envelope,
        native_track=native,
        text_observations=text,
        visible_mask=visible,
        failure=failure,
    )


def run_sam_stage(
    context: LoopContext,
    semantic_plan: SemanticPlan,
    backend: SamBackend,
    resource_path: Path,
    *,
    frame_shape: tuple[int, int],
    mask_config: MaskConfig,
) -> SamStageResult:
    """Execute Stage 3 for target then receiver using only the primary query."""

    if semantic_plan.episode != context.episode:
        raise SamStageError("SemanticPlan and LoopContext refer to different episodes")
    target = _run_role(
        "target",
        semantic=semantic_plan.target,
        output_window=context.events.target_window,
        padding=mask_config.target_envelope_padding_px,
        context=context,
        backend=backend,
        resource_path=resource_path,
        frame_shape=frame_shape,
    )
    receiver = _run_role(
        "receiver",
        semantic=semantic_plan.receiver,
        output_window=context.events.receiver_window,
        padding=mask_config.receiver_envelope_padding_px,
        context=context,
        backend=backend,
        resource_path=resource_path,
        frame_shape=frame_shape,
    )
    return SamStageResult(
        frame_count=context.frame_count,
        frame_shape=frame_shape,
        target=target,
        receiver=receiver,
    )


def save_sam_artifacts(
    store: ArtifactStore,
    run_id: str,
    context: LoopContext,
    semantic_plan: SemanticPlan,
    result: SamStageResult,
    *,
    seed_images: Mapping[int, Image.Image],
) -> MaskRun:
    """Persist Stage-3 diagnostics, compatible masks, and provenance."""

    episode_dir = store.episode_dir(run_id, context.episode)
    role_results: list[RoleMaskResult] = []
    role_data = (result.target, result.receiver)
    for index, data in enumerate(role_data):
        role_name = INSTANCE_NAMES[index]
        role_dir = episode_dir / role_name
        seed_rgb_path: str | None = None
        seed_mask_path: str | None = None
        envelope_path: str | None = None
        native_path: str | None = None
        text_path: str | None = None
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
            text_file = store.write_npz(
                role_dir / "text_observations.npz",
                masks=data.text_observations,
            )
            native_path = str(native_file.relative_to(episode_dir))
            text_path = str(text_file.relative_to(episode_dir))
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
                text_observation_path=text_path,
                nonempty_frames=len(data.nonempty_frame_ids),
                failure=data.failure,
            )
        )

    annotation_status = np.asarray(
        [
            "valid" if result.target.status is MaskStatus.OK else "failed",
            "valid" if result.receiver.status is MaskStatus.OK else "failed",
            "not_annotated",
            "not_annotated",
        ]
    )
    masks_path = store.write_npz(
        episode_dir / "masks.npz",
        format_version=np.asarray("robotwin_visible_masks_v1"),
        frame_count=np.asarray(result.frame_count, dtype=np.int64),
        masks=result.masks,
        instance_names=np.asarray(INSTANCE_NAMES),
        roles=np.asarray(ROLES),
        annotation_status=annotation_status,
    )
    provenance = {
        "format_version": "robotwin_frame_provenance_v1",
        "composition": "native_track & same_frame_text & canonical_envelope",
        "channels": {
            "target_0": {
                "status": result.target.status.value,
                "seed_frame_id": result.target.seed_frame_id,
                "primary_query": result.target.primary_query,
                "failure": result.target.failure,
                "output_window": result.target.output_window.to_json(),
                "nonempty_frame_ids": list(result.target.nonempty_frame_ids),
            },
            "receiver_0": {
                "status": result.receiver.status.value,
                "seed_frame_id": result.receiver.seed_frame_id,
                "primary_query": result.receiver.primary_query,
                "failure": result.receiver.failure,
                "output_window": result.receiver.output_window.to_json(),
                "nonempty_frame_ids": list(result.receiver.nonempty_frame_ids),
            },
            "gripper_left": {"status": "not_annotated"},
            "gripper_right": {"status": "not_annotated"},
        },
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
    manifest.update(
        {
            "semantic_prompt_sha256": semantic_plan.prompt_sha256,
            "algorithm": {
                "seed": "sam3_text_only_primary_query",
                "propagation": "sam3_native_mask_forward_backward",
                "visibility": "native_track & same_frame_text & canonical_envelope",
                "automatic_query_fallback": False,
                "amodal_completion": False,
            },
            "artifacts": {
                "masks": str(masks_path.relative_to(episode_dir)),
                "frame_provenance": str(provenance_path.relative_to(episode_dir)),
            },
        }
    )
    store.write_json(episode_dir / "run_manifest.json", manifest)
    return mask_run
