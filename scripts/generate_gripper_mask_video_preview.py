#!/usr/bin/env python3
"""Propagate one reviewed gripper seed and render a full-trajectory preview."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import cv2
import numpy as np
from PIL import Image

from robotwin_annotation_v2.adapters import (
    RoboTwinDataset,
    Sam3Adapter,
    sam3_video_resource,
)
from robotwin_annotation_v2.config import load_config
from robotwin_annotation_v2.pipeline import (
    DEFAULT_GRIPPER_ROI_GEOMETRY,
    GripperRoiGeometry,
    GripperTrackResult,
    ProjectedGripperRoi,
    compose_gripper_track,
    project_gripper_roi,
)
from robotwin_annotation_v2.models import EpisodeRef, LoopEvents
from robotwin_annotation_v2.pipeline import build_loop_context


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CYAN = np.asarray([15, 230, 185], dtype=np.float32)
MAGENTA = np.asarray([235, 50, 190], dtype=np.float32)
ORANGE = np.asarray([255, 126, 35], dtype=np.float32)
BLUE = np.asarray([70, 105, 255], dtype=np.float32)
YELLOW_RGB = (255, 240, 68)
RED_RGB = (255, 59, 59)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "pilot_move_pillbottle_pad.yaml",
    )
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument("--seed-frame", type=int, required=True)
    parser.add_argument("--gripper-seed-archive", type=Path, required=True)
    parser.add_argument("--gripper-seed-key", required=True)
    parser.add_argument("--object-track-archive", type=Path, required=True)
    parser.add_argument("--target-track-key", required=True)
    parser.add_argument("--receiver-track-key", required=True)
    parser.add_argument("--object-track-provenance", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sam3-gpu", type=int, default=0)
    parser.add_argument("--temp-root", type=Path, default=Path("/tmp"))
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="fast")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_array(path: Path, key: str, *, dimensions: int) -> np.ndarray:
    source = path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"mask archive does not exist: {source}")
    with np.load(source, allow_pickle=False) as archive:
        if key not in archive.files:
            raise KeyError(f"mask archive has no key {key!r}: {source}")
        value = np.asarray(archive[key], dtype=bool)
    if value.ndim != dimensions:
        raise ValueError(f"{key} must be {dimensions}-D, got {value.shape}")
    return value


def _polygon_mask(
    roi: ProjectedGripperRoi,
    frame_shape: tuple[int, int],
) -> np.ndarray:
    mask = np.zeros(frame_shape, dtype=np.uint8)
    polygon = np.rint(roi.hull_pixels_xy).astype(np.int32)
    if len(polygon) >= 3:
        cv2.fillConvexPoly(mask, polygon, 1)
    return mask.astype(bool)


def _build_roi_track(
    eef_states: np.ndarray,
    gripper_states: np.ndarray,
    events: LoopEvents,
    *,
    frame_shape: tuple[int, int],
    geometry: GripperRoiGeometry = DEFAULT_GRIPPER_ROI_GEOMETRY,
) -> tuple[np.ndarray, tuple[ProjectedGripperRoi | None, ...]]:
    """Project one configurable ROI geometry across the active action window."""

    frame_count = eef_states.shape[0]
    arm_index = 0 if events.active_arm == "left" else 1
    masks = np.zeros((frame_count, *frame_shape), dtype=bool)
    rois: list[ProjectedGripperRoi | None] = [None] * frame_count
    for frame_id in range(events.t_move_start, events.t_open_done + 1):
        roi = project_gripper_roi(
            eef_states[frame_id, arm_index],
            gripper_states[frame_id, arm_index],
            geometry=geometry,
        )
        masks[frame_id] = _polygon_mask(roi, frame_shape)
        rois[frame_id] = roi
    return masks, tuple(rois)


def _phase(frame_id: int, events: LoopEvents) -> str:
    if frame_id < events.t_move_start or frame_id > events.t_open_done:
        return "inactive"
    if frame_id < events.t_close_start:
        return "approach"
    if frame_id <= events.t_close_done:
        return "close"
    if frame_id < events.t_open_start:
        return "transport"
    return "release"


def _blend(frame: np.ndarray, mask: np.ndarray, color: np.ndarray, alpha: float) -> None:
    if mask.any():
        frame[mask] = frame[mask] * (1.0 - alpha) + color * alpha


def _label(frame: np.ndarray, text: str) -> None:
    cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, 18), (12, 12, 12), -1)
    cv2.putText(
        frame,
        text,
        (4, 13),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _draw_roi(frame: np.ndarray, roi: ProjectedGripperRoi | None) -> None:
    if roi is None:
        return
    polygon = np.rint(roi.hull_pixels_xy).astype(np.int32).reshape(-1, 1, 2)
    if len(polygon) >= 3:
        cv2.polylines(frame, [polygon], True, YELLOW_RGB, 2, cv2.LINE_AA)
    x, y = np.rint(roi.tcp_pixel_xy).astype(int)
    cv2.line(frame, (x - 4, y), (x + 4, y), RED_RGB, 2, cv2.LINE_AA)
    cv2.line(frame, (x, y - 4), (x, y + 4), RED_RGB, 2, cv2.LINE_AA)


def _review_frame(
    rgb: np.ndarray,
    *,
    episode_id: int,
    frame_id: int,
    events: LoopEvents,
    roi: ProjectedGripperRoi | None,
    native_mask: np.ndarray,
    result: GripperTrackResult,
) -> np.ndarray:
    source = np.asarray(rgb, dtype=np.uint8)
    raw_panel = source.copy()
    _label(
        raw_panel,
        f"RGB ep{episode_id} f{frame_id} {_phase(frame_id, events)}",
    )

    candidate_panel = source.astype(np.float32)
    _blend(
        candidate_panel,
        native_mask & ~result.candidate_mask[frame_id],
        MAGENTA,
        0.48,
    )
    _blend(candidate_panel, result.candidate_mask[frame_id], CYAN, 0.60)
    candidate_panel = np.clip(candidate_panel, 0, 255).astype(np.uint8)
    _draw_roi(candidate_panel, roi)
    _label(
        candidate_panel,
        f"native={int(native_mask.sum())} roi-crop={int(result.candidate_mask[frame_id].sum())}",
    )

    final_panel = source.astype(np.float32)
    _blend(final_panel, result.target_removed[frame_id], ORANGE, 0.64)
    _blend(final_panel, result.receiver_removed[frame_id], BLUE, 0.64)
    _blend(final_panel, result.gripper_mask[frame_id], CYAN, 0.68)
    final_panel = np.clip(final_panel, 0, 255).astype(np.uint8)
    _draw_roi(final_panel, roi)
    _label(
        final_panel,
        f"gripper={int(result.gripper_mask[frame_id].sum())} "
        f"-obj={int(result.removed_mask[frame_id].sum())}",
    )

    binary_panel = np.zeros_like(source)
    binary_panel[result.gripper_mask[frame_id]] = 255
    _label(binary_panel, "binary visible gripper mask")
    return np.concatenate((raw_panel, candidate_panel, final_panel, binary_panel), axis=1)


def _stream_rate(stream: Any) -> Fraction:
    rate = stream.average_rate or stream.base_rate
    if rate is None or rate <= 0:
        raise ValueError("source video does not expose a positive frame rate")
    return Fraction(rate.numerator, rate.denominator)


def _sample_frames(start: int, end: int, count: int = 12) -> tuple[int, ...]:
    return tuple(
        sorted(
            set(
                np.linspace(start, end, num=min(count, end - start + 1))
                .round()
                .astype(int)
                .tolist()
            )
        )
    )


def _save_contact_sheet(
    panels: dict[int, Image.Image],
    path: Path,
) -> None:
    ordered = [panels[key] for key in sorted(panels)]
    if not ordered:
        raise ValueError("contact sheet has no panels")
    panel_size = (640, 120)
    rows = (len(ordered) + 1) // 2
    sheet = Image.new("RGB", (panel_size[0] * 2, panel_size[1] * rows), "#111111")
    for index, panel in enumerate(ordered):
        resized = panel.resize(panel_size, Image.Resampling.LANCZOS)
        sheet.paste(resized, ((index % 2) * panel_size[0], (index // 2) * panel_size[1]))
    sheet.save(path, format="JPEG", quality=82, optimize=True)


def _render_video(
    video_path: Path,
    output_path: Path,
    contact_sheet_path: Path,
    *,
    episode_id: int,
    frame_count: int,
    events: LoopEvents,
    rois: tuple[ProjectedGripperRoi | None, ...],
    native_track: np.ndarray,
    result: GripperTrackResult,
    crf: int,
    preset: str,
) -> dict[str, Any]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to encode review videos")
    if output_path.exists() or contact_sheet_path.exists():
        raise FileExistsError("review output already exists")
    temporary = output_path.with_name(f".{output_path.name}.tmp.mp4")
    sample_ids = set(_sample_frames(events.t_move_start, events.t_open_done))
    panels: dict[int, Image.Image] = {}
    dark_fractions: list[float] = []

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        frame_rate = _stream_rate(stream)
        height, width = native_track.shape[1:]
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{width * 4}x{height}",
            "-pix_fmt",
            "rgb24",
            "-r",
            f"{frame_rate.numerator}/{frame_rate.denominator}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(temporary),
        ]
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        if process.stdin is None or process.stderr is None:
            process.kill()
            raise RuntimeError("failed to open ffmpeg pipes")
        decoded = 0
        try:
            for frame_id, video_frame in enumerate(container.decode(stream)):
                if frame_id >= frame_count:
                    break
                rgb = video_frame.to_ndarray(format="rgb24")
                review = _review_frame(
                    rgb,
                    episode_id=episode_id,
                    frame_id=frame_id,
                    events=events,
                    roi=rois[frame_id],
                    native_mask=native_track[frame_id],
                    result=result,
                )
                process.stdin.write(review.tobytes())
                if frame_id in sample_ids:
                    panels[frame_id] = Image.fromarray(review)
                final_mask = result.gripper_mask[frame_id]
                if final_mask.any():
                    dark_fractions.append(
                        float((rgb[final_mask].max(axis=1) < 70).mean())
                    )
                decoded += 1
            process.stdin.close()
            stderr = process.stderr.read()
            return_code = process.wait()
            if return_code != 0:
                raise RuntimeError(stderr.decode(errors="replace").strip())
        except BaseException:
            process.kill()
            process.wait()
            temporary.unlink(missing_ok=True)
            raise
    if decoded != frame_count:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"decoded {decoded} usable frames, expected {frame_count}")
    temporary.replace(output_path)
    _save_contact_sheet(panels, contact_sheet_path)
    return {
        "frame_count": decoded,
        "frame_rate": f"{frame_rate.numerator}/{frame_rate.denominator}",
        "duration_seconds": float(Fraction(decoded, 1) / frame_rate),
        "review_width": width * 4,
        "review_height": height,
        "contact_sheet_frames": sorted(panels),
        "gripper_dark_fraction_median": (
            None if not dark_fractions else float(np.median(dark_fractions))
        ),
    }


def _track_summary(track: np.ndarray, start: int, end: int) -> dict[str, Any]:
    window = np.asarray(track[start : end + 1], dtype=bool)
    areas = window.reshape(window.shape[0], -1).sum(axis=1)
    present = areas > 0
    adjacent_ious: list[float] = []
    for left, right in zip(window, window[1:], strict=False):
        if not left.any() or not right.any():
            continue
        union = int((left | right).sum())
        adjacent_ious.append(int((left & right).sum()) / union)
    nonempty_areas = areas[present]
    return {
        "window_inclusive": [start, end],
        "window_frames": len(window),
        "nonempty_frames": int(present.sum()),
        "coverage": float(present.mean()),
        "pixels_min_nonempty": (
            None if not nonempty_areas.size else int(nonempty_areas.min())
        ),
        "pixels_median_nonempty": (
            None if not nonempty_areas.size else float(np.median(nonempty_areas))
        ),
        "pixels_max": int(areas.max()),
        "adjacent_iou_mean": (
            None if not adjacent_ious else float(np.mean(adjacent_ious))
        ),
        "adjacent_iou_p05": (
            None if not adjacent_ious else float(np.quantile(adjacent_ious, 0.05))
        ),
    }


def main() -> None:
    args = _parse_args()
    if not 0 <= args.crf <= 51:
        raise ValueError("--crf must be between 0 and 51")
    config = load_config(args.config)
    if args.episode_id not in config.dataset.regression_episode_ids:
        raise ValueError("--episode-id is outside the configured coverage20 set")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")

    dataset = RoboTwinDataset(
        config.dataset.root,
        task=config.dataset.task,
        camera=config.dataset.camera,
        manifest_path=config.dataset.manifest,
    )
    ref = EpisodeRef(config.dataset.task, args.episode_id, config.dataset.camera)
    state = dataset.load_state(ref)
    context = build_loop_context(dataset, ref)
    events = context.events
    frame_shape = tuple(int(value) for value in dataset.manifest["frame_shape_hw"])
    active_window = (events.t_move_start, events.t_open_done)
    if not active_window[0] <= args.seed_frame <= active_window[1]:
        raise ValueError("--seed-frame must be inside the active action window")

    seed_archive = args.gripper_seed_archive.expanduser().resolve()
    object_archive = args.object_track_archive.expanduser().resolve()
    original_seed = _load_array(seed_archive, args.gripper_seed_key, dimensions=2)
    target_track = _load_array(object_archive, args.target_track_key, dimensions=3)
    receiver_track = _load_array(object_archive, args.receiver_track_key, dimensions=3)
    expected_track_shape = (state.frame_count, *frame_shape)
    if original_seed.shape != frame_shape:
        raise ValueError(f"gripper seed has shape {original_seed.shape}, expected {frame_shape}")
    if target_track.shape != expected_track_shape or receiver_track.shape != expected_track_shape:
        raise ValueError(
            "known-object tracks must match the usable episode shape "
            f"{expected_track_shape}"
        )

    roi_track, rois = _build_roi_track(
        state.eef_states,
        state.gripper_states,
        events,
        frame_shape=frame_shape,
    )
    seed_mask = (
        original_seed
        & roi_track[args.seed_frame]
        & ~target_track[args.seed_frame]
        & ~receiver_track[args.seed_frame]
    )
    if not seed_mask.any():
        raise ValueError("reviewed gripper seed is empty after exact constraints")

    total_started = time.perf_counter()
    model_started = time.perf_counter()
    adapter = Sam3Adapter(
        checkpoint_path=config.sam3.checkpoint,
        gpus=(args.sam3_gpu,),
    )
    model_load_seconds = time.perf_counter() - model_started
    try:
        resource_started = time.perf_counter()
        with sam3_video_resource(
            state.paths.video,
            minimum_frame_count=(
                state.frame_count + int(dataset.manifest["raw_video_frame_surplus"])
            ),
            temp_root=args.temp_root,
        ) as resource:
            frame_extract_seconds = time.perf_counter() - resource_started
            propagation_started = time.perf_counter()
            native_track = adapter.propagate_mask(
                resource,
                seed_mask,
                seed_frame=args.seed_frame,
                frame_count=state.frame_count,
                frame_shape=frame_shape,
                tracking_window=active_window,
            )
            propagation_seconds = time.perf_counter() - propagation_started
    finally:
        adapter.shutdown()

    result = compose_gripper_track(
        native_track,
        roi_track,
        target_track,
        receiver_track,
        active_window=active_window,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    archive_path = output_dir / f"episode_{args.episode_id:06d}_gripper_masks.npz"
    np.savez_compressed(
        archive_path,
        format_version=np.asarray("robotwin_gripper_mask_video_preview_v1"),
        episode_index=np.asarray(args.episode_id, dtype=np.int64),
        seed_frame=np.asarray(args.seed_frame, dtype=np.int64),
        active_window=np.asarray(active_window, dtype=np.int64),
        seed_mask=seed_mask,
        native_track=native_track,
        roi_track=result.roi_mask,
        candidate_track=result.candidate_mask,
        target_track=target_track,
        receiver_track=receiver_track,
        target_removed=result.target_removed,
        receiver_removed=result.receiver_removed,
        gripper_track=result.gripper_mask,
    )

    video_path = output_dir / f"episode_{args.episode_id:06d}_gripper_review.mp4"
    contact_sheet_path = output_dir / f"episode_{args.episode_id:06d}_contact_sheet.jpg"
    render_started = time.perf_counter()
    render = _render_video(
        state.paths.video,
        video_path,
        contact_sheet_path,
        episode_id=args.episode_id,
        frame_count=state.frame_count,
        events=events,
        rois=rois,
        native_track=native_track,
        result=result,
        crf=args.crf,
        preset=args.preset,
    )
    render_seconds = time.perf_counter() - render_started
    window_frames = active_window[1] - active_window[0] + 1
    manifest = {
        "format": "robotwin_gripper_mask_video_preview_v1",
        "status": "review_required",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract": {
            "visible_only": True,
            "amodal_completion": False,
            "morphology": "none",
            "operation": "native_gripper_track & pose_roi & ~(target | receiver)",
            "outside_active_window": "empty",
        },
        "episode": {
            "task": ref.task,
            "episode_index": ref.episode_index,
            "camera": ref.camera,
            "frame_count": state.frame_count,
            "active_arm": events.active_arm,
            "events": events.to_json(),
            "source_video": str(state.paths.video),
            "source_video_sha256": _sha256(state.paths.video),
        },
        "seed": {
            "frame": args.seed_frame,
            "source_archive": str(seed_archive),
            "source_archive_sha256": _sha256(seed_archive),
            "source_key": args.gripper_seed_key,
            "original_pixels": int(original_seed.sum()),
            "constrained_pixels": int(seed_mask.sum()),
        },
        "known_objects": {
            "source_archive": str(object_archive),
            "source_archive_sha256": _sha256(object_archive),
            "target_key": args.target_track_key,
            "receiver_key": args.receiver_track_key,
            "provenance": args.object_track_provenance,
        },
        "sam3": {
            "checkpoint": str(config.sam3.checkpoint),
            "checkpoint_sha256": _sha256(config.sam3.checkpoint),
            "visible_gpu_index": args.sam3_gpu,
            "propagation": "sam3_native_mask_forward_backward",
        },
        "tracks": {
            "native": _track_summary(native_track, *active_window),
            "pose_cropped_candidate": _track_summary(
                result.candidate_mask,
                *active_window,
            ),
            "final_gripper": _track_summary(result.gripper_mask, *active_window),
            "target_removed": _track_summary(result.target_removed, *active_window),
            "receiver_removed": _track_summary(
                result.receiver_removed,
                *active_window,
            ),
        },
        "timing": {
            "model_load_seconds": model_load_seconds,
            "frame_extract_seconds": frame_extract_seconds,
            "native_propagation_seconds": propagation_seconds,
            "native_propagation_fps": window_frames / propagation_seconds,
            "render_seconds": render_seconds,
            "total_seconds": time.perf_counter() - total_started,
        },
        "render": render,
        "artifacts": {
            "mask_archive": {
                "path": archive_path.name,
                "sha256": _sha256(archive_path),
                "bytes": archive_path.stat().st_size,
            },
            "review_video": {
                "path": video_path.name,
                "sha256": _sha256(video_path),
                "bytes": video_path.stat().st_size,
            },
            "contact_sheet": {
                "path": contact_sheet_path.name,
                "sha256": _sha256(contact_sheet_path),
                "bytes": contact_sheet_path.stat().st_size,
            },
        },
        "implementation": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": _sha256(Path(__file__).resolve()),
        },
        "review_questions": [
            "Does the final mask include wrist or forearm pixels?",
            "Does it retain the visible gripper during grasp and release?",
            "Does known-object exclusion leave target or receiver contamination?",
            "Does native propagation flicker or switch identity?",
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "episode": args.episode_id,
                "manifest": str(manifest_path),
                "video": str(video_path),
                "native_propagation_fps": manifest["timing"][
                    "native_propagation_fps"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
