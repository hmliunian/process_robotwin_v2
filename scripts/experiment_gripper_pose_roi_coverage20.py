#!/usr/bin/env python3
"""Audit state-projected gripper ROIs on coverage20, optionally with SAM3 seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw

from robotwin_annotation_v2.adapters import RoboTwinDataset
from robotwin_annotation_v2.config import load_config
from robotwin_annotation_v2.experiments import (
    CAM_HIGH_CALIBRATION,
    DEFAULT_GRIPPER_ROI_GEOMETRY,
    ObjectExclusionResult,
    ProjectedGripperRoi,
    exclude_known_objects,
    project_gripper_roi,
)
from robotwin_annotation_v2.models import EpisodeRef, LoopEvents
from robotwin_annotation_v2.pipeline import build_loop_context


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRAME_SHAPE = (240, 320)
PANEL_SIZE = (160, 120)
GEOMETRY_KEYFRAME_COUNT = 7


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable_dataclass(value: Any) -> dict[str, Any]:
    return {key: float(item) for key, item in asdict(value).items()}


def _polygon_mask(roi: ProjectedGripperRoi) -> np.ndarray:
    mask = np.zeros(FRAME_SHAPE, dtype=np.uint8)
    polygon = np.rint(roi.hull_pixels_xy).astype(np.int32)
    if len(polygon) >= 3:
        cv2.fillConvexPoly(mask, polygon, 1)
    return mask.astype(bool)


def _clipped_bbox(roi: ProjectedGripperRoi) -> tuple[int, int, int, int] | None:
    height, width = FRAME_SHAPE
    x0 = max(0, min(width, int(np.floor(roi.bbox_xyxy[0]))))
    y0 = max(0, min(height, int(np.floor(roi.bbox_xyxy[1]))))
    x1 = max(0, min(width, int(np.ceil(roi.bbox_xyxy[2]))))
    y1 = max(0, min(height, int(np.ceil(roi.bbox_xyxy[3]))))
    return None if x1 <= x0 or y1 <= y0 else (x0, y0, x1, y1)


def _point_in_frame(point_xy: np.ndarray) -> bool:
    height, width = FRAME_SHAPE
    return 0 <= point_xy[0] < width and 0 <= point_xy[1] < height


def _phase(frame_id: int, events: LoopEvents) -> str:
    if frame_id < events.t_close_start:
        return "approach"
    if frame_id <= events.t_close_done:
        return "close"
    if frame_id < events.t_open_start:
        return "transport"
    return "release"


def _draw_cross(draw: ImageDraw.ImageDraw, xy: np.ndarray, color: str) -> None:
    x, y = (float(value) for value in xy)
    draw.line((x - 4, y, x + 4, y), fill=color, width=2)
    draw.line((x, y - 4, x, y + 4), fill=color, width=2)


def _geometry_overlay(
    image: Image.Image,
    roi: ProjectedGripperRoi,
    *,
    label: str,
) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    hull = [tuple(float(value) for value in point) for point in roi.hull_pixels_xy]
    if len(hull) >= 3:
        draw.line(hull + [hull[0]], fill="#36ff68", width=2)
    bbox = tuple(float(value) for value in roi.bbox_xyxy)
    draw.rectangle(bbox, outline="#ffd83d", width=1)
    _draw_cross(draw, roi.eef_pixel_xy, "#4aa8ff")
    _draw_cross(draw, roi.tcp_pixel_xy, "#ff3b3b")
    draw.rectangle((0, 0, min(319, 8 + 6 * len(label)), 15), fill="#111111")
    draw.text((3, 2), label, fill="white")
    return canvas


def _save_compact_jpeg(image: Image.Image, path: Path, *, max_bytes: int = 300_000) -> int:
    for quality in (70, 64, 58, 52):
        image.save(path, format="JPEG", quality=quality, optimize=True)
        if path.stat().st_size <= max_bytes:
            return quality
    return 52


def _episode_geometry(
    dataset: RoboTwinDataset,
    ref: EpisodeRef,
) -> tuple[dict[str, Any], list[Image.Image]]:
    state = dataset.load_state(ref)
    context = build_loop_context(dataset, ref)
    events = context.events
    arm_index = 0 if events.active_arm == "left" else 1
    active_frames = range(events.t_move_start, events.t_open_done + 1)
    rois: dict[int, ProjectedGripperRoi] = {}
    roi_pixels: dict[int, int] = {}
    for frame_id in active_frames:
        roi = project_gripper_roi(
            state.eef_states[frame_id, arm_index],
            state.gripper_states[frame_id, arm_index],
        )
        rois[frame_id] = roi
        roi_pixels[frame_id] = int(_polygon_mask(roi).sum())

    tcp_in_frame = [frame for frame, roi in rois.items() if _point_in_frame(roi.tcp_pixel_xy)]
    roi_intersects = [frame for frame, pixels in roi_pixels.items() if pixels > 0]
    roi_fully_in_frame = [
        frame
        for frame, roi in rois.items()
        if all(_point_in_frame(point) for point in roi.hull_pixels_xy)
    ]
    first_tcp = tcp_in_frame[0] if tcp_in_frame else events.t_close_start
    keyframes = (
        events.t_move_start,
        first_tcp,
        max(events.t_move_start, events.t_close_start - 1),
        events.t_close_done,
        (events.t_close_done + events.t_open_start) // 2,
        events.t_open_start,
        events.t_open_done,
    )
    if len(keyframes) != GEOMETRY_KEYFRAME_COUNT:
        raise AssertionError("geometry keyframe contract changed")
    frames = dataset.read_frames(ref, keyframes)
    panels = [
        _geometry_overlay(
            frames[frame_id],
            rois[frame_id],
            label=f"ep{ref.episode_index} {events.active_arm[0]} f{frame_id} {_phase(frame_id, events)}",
        ).resize(PANEL_SIZE, Image.Resampling.LANCZOS)
        for frame_id in keyframes
    ]
    pixels = np.asarray(list(roi_pixels.values()), dtype=np.int64)
    record = {
        "episode_index": ref.episode_index,
        "variant": (
            "clean" if ref.episode_index < 7200 else "randomized"
        ),
        "frame_count": state.frame_count,
        "events": events.to_json(),
        "active_window_inclusive": [events.t_move_start, events.t_open_done],
        "keyframes": list(keyframes),
        "first_tcp_in_frame": tcp_in_frame[0] if tcp_in_frame else None,
        "first_roi_intersection": roi_intersects[0] if roi_intersects else None,
        "tcp_in_frame_count": len(tcp_in_frame),
        "roi_intersection_count": len(roi_intersects),
        "roi_fully_in_frame_count": len(roi_fully_in_frame),
        "roi_pixels_min": int(pixels.min()),
        "roi_pixels_median": float(np.median(pixels)),
        "roi_pixels_max": int(pixels.max()),
    }
    return record, panels


def _geometry_audit(
    dataset: RoboTwinDataset,
    episode_ids: Iterable[int],
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    episode_records: list[dict[str, Any]] = []
    episode_panels: list[list[Image.Image]] = []
    for episode_id in episode_ids:
        ref = EpisodeRef(dataset.task, int(episode_id), dataset.camera)
        record, panels = _episode_geometry(dataset, ref)
        episode_records.append(record)
        episode_panels.append(panels)

    artifacts: list[dict[str, Any]] = []
    for group_start in range(0, len(episode_records), 5):
        group = episode_panels[group_start : group_start + 5]
        sheet = Image.new(
            "RGB",
            (PANEL_SIZE[0] * GEOMETRY_KEYFRAME_COUNT, PANEL_SIZE[1] * len(group)),
            "#151515",
        )
        for row, panels in enumerate(group):
            for column, panel in enumerate(panels):
                sheet.paste(panel, (column * PANEL_SIZE[0], row * PANEL_SIZE[1]))
        path = output_dir / f"geometry_review_{group_start // 5:02d}.jpg"
        quality = _save_compact_jpeg(sheet, path)
        artifacts.append(
            {
                "path": path.name,
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
                "width": sheet.width,
                "height": sheet.height,
                "jpeg_quality": quality,
                "episode_ids": [
                    item["episode_index"]
                    for item in episode_records[group_start : group_start + 5]
                ],
            }
        )
    return episode_records, artifacts


def _mask_overlay(
    image: Image.Image,
    roi: ProjectedGripperRoi,
    raw_mask: np.ndarray,
    cropped_mask: np.ndarray,
    *,
    label: str,
) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    raw_only = raw_mask & ~cropped_mask
    base[raw_only] = base[raw_only] * 0.52 + np.asarray([235, 50, 190]) * 0.48
    base[cropped_mask] = base[cropped_mask] * 0.40 + np.asarray([20, 230, 180]) * 0.60
    canvas = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(canvas)
    hull = [tuple(float(value) for value in point) for point in roi.hull_pixels_xy]
    if len(hull) >= 3:
        draw.line(hull + [hull[0]], fill="#fff044", width=2)
    _draw_cross(draw, roi.tcp_pixel_xy, "#ff3b3b")
    draw.rectangle((0, 0, min(319, 8 + 6 * len(label)), 15), fill="#111111")
    draw.text((3, 2), label, fill="white")
    return canvas


def _draw_mask_contours(
    draw: ImageDraw.ImageDraw,
    mask: np.ndarray,
    *,
    color: str,
) -> None:
    contours, _hierarchy = cv2.findContours(
        np.asarray(mask, dtype=np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    for contour in contours:
        points = [tuple(int(value) for value in point) for point in contour[:, 0, :]]
        if len(points) >= 2:
            draw.line(points + [points[0]], fill=color, width=1)


def _object_exclusion_overlay(
    image: Image.Image,
    roi: ProjectedGripperRoi,
    target_mask: np.ndarray,
    receiver_mask: np.ndarray,
    result: ObjectExclusionResult,
    *,
    label: str,
) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    base[result.target_removed] = (
        base[result.target_removed] * 0.38 + np.asarray([255, 126, 35]) * 0.62
    )
    base[result.receiver_removed] = (
        base[result.receiver_removed] * 0.38 + np.asarray([70, 105, 255]) * 0.62
    )
    base[result.gripper_mask] = (
        base[result.gripper_mask] * 0.36 + np.asarray([15, 230, 185]) * 0.64
    )
    canvas = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(canvas)
    _draw_mask_contours(draw, target_mask, color="#ff7e23")
    _draw_mask_contours(draw, receiver_mask, color="#4669ff")
    hull = [tuple(float(value) for value in point) for point in roi.hull_pixels_xy]
    if len(hull) >= 3:
        draw.line(hull + [hull[0]], fill="#fff044", width=2)
    _draw_cross(draw, roi.tcp_pixel_xy, "#ff3b3b")
    draw.rectangle((0, 0, min(319, 8 + 6 * len(label)), 15), fill="#111111")
    draw.text((3, 2), label, fill="white")
    return canvas


def _load_object_seeds(
    run_root: Path,
    ref: EpisodeRef,
) -> tuple[dict[str, tuple[int, np.ndarray]], dict[str, Any]]:
    episode_dir = (
        run_root
        / ref.task
        / f"episode_{ref.episode_id}"
        / ref.camera
    )
    manifest_path = episode_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    episode = manifest.get("episode", {})
    expected_episode = {
        "task": ref.task,
        "episode_index": ref.episode_index,
        "camera": ref.camera,
    }
    for key, expected in expected_episode.items():
        if episode.get(key) != expected:
            raise ValueError(
                f"object-mask manifest {manifest_path} has {key}={episode.get(key)!r}, "
                f"expected {expected!r}"
            )

    roles = {item.get("role"): item for item in manifest.get("roles", [])}
    seeds: dict[str, tuple[int, np.ndarray]] = {}
    seed_provenance: dict[str, Any] = {}
    for role in ("target", "receiver"):
        record = roles.get(role)
        if record is None or record.get("status") != "ok":
            raise ValueError(f"{manifest_path} has no approved {role} seed")
        seed_frame = record.get("seed_frame_id")
        relative_path = record.get("seed_mask_path")
        if not isinstance(seed_frame, int) or not isinstance(relative_path, str):
            raise ValueError(f"{manifest_path} has an invalid {role} seed contract")
        seed_path = episode_dir / relative_path
        with Image.open(seed_path) as image:
            seed_mask = np.asarray(image.convert("L"), dtype=np.uint8) != 0
        if seed_mask.shape != FRAME_SHAPE or not seed_mask.any():
            raise ValueError(f"invalid {role} seed mask at {seed_path}")
        seeds[role] = (seed_frame, seed_mask)
        seed_provenance[role] = {
            "seed_frame": seed_frame,
            "seed_mask": str(seed_path),
            "seed_mask_sha256": _sha256(seed_path),
            "seed_pixels": int(seed_mask.sum()),
            "primary_query": record.get("primary_query"),
        }
    return seeds, {
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "run_id": manifest.get("run_id"),
        "seeds": seed_provenance,
    }


def _track_summary(track: np.ndarray, start: int, end: int) -> dict[str, Any]:
    pixels = track[start : end + 1].reshape(end - start + 1, -1).sum(axis=1)
    nonempty_offsets = np.flatnonzero(pixels > 0)
    nonempty_pixels = pixels[nonempty_offsets]
    return {
        "window_inclusive": [start, end],
        "nonempty_frames": int(nonempty_offsets.size),
        "first_nonempty_frame": (
            None if nonempty_offsets.size == 0 else int(start + nonempty_offsets[0])
        ),
        "last_nonempty_frame": (
            None if nonempty_offsets.size == 0 else int(start + nonempty_offsets[-1])
        ),
        "pixels_min_nonempty": (
            None if nonempty_pixels.size == 0 else int(nonempty_pixels.min())
        ),
        "pixels_median_nonempty": (
            None if nonempty_pixels.size == 0 else float(np.median(nonempty_pixels))
        ),
        "pixels_max": int(pixels.max()),
    }


def _object_exclusion_audit(
    dataset: RoboTwinDataset,
    episode_ids: Iterable[int],
    output_dir: Path,
    *,
    checkpoint: Path,
    sam3_source_root: Path,
    gpu: int,
    candidate_run_dir: Path,
    object_mask_run_root: Path,
    target_seed_query: str | None,
    target_seed_box_xyxy: tuple[float, float, float, float] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    source = sam3_source_root.resolve()
    if not (source / "robotwin_annotate" / "sam3_masks.py").is_file():
        raise FileNotFoundError(f"SAM3 source root is invalid: {source}")
    sys.path.insert(0, str(source))
    from robotwin_annotate.sam3_masks import (  # type: ignore[import-not-found]
        Sam3EpisodeSegmenter,
        sam3_resource_path,
    )

    candidate_dir = candidate_run_dir.resolve()
    candidate_manifest_path = candidate_dir / "manifest.json"
    candidate_manifest = json.loads(
        candidate_manifest_path.read_text(encoding="utf-8")
    )
    candidate_episodes = {
        int(item["episode_index"]): item
        for item in candidate_manifest.get("episodes", [])
    }
    missing = sorted(set(int(value) for value in episode_ids) - candidate_episodes.keys())
    if missing:
        raise ValueError(f"candidate run has no episodes: {missing}")
    candidate_archives = [
        item["path"]
        for item in candidate_manifest.get("artifacts", [])
        if str(item.get("path", "")).endswith(".npz")
    ]
    if len(candidate_archives) != 1:
        raise ValueError("candidate run must declare exactly one NPZ artifact")
    candidate_archive_path = candidate_dir / candidate_archives[0]

    segmenter = Sam3EpisodeSegmenter(
        gpus_to_use=[gpu],
        checkpoint_path=checkpoint,
    )
    records: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    object_sources: list[dict[str, Any]] = []
    try:
        with np.load(candidate_archive_path) as candidate_arrays:
            for episode_id in episode_ids:
                ref = EpisodeRef(dataset.task, int(episode_id), dataset.camera)
                state = dataset.load_state(ref)
                context = build_loop_context(dataset, ref)
                events = context.events
                arm_index = 0 if events.active_arm == "left" else 1
                keyframes = tuple(
                    int(item["frame"])
                    for item in candidate_episodes[int(episode_id)]["frames"]
                )
                images = dataset.read_frames(ref, keyframes)
                seeds, object_source = _load_object_seeds(object_mask_run_root, ref)

                raw_surplus = int(dataset.manifest["raw_video_frame_surplus"])
                tracks: dict[str, np.ndarray] = {}
                with sam3_resource_path(
                    state.paths.video,
                    expected_num_frames=state.frame_count + raw_surplus,
                    temp_root=Path("/tmp"),
                ) as resource:
                    if target_seed_query is not None or target_seed_box_xyxy is not None:
                        target_frame, _original_target_seed = seeds["target"]
                        if target_seed_box_xyxy is not None:
                            x0, y0, x1, y1 = target_seed_box_xyxy
                            if not (0 <= x0 < x1 <= 320 and 0 <= y0 < y1 <= 240):
                                raise ValueError(
                                    "--target-seed-box must be a valid pixel box "
                                    "inside the 320x240 frame"
                                )
                            normalized_box = (
                                x0 / 320.0,
                                y0 / 240.0,
                                x1 / 320.0,
                                y1 / 240.0,
                            )
                            override_seed = segmenter.segment_box_seed(
                                resource,
                                normalized_box,
                                seed_frame=target_frame,
                                frame_shape=FRAME_SHAPE,
                            )
                            override_contract: dict[str, Any] = {
                                "method": "known_pixel_box",
                                "bbox_xyxy": list(target_seed_box_xyxy),
                            }
                        else:
                            assert target_seed_query is not None
                            override_seed = segmenter.segment_text_seed(
                                resource,
                                target_seed_query,
                                seed_frame=target_frame,
                                frame_shape=FRAME_SHAPE,
                            )
                            override_contract = {
                                "method": "text_query",
                                "query": target_seed_query,
                            }
                        if not override_seed.any():
                            raise ValueError(
                                f"target seed override returned empty for episode {episode_id}"
                            )
                        seeds["target"] = (target_frame, override_seed)
                        object_source["target_seed_override"] = {
                            **override_contract,
                            "seed_frame": target_frame,
                            "seed_pixels": int(override_seed.sum()),
                            "mask_sha256": hashlib.sha256(
                                np.ascontiguousarray(override_seed, dtype=np.uint8)
                            ).hexdigest(),
                        }
                    for role in ("target", "receiver"):
                        seed_frame, seed_mask = seeds[role]
                        tracks[role] = segmenter.segment_mask_prompt(
                            resource,
                            seed_mask,
                            seed_frame=seed_frame,
                            num_frames=state.frame_count,
                            frame_shape=FRAME_SHAPE,
                            tracking_range=(seed_frame, events.t_open_done),
                        )

                object_sources.append(
                    {"episode_index": int(episode_id), **object_source}
                )

                prefix = f"episode_{episode_id}"
                arrays[f"{prefix}_target_seed"] = seeds["target"][1]
                arrays[f"{prefix}_receiver_seed"] = seeds["receiver"][1]
                arrays[f"{prefix}_target_track"] = tracks["target"]
                arrays[f"{prefix}_receiver_track"] = tracks["receiver"]
                panels: list[Image.Image] = []
                frame_records: list[dict[str, Any]] = []
                for frame_id in keyframes:
                    key = f"{prefix}_frame_{frame_id:06d}"
                    candidate_key = f"{key}_cropped"
                    if candidate_key not in candidate_arrays:
                        raise ValueError(f"candidate archive is missing {candidate_key}")
                    candidate = np.asarray(candidate_arrays[candidate_key], dtype=bool)
                    target = tracks["target"][frame_id]
                    receiver = tracks["receiver"][frame_id]
                    result = exclude_known_objects(candidate, target, receiver)
                    arrays[f"{key}_candidate"] = candidate
                    arrays[f"{key}_target"] = target
                    arrays[f"{key}_receiver"] = receiver
                    arrays[f"{key}_removed"] = result.removed_mask
                    arrays[f"{key}_gripper"] = result.gripper_mask

                    rgb = np.asarray(images[frame_id], dtype=np.uint8)
                    residual_dark_fraction = (
                        float(
                            (rgb[result.gripper_mask].max(axis=1) < 70).mean()
                        )
                        if result.gripper_mask.any()
                        else None
                    )
                    candidate_pixels = int(candidate.sum())
                    removed_pixels = int(result.removed_mask.sum())
                    frame_record = {
                        "frame": frame_id,
                        "phase": _phase(frame_id, events),
                        "candidate_pixels": candidate_pixels,
                        "target_track_pixels": int(target.sum()),
                        "receiver_track_pixels": int(receiver.sum()),
                        "target_receiver_overlap_pixels": int((target & receiver).sum()),
                        "target_removed_pixels": int(result.target_removed.sum()),
                        "receiver_removed_pixels": int(result.receiver_removed.sum()),
                        "removed_pixels": removed_pixels,
                        "removed_fraction": (
                            float(removed_pixels / candidate_pixels)
                            if candidate_pixels
                            else None
                        ),
                        "gripper_residual_pixels": int(result.gripper_mask.sum()),
                        "gripper_residual_dark_fraction": residual_dark_fraction,
                    }
                    frame_records.append(frame_record)
                    roi = project_gripper_roi(
                        state.eef_states[frame_id, arm_index],
                        state.gripper_states[frame_id, arm_index],
                    )
                    label = (
                        f"f{frame_id} {frame_record['phase']} c={candidate_pixels} "
                        f"-t={frame_record['target_removed_pixels']} "
                        f"-r={frame_record['receiver_removed_pixels']} "
                        f"g={frame_record['gripper_residual_pixels']}"
                    )
                    panels.append(
                        _object_exclusion_overlay(
                            images[frame_id],
                            roi,
                            target,
                            receiver,
                            result,
                            label=label,
                        ).resize((280, 210), Image.Resampling.LANCZOS)
                    )

                sheet = Image.new("RGB", (1120, 420), "#151515")
                for index, panel in enumerate(panels):
                    sheet.paste(panel, ((index % 4) * 280, (index // 4) * 210))
                review_path = output_dir / f"object_exclusion_review_ep{episode_id}.jpg"
                quality = _save_compact_jpeg(sheet, review_path)
                artifacts.append(
                    {
                        "path": review_path.name,
                        "sha256": _sha256(review_path),
                        "bytes": review_path.stat().st_size,
                        "width": sheet.width,
                        "height": sheet.height,
                        "jpeg_quality": quality,
                        "episode_id": int(episode_id),
                    }
                )
                records.append(
                    {
                        "episode_index": int(episode_id),
                        "variant": (
                            "clean" if int(episode_id) < 7200 else "randomized"
                        ),
                        "events": events.to_json(),
                        "tracks": {
                            role: _track_summary(
                                track,
                                events.t_move_start,
                                events.t_open_done,
                            )
                            for role, track in tracks.items()
                        },
                        "frames": frame_records,
                    }
                )
    finally:
        segmenter.shutdown()

    archive_path = output_dir / "object_exclusion_masks.npz"
    np.savez_compressed(archive_path, **arrays)
    artifacts.append(
        {
            "path": archive_path.name,
            "sha256": _sha256(archive_path),
            "bytes": archive_path.stat().st_size,
        }
    )
    provenance = {
        "candidate_manifest": str(candidate_manifest_path),
        "candidate_manifest_sha256": _sha256(candidate_manifest_path),
        "candidate_masks": str(candidate_archive_path),
        "candidate_masks_sha256": _sha256(candidate_archive_path),
        "object_mask_run_root": str(object_mask_run_root.resolve()),
        "object_sources": object_sources,
        "target_seed_query_override": target_seed_query,
        "target_seed_box_xyxy_override": target_seed_box_xyxy,
        "operation": "candidate & ~(target_track | receiver_track)",
        "morphology": "none",
        "overlay_colors": {
            "gripper_residual": "cyan",
            "target_removed": "orange",
            "receiver_removed": "blue",
            "projected_roi": "yellow",
        },
    }
    return records, artifacts, provenance


def _sam3_seed_audit(
    dataset: RoboTwinDataset,
    episode_ids: Iterable[int],
    output_dir: Path,
    *,
    checkpoint: Path,
    sam3_source_root: Path,
    gpu: int,
    prompt_text: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source = sam3_source_root.resolve()
    if not (source / "robotwin_annotate" / "sam3_masks.py").is_file():
        raise FileNotFoundError(f"SAM3 source root is invalid: {source}")
    sys.path.insert(0, str(source))
    from robotwin_annotate.sam3_masks import (  # type: ignore[import-not-found]
        Sam3EpisodeSegmenter,
        sam3_resource_path,
    )

    segmenter = Sam3EpisodeSegmenter(
        gpus_to_use=[gpu],
        checkpoint_path=checkpoint,
    )
    records: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    try:
        for episode_id in episode_ids:
            ref = EpisodeRef(dataset.task, int(episode_id), dataset.camera)
            state = dataset.load_state(ref)
            context = build_loop_context(dataset, ref)
            events = context.events
            arm_index = 0 if events.active_arm == "left" else 1
            all_rois = {
                frame_id: project_gripper_roi(
                    state.eef_states[frame_id, arm_index],
                    state.gripper_states[frame_id, arm_index],
                )
                for frame_id in range(events.t_move_start, events.t_open_done + 1)
            }
            tcp_visible = [
                frame_id
                for frame_id, roi in all_rois.items()
                if _point_in_frame(roi.tcp_pixel_xy)
            ]
            first_tcp = tcp_visible[0] if tcp_visible else events.t_close_start
            keyframes = (
                first_tcp,
                max(first_tcp, events.t_close_start - 1),
                events.t_close_done,
                (events.t_close_done + events.t_open_start) // 2,
                max(events.t_close_done + 1, events.t_open_start - 1),
                events.t_open_start,
                events.t_open_done,
            )
            images = dataset.read_frames(ref, keyframes)
            panels: list[Image.Image] = []
            frame_records: list[dict[str, Any]] = []
            raw_surplus = int(dataset.manifest["raw_video_frame_surplus"])
            with sam3_resource_path(
                state.paths.video,
                expected_num_frames=state.frame_count + raw_surplus,
                temp_root=Path("/tmp"),
            ) as resource:
                for frame_id in keyframes:
                    roi = all_rois[frame_id]
                    bbox = _clipped_bbox(roi)
                    if bbox is None:
                        raw_mask = np.zeros(FRAME_SHAPE, dtype=bool)
                    else:
                        x0, y0, x1, y1 = bbox
                        normalized_box = (
                            x0 / 320.0,
                            y0 / 240.0,
                            x1 / 320.0,
                            y1 / 240.0,
                        )
                        if prompt_text is None:
                            raw_mask = segmenter.segment_box_seed(
                                resource,
                                normalized_box,
                                seed_frame=frame_id,
                                frame_shape=FRAME_SHAPE,
                            )
                        else:
                            raw_mask = segmenter.segment_text_box_seed(
                                resource,
                                prompt_text,
                                normalized_box,
                                seed_frame=frame_id,
                                frame_shape=FRAME_SHAPE,
                            )
                    roi_mask = _polygon_mask(roi)
                    cropped = raw_mask & roi_mask
                    rgb = np.asarray(images[frame_id], dtype=np.uint8)
                    dark_fraction = (
                        float((rgb[cropped].max(axis=1) < 70).mean())
                        if cropped.any()
                        else None
                    )
                    key = f"episode_{episode_id}_frame_{frame_id:06d}"
                    arrays[f"{key}_raw"] = raw_mask
                    arrays[f"{key}_roi"] = roi_mask
                    arrays[f"{key}_cropped"] = cropped
                    frame_record = {
                        "frame": frame_id,
                        "phase": _phase(frame_id, events),
                        "bbox_xyxy": None if bbox is None else list(bbox),
                        "raw_pixels": int(raw_mask.sum()),
                        "roi_pixels": int(roi_mask.sum()),
                        "cropped_pixels": int(cropped.sum()),
                        "raw_outside_roi_pixels": int((raw_mask & ~roi_mask).sum()),
                        "cropped_dark_fraction": dark_fraction,
                    }
                    frame_records.append(frame_record)
                    label = (
                        f"f{frame_id} {frame_record['phase']} "
                        f"raw={frame_record['raw_pixels']} crop={frame_record['cropped_pixels']}"
                    )
                    panels.append(
                        _mask_overlay(
                            images[frame_id],
                            roi,
                            raw_mask,
                            cropped,
                            label=label,
                        ).resize((280, 210), Image.Resampling.LANCZOS)
                    )
            sheet = Image.new("RGB", (1120, 420), "#151515")
            for index, panel in enumerate(panels):
                sheet.paste(panel, ((index % 4) * 280, (index // 4) * 210))
            review_path = output_dir / f"sam3_seed_review_ep{episode_id}.jpg"
            quality = _save_compact_jpeg(sheet, review_path)
            artifacts.append(
                {
                    "path": review_path.name,
                    "sha256": _sha256(review_path),
                    "bytes": review_path.stat().st_size,
                    "width": sheet.width,
                    "height": sheet.height,
                    "jpeg_quality": quality,
                    "episode_id": int(episode_id),
                }
            )
            records.append(
                {
                    "episode_index": int(episode_id),
                    "variant": "clean" if int(episode_id) < 7200 else "randomized",
                    "events": events.to_json(),
                    "frames": frame_records,
                }
            )
    finally:
        segmenter.shutdown()

    archive_path = output_dir / "sam3_seed_masks.npz"
    np.savez_compressed(archive_path, **arrays)
    artifacts.append(
        {
            "path": archive_path.name,
            "sha256": _sha256(archive_path),
            "bytes": archive_path.stat().st_size,
        }
    )
    return records, artifacts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "pilot_move_pillbottle_pad.yaml",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("geometry", "sam3-seeds", "object-exclusion"),
        default="geometry",
    )
    parser.add_argument("--episode-ids", type=int, nargs="*")
    parser.add_argument("--sam3-gpu", type=int, default=0)
    parser.add_argument(
        "--sam3-text",
        help="Optional text submitted jointly with the projected visual box",
    )
    parser.add_argument(
        "--sam3-source-root",
        type=Path,
        default=PROJECT_ROOT.parent / "process_data",
    )
    parser.add_argument(
        "--candidate-run-dir",
        type=Path,
        help="SAM3 seed run supplying projected-ROI candidate masks",
    )
    parser.add_argument(
        "--object-mask-run-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "runs" / "coverage20-e2e-v1",
        help="Existing run root supplying approved target/receiver seed masks",
    )
    parser.add_argument(
        "--target-seed-query",
        help="Optional SAM3 text query replacing the stored target seed mask",
    )
    parser.add_argument(
        "--target-seed-box",
        type=float,
        nargs=4,
        metavar=("X0", "Y0", "X1", "Y1"),
        help="Optional known target box in 320x240 seed-frame pixels",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.target_seed_query is not None and args.target_seed_box is not None:
        raise ValueError("--target-seed-query and --target-seed-box are exclusive")
    config = load_config(args.config)
    episode_ids = tuple(args.episode_ids or config.dataset.regression_episode_ids)
    unknown = sorted(set(episode_ids) - set(config.dataset.regression_episode_ids))
    if unknown:
        raise ValueError(f"episode ids are outside coverage20: {unknown}")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    dataset = RoboTwinDataset(
        config.dataset.root,
        task=config.dataset.task,
        camera=config.dataset.camera,
        manifest_path=config.dataset.manifest,
    )

    if args.mode == "geometry":
        records, artifacts = _geometry_audit(dataset, episode_ids, output_dir)
        status = "geometry_review_required"
        extra: dict[str, Any] = {}
    elif args.mode == "sam3-seeds":
        records, artifacts = _sam3_seed_audit(
            dataset,
            episode_ids,
            output_dir,
            checkpoint=config.sam3.checkpoint,
            sam3_source_root=args.sam3_source_root,
            gpu=args.sam3_gpu,
            prompt_text=args.sam3_text,
        )
        status = "sam3_seed_review_required"
        sam_source = args.sam3_source_root / "robotwin_annotate" / "sam3_masks.py"
        extra = {
            "sam3": {
                "checkpoint": str(config.sam3.checkpoint),
                "checkpoint_sha256": _sha256(config.sam3.checkpoint),
                "source": str(sam_source),
                "source_sha256": _sha256(sam_source),
                "gpu": args.sam3_gpu,
                "prompt_text": args.sam3_text,
                "pixel_contract": (
                    "direct same-frame SAM3 "
                    + ("joint text+visual-box" if args.sam3_text else "visual-box")
                    + " mask AND projected 3-D ROI"
                ),
            }
        }
    else:
        if args.candidate_run_dir is None:
            raise ValueError("--candidate-run-dir is required for object-exclusion")
        records, artifacts, exclusion_provenance = _object_exclusion_audit(
            dataset,
            episode_ids,
            output_dir,
            checkpoint=config.sam3.checkpoint,
            sam3_source_root=args.sam3_source_root,
            gpu=args.sam3_gpu,
            candidate_run_dir=args.candidate_run_dir,
            object_mask_run_root=args.object_mask_run_root,
            target_seed_query=args.target_seed_query,
            target_seed_box_xyxy=(
                None
                if args.target_seed_box is None
                else tuple(float(value) for value in args.target_seed_box)
            ),
        )
        status = "object_exclusion_review_required"
        sam_source = args.sam3_source_root / "robotwin_annotate" / "sam3_masks.py"
        extra = {
            "sam3": {
                "checkpoint": str(config.sam3.checkpoint),
                "checkpoint_sha256": _sha256(config.sam3.checkpoint),
                "source": str(sam_source),
                "source_sha256": _sha256(sam_source),
                "gpu": args.sam3_gpu,
                "pixel_contract": (
                    "approved object seed mask -> native full-window identity track"
                ),
            },
            "object_exclusion": exclusion_provenance,
        }

    module_path = PROJECT_ROOT / "src" / "robotwin_annotation_v2" / "experiments" / "gripper_pose_roi.py"
    manifest = {
        "format": f"robotwin_gripper_pose_roi_coverage20_{args.mode}_v1",
        "status": status,
        "dataset": {
            "root": str(config.dataset.root),
            "task": config.dataset.task,
            "camera": config.dataset.camera,
            "episode_ids": list(episode_ids),
            "manifest": str(config.dataset.manifest),
            "manifest_sha256": _sha256(config.dataset.manifest),
        },
        "camera_calibration": {
            "intrinsic_cv": CAM_HIGH_CALIBRATION.intrinsic_cv.tolist(),
            "extrinsic_cv": CAM_HIGH_CALIBRATION.extrinsic_cv.tolist(),
        },
        "roi_geometry": _jsonable_dataclass(DEFAULT_GRIPPER_ROI_GEOMETRY),
        "episodes": records,
        "artifacts": artifacts,
        "implementation": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": _sha256(Path(__file__).resolve()),
            "geometry_module": str(module_path),
            "geometry_module_sha256": _sha256(module_path),
        },
        **extra,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": status, "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
