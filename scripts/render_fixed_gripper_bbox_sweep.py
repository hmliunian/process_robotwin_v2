#!/usr/bin/env python3
"""Render adaptive-width and fixed-width 3-D gripper ROI projections side by side."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from robotwin_annotation_v2.adapters import RoboTwinDataset
from robotwin_annotation_v2.config import load_config
from robotwin_annotation_v2.experiments import (
    DEFAULT_GRIPPER_ROI_GEOMETRY,
    GripperRoiGeometry,
    project_gripper_roi,
)
from robotwin_annotation_v2.models import EpisodeRef
from robotwin_annotation_v2.pipeline import build_loop_context


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FRAMES = ((7152, 93), (7188, 124), (7274, 102), (7317, 20), (7674, 103))
COLORS_RGB = ((40, 220, 155), (255, 205, 40), (235, 65, 210))


def _frame_spec(value: str) -> tuple[int, int]:
    try:
        episode, frame = (int(item) for item in value.split(":", maxsplit=1))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("frame must use EPISODE:FRAME") from exc
    if episode < 0 or frame < 0:
        raise argparse.ArgumentTypeError("episode and frame must be non-negative")
    return episode, frame


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "pilot_move_pillbottle_pad.yaml",
    )
    parser.add_argument(
        "--frame",
        dest="frames",
        action="append",
        type=_frame_spec,
        help="Representative EPISODE:FRAME; repeat as needed",
    )
    parser.add_argument("--axial-back-m", type=float, nargs="+", default=(0.08, 0.10, 0.12))
    parser.add_argument("--axial-front-m", type=float, default=0.06)
    parser.add_argument("--fixed-half-width-m", type=float, default=0.085)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _panel(
    rgb: np.ndarray,
    pose: np.ndarray,
    gripper_open: float,
    geometry: GripperRoiGeometry,
    *,
    title: str,
    color_rgb: tuple[int, int, int],
) -> np.ndarray:
    roi = project_gripper_roi(pose, gripper_open, geometry=geometry)
    result = np.asarray(rgb, dtype=np.uint8).copy()
    polygon = np.rint(roi.hull_pixels_xy).astype(np.int32).reshape(-1, 1, 2)
    if len(polygon) >= 3:
        fill = result.copy()
        cv2.fillConvexPoly(fill, polygon, color_rgb)
        result = cv2.addWeighted(fill, 0.18, result, 0.82, 0.0)
        cv2.polylines(result, [polygon], True, color_rgb, 2, cv2.LINE_AA)
    eef_x, eef_y = np.rint(roi.eef_pixel_xy).astype(int)
    tcp_x, tcp_y = np.rint(roi.tcp_pixel_xy).astype(int)
    cv2.drawMarker(result, (eef_x, eef_y), (50, 120, 255), cv2.MARKER_CROSS, 11, 2)
    cv2.drawMarker(result, (tcp_x, tcp_y), (255, 55, 55), cv2.MARKER_CROSS, 11, 2)
    cv2.rectangle(result, (0, 0), (result.shape[1] - 1, 31), (10, 10, 10), -1)
    cv2.putText(
        result,
        title,
        (5, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return result


def _validate_geometry_args(args: argparse.Namespace) -> None:
    values = (*args.axial_back_m, args.axial_front_m, args.fixed_half_width_m)
    if not all(np.isfinite(value) and value > 0.0 for value in values):
        raise ValueError("ROI dimensions must be finite and positive")
    if len(args.axial_back_m) != len(set(args.axial_back_m)):
        raise ValueError("axial back values must be unique")


def main() -> None:
    args = _parse_args()
    _validate_geometry_args(args)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    config = load_config(args.config)
    dataset = RoboTwinDataset(
        config.dataset.root,
        task=config.dataset.task,
        camera=config.dataset.camera,
        manifest_path=config.dataset.manifest,
    )
    frame_specs = tuple(args.frames or DEFAULT_FRAMES)
    rows: list[np.ndarray] = []
    records: list[dict[str, object]] = []
    fixed_geometries = tuple(
        replace(
            DEFAULT_GRIPPER_ROI_GEOMETRY,
            axial_back_m=back,
            axial_front_m=args.axial_front_m,
            closed_half_width_m=args.fixed_half_width_m,
            open_half_width_m=args.fixed_half_width_m,
        )
        for back in args.axial_back_m
    )
    for episode_id, frame_id in frame_specs:
        ref = EpisodeRef(config.dataset.task, episode_id, config.dataset.camera)
        state = dataset.load_state(ref)
        if frame_id >= state.frame_count:
            raise ValueError(f"episode {episode_id} has no frame {frame_id}")
        context = build_loop_context(dataset, ref)
        arm_index = 0 if context.events.active_arm == "left" else 1
        pose = state.eef_states[frame_id, arm_index]
        opening = float(state.gripper_states[frame_id, arm_index])
        rgb = np.asarray(dataset.read_frames(ref, (frame_id,))[frame_id], dtype=np.uint8)
        adaptive = replace(
            DEFAULT_GRIPPER_ROI_GEOMETRY,
            axial_back_m=args.axial_back_m[0],
            axial_front_m=args.axial_front_m,
        )
        adaptive_width = adaptive.closed_half_width_m + opening * (
            adaptive.open_half_width_m - adaptive.closed_half_width_m
        )
        panels = [
            _panel(
                rgb,
                pose,
                opening,
                adaptive,
                title=f"adaptive back={adaptive.axial_back_m:.3f} width={adaptive_width:.3f}",
                color_rgb=(35, 210, 235),
            )
        ]
        for geometry, color in zip(fixed_geometries, COLORS_RGB, strict=True):
            start_from_eef = geometry.tcp_offset_m - geometry.axial_back_m
            panels.append(
                _panel(
                    rgb,
                    pose,
                    opening,
                    geometry,
                    title=(
                        f"fixed back={geometry.axial_back_m:.3f} "
                        f"EEF-start={start_from_eef:+.3f} width={args.fixed_half_width_m:.3f}"
                    ),
                    color_rgb=color,
                )
            )
        row = np.concatenate(panels, axis=1)
        row_path = output_dir / f"episode_{episode_id:06d}_frame_{frame_id:04d}.jpg"
        Image.fromarray(row).save(row_path, format="JPEG", quality=91)
        rows.append(row)
        records.append(
            {
                "episode": episode_id,
                "frame": frame_id,
                "active_arm": context.events.active_arm,
                "gripper_open": opening,
                "adaptive_half_width_m": adaptive_width,
                "image": row_path.name,
            }
        )
    overview = np.concatenate(rows, axis=0)
    overview_path = output_dir / "fixed_gripper_bbox_sweep.jpg"
    Image.fromarray(overview).save(overview_path, format="JPEG", quality=91)
    manifest = {
        "format": "robotwin_fixed_gripper_bbox_sweep_v1",
        "config": str(args.config.resolve()),
        "adaptive_geometry": asdict(
            replace(
                DEFAULT_GRIPPER_ROI_GEOMETRY,
                axial_back_m=args.axial_back_m[0],
                axial_front_m=args.axial_front_m,
            )
        ),
        "fixed_geometries": [asdict(item) for item in fixed_geometries],
        "frames": records,
        "overview": overview_path.name,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(overview_path)


if __name__ == "__main__":
    main()
