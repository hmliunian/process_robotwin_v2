#!/usr/bin/env python3
"""Compare published masks with their saved SAM3 native tracks."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from robotwin_annotation_v2.config import MaskConfig, load_config
from robotwin_annotation_v2.models import FrameWindow
from robotwin_annotation_v2.pipeline import compose_visible_mask, evaluate_temporal_mask

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "pilot_move_pillbottle_pad.yaml",
    )
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "rendered_videos"
            / "coverage20_best_current"
            / "manifest.json"
        ),
        help="Render manifest that pins one source masks.npz per episode",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    return parser.parse_args()


def _read_seed(path: Path | None) -> np.ndarray | None:
    if path is None or not path.is_file():
        return None
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > 0


def _qc_payload(
    masks: np.ndarray,
    window: FrameWindow,
    config: MaskConfig,
    *,
    reference_mask: np.ndarray | None,
) -> dict[str, Any]:
    return evaluate_temporal_mask(
        masks,
        window,
        config,
        reference_mask=reference_mask,
    ).to_json()


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["role"], "published")].append(record["published"])
        grouped[(record["role"], "sam3_native")].append(record["sam3_native"])

    output: dict[str, Any] = {}
    for (role, method), values in sorted(grouped.items()):
        total_frames = sum(int(value["window_frames"]) for value in values)
        nonempty_frames = sum(int(value["nonempty_frames"]) for value in values)
        adjacent = [
            float(value["adjacent_iou_mean"])
            for value in values
            if value["adjacent_iou_mean"] is not None
        ]
        output[f"{role}_{method}"] = {
            "role_tracks": len(values),
            "window_frames": total_frames,
            "nonempty_frames": nonempty_frames,
            "coverage": nonempty_frames / total_frames if total_frames else 0.0,
            "presence_transitions": sum(
                int(value["presence_transitions"]) for value in values
            ),
            "internal_missing_frames": sum(
                int(value["internal_missing_frames"]) for value in values
            ),
            "mean_adjacent_iou": float(np.mean(adjacent)) if adjacent else None,
            "qc_status_counts": dict(Counter(str(value["status"]) for value in values)),
        }
    return output


def build_report(
    selection_manifest: Path,
    mask_config: MaskConfig,
) -> dict[str, Any]:
    selection = json.loads(selection_manifest.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for episode in selection["episodes"]:
        masks_path = Path(episode["source_masks"])
        episode_dir = masks_path.parent
        run_manifest = json.loads(
            (episode_dir / "run_manifest.json").read_text(encoding="utf-8")
        )
        with np.load(masks_path, allow_pickle=False) as archive:
            published = np.asarray(archive["masks"], dtype=bool)
        for channel_index, role_record in enumerate(run_manifest["roles"]):
            native_relative = role_record.get("native_track_path")
            if not native_relative:
                continue
            native_path = episode_dir / native_relative
            if not native_path.is_file():
                continue
            with np.load(native_path, allow_pickle=False) as archive:
                native = np.asarray(archive["masks"], dtype=bool)
            start, end = (int(value) for value in role_record["output_window"])
            window = FrameWindow(start, end)
            seed_relative = role_record.get("seed_mask_path")
            seed_path = None if not seed_relative else episode_dir / seed_relative
            seed = _read_seed(seed_path)
            native_windowed = compose_visible_mask(native, window)
            records.append(
                {
                    "episode_index": int(episode["episode_index"]),
                    "run_id": str(episode["run_id"]),
                    "role": str(role_record["role"]),
                    "source_role_status": str(role_record["status"]),
                    "source_masks": str(masks_path),
                    "published": _qc_payload(
                        published[channel_index],
                        window,
                        mask_config,
                        reference_mask=seed,
                    ),
                    "sam3_native": _qc_payload(
                        native_windowed,
                        window,
                        mask_config,
                        reference_mask=seed,
                    ),
                }
            )
    return {
        "format_version": "robotwin_temporal_tracking_benchmark_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "selection_manifest": str(selection_manifest.resolve()),
        "methods": {
            "published": "saved masks.npz composition",
            "sam3_native": "saved native_track.npz clipped to role output window",
        },
        "thresholds": {
            "minimum_adjacent_iou_p05": (
                mask_config.temporal_qc_min_adjacent_iou_p05
            ),
            "maximum_centroid_jump_p95_px": (
                mask_config.temporal_qc_max_centroid_jump_p95_px
            ),
            "maximum_area_ratio_jump_p95": (
                mask_config.temporal_qc_max_area_ratio_jump_p95
            ),
            "quarantine_signal_count": (
                mask_config.temporal_qc_quarantine_signal_count
            ),
        },
        "summary": _summary(records),
        "records": records,
    }


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    selection_manifest = args.selection_manifest.expanduser().resolve()
    report = build_report(selection_manifest, config.mask)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(rendered, end="")
        return
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
