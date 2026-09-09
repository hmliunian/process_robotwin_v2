"""Build cross-task contact sheets from the final episode results index."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

from robotwin_annotation_v2.adapters.artifact_store import ArtifactStore
from robotwin_annotation_v2.adapters.rendering import _decode_selected, file_sha256

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MOMENTS = {"overall_index": 0.5, "early": 0.25, "late": 1.0}
STATUS_STYLE = {
    "auto_complete": ("AUTO QC", "#69d39c"),
    "needs_temporal_review": ("REVIEW", "#f4be55"),
    "failed": ("FAILED", "#ff7e83"),
}
TILE_WIDTH, IMAGE_HEIGHT = 320, 240
LABEL_HEIGHT, GAP, MARGIN, HEADER = 28, 12, 24, 180


def render_batch_sheets(
    results_path: Path,
    datasets_root: Path,
    output_dir: Path,
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Sample final overlays, keeping failed episodes visible as labeled raw RGB."""

    report = json.loads(results_path.read_text(encoding="utf-8"))
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, int]] = set()
    for record in report["episodes"]:
        key = (record["task"], record["episode"])
        if key in seen:
            raise ValueError(f"duplicate episode: {key}")
        seen.add(key)
        status = record["status"]
        if status not in STATUS_STYLE or record["frame_count"] < 1:
            raise ValueError(f"invalid terminal episode: {key}")
        if bool(record["objects_ready"]) != (status != "failed"):
            raise ValueError(f"inconsistent output status: {key}")
        groups[record["task"]].append(record)
    if not groups:
        raise ValueError("results index has no episodes")

    columns = min(10, max(len(group) for group in groups.values()))
    card_height = IMAGE_HEIGHT + 2 * LABEL_HEIGHT
    width = 2 * MARGIN + columns * TILE_WIDTH + (columns - 1) * GAP
    height = (
        HEADER
        + MARGIN
        + sum(
            48 + math.ceil(len(group) / columns) * (card_height + GAP) for group in groups.values()
        )
    )
    font = ImageFont.truetype("DejaVuSans.ttf", 18)
    title_font = ImageFont.truetype("DejaVuSans.ttf", 28)
    counts = Counter(record["status"] for group in groups.values() for record in group)
    summary = " | ".join(f"{counts[status]} {label}" for status, (label, _) in STATUS_STYLE.items())
    sheets = {}
    for name, fraction in MOMENTS.items():
        canvas = Image.new("RGB", (width, height), "#101823")
        draw = ImageDraw.Draw(canvas)
        draw.text((MARGIN, 20), "OBJECT MASKS | FULL BATCH REVIEW", font=title_font, fill="white")
        draw.text(
            (MARGIN, 60),
            f"{len(seen)} clips | {summary} | sample: {fraction:.0%}",
            font=font,
            fill="#d0dae6",
        )
        draw.text((MARGIN, 96), "MASK: green = target", font=font, fill="#69d39c")
        draw.text((MARGIN + 290, 96), "blue = receiver", font=font, fill="#78aeff")
        draw.text(
            (MARGIN, 132),
            "Failed clips show RAW RGB. Frame samples are not frame-by-frame acceptance.",
            font=font,
            fill="#d0dae6",
        )
        sheets[name] = canvas

    samples = []
    task_y = HEADER
    for task, records in groups.items():
        task_counts = Counter(record["status"] for record in records)
        subtitle = " | ".join(
            f"{task_counts[status]} {label}" for status, (label, _) in STATUS_STYLE.items()
        )
        for canvas in sheets.values():
            ImageDraw.Draw(canvas).text(
                (MARGIN, task_y), f"{task} | {subtitle}", font=title_font, fill="white"
            )
        for index, record in enumerate(sorted(records, key=lambda item: item["episode"])):
            episode = record["episode"]
            raw = record["status"] == "failed"
            if raw:
                video = (
                    datasets_root
                    / task
                    / "videos/chunk-000/observation.images.cam_high"
                    / f"episode_{episode:06d}.mp4"
                )
            else:
                if not record.get("video"):
                    raise ValueError(f"missing overlay path: {task} episode {episode}")
                video = project_root / record["video"]
            if not video.is_file():
                raise FileNotFoundError(video)
            frame_ids = {
                name: int((record["frame_count"] - 1) * fraction)
                for name, fraction in MOMENTS.items()
            }
            frames = _decode_selected(video, set(frame_ids.values()))
            label, color = STATUS_STYLE[record["status"]]
            x = MARGIN + (index % columns) * (TILE_WIDTH + GAP)
            y = task_y + 48 + (index // columns) * (card_height + GAP)
            source_kind = "RAW RGB" if raw else "overlay"
            for name, frame_id in frame_ids.items():
                canvas = sheets[name]
                thumbnail = ImageOps.pad(frames[frame_id], (TILE_WIDTH, IMAGE_HEIGHT))
                canvas.paste(thumbnail, (x, y + LABEL_HEIGHT))
                draw = ImageDraw.Draw(canvas)
                draw.rectangle(
                    (x, y, x + TILE_WIDTH - 1, y + card_height - 1), outline=color, width=2
                )
                draw.text(
                    (x + 7, y + 3),
                    f"ep {episode:03d} | {record['active_arm'][0].upper()} | frame {frame_id}",
                    font=font,
                    fill="white",
                )
                draw.text(
                    (x + 7, y + LABEL_HEIGHT + IMAGE_HEIGHT + 3),
                    f"{label} | {source_kind} | {record['run_id'].rsplit('-', 1)[-1]}",
                    font=font,
                    fill=color,
                )
            samples.append(
                {
                    "task": task,
                    "episode": episode,
                    "run_id": record["run_id"],
                    "status": record["status"],
                    "video": str(video),
                    "source_kind": source_kind,
                    "frame_ids": frame_ids,
                }
            )
        task_y += 48 + math.ceil(len(records) / columns) * (card_height + GAP)

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, canvas in sheets.items():
        canvas.save(output_dir / f"{name}.jpg", quality=94, subsampling=0)
    manifest = {
        "results_index": str(results_path),
        "results_sha256": file_sha256(results_path),
        "counts": dict(counts),
        "sheets": [f"{name}.jpg" for name in sheets],
        "width": width,
        "height": height,
        "samples": samples,
    }
    ArtifactStore.write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--datasets-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.results.parent / "review_sheets"
    manifest = render_batch_sheets(args.results, args.datasets_root, output_dir)
    print(json.dumps({"output_dir": str(output_dir), "counts": manifest["counts"]}))


if __name__ == "__main__":
    main()
