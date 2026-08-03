#!/usr/bin/env python3
"""Build compact early/late contact sheets from rendered tracking videos."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import av
from PIL import Image, ImageDraw


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--columns", type=int, default=4)
    return parser.parse_args()


def _decode_selected(video_path: Path, frame_ids: set[int]) -> dict[int, Image.Image]:
    selected: dict[int, Image.Image] = {}
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame_id, frame in enumerate(container.decode(stream)):
            if frame_id in frame_ids:
                selected[frame_id] = Image.fromarray(frame.to_ndarray(format="rgb24"))
            if len(selected) == len(frame_ids):
                break
    missing = sorted(frame_ids - set(selected))
    if missing:
        raise ValueError(f"video is missing requested frames {missing}: {video_path}")
    return selected


def build_sheets(render_manifest: Path, output_dir: Path, columns: int) -> list[Path]:
    if columns < 1:
        raise ValueError("columns must be positive")
    render = json.loads(render_manifest.read_text(encoding="utf-8"))
    video_root = render_manifest.parent
    pages: dict[str, list[Image.Image]] = {
        "target_early": [],
        "target_late": [],
        "receiver_early": [],
        "receiver_late": [],
    }
    for episode in render["episodes"]:
        episode_index = int(episode["episode_index"])
        source_dir = Path(episode["source_masks"]).parent
        run_manifest = json.loads(
            (source_dir / "run_manifest.json").read_text(encoding="utf-8")
        )
        wanted: dict[str, int] = {}
        statuses: dict[str, str] = {}
        for role_record in run_manifest["roles"]:
            role = str(role_record["role"])
            start, end = (int(value) for value in role_record["output_window"])
            wanted[f"{role}_early"] = start + (end - start) // 4
            wanted[f"{role}_late"] = end
            statuses[role] = str(role_record["status"])
        frames = _decode_selected(
            video_root / episode["output_video"],
            set(wanted.values()),
        )
        for page, frame_id in wanted.items():
            role = page.removesuffix("_early").removesuffix("_late")
            image = frames[frame_id].copy()
            draw = ImageDraw.Draw(image)
            label = f"ep {episode_index} f{frame_id} {statuses[role]}"
            draw.rectangle((0, 0, 210, 20), fill=(0, 0, 0))
            draw.text((3, 3), label, fill=(255, 255, 255))
            pages[page].append(image)

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for name, images in pages.items():
        if not images:
            continue
        width, height = images[0].size
        rows = math.ceil(len(images) / columns)
        sheet = Image.new("RGB", (width * columns, height * rows), color=(20, 20, 20))
        for index, image in enumerate(images):
            sheet.paste(image, ((index % columns) * width, (index // columns) * height))
        output_path = output_dir / f"{name}.jpg"
        sheet.save(output_path, quality=92)
        outputs.append(output_path)
    return outputs


def main() -> None:
    args = _parse_args()
    outputs = build_sheets(
        args.render_manifest.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
        args.columns,
    )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
