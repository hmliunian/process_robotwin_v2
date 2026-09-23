"""Conservative single-arm screening using time-labeled UMI head and wrist previews."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from .qwen_client import OpenAICompatibleQwenClient, image_data_url


@dataclass(frozen=True)
class UmiPreview:
    path: Path
    label: str
    time_ms: int


def load_umi_previews(source_dir: Path) -> tuple[UmiPreview, ...]:
    """Require all three views; retain all head frames and four frames per wrist."""

    manifest = json.loads((source_dir / "frames/manifest.json").read_text())
    previews = []
    for view, count in (("head", 8), ("left_wrist", 4), ("right_wrist", 4)):
        lanes = sorted(
            (lane for lane in manifest["lanes"] if lane["laneKey"].split("/")[0] == view),
            key=lambda lane: lane["laneKey"],
        )
        if not lanes or len(lanes[0]["frames"]) < 4:
            raise ValueError(f"insufficient {view} preview evidence")
        lane = lanes[0]
        frames = lane["frames"]
        indices = sorted({round(i * (len(frames) - 1) / (count - 1)) for i in range(count)})
        for index in indices:
            frame = frames[index]
            path = source_dir / "frames" / Path(*Path(frame["key"]).parts[-2:])
            if not path.resolve().is_relative_to((source_dir / "frames").resolve()):
                raise ValueError("preview path escapes source frames")
            if not path.is_file():
                raise FileNotFoundError(path)
            time_ms = int(frame["timeMs"])
            previews.append(UmiPreview(path, f"{lane['laneKey']}@{time_ms}ms", time_ms))
    return tuple(previews)


def preview_identity(previews: tuple[UmiPreview, ...]) -> list[dict[str, Any]]:
    return [
        {
            "label": frame.label,
            "time_ms": frame.time_ms,
            "sha256": hashlib.sha256(frame.path.read_bytes()).hexdigest(),
        }
        for frame in previews
    ]


def parse_arm_review(content: str, previews: tuple[UmiPreview, ...]) -> dict[str, Any]:
    """Reject ungrounded, incomplete, or non-finite model answers."""

    value = json.loads(content)
    if not isinstance(value, dict) or value.get("decision") not in {
        "left",
        "right",
        "bimanual",
        "uncertain",
        "no_action",
    }:
        raise ValueError("invalid single-arm decision")
    confidence = value.get("confidence")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not 0 <= confidence <= 1
    ):
        raise ValueError("invalid single-arm confidence")
    labels = {frame.label: frame.time_ms for frame in previews}
    evidence = value.get("evidence_frames")
    if (
        not isinstance(evidence, list)
        or any(not isinstance(label, str) or label not in labels for label in evidence)
        or len({labels[label] for label in evidence}) < 2
    ):
        raise ValueError("single-arm review requires evidence at two distinct times")
    for key in ("left_hand", "right_hand", "reason"):
        if not isinstance(value.get(key), str) or not value[key].strip():
            raise ValueError(f"single-arm review missing {key}")
    return value


def screen_umi_episode(
    client: OpenAICompatibleQwenClient,
    *,
    previews: tuple[UmiPreview, ...],
    prompt: str,
    task_text: str,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [
        {"type": "text", "text": prompt.replace("{task_text}", task_text)}
    ]
    for frame in previews:
        with Image.open(frame.path) as image:
            rgb = image.convert("RGB")
            side = 512 if frame.label.startswith("head/") else 384
            rgb.thumbnail((side, side))
            content.extend(
                [
                    {"type": "text", "text": frame.label},
                    {"type": "image_url", "image_url": {"url": image_data_url(rgb)}},
                ]
            )
    answer = client.complete([{"role": "user", "content": content}], max_tokens=900)
    if answer.finish_reason not in {"stop", None}:
        raise ValueError(f"incomplete single-arm response: {answer.finish_reason}")
    return {
        **parse_arm_review(answer.content, previews),
        "review_source": "qwen_multiframe_screen",
        "model": answer.model,
        "raw_response": answer.content,
    }
