"""Strict Qwen contract for open-set role bounding-box localization."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Literal

from PIL import Image

from ..adapters.qwen_client import image_data_url

_PROMPT_FIELDS = frozenset({"task", "task_text", "episode_id", "role", "seed_frame_id"})
_PLACEHOLDER_PATTERN = re.compile(r"\{([a-z][a-z0-9_]*)\}")
_RESPONSE_FIELDS = frozenset({"status", "bbox_xyxy", "confidence", "reason"})

BboxStatus = Literal["ok", "ambiguous", "not_visible"]


class BboxLocalizationError(RuntimeError):
    """The bbox-localization prompt or Qwen response violated its contract."""


@dataclass(frozen=True)
class BboxLocalization:
    status: BboxStatus
    bbox_xyxy: tuple[float, float, float, float] | None
    confidence: float
    reason: str

    def to_json(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "bbox_xyxy": None if self.bbox_xyxy is None else list(self.bbox_xyxy),
            "confidence": self.confidence,
            "reason": self.reason,
        }


def _reject_json_constant(value: str) -> None:
    raise BboxLocalizationError(f"Qwen response contains non-finite JSON number {value}")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise BboxLocalizationError(f"Qwen response contains duplicate field {key!r}")
        output[key] = value
    return output


def _json_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BboxLocalizationError(f"{field} must be a JSON number")
    converted = float(value)
    if not math.isfinite(converted):
        raise BboxLocalizationError(f"{field} must be finite")
    return converted


def parse_bbox_localization(raw_response: str) -> BboxLocalization:
    """Parse one strict response without repairing or clamping Qwen coordinates."""

    text = raw_response.strip()
    if not text:
        raise BboxLocalizationError("Qwen response is empty")
    if text.startswith("```"):
        raise BboxLocalizationError("Qwen response must be raw JSON without a Markdown fence")
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except BboxLocalizationError:
        raise
    except json.JSONDecodeError as exc:
        raise BboxLocalizationError(f"Qwen response is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise BboxLocalizationError("Qwen response must be a JSON object")
    fields = frozenset(payload)
    if fields != _RESPONSE_FIELDS:
        raise BboxLocalizationError(
            "Qwen response fields do not match schema; "
            f"missing={sorted(_RESPONSE_FIELDS - fields)}, "
            f"extra={sorted(fields - _RESPONSE_FIELDS)}"
        )

    status = payload["status"]
    if status not in {"ok", "ambiguous", "not_visible"}:
        raise BboxLocalizationError("status must be ok, ambiguous, or not_visible")

    confidence = _json_number(payload["confidence"], field="confidence")
    if not 0.0 <= confidence <= 1.0:
        raise BboxLocalizationError("confidence must be in [0, 1]")

    reason = payload["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise BboxLocalizationError("reason must be a non-empty string")
    normalized_reason = " ".join(reason.split())

    raw_box = payload["bbox_xyxy"]
    bbox: tuple[float, float, float, float] | None
    if status == "ok":
        if not isinstance(raw_box, list) or len(raw_box) != 4:
            raise BboxLocalizationError("ok status requires bbox_xyxy with exactly four numbers")
        values = (
            _json_number(raw_box[0], field="bbox_xyxy[0]"),
            _json_number(raw_box[1], field="bbox_xyxy[1]"),
            _json_number(raw_box[2], field="bbox_xyxy[2]"),
            _json_number(raw_box[3], field="bbox_xyxy[3]"),
        )
        x0, y0, x1, y1 = values
        if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
            raise BboxLocalizationError(
                "bbox_xyxy must be an ordered normalized box in [0, 1]; "
                "coordinates are rejected rather than clamped"
            )
        bbox = values
    else:
        if raw_box is not None:
            raise BboxLocalizationError(f"{status} status requires bbox_xyxy=null")
        bbox = None

    return BboxLocalization(
        status=status,
        bbox_xyxy=bbox,
        confidence=confidence,
        reason=normalized_reason,
    )


def render_bbox_prompt(
    template: str,
    *,
    task: str,
    task_text: str,
    episode_id: str,
    role: str,
    seed_frame_id: int,
) -> str:
    placeholders = frozenset(_PLACEHOLDER_PATTERN.findall(template))
    missing = sorted(_PROMPT_FIELDS - placeholders)
    unknown = sorted(placeholders - _PROMPT_FIELDS)
    if missing or unknown:
        raise BboxLocalizationError(
            f"bbox prompt placeholders do not match contract; missing={missing}, unknown={unknown}"
        )
    replacements = {
        "task": task,
        "task_text": task_text,
        "episode_id": episode_id,
        "role": role,
        "seed_frame_id": str(seed_frame_id),
    }
    rendered = template
    for field, value in replacements.items():
        rendered = rendered.replace(f"{{{field}}}", value)
    return rendered


def build_bbox_messages(prompt: str, image: Image.Image) -> list[dict[str, Any]]:
    if not prompt.strip():
        raise BboxLocalizationError("rendered bbox prompt must be non-empty")
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_url(image)},
                },
            ],
        }
    ]
