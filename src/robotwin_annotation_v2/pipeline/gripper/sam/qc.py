"""Rendering, prompt construction, and Qwen selection for gripper seeds."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import cv2
import numpy as np
from PIL import Image, ImageDraw

from ....adapters.qwen_client import QwenCompletion, image_data_url
from ....models.loop_context import LoopContext
from ....models.mask_qc import MaskQCStatus
from ....models.timeline import LoopEvents
from ...mask_qc import parse_mask_qc_response
from ...object_mask.qc import MaskQCError
from .candidates import GripperSeedCandidate
from .geometry import ProjectedGripperRoi

NDArray = np.ndarray[Any, Any]

CYAN = np.asarray([15, 230, 185], dtype=np.uint8)
ORANGE = np.asarray([255, 126, 35], dtype=np.uint8)
BLUE = np.asarray([70, 105, 255], dtype=np.uint8)
YELLOW_RGB = (255, 240, 68)
RED_RGB = (255, 59, 59)


class GripperQwenClient(Protocol):
    model_id: str

    def health(self) -> dict[str, Any]: ...

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
    ) -> QwenCompletion: ...


@dataclass(frozen=True)
class GripperSeedQCResult:
    status: MaskQCStatus
    selected_candidate: str | None
    confidence: float | None
    reason: str
    candidates: tuple[GripperSeedCandidate, ...]
    model: str | None = None
    raw_response: str | None = None
    rendered_prompt: str | None = None
    health: dict[str, Any] | None = None
    forced_fallback: bool = False

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("gripper QC reason must be non-empty")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("gripper QC confidence must be between 0 and 1")
        if (self.status is MaskQCStatus.PASSED) != (self.selected_candidate is not None):
            raise ValueError("only passed gripper QC may select a candidate")
        if self.forced_fallback and self.selected_candidate is None:
            raise ValueError("a forced gripper fallback must select a candidate")

    @property
    def selected(self) -> GripperSeedCandidate | None:
        if self.selected_candidate is None:
            return None
        return next(
            candidate
            for candidate in self.candidates
            if candidate.candidate_id == self.selected_candidate
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "format_version": "robotwin_gripper_seed_qwen_qc_v1",
            "status": self.status.value,
            "selected_candidate": self.selected_candidate,
            "confidence": self.confidence,
            "reason": self.reason,
            "model": self.model,
            "raw_response": self.raw_response,
            "rendered_prompt": self.rendered_prompt,
            "health": self.health,
            "forced_fallback": self.forced_fallback,
            "candidates": [candidate.to_json() for candidate in self.candidates],
        }


def _outline(mask: NDArray, radius: int = 2) -> NDArray:
    value = np.asarray(mask, dtype=np.uint8)
    if not value.any():
        return value.astype(bool)
    kernel_size = 2 * radius + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    dilated = cv2.dilate(value, kernel, iterations=1)
    eroded = cv2.erode(value, np.ones((3, 3), dtype=np.uint8), iterations=1)
    outline: NDArray = dilated != eroded
    return outline


def render_gripper_candidate_panel(
    rgb: NDArray | Image.Image,
    candidate: GripperSeedCandidate,
    roi: ProjectedGripperRoi,
) -> Image.Image:
    source = np.asarray(
        rgb.convert("RGB") if isinstance(rgb, Image.Image) else rgb,
        dtype=np.uint8,
    )
    if source.shape != (*candidate.clean_mask.shape, 3):
        raise ValueError("candidate panel RGB shape does not match candidate mask")
    canvas = source.copy()
    canvas[_outline(candidate.target_removed)] = ORANGE
    canvas[_outline(candidate.receiver_removed)] = BLUE
    canvas[_outline(candidate.clean_mask)] = CYAN
    polygon = np.rint(roi.hull_pixels_xy).astype(np.int32).reshape(-1, 1, 2)
    if len(polygon) >= 3:
        cv2.polylines(canvas, [polygon], True, YELLOW_RGB, 1, cv2.LINE_AA)
    x, y = np.rint(roi.tcp_pixel_xy).astype(int)
    cv2.line(canvas, (x - 4, y), (x + 4, y), RED_RGB, 1, cv2.LINE_AA)
    cv2.line(canvas, (x, y - 4), (x, y + 4), RED_RGB, 1, cv2.LINE_AA)
    panel = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, panel.width, 20), fill=(0, 0, 0))
    text = (
        f"{candidate.candidate_id} f{candidate.frame_id} {candidate.phase} "
        f"{candidate.prompt_mode} clean={candidate.clean_pixels}"
    )
    draw.text((4, 4), text, fill=(255, 255, 255))
    return panel


def render_gripper_candidate_sheet(
    candidates: Sequence[GripperSeedCandidate],
    panels: Mapping[str, Image.Image],
    *,
    columns: int = 4,
) -> Image.Image:
    if columns < 1 or not candidates:
        raise ValueError("candidate sheet requires candidates and positive columns")
    panel_width, panel_height = next(iter(panels.values())).size
    rows = math.ceil(len(candidates) / columns)
    sheet = Image.new("RGB", (panel_width * columns, panel_height * rows), "#111111")
    for index, candidate in enumerate(candidates):
        sheet.paste(
            panels[candidate.candidate_id],
            ((index % columns) * panel_width, (index // columns) * panel_height),
        )
    return sheet


def _candidate_record(candidate: GripperSeedCandidate) -> str:
    dark = "null" if candidate.dark_fraction is None else f"{candidate.dark_fraction:.3f}"
    tcp = "null" if candidate.tcp_distance_px is None else f"{candidate.tcp_distance_px:.1f}"
    return (
        f"- {candidate.candidate_id}: frame={candidate.frame_id}, phase={candidate.phase}, "
        f"method={candidate.prompt_mode}, clean_pixels={candidate.clean_pixels}, "
        f"target_removed={candidate.target_removed_pixels}, "
        f"receiver_removed={candidate.receiver_removed_pixels}, "
        f"dark_fraction={dark}, tcp_distance_px={tcp}"
    )


def _fallback_candidate(candidates: Sequence[GripperSeedCandidate]) -> GripperSeedCandidate:
    """Select a deterministic least-bad seed when Qwen cannot decide."""

    if not candidates:
        raise ValueError("gripper fallback requires at least one candidate")

    def score(candidate: GripperSeedCandidate) -> tuple[float, ...]:
        return (
            float(candidate.basic_valid),
            float(candidate.clean_pixels),
            float(candidate.largest_component_fraction or 0.0),
            float(candidate.dark_fraction or 0.0),
            float(-candidate.component_count),
            float(-(candidate.tcp_distance_px or 1e9)),
        )

    return max(candidates, key=score)


def build_gripper_qwen_request(
    context: LoopContext,
    candidates: Sequence[GripperSeedCandidate],
    panels: Mapping[str, Image.Image],
    context_images: Mapping[int, Image.Image],
    *,
    prompt_template: str,
) -> tuple[str, list[dict[str, Any]]]:
    valid = tuple(candidate for candidate in candidates if candidate.basic_valid)
    if not valid:
        raise ValueError("Qwen request requires at least one valid gripper candidate")
    candidate_marker = "{candidate_panels}"
    context_marker = "{context_frames}"
    if prompt_template.count(candidate_marker) != 1:
        raise ValueError("gripper QC template must contain {candidate_panels} once")
    if prompt_template.count(context_marker) != 1:
        raise ValueError("gripper QC template must contain {context_frames} once")
    records = "\n".join(_candidate_record(candidate) for candidate in valid)
    context_ids = tuple(sorted(context_images))
    events = cast(LoopEvents, context.events)
    replacements = {
        "task_text": context.task_text,
        "active_arm": events.active_arm,
        "candidate_ids": ", ".join(candidate.candidate_id for candidate in valid),
        "candidate_records": records,
        "move_start": str(events.t_move_start),
        "close_start": str(events.t_close_start),
        "close_done": str(events.t_close_done),
        "open_start": str(events.t_open_start),
        "open_done": str(events.t_open_done),
    }
    partial = prompt_template
    for key, value in replacements.items():
        partial = partial.replace(f"{{{key}}}", value)
    candidate_prefix, remainder = partial.split(candidate_marker)
    candidate_suffix, context_suffix = remainder.split(context_marker)
    content: list[dict[str, Any]] = [{"type": "text", "text": candidate_prefix}]
    for candidate in valid:
        content.append(
            {
                "type": "text",
                "text": (
                    f"[candidate {candidate.candidate_id}; frame={candidate.frame_id}; "
                    f"phase={candidate.phase}; method={candidate.prompt_mode}]"
                ),
            }
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_data_url(panels[candidate.candidate_id])},
            }
        )
    content.append({"type": "text", "text": candidate_suffix})
    for frame_id in context_ids:
        content.append({"type": "text", "text": f"[raw context frame_id={frame_id}]"})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_data_url(context_images[frame_id])},
            }
        )
    content.append({"type": "text", "text": context_suffix})
    rendered = partial.replace(
        candidate_marker,
        "[candidate images: " + replacements["candidate_ids"] + "]",
    ).replace(
        context_marker,
        "[raw context images: " + ", ".join(map(str, context_ids)) + "]",
    )
    return rendered, [{"role": "user", "content": content}]


def run_gripper_seed_qc(
    context: LoopContext,
    candidates: Sequence[GripperSeedCandidate],
    panels: Mapping[str, Image.Image],
    context_images: Mapping[int, Image.Image],
    *,
    prompt_template_path: Path,
    client: GripperQwenClient,
    max_tokens: int = 200,
    max_attempts: int = 2,
    minimum_confidence: float = 0.70,
) -> GripperSeedQCResult:
    candidate_tuple = tuple(candidates)
    valid = tuple(candidate for candidate in candidate_tuple if candidate.basic_valid)

    def forced_result(
        reason: str,
        *,
        confidence: float | None = None,
        model: str | None = None,
        raw_response: str | None = None,
        rendered_prompt: str | None = None,
        health: dict[str, Any] | None = None,
    ) -> GripperSeedQCResult:
        pool = valid or candidate_tuple
        if not pool:
            return GripperSeedQCResult(
                status=MaskQCStatus.REJECTED,
                selected_candidate=None,
                confidence=confidence,
                reason=reason,
                candidates=candidate_tuple,
                model=model,
                raw_response=raw_response,
                rendered_prompt=rendered_prompt,
                health=health,
            )
        selected = _fallback_candidate(pool)
        return GripperSeedQCResult(
            status=MaskQCStatus.PASSED,
            selected_candidate=selected.candidate_id,
            confidence=confidence,
            reason=f"forced fallback candidate {selected.candidate_id}; {reason}",
            candidates=candidate_tuple,
            model=model,
            raw_response=raw_response,
            rendered_prompt=rendered_prompt,
            health=health,
            forced_fallback=True,
        )

    if not valid:
        return forced_result("all_gripper_candidates_failed_basic_checks")
    if max_tokens < 1 or max_attempts < 1:
        raise ValueError("gripper QC token and attempt limits must be positive")
    if not 0.0 <= minimum_confidence <= 1.0:
        raise ValueError("gripper QC minimum confidence must be in [0, 1]")
    if not prompt_template_path.is_file():
        return forced_result(f"gripper QC prompt is missing: {prompt_template_path}")
    try:
        health = client.health()
    except Exception as exc:  # noqa: BLE001 - external Qwen client boundary
        return forced_result(f"gripper QC health failed: {exc}")
    rendered, messages = build_gripper_qwen_request(
        context,
        valid,
        panels,
        context_images,
        prompt_template=prompt_template_path.read_text(encoding="utf-8"),
    )
    completion: QwenCompletion | None = None
    request_error: Exception | None = None
    for _attempt in range(max_attempts):
        try:
            completion = client.complete(messages, max_tokens=max_tokens)
            break
        except Exception as exc:  # noqa: BLE001 - external Qwen client boundary
            request_error = exc
    if completion is None:
        assert request_error is not None
        return forced_result(
            f"gripper QC request failed after {max_attempts} attempt(s): {request_error}",
            rendered_prompt=rendered,
            health=health,
        )
    try:
        decision, selected, confidence, reason = parse_mask_qc_response(
            completion.content,
            candidate_ids=tuple(candidate.candidate_id for candidate in valid),
        )
    except MaskQCError as exc:
        return forced_result(
            str(exc),
            model=completion.model,
            raw_response=completion.content,
            rendered_prompt=rendered,
            health=health,
        )
    if decision == "reject_all":
        return forced_result(
            f"Qwen decision was reject_all: {reason}",
            confidence=confidence,
            model=completion.model,
            raw_response=completion.content,
            rendered_prompt=rendered,
            health=health,
        )
    if decision == "ambiguous":
        return forced_result(
            f"Qwen decision was ambiguous: {reason}",
            confidence=confidence,
            model=completion.model,
            raw_response=completion.content,
            rendered_prompt=rendered,
            health=health,
        )
    if confidence < minimum_confidence:
        return forced_result(
            f"Qwen confidence {confidence:.3f} is below minimum "
            f"{minimum_confidence:.3f}: {reason}",
            confidence=confidence,
            model=completion.model,
            raw_response=completion.content,
            rendered_prompt=rendered,
            health=health,
        )
    return GripperSeedQCResult(
        status=MaskQCStatus.PASSED,
        selected_candidate=selected,
        confidence=confidence,
        reason=reason,
        candidates=candidate_tuple,
        model=completion.model,
        raw_response=completion.content,
        rendered_prompt=rendered,
        health=health,
        forced_fallback=False,
    )


__all__ = [
    "GripperQwenClient",
    "GripperSeedQCResult",
    "build_gripper_qwen_request",
    "render_gripper_candidate_panel",
    "render_gripper_candidate_sheet",
    "run_gripper_seed_qc",
]
