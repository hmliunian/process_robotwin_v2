"""Stage 3.5: compare SAM3 seed-mask candidates with Qwen and basic QC."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
from PIL import Image, ImageDraw

from ..adapters.artifact_store import ArtifactStore
from ..adapters.qwen_client import QwenCompletion, image_data_url
from ..config import MaskConfig
from ..models import (
    LoopContext,
    MaskCandidateInfo,
    MaskQCResult,
    MaskQCStatus,
    RoleMaskQC,
    RoleSemanticPlan,
    SemanticPlan,
    SemanticStatus,
)
from ..models.loop_context import RoleName


_CANDIDATE_MARKER = "{candidate_panels}"
_CONTEXT_MARKER = "{context_frames}"
_PLACEHOLDER_PATTERN = re.compile(r"\{([a-z][a-z0-9_]*)\}")
_QC_FIELDS = frozenset(
    {"decision", "selected_candidate", "confidence", "reason"}
)
_PANEL_COLORS = (
    (232, 67, 55),
    (35, 116, 224),
    (238, 176, 38),
    (153, 78, 189),
    (20, 160, 160),
    (220, 90, 150),
)
_ROLES: tuple[RoleName, RoleName] = ("target", "receiver")


class MaskQCError(RuntimeError):
    """The candidate-mask QC request or response violated its contract."""

    def __init__(
        self,
        message: str,
        *,
        rendered_prompt: str | None = None,
        raw_response: str | None = None,
    ) -> None:
        super().__init__(message)
        self.rendered_prompt = rendered_prompt
        self.raw_response = raw_response


class MaskQCBackend(Protocol):
    def text_query_masks(
        self,
        resource_path: Path,
        texts: Sequence[str],
        *,
        frame_id: int,
        frame_count: int,
        frame_shape: tuple[int, int],
    ) -> dict[str, np.ndarray]: ...


class MaskQCClient(Protocol):
    model_id: str

    def health(self) -> dict[str, Any]: ...

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
    ) -> QwenCompletion: ...


@dataclass(frozen=True)
class _Candidate:
    candidate_id: str
    query_field: str
    query: str
    mask: np.ndarray
    info: MaskCandidateInfo


@dataclass(frozen=True)
class _RoleExecution:
    report: RoleMaskQC
    candidates: tuple[_Candidate, ...]
    panels: tuple[Image.Image, ...] = ()


def _normalize_text(value: str) -> str:
    return " ".join(value.split())


def _component_count(mask: np.ndarray) -> int:
    """Count 4-connected components without requiring OpenCV/scipy."""

    remaining = np.asarray(mask, dtype=bool).copy()
    if remaining.ndim != 2:
        raise ValueError("mask must be 2-D")
    count = 0
    height, width = remaining.shape
    while remaining.any():
        row, column = np.argwhere(remaining)[0]
        count += 1
        stack = [(int(row), int(column))]
        remaining[row, column] = False
        while stack:
            current_row, current_column = stack.pop()
            for next_row, next_column in (
                (current_row - 1, current_column),
                (current_row + 1, current_column),
                (current_row, current_column - 1),
                (current_row, current_column + 1),
            ):
                if (
                    0 <= next_row < height
                    and 0 <= next_column < width
                    and remaining[next_row, next_column]
                ):
                    remaining[next_row, next_column] = False
                    stack.append((next_row, next_column))
    return count


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep the largest 4-connected component of a binary proposal."""

    remaining = np.asarray(mask, dtype=bool).copy()
    if remaining.ndim != 2:
        raise ValueError("mask must be 2-D")
    largest: list[tuple[int, int]] = []
    height, width = remaining.shape
    while remaining.any():
        row, column = np.argwhere(remaining)[0]
        component: list[tuple[int, int]] = []
        stack = [(int(row), int(column))]
        remaining[row, column] = False
        while stack:
            current_row, current_column = stack.pop()
            component.append((current_row, current_column))
            for next_row, next_column in (
                (current_row - 1, current_column),
                (current_row + 1, current_column),
                (current_row, current_column - 1),
                (current_row, current_column + 1),
            ):
                if (
                    0 <= next_row < height
                    and 0 <= next_column < width
                    and remaining[next_row, next_column]
                ):
                    remaining[next_row, next_column] = False
                    stack.append((next_row, next_column))
        if len(component) > len(largest):
            largest = component
    output = np.zeros_like(remaining)
    for row, column in largest:
        output[row, column] = True
    return output


def _blue_planar_region(seed_image: Image.Image, frame_shape: tuple[int, int]) -> np.ndarray:
    """Build a coordinate-free proposal for a saturated blue receiver region."""

    rgb = np.asarray(seed_image.convert("RGB"), dtype=np.int16)
    if rgb.shape[:2] != frame_shape:
        raise MaskQCError(
            f"seed RGB shape {rgb.shape[:2]} does not match expected {frame_shape}"
        )
    red, green, blue = (rgb[..., index] for index in range(3))
    saturated_blue = (
        (blue >= 80)
        & ((blue - red) >= 30)
        & ((blue - green) >= 20)
    )
    return _largest_component(saturated_blue)


def _dilate(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    value = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return value.copy()
    padded = np.pad(value, radius)
    output = np.zeros_like(value)
    height, width = value.shape
    for row_offset in range(2 * radius + 1):
        for column_offset in range(2 * radius + 1):
            output |= padded[
                row_offset : row_offset + height,
                column_offset : column_offset + width,
            ]
    return output


def _panel_image(
    seed_image: Image.Image,
    candidate: _Candidate,
    *,
    scale: int = 2,
) -> Image.Image:
    """Render a contour-only candidate panel so object appearance stays visible."""

    rgb = seed_image.convert("RGB")
    width, height = rgb.size
    if candidate.mask.shape != (height, width):
        raise MaskQCError(
            f"candidate {candidate.candidate_id} mask shape {candidate.mask.shape} "
            f"does not match image {(height, width)}"
        )
    array = np.asarray(rgb, dtype=np.uint8).copy()
    mask = np.asarray(candidate.mask, dtype=bool)
    outline = _dilate(mask, 3) & ~mask
    color = np.asarray(
        _PANEL_COLORS[(ord(candidate.candidate_id) - ord("A")) % len(_PANEL_COLORS)],
        dtype=np.uint8,
    )
    array[outline] = color
    # A thin inner edge makes tiny masks visible without hiding their texture.
    inner = mask & ~np.roll(mask, 1, axis=0)
    array[inner] = color
    panel = Image.fromarray(array, mode="RGB")
    if scale != 1:
        panel = panel.resize((width * scale, height * scale), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(panel)
    band_height = max(24, 16 * scale)
    draw.rectangle((0, 0, panel.width, band_height), fill=(0, 0, 0))
    draw.text(
        (6, 4),
        f"candidate {candidate.candidate_id}",
        fill=(255, 255, 255),
    )
    return panel


def _candidate_info(
    candidate_id: str,
    query_field: str,
    query: str,
    mask: np.ndarray,
    *,
    min_area_fraction: float,
    max_area_fraction: float,
    duplicate_of: str | None = None,
) -> MaskCandidateInfo:
    value = np.asarray(mask, dtype=bool)
    if value.ndim != 2:
        raise MaskQCError(f"candidate {candidate_id} mask must be 2-D")
    area_fraction = float(value.mean())
    nonempty = bool(value.any())
    components = _component_count(value) if nonempty else 0
    if not nonempty:
        reason = "empty_seed_mask"
    elif area_fraction < min_area_fraction:
        reason = "seed_mask_too_small"
    elif area_fraction > max_area_fraction:
        reason = "seed_mask_too_large"
    elif duplicate_of is not None:
        reason = "duplicate_candidate_mask"
    else:
        reason = None
    return MaskCandidateInfo(
        candidate_id=candidate_id,
        query_field=query_field,
        query=query,
        nonempty=nonempty,
        area_fraction=area_fraction,
        component_count=components,
        basic_valid=reason is None,
        basic_reason=reason,
        duplicate_of=duplicate_of,
    )


def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    union = np.asarray(first, dtype=bool) | np.asarray(second, dtype=bool)
    if not union.any():
        return 1.0
    intersection = np.asarray(first, dtype=bool) & np.asarray(second, dtype=bool)
    return float(intersection.sum() / union.sum())


def _context_items(
    context: LoopContext,
    role: RoleName,
    seed_frame_id: int,
    context_images: Mapping[int, Image.Image],
    *,
    limit: int = 2,
) -> tuple[tuple[int, Image.Image], ...]:
    eligible = [
        frame.frame_id
        for frame in context.semantic_frames
        if frame.frame_id != seed_frame_id
        and role in frame.eligible_roles
        and frame.frame_id in context_images
    ]
    if len(eligible) > limit:
        indices = tuple(
            dict.fromkeys(
                int(round(value))
                for value in np.linspace(0, len(eligible) - 1, num=limit)
            )
        )
        eligible = [eligible[index] for index in indices]
    return tuple((frame_id, context_images[frame_id]) for frame_id in eligible)


def _decode_response(raw_response: str) -> dict[str, Any]:
    text = raw_response.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) < 3 or lines[-1].strip() != "```":
            raise MaskQCError("mask QC response contains an incomplete JSON fence")
        if lines[0].strip() not in {"```", "```json"}:
            raise MaskQCError("mask QC response fence must contain JSON")
        text = "\n".join(lines[1:-1]).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise MaskQCError(f"mask QC response is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise MaskQCError("mask QC response must be a JSON object")
    fields = frozenset(payload)
    if fields != _QC_FIELDS:
        raise MaskQCError(
            "mask QC response fields do not match schema; "
            f"missing={sorted(_QC_FIELDS - fields)}, extra={sorted(fields - _QC_FIELDS)}"
        )
    return payload


def parse_mask_qc_response(
    raw_response: str,
    *,
    candidate_ids: Sequence[str],
) -> tuple[str, str | None, float, str]:
    """Parse and validate Qwen's small candidate-selection response."""

    payload = _decode_response(raw_response)
    decision = payload["decision"]
    if decision not in {"accept", "reject_all", "ambiguous"}:
        raise MaskQCError("mask QC decision must be accept, reject_all, or ambiguous")
    selected = payload["selected_candidate"]
    if selected is not None:
        if not isinstance(selected, str):
            raise MaskQCError("selected_candidate must be a string or null")
        selected = selected.strip().upper()
        if selected not in set(candidate_ids):
            raise MaskQCError(
                f"selected_candidate={selected!r} is not one of {tuple(candidate_ids)}"
            )
    if decision == "accept" and selected is None:
        raise MaskQCError("accept decision requires selected_candidate")
    if decision != "accept" and selected is not None:
        raise MaskQCError("reject_all/ambiguous decisions must not select a candidate")
    confidence = payload["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise MaskQCError("confidence must be a number")
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise MaskQCError("confidence must be finite and between 0 and 1")
    reason = payload["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise MaskQCError("reason must be a non-empty string")
    return decision, selected, confidence, _normalize_text(reason)


def _render_request(
    context: LoopContext,
    *,
    role: RoleName,
    seed_frame_id: int,
    seed_image: Image.Image,
    candidates: Sequence[_Candidate],
    context_images: Mapping[int, Image.Image],
    template: str,
) -> tuple[str, list[dict[str, Any]], tuple[Image.Image, ...]]:
    if template.count(_CANDIDATE_MARKER) != 1:
        raise MaskQCError(
            f"mask QC template must contain {_CANDIDATE_MARKER!r} exactly once"
        )
    if template.count(_CONTEXT_MARKER) != 1:
        raise MaskQCError(
            f"mask QC template must contain {_CONTEXT_MARKER!r} exactly once"
        )
    candidate_ids = ", ".join(candidate.candidate_id for candidate in candidates)
    replacements = {
        "task_text": context.task_text,
        "role": role,
        "seed_frame_id": str(seed_frame_id),
        "candidate_ids": candidate_ids,
        "close_done": str(context.events.t_close_done),
        "open_start": str(context.events.t_open_start),
        "open_done": str(context.events.t_open_done),
    }
    placeholders = set(_PLACEHOLDER_PATTERN.findall(template))
    unknown = sorted(
        placeholders
        - set(replacements)
        - {"candidate_panels", "context_frames"}
    )
    if unknown:
        raise MaskQCError(f"unknown mask QC template placeholder(s): {', '.join(unknown)}")
    rendered = template
    for key, value in replacements.items():
        rendered = rendered.replace(f"{{{key}}}", value)
    rendered = rendered.replace(_CANDIDATE_MARKER, f"[candidate images: {candidate_ids}]")
    context_items = _context_items(
        context,
        role,
        seed_frame_id,
        context_images,
    )
    context_ids = ", ".join(str(frame_id) for frame_id, _image in context_items) or "none"
    rendered = rendered.replace(_CONTEXT_MARKER, f"[context images: {context_ids}]")

    partially = template
    for key, value in replacements.items():
        partially = partially.replace(f"{{{key}}}", value)
    candidate_prefix, candidate_suffix = partially.split(_CANDIDATE_MARKER)
    candidate_text, context_suffix = candidate_suffix.split(_CONTEXT_MARKER)
    content: list[dict[str, Any]] = [
        {"type": "text", "text": candidate_prefix},
    ]
    panels = tuple(_panel_image(seed_image, candidate) for candidate in candidates)
    for panel in panels:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_data_url(panel)},
            }
        )
    content.append({"type": "text", "text": candidate_text})
    # First/last context frames preserve the motion evidence while bounding image tokens.
    for frame_id, image in context_items:
        frame = next(
            semantic_frame
            for semantic_frame in context.semantic_frames
            if semantic_frame.frame_id == frame_id
        )
        content.append(
            {
                "type": "text",
                "text": (
                    f"[context frame_id={frame_id}; purpose={frame.purpose.value}; "
                    f"eligible_roles={','.join(frame.eligible_roles)}]"
                ),
            },
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_data_url(image.convert("RGB"))},
            }
        )
    content.append({"type": "text", "text": context_suffix})
    return rendered, [{"role": "user", "content": content}], panels


def _error_report(
    role: RoleName,
    status: MaskQCStatus,
    reason: str,
    *,
    model: str | None = None,
    raw_response: str | None = None,
    rendered_prompt: str | None = None,
    candidates: tuple[MaskCandidateInfo, ...] = (),
) -> RoleMaskQC:
    return RoleMaskQC(
        role=role,
        status=status,
        selected_candidate=None,
        selected_query_field=None,
        selected_query=None,
        confidence=None,
        reason=_normalize_text(reason),
        candidates=candidates,
        model=model,
        raw_response=raw_response,
        rendered_prompt=rendered_prompt,
    )


def _run_role_qc(
    context: LoopContext,
    role: RoleName,
    semantic: RoleSemanticPlan,
    *,
    backend: MaskQCBackend,
    resource_path: Path,
    seed_image: Image.Image,
    context_images: Mapping[int, Image.Image],
    frame_shape: tuple[int, int],
    mask_config: MaskConfig,
    client: MaskQCClient,
) -> _RoleExecution:
    if semantic.status is SemanticStatus.NO_CLEAR_SEED:
        return _RoleExecution(
            _error_report(role, MaskQCStatus.REJECTED, "semantic_plan_no_clear_seed"),
            (),
        )
    assert semantic.seed_frame_id is not None
    assert semantic.query_bank is not None
    ordered_fields = semantic.query_bank.recommended_order
    blue_prior = np.zeros(frame_shape, dtype=bool)
    if role == "receiver" and any(
        "blue" in query.split()
        for query in (
            getattr(semantic.query_bank, field)
            for field in ordered_fields
        )
        if query is not None
    ):
        blue_prior = _blue_planar_region(seed_image, frame_shape)
    reserve_prior = bool(blue_prior.any()) and mask_config.qc_max_candidates >= 2
    text_limit = mask_config.qc_max_candidates - int(reserve_prior)
    fields = ordered_fields[:text_limit]
    queries = tuple(getattr(semantic.query_bank, field) for field in fields)
    if any(query is None for query in queries):
        raise MaskQCError(f"{role} query bank contains an ordered null candidate")
    query_masks = backend.text_query_masks(
        resource_path,
        queries,  # type: ignore[arg-type]
        frame_id=semantic.seed_frame_id,
        frame_count=context.frame_count,
        frame_shape=frame_shape,
    )
    candidates: list[_Candidate] = []
    infos: list[MaskCandidateInfo] = []
    for index, (field, query_value) in enumerate(zip(fields, queries, strict=True)):
        assert query_value is not None
        query = query_value
        candidate_id = chr(ord("A") + index)
        mask = np.asarray(
            query_masks[query],
            dtype=bool,
        )
        if mask.shape != frame_shape:
            raise MaskQCError(
                f"{role} candidate {candidate_id} has shape {mask.shape}, expected {frame_shape}"
            )
        info = _candidate_info(
            candidate_id,
            field,
            query,
            mask,
            min_area_fraction=mask_config.qc_min_area_fraction,
            max_area_fraction=mask_config.qc_max_area_fraction,
        )
        if info.basic_valid:
            duplicate = next(
                (
                    previous
                    for previous in candidates
                    if previous.info.basic_valid
                    and _mask_iou(previous.mask, mask)
                    >= mask_config.qc_duplicate_iou_threshold
                ),
                None,
            )
            if duplicate is not None:
                info = _candidate_info(
                    candidate_id,
                    field,
                    query,
                    mask,
                    min_area_fraction=mask_config.qc_min_area_fraction,
                    max_area_fraction=mask_config.qc_max_area_fraction,
                    duplicate_of=duplicate.candidate_id,
                )
        candidates.append(_Candidate(candidate_id, field, query, mask, info))
        infos.append(info)
    if reserve_prior:
        candidate_id = chr(ord("A") + len(candidates))
        query = "blue planar region"
        info = _candidate_info(
            candidate_id,
            "blue_region_prior",
            query,
            blue_prior,
            min_area_fraction=mask_config.qc_min_area_fraction,
            max_area_fraction=mask_config.qc_max_area_fraction,
        )
        if info.basic_valid:
            duplicate = next(
                (
                    previous
                    for previous in candidates
                    if previous.info.basic_valid
                    and _mask_iou(previous.mask, blue_prior)
                    >= mask_config.qc_duplicate_iou_threshold
                ),
                None,
            )
            if duplicate is not None:
                info = _candidate_info(
                    candidate_id,
                    "blue_region_prior",
                    query,
                    blue_prior,
                    min_area_fraction=mask_config.qc_min_area_fraction,
                    max_area_fraction=mask_config.qc_max_area_fraction,
                    duplicate_of=duplicate.candidate_id,
                )
        candidates.append(
            _Candidate(candidate_id, "blue_region_prior", query, blue_prior, info)
        )
        infos.append(info)
    candidate_tuple = tuple(candidates)
    info_tuple = tuple(infos)
    valid = tuple(candidate for candidate in candidate_tuple if candidate.info.basic_valid)
    panels = tuple(_panel_image(seed_image, candidate) for candidate in candidate_tuple)
    if not valid:
        return _RoleExecution(
            _error_report(
                role,
                MaskQCStatus.REJECTED,
                "all_candidate_masks_failed_basic_checks",
                candidates=info_tuple,
            ),
            candidate_tuple,
            panels,
        )
    template_path = mask_config.qc_prompt_template
    if template_path is None or not template_path.is_file():
        return _RoleExecution(
            _error_report(
                role,
                MaskQCStatus.ERROR,
                f"mask QC prompt template is missing: {template_path}",
                candidates=info_tuple,
            ),
            candidate_tuple,
            panels,
        )
    template = template_path.read_text(encoding="utf-8")
    rendered_prompt, messages, _panels = _render_request(
        context,
        role=role,
        seed_frame_id=semantic.seed_frame_id,
        seed_image=seed_image,
        candidates=valid,
        context_images=context_images,
        template=template,
    )
    completion: QwenCompletion | None = None
    request_error: Exception | None = None
    for _attempt in range(mask_config.qc_max_attempts):
        try:
            completion = client.complete(messages, max_tokens=mask_config.qc_max_tokens)
            break
        except Exception as exc:
            request_error = exc
    if completion is None:
        assert request_error is not None
        return _RoleExecution(
            _error_report(
                role,
                MaskQCStatus.ERROR,
                (
                    f"mask QC request failed after {mask_config.qc_max_attempts} "
                    f"attempt(s): {request_error}"
                ),
                model=getattr(client, "model_id", None),
                rendered_prompt=rendered_prompt,
                candidates=info_tuple,
            ),
            candidate_tuple,
            panels,
        )
    try:
        decision, selected_id, confidence, reason = parse_mask_qc_response(
            completion.content,
            candidate_ids=tuple(candidate.candidate_id for candidate in valid),
        )
    except MaskQCError as exc:
        return _RoleExecution(
            _error_report(
                role,
                MaskQCStatus.ERROR,
                str(exc),
                model=completion.model,
                raw_response=completion.content,
                rendered_prompt=rendered_prompt,
                candidates=info_tuple,
            ),
            candidate_tuple,
            panels,
        )
    if decision == "reject_all":
        status = MaskQCStatus.REJECTED
    elif decision == "ambiguous":
        status = MaskQCStatus.AMBIGUOUS
    else:
        assert selected_id is not None
        selected = next(
            candidate
            for candidate in candidate_tuple
            if candidate.candidate_id == selected_id
        )
        if not selected.info.basic_valid:
            status = MaskQCStatus.AMBIGUOUS
            reason = f"Qwen selected a mechanically invalid candidate: {selected.info.basic_reason}"
        elif confidence < mask_config.qc_min_confidence:
            status = MaskQCStatus.AMBIGUOUS
            reason = (
                f"Qwen confidence {confidence:.3f} is below the configured minimum "
                f"{mask_config.qc_min_confidence:.3f}"
            )
        else:
            status = MaskQCStatus.PASSED
    if status is MaskQCStatus.PASSED:
        assert selected_id is not None
        selected = next(
            candidate
            for candidate in candidate_tuple
            if candidate.candidate_id == selected_id
        )
        report = RoleMaskQC(
            role=role,
            status=status,
            selected_candidate=selected.candidate_id,
            selected_query_field=selected.query_field,
            selected_query=selected.query,
            confidence=confidence,
            reason=reason,
            candidates=info_tuple,
            model=completion.model,
            raw_response=completion.content,
            rendered_prompt=rendered_prompt,
        )
    else:
        report = RoleMaskQC(
            role=role,
            status=status,
            selected_candidate=None,
            selected_query_field=None,
            selected_query=None,
            confidence=confidence,
            reason=reason,
            candidates=info_tuple,
            model=completion.model,
            raw_response=completion.content,
            rendered_prompt=rendered_prompt,
        )
    return _RoleExecution(report, candidate_tuple, panels)


def run_mask_qc_stage(
    context: LoopContext,
    semantic_plan: SemanticPlan,
    backend: MaskQCBackend,
    resource_path: Path,
    *,
    seed_images: Mapping[int, Image.Image],
    context_images: Mapping[int, Image.Image],
    frame_shape: tuple[int, int],
    mask_config: MaskConfig,
    client: MaskQCClient,
    check_health: bool = True,
) -> MaskQCResult:
    """Generate query-bank seed candidates and fail closed on uncertain identity."""

    if semantic_plan.episode != context.episode:
        raise MaskQCError("SemanticPlan and LoopContext refer to different episodes")
    health: dict[str, Any] = {}
    if check_health:
        try:
            health = client.health()
        except Exception as exc:
            error = str(exc)
            target = _error_report("target", MaskQCStatus.ERROR, f"mask QC health failed: {error}")
            receiver = _error_report(
                "receiver", MaskQCStatus.ERROR, f"mask QC health failed: {error}"
            )
            return MaskQCResult(target, receiver, {}, {"status": "error", "error": error})
    executions: list[_RoleExecution] = []
    for role, semantic in (
        (_ROLES[0], semantic_plan.target),
        (_ROLES[1], semantic_plan.receiver),
    ):
        if semantic.seed_frame_id is None:
            seed_image = Image.new("RGB", (frame_shape[1], frame_shape[0]))
        else:
            try:
                seed_image = seed_images[semantic.seed_frame_id]
            except KeyError as exc:
                raise MaskQCError(
                    f"missing seed RGB image for {role} frame {semantic.seed_frame_id}"
                ) from exc
        executions.append(
            _run_role_qc(
                context,
                role,
                semantic,
                backend=backend,
                resource_path=resource_path,
                seed_image=seed_image,
                context_images=context_images,
                frame_shape=frame_shape,
                mask_config=mask_config,
                client=client,
            )
        )
    selected_masks: dict[RoleName, np.ndarray] = {}
    candidate_masks: dict[RoleName, dict[str, np.ndarray]] = {}
    candidate_panels: dict[RoleName, dict[str, Image.Image]] = {}
    for role, execution in zip(_ROLES, executions, strict=True):
        candidate_masks[role] = {
            candidate.candidate_id: candidate.mask for candidate in execution.candidates
        }
        candidate_panels[role] = {
            candidate.candidate_id: panel
            for candidate, panel in zip(
                execution.candidates,
                execution.panels,
                strict=True,
            )
        }
        report = execution.report
        if report.status is not MaskQCStatus.PASSED or report.selected_candidate is None:
            continue
        selected = next(
            candidate
            for candidate in execution.candidates
            if candidate.candidate_id == report.selected_candidate
        )
        selected_masks[role] = selected.mask
    return MaskQCResult(
        target=executions[0].report,
        receiver=executions[1].report,
        selected_masks=selected_masks,
        health=health,
        candidate_masks=candidate_masks,
        candidate_panels=candidate_panels,
    )


def save_mask_qc_artifacts(
    store: ArtifactStore,
    run_id: str,
    context: LoopContext,
    result: MaskQCResult,
    *,
    candidate_masks: Mapping[str, Mapping[str, np.ndarray]] | None = None,
) -> Path:
    """Persist QC decisions and optional candidate masks for later review."""

    episode_dir = store.episode_dir(run_id, context.episode)
    reports = result.to_json()
    reports["episode"] = context.episode.to_json()
    masks_to_save = result.candidate_masks if candidate_masks is None else candidate_masks
    mask_paths: dict[str, dict[str, str]] = {}
    if masks_to_save:
        for role, masks in masks_to_save.items():
            mask_paths[role] = {}
            for candidate_id, mask in masks.items():
                path = store.write_png(
                    episode_dir / role / "qc_candidates" / f"candidate_{candidate_id}.mask.png",
                    np.asarray(mask, dtype=bool),
                )
                mask_paths[role][candidate_id] = str(path.relative_to(episode_dir))
    panel_paths: dict[str, dict[str, str]] = {}
    for role, panels in result.candidate_panels.items():
        panel_paths[role] = {}
        for candidate_id, panel in panels.items():
            path = store.write_png(
                episode_dir / role / "qc_candidates" / f"candidate_{candidate_id}.overlay.png",
                np.asarray(panel.convert("RGB")),
                rgb=True,
            )
            panel_paths[role][candidate_id] = str(path.relative_to(episode_dir))
    reports["artifacts"] = {
        "candidate_masks": mask_paths,
        "candidate_panels": panel_paths,
    }
    report_path = store.write_json(episode_dir / "mask_qc.json", reports)
    return report_path
