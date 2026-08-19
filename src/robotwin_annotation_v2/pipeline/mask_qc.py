"""Stage 3.5: compare SAM3 seed-mask candidates with Qwen and basic QC."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from PIL import Image, ImageDraw

from ..adapters.artifact_store import ArtifactStore
from ..adapters.qwen_client import QwenCompletion, image_data_url
from ..config import MaskConfig
from ..models import (
    LoopContext,
    MaskCandidateInfo,
    MaskQCAttempt,
    MaskQCAttemptMethod,
    MaskQCResult,
    MaskQCStatus,
    RoleMaskQC,
    RoleSemanticPlan,
    SemanticPlan,
    SemanticStatus,
)
from ..models.loop_context import RoleName
from .bbox_localization import (
    BboxLocalizationError,
    build_bbox_messages,
    parse_bbox_localization,
    render_bbox_prompt,
)
from .object_mask.planner import QueryCandidate, plan_role_queries
from .object_mask.proposals import blue_planar_region
from .object_mask.qc import MaskQCError, candidate_info, mask_iou
from .prompt_context import timeline_prompt_fields

_CANDIDATE_MARKER = "{candidate_panels}"
_CONTEXT_MARKER = "{context_frames}"
_PLACEHOLDER_PATTERN = re.compile(r"\{([a-z][a-z0-9_]*)\}")
_QC_FIELDS = frozenset({"decision", "selected_candidate", "confidence", "reason"})
_PANEL_COLORS = (
    (232, 67, 55),
    (35, 116, 224),
    (238, 176, 38),
    (153, 78, 189),
    (20, 160, 160),
    (220, 90, 150),
)


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


class BboxMaskBackend(Protocol):
    def box_mask(
        self,
        resource_path: Path,
        box_xyxy: Sequence[float],
        *,
        frame_id: int,
        frame_count: int,
        frame_shape: tuple[int, int],
    ) -> np.ndarray: ...


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
    seed_frame_id: int
    mask: np.ndarray
    info: MaskCandidateInfo


@dataclass(frozen=True)
class _RoleAttemptExecution:
    seed_frame_id: int
    report: RoleMaskQC
    candidates: tuple[_Candidate, ...]
    panels: tuple[Image.Image, ...] = ()
    method: MaskQCAttemptMethod = MaskQCAttemptMethod.TEXT_QUERY
    provenance: dict[str, Any] = dataclass_field(default_factory=dict)


@dataclass(frozen=True)
class _RoleExecution:
    report: RoleMaskQC
    candidates: tuple[_Candidate, ...]
    panels: tuple[Image.Image, ...] = ()
    attempts: tuple[_RoleAttemptExecution, ...] = ()


def _normalize_text(value: str) -> str:
    return " ".join(value.split())


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
        _PANEL_COLORS[
            sum(ord(character) for character in candidate.candidate_id) % len(_PANEL_COLORS)
        ],
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


def _context_items(
    context: LoopContext,
    role: RoleName,
    seed_frame_id: int,
    context_images: Mapping[int, Image.Image],
    *,
    limit: int = 2,
) -> tuple[tuple[int, Image.Image], ...]:
    eligible = [
        frame
        for frame in context.semantic_frames
        if frame.frame_id != seed_frame_id
        and role in frame.eligible_roles
        and frame.frame_id in context_images
    ]

    def sample(frame_ids: list[int], count: int) -> list[int]:
        if len(frame_ids) <= count:
            return frame_ids
        indices = tuple(
            dict.fromkeys(round(value) for value in np.linspace(0, len(frame_ids) - 1, num=count))
        )
        return [frame_ids[index] for index in indices]

    # Close/hold/place evidence identifies the manipulated instance more
    # reliably than another static pre-grasp view.  Only use spare slots for
    # additional seed candidates when fewer action-context frames exist.
    evidence = [frame.frame_id for frame in eligible if not frame.seed_eligible]
    selected = sample(evidence, limit)
    if len(selected) < limit:
        supporting_seeds = [frame.frame_id for frame in eligible if frame.seed_eligible]
        selected.extend(sample(supporting_seeds, limit - len(selected)))
    selected = sorted(set(selected))
    return tuple((frame_id, context_images[frame_id]) for frame_id in selected)


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
        raise MaskQCError(f"mask QC template must contain {_CANDIDATE_MARKER!r} exactly once")
    if template.count(_CONTEXT_MARKER) != 1:
        raise MaskQCError(f"mask QC template must contain {_CONTEXT_MARKER!r} exactly once")
    candidate_ids = ", ".join(candidate.candidate_id for candidate in candidates)
    replacements = {
        "task_text": context.task_text,
        "role": role,
        "seed_frame_id": str(seed_frame_id),
        "candidate_ids": candidate_ids,
        **timeline_prompt_fields(context),
    }
    placeholders = set(_PLACEHOLDER_PATTERN.findall(template))
    unknown = sorted(placeholders - set(replacements) - {"candidate_panels", "context_frames"})
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


def _role_query_candidates(
    context: LoopContext,
    role: RoleName,
    semantic: RoleSemanticPlan,
    mask_config: MaskConfig,
) -> tuple[QueryCandidate, ...]:
    """Compatibility shim for the canonical object-mask query planner."""

    assert semantic.query_bank is not None
    return plan_role_queries(
        context,
        role,
        semantic,
        query_fallback_enabled=mask_config.qc_query_fallback_enabled,
    )


def _visual_report_from_completion(
    role: RoleName,
    *,
    candidates: tuple[_Candidate, ...],
    valid: tuple[_Candidate, ...],
    completion: QwenCompletion,
    rendered_prompt: str,
    mask_config: MaskConfig,
) -> RoleMaskQC:
    """Convert one visual-QC completion into the normal fail-closed report."""

    info_tuple = tuple(candidate.info for candidate in candidates)
    try:
        decision, selected_id, confidence, reason = parse_mask_qc_response(
            completion.content,
            candidate_ids=tuple(candidate.candidate_id for candidate in valid),
        )
    except MaskQCError as exc:
        return _error_report(
            role,
            MaskQCStatus.ERROR,
            str(exc),
            model=completion.model,
            raw_response=completion.content,
            rendered_prompt=rendered_prompt,
            candidates=info_tuple,
        )
    if decision == "reject_all":
        status = MaskQCStatus.REJECTED
    elif decision == "ambiguous":
        status = MaskQCStatus.AMBIGUOUS
    else:
        assert selected_id is not None
        selected = next(
            candidate for candidate in candidates if candidate.candidate_id == selected_id
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
            candidate for candidate in candidates if candidate.candidate_id == selected_id
        )
        return RoleMaskQC(
            role=role,
            status=status,
            selected_candidate=selected.candidate_id,
            selected_query_field=selected.query_field,
            selected_query=selected.query,
            selected_seed_frame_id=selected.seed_frame_id,
            confidence=confidence,
            reason=reason,
            candidates=info_tuple,
            model=completion.model,
            raw_response=completion.content,
            rendered_prompt=rendered_prompt,
        )
    return RoleMaskQC(
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


def _evaluate_candidate_visual_qc(
    context: LoopContext,
    role: RoleName,
    *,
    seed_frame_id: int,
    candidates: tuple[_Candidate, ...],
    seed_image: Image.Image,
    context_images: Mapping[int, Image.Image],
    mask_config: MaskConfig,
    client: MaskQCClient,
    method: MaskQCAttemptMethod,
    provenance: dict[str, Any] | None = None,
) -> _RoleAttemptExecution:
    """Apply the same mechanical and visual gate to text- and bbox-seeded masks."""

    info_tuple = tuple(candidate.info for candidate in candidates)
    valid = tuple(candidate for candidate in candidates if candidate.info.basic_valid)
    panels = tuple(_panel_image(seed_image, candidate) for candidate in candidates)
    attempt_provenance = {} if provenance is None else provenance
    if not valid:
        return _RoleAttemptExecution(
            seed_frame_id,
            _error_report(
                role,
                MaskQCStatus.REJECTED,
                "all_candidate_masks_failed_basic_checks",
                candidates=info_tuple,
            ),
            candidates,
            panels,
            method,
            attempt_provenance,
        )
    template_path = mask_config.qc_prompt_template
    if template_path is None or not template_path.is_file():
        return _RoleAttemptExecution(
            seed_frame_id,
            _error_report(
                role,
                MaskQCStatus.ERROR,
                f"mask QC prompt template is missing: {template_path}",
                candidates=info_tuple,
            ),
            candidates,
            panels,
            method,
            attempt_provenance,
        )
    try:
        template = template_path.read_text(encoding="utf-8")
        rendered_prompt, messages, _panels = _render_request(
            context,
            role=role,
            seed_frame_id=seed_frame_id,
            seed_image=seed_image,
            candidates=valid,
            context_images=context_images,
            template=template,
        )
    except (OSError, MaskQCError) as exc:
        return _RoleAttemptExecution(
            seed_frame_id,
            _error_report(
                role,
                MaskQCStatus.ERROR,
                f"mask QC request could not be rendered: {exc}",
                candidates=info_tuple,
            ),
            candidates,
            panels,
            method,
            attempt_provenance,
        )
    completion: QwenCompletion | None = None
    request_error: Exception | None = None
    for _attempt in range(mask_config.qc_max_attempts):
        try:
            completion = client.complete(messages, max_tokens=mask_config.qc_max_tokens)
            break
        except Exception as exc:  # noqa: BLE001 - external QC client boundary
            request_error = exc
    if completion is None:
        assert request_error is not None
        return _RoleAttemptExecution(
            seed_frame_id,
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
            candidates,
            panels,
            method,
            attempt_provenance,
        )
    report = _visual_report_from_completion(
        role,
        candidates=candidates,
        valid=valid,
        completion=completion,
        rendered_prompt=rendered_prompt,
        mask_config=mask_config,
    )

    return _RoleAttemptExecution(
        seed_frame_id,
        report,
        candidates,
        panels,
        method,
        attempt_provenance,
    )


def _run_role_qc_at_seed(
    context: LoopContext,
    role: RoleName,
    *,
    seed_frame_id: int,
    query_candidates: tuple[QueryCandidate, ...],
    backend: MaskQCBackend,
    resource_path: Path,
    seed_image: Image.Image,
    context_images: Mapping[int, Image.Image],
    frame_shape: tuple[int, int],
    mask_config: MaskConfig,
    client: MaskQCClient,
) -> _RoleAttemptExecution:
    def generation_error(reason: str) -> _RoleAttemptExecution:
        return _RoleAttemptExecution(
            seed_frame_id,
            _error_report(
                role,
                MaskQCStatus.ERROR,
                f"text candidate generation failed: {reason}",
            ),
            (),
        )

    blue_prior = np.zeros(frame_shape, dtype=bool)
    if role == "receiver" and any(
        "blue" in query.split() for query in (candidate.query for candidate in query_candidates)
    ):
        try:
            blue_prior = blue_planar_region(seed_image, frame_shape)
        except MaskQCError as exc:
            return generation_error(str(exc))
    reserve_prior = bool(blue_prior.any()) and mask_config.qc_max_candidates >= 2
    text_limit = mask_config.qc_max_candidates - int(reserve_prior)
    selected_queries = query_candidates[:text_limit]
    fields = tuple(candidate.field for candidate in selected_queries)
    queries = tuple(candidate.query for candidate in selected_queries)
    try:
        query_masks = backend.text_query_masks(
            resource_path,
            queries,
            frame_id=seed_frame_id,
            frame_count=context.frame_count,
            frame_shape=frame_shape,
        )
    except Exception as exc:  # noqa: BLE001 - external SAM backend boundary
        return generation_error(f"{type(exc).__name__}: {exc}")
    if not isinstance(query_masks, Mapping):
        return generation_error("text backend response must be a query-to-mask mapping")
    missing_queries = tuple(query for query in queries if query not in query_masks)
    if missing_queries:
        return generation_error(
            "text backend omitted mask(s) for query: " + ", ".join(missing_queries)
        )
    candidates: list[_Candidate] = []
    for index, (field, query_value) in enumerate(zip(fields, queries, strict=True)):
        assert query_value is not None
        query = query_value
        candidate_id = chr(ord("A") + index)
        try:
            mask = np.asarray(query_masks[query], dtype=bool)
        except Exception as exc:  # noqa: BLE001 - external SAM backend value boundary
            return generation_error(
                f"query {query!r} mask could not be converted: {type(exc).__name__}: {exc}"
            )
        if mask.shape != frame_shape:
            return generation_error(
                f"{role} candidate {candidate_id} has shape {mask.shape}, expected {frame_shape}"
            )
        info = candidate_info(
            candidate_id,
            field,
            query,
            mask,
            min_area_fraction=mask_config.qc_min_area_fraction,
            max_area_fraction=mask_config.qc_max_area_fraction,
            seed_frame_id=seed_frame_id,
        )
        if info.basic_valid:
            duplicate = next(
                (
                    previous
                    for previous in candidates
                    if previous.info.basic_valid
                    and mask_iou(previous.mask, mask) >= mask_config.qc_duplicate_iou_threshold
                ),
                None,
            )
            if duplicate is not None:
                info = candidate_info(
                    candidate_id,
                    field,
                    query,
                    mask,
                    min_area_fraction=mask_config.qc_min_area_fraction,
                    max_area_fraction=mask_config.qc_max_area_fraction,
                    duplicate_of=duplicate.candidate_id,
                    seed_frame_id=seed_frame_id,
                )
        candidates.append(_Candidate(candidate_id, field, query, seed_frame_id, mask, info))
    if reserve_prior:
        candidate_id = chr(ord("A") + len(candidates))
        query = "blue planar region"
        info = candidate_info(
            candidate_id,
            "blue_region_prior",
            query,
            blue_prior,
            min_area_fraction=mask_config.qc_min_area_fraction,
            max_area_fraction=mask_config.qc_max_area_fraction,
            seed_frame_id=seed_frame_id,
        )
        if info.basic_valid:
            duplicate = next(
                (
                    previous
                    for previous in candidates
                    if previous.info.basic_valid
                    and mask_iou(previous.mask, blue_prior)
                    >= mask_config.qc_duplicate_iou_threshold
                ),
                None,
            )
            if duplicate is not None:
                info = candidate_info(
                    candidate_id,
                    "blue_region_prior",
                    query,
                    blue_prior,
                    min_area_fraction=mask_config.qc_min_area_fraction,
                    max_area_fraction=mask_config.qc_max_area_fraction,
                    duplicate_of=duplicate.candidate_id,
                    seed_frame_id=seed_frame_id,
                )
        candidates.append(
            _Candidate(
                candidate_id,
                "blue_region_prior",
                query,
                seed_frame_id,
                blue_prior,
                info,
            )
        )
    return _evaluate_candidate_visual_qc(
        context,
        role,
        seed_frame_id=seed_frame_id,
        candidates=tuple(candidates),
        seed_image=seed_image,
        context_images=context_images,
        mask_config=mask_config,
        client=client,
        method=MaskQCAttemptMethod.TEXT_QUERY,
    )


def _run_bbox_qc_at_seed(
    context: LoopContext,
    role: RoleName,
    *,
    seed_frame_id: int,
    backend: MaskQCBackend,
    resource_path: Path,
    seed_image: Image.Image,
    context_images: Mapping[int, Image.Image],
    frame_shape: tuple[int, int],
    mask_config: MaskConfig,
    client: MaskQCClient,
) -> _RoleAttemptExecution:
    """Generate one Qwen-box/SAM candidate and pass it through normal visual QC."""

    method = MaskQCAttemptMethod.BBOX_FALLBACK
    provenance: dict[str, Any] = {
        "candidate_generation": "qwen_bbox_to_sam_box",
        "sam_ran": False,
    }
    template_path = mask_config.qc_bbox_prompt_template
    if template_path is None or not template_path.is_file():
        return _RoleAttemptExecution(
            seed_frame_id,
            _error_report(
                role,
                MaskQCStatus.ERROR,
                f"bbox localization prompt template is missing: {template_path}",
            ),
            (),
            (),
            method,
            provenance,
        )
    provenance["localization_prompt_template"] = str(template_path)
    try:
        template = template_path.read_text(encoding="utf-8")
        rendered_prompt = render_bbox_prompt(
            template,
            task=context.episode.task,
            task_text=context.task_text,
            episode_id=context.episode.episode_id,
            role=role,
            seed_frame_id=seed_frame_id,
        )
        messages = build_bbox_messages(rendered_prompt, seed_image.convert("RGB"))
    except (OSError, BboxLocalizationError) as exc:
        return _RoleAttemptExecution(
            seed_frame_id,
            _error_report(
                role,
                MaskQCStatus.ERROR,
                f"bbox localization request could not be rendered: {exc}",
            ),
            (),
            (),
            method,
            provenance,
        )

    completion: QwenCompletion | None = None
    request_error: Exception | None = None
    for _attempt in range(mask_config.qc_max_attempts):
        try:
            completion = client.complete(messages, max_tokens=mask_config.qc_bbox_max_tokens)
            break
        except Exception as exc:  # noqa: BLE001 - external localization client boundary
            request_error = exc
    if completion is None:
        assert request_error is not None
        provenance["localization_rendered_prompt"] = rendered_prompt
        return _RoleAttemptExecution(
            seed_frame_id,
            _error_report(
                role,
                MaskQCStatus.ERROR,
                (
                    f"bbox localization request failed after {mask_config.qc_max_attempts} "
                    f"attempt(s): {request_error}"
                ),
                model=getattr(client, "model_id", None),
                rendered_prompt=rendered_prompt,
            ),
            (),
            (),
            method,
            provenance,
        )

    provenance.update(
        {
            "localization_model": completion.model,
            "localization_raw_response": completion.content,
            "localization_rendered_prompt": rendered_prompt,
        }
    )
    try:
        localization = parse_bbox_localization(completion.content)
    except BboxLocalizationError as exc:
        return _RoleAttemptExecution(
            seed_frame_id,
            _error_report(
                role,
                MaskQCStatus.ERROR,
                f"invalid bbox localization response: {exc}",
                model=completion.model,
                raw_response=completion.content,
                rendered_prompt=rendered_prompt,
            ),
            (),
            (),
            method,
            provenance,
        )
    provenance["localization"] = localization.to_json()
    if localization.bbox_xyxy is None:
        status = (
            MaskQCStatus.AMBIGUOUS if localization.status == "ambiguous" else MaskQCStatus.REJECTED
        )
        return _RoleAttemptExecution(
            seed_frame_id,
            _error_report(
                role,
                status,
                f"bbox localization {localization.status}: {localization.reason}",
                model=completion.model,
                raw_response=completion.content,
                rendered_prompt=rendered_prompt,
            ),
            (),
            (),
            method,
            provenance,
        )

    try:
        mask = np.asarray(
            cast(BboxMaskBackend, backend).box_mask(
                resource_path,
                localization.bbox_xyxy,
                frame_id=seed_frame_id,
                frame_count=context.frame_count,
                frame_shape=frame_shape,
            ),
            dtype=bool,
        )
    except Exception as exc:  # noqa: BLE001 - external SAM backend boundary
        return _RoleAttemptExecution(
            seed_frame_id,
            _error_report(
                role,
                MaskQCStatus.ERROR,
                f"SAM bbox candidate generation failed: {exc}",
            ),
            (),
            (),
            method,
            provenance,
        )
    if mask.shape != frame_shape:
        return _RoleAttemptExecution(
            seed_frame_id,
            _error_report(
                role,
                MaskQCStatus.ERROR,
                f"{role} bbox candidate has shape {mask.shape}, expected {frame_shape}",
            ),
            (),
            (),
            method,
            provenance,
        )

    provenance.update(
        {
            "sam_ran": True,
            "sam_prompt": {
                "type": "normalized_box_xyxy",
                "bbox_xyxy": list(localization.bbox_xyxy),
                "coordinates_clamped": False,
            },
        }
    )
    candidate_id = "BBOX"
    query_field = "bbox_fallback"
    query = f"Qwen-localized {role} bounding box"
    info = candidate_info(
        candidate_id,
        query_field,
        query,
        mask,
        min_area_fraction=mask_config.qc_min_area_fraction,
        max_area_fraction=mask_config.qc_max_area_fraction,
        seed_frame_id=seed_frame_id,
    )
    candidate = _Candidate(
        candidate_id,
        query_field,
        query,
        seed_frame_id,
        mask,
        info,
    )
    return _evaluate_candidate_visual_qc(
        context,
        role,
        seed_frame_id=seed_frame_id,
        candidates=(candidate,),
        seed_image=seed_image,
        context_images=context_images,
        mask_config=mask_config,
        client=client,
        method=method,
        provenance=provenance,
    )


def _attempt_report(execution: _RoleAttemptExecution) -> MaskQCAttempt:
    report = execution.report
    return MaskQCAttempt(
        seed_frame_id=execution.seed_frame_id,
        status=report.status,
        selected_candidate=report.selected_candidate,
        selected_query_field=report.selected_query_field,
        selected_query=report.selected_query,
        confidence=report.confidence,
        reason=report.reason,
        method=execution.method,
        candidates=report.candidates,
        model=report.model,
        raw_response=report.raw_response,
        rendered_prompt=report.rendered_prompt,
        provenance=execution.provenance,
    )


def _finalize_role_execution(
    final: _RoleAttemptExecution,
    attempts: Sequence[_RoleAttemptExecution],
    *,
    reason: str | None = None,
) -> _RoleExecution:
    report = final.report
    if reason is not None:
        report = replace(report, reason=_normalize_text(reason))
    report = replace(
        report,
        attempts=tuple(_attempt_report(attempt) for attempt in attempts),
    )
    return _RoleExecution(
        report=report,
        candidates=final.candidates,
        panels=final.panels,
        attempts=tuple(attempts),
    )


def _run_role_qc(
    context: LoopContext,
    role: RoleName,
    semantic: RoleSemanticPlan,
    *,
    backend: MaskQCBackend,
    resource_path: Path,
    seed_images: Mapping[int, Image.Image],
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
    query_candidates = _role_query_candidates(context, role, semantic, mask_config)
    seed_frame_ids = [semantic.seed_frame_id]
    if mask_config.qc_seed_fallback_enabled:
        seed_frame_ids.extend(
            frame_id
            for frame_id in context.seed_candidates(role)
            if frame_id != semantic.seed_frame_id
        )

    executions: list[_RoleAttemptExecution] = []
    for seed_frame_id in seed_frame_ids:
        try:
            seed_image = seed_images[seed_frame_id]
        except KeyError as exc:
            raise MaskQCError(f"missing seed RGB image for {role} frame {seed_frame_id}") from exc
        execution = _run_role_qc_at_seed(
            context,
            role,
            seed_frame_id=seed_frame_id,
            query_candidates=query_candidates,
            backend=backend,
            resource_path=resource_path,
            seed_image=seed_image,
            context_images=context_images,
            frame_shape=frame_shape,
            mask_config=mask_config,
            client=client,
        )
        executions.append(execution)
        if execution.report.status in {MaskQCStatus.PASSED, MaskQCStatus.ERROR}:
            return _finalize_role_execution(execution, executions)

    bbox_seed_frame_ids: list[int] = []
    if mask_config.qc_bbox_fallback_enabled:
        # This block is deliberately after the complete text/seed loop.  A box
        # proposal must never pre-empt a text candidate that passed visual QC.
        bbox_seed_frame_ids = list(seed_frame_ids)
        for seed_frame_id in bbox_seed_frame_ids:
            seed_image = seed_images[seed_frame_id]
            execution = _run_bbox_qc_at_seed(
                context,
                role,
                seed_frame_id=seed_frame_id,
                backend=backend,
                resource_path=resource_path,
                seed_image=seed_image,
                context_images=context_images,
                frame_shape=frame_shape,
                mask_config=mask_config,
                client=client,
            )
            executions.append(execution)
            if execution.report.status in {MaskQCStatus.PASSED, MaskQCStatus.ERROR}:
                return _finalize_role_execution(execution, executions)

    meaningful = [
        execution
        for execution in executions
        if any(candidate.info.basic_valid for candidate in execution.candidates)
    ]
    final = meaningful[-1] if meaningful else executions[-1]
    attempted = ",".join(str(frame_id) for frame_id in seed_frame_ids)
    bbox_attempted = ",".join(str(frame_id) for frame_id in bbox_seed_frame_ids)
    suffix = f"; text seed frames: {attempted}"
    if bbox_seed_frame_ids:
        suffix += f"; bbox fallback seed frames: {bbox_attempted}"
    return _finalize_role_execution(
        final,
        executions,
        reason=f"{final.report.reason}{suffix}",
    )


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
    if semantic_plan.annotation_mode is not context.annotation_mode:
        raise MaskQCError("SemanticPlan and LoopContext use different annotation modes")
    health: dict[str, Any] = {}
    if check_health:
        try:
            health = client.health()
        except Exception as exc:  # noqa: BLE001 - health failures must fail closed
            error = str(exc)
            reports = tuple(
                _error_report(
                    semantic.role,
                    MaskQCStatus.ERROR,
                    f"mask QC health failed: {error}",
                )
                for semantic in semantic_plan.role_plans
            )
            return MaskQCResult(
                role_reports=reports,
                selected_masks={},
                health={"status": "error", "error": error},
            )
    executions: list[_RoleExecution] = []
    for semantic in semantic_plan.role_plans:
        role = semantic.role
        executions.append(
            _run_role_qc(
                context,
                role,
                semantic,
                backend=backend,
                resource_path=resource_path,
                seed_images=seed_images,
                context_images=context_images,
                frame_shape=frame_shape,
                mask_config=mask_config,
                client=client,
            )
        )
    selected_masks: dict[RoleName, np.ndarray] = {}
    candidate_masks: dict[RoleName, dict[str, np.ndarray]] = {}
    candidate_panels: dict[RoleName, dict[str, Image.Image]] = {}
    attempt_candidate_masks: dict[RoleName, dict[int, dict[str, np.ndarray]]] = {}
    attempt_candidate_panels: dict[RoleName, dict[int, dict[str, Image.Image]]] = {}
    for semantic, execution in zip(semantic_plan.role_plans, executions, strict=True):
        role = semantic.role
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
        role_attempt_masks: dict[int, dict[str, np.ndarray]] = {}
        role_attempt_panels: dict[int, dict[str, Image.Image]] = {}
        for attempt in execution.attempts:
            masks_at_seed = role_attempt_masks.setdefault(attempt.seed_frame_id, {})
            panels_at_seed = role_attempt_panels.setdefault(attempt.seed_frame_id, {})
            for candidate, panel in zip(
                attempt.candidates,
                attempt.panels,
                strict=True,
            ):
                if candidate.candidate_id in masks_at_seed:
                    raise MaskQCError(
                        f"duplicate {role} candidate id {candidate.candidate_id!r} "
                        f"at seed frame {attempt.seed_frame_id}"
                    )
                masks_at_seed[candidate.candidate_id] = candidate.mask
                panels_at_seed[candidate.candidate_id] = panel
        attempt_candidate_masks[role] = role_attempt_masks
        attempt_candidate_panels[role] = role_attempt_panels
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
        role_reports=tuple(execution.report for execution in executions),
        selected_masks=selected_masks,
        health=health,
        candidate_masks=candidate_masks,
        candidate_panels=candidate_panels,
        attempt_candidate_masks=attempt_candidate_masks,
        attempt_candidate_panels=attempt_candidate_panels,
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
    attempt_paths: dict[str, dict[str, dict[str, Any]]] = {}
    for report in result.role_reports:
        role = report.role
        mask_attempts = result.attempt_candidate_masks.get(role, {})
        panel_attempts = result.attempt_candidate_panels.get(role, {})
        frame_ids = sorted(set(mask_attempts) | set(panel_attempts))
        if not frame_ids:
            continue
        attempt_paths[role] = {}
        for seed_frame_id in frame_ids:
            frame_key = f"frame_{seed_frame_id:06d}"
            frame_dir = episode_dir / role / "qc_candidates" / frame_key
            frame_mask_paths: dict[str, str] = {}
            for candidate_id, mask in mask_attempts.get(seed_frame_id, {}).items():
                path = store.write_png(
                    frame_dir / f"candidate_{candidate_id}.mask.png",
                    np.asarray(mask, dtype=bool),
                )
                frame_mask_paths[candidate_id] = str(path.relative_to(episode_dir))
            frame_panel_paths: dict[str, str] = {}
            for candidate_id, panel in panel_attempts.get(seed_frame_id, {}).items():
                path = store.write_png(
                    frame_dir / f"candidate_{candidate_id}.overlay.png",
                    np.asarray(panel.convert("RGB")),
                    rgb=True,
                )
                frame_panel_paths[candidate_id] = str(path.relative_to(episode_dir))
            attempt_paths[role][frame_key] = {
                "seed_frame_id": seed_frame_id,
                "candidate_masks": frame_mask_paths,
                "candidate_panels": frame_panel_paths,
            }
    reports["artifacts"] = {
        "candidate_masks": mask_paths,
        "candidate_panels": panel_paths,
        "attempts": attempt_paths,
    }
    report_path = store.write_json(episode_dir / "mask_qc.json", reports)
    return report_path
