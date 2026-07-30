"""Stage 2: turn sparse episode frames into a validated semantic plan."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from PIL import Image

from ..adapters.qwen_client import QwenCompletion, image_data_url
from ..config import QwenConfig
from ..models import (
    CANDIDATE_FIELDS,
    LoopContext,
    QueryBank,
    RoleSemanticPlan,
    SemanticPlan,
    SemanticPlanError,
    SemanticStatus,
)


_FRAME_MARKER = "{labeled_multimodal_frames}"
_PLACEHOLDER_PATTERN = re.compile(r"\{([a-z][a-z0-9_]*)\}")
_ROLE_FIELDS = frozenset(
    {
        "status",
        "seed_frame_id",
        *CANDIDATE_FIELDS,
        "recommended_order",
        "exclude",
        "reason",
    }
)


class QwenStageError(RuntimeError):
    """The prompt, response, or selected seed violates the Stage-2 contract."""

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


class QwenClient(Protocol):
    model_id: str

    def health(self) -> dict[str, Any]: ...

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
    ) -> QwenCompletion: ...


@dataclass(frozen=True)
class RenderedQwenRequest:
    rendered_prompt: str
    messages: list[dict[str, Any]]
    input_frame_ids: tuple[int, ...]


@dataclass(frozen=True)
class QwenStageResult:
    semantic_plan: SemanticPlan
    rendered_prompt: str
    health: dict[str, Any]


def _frame_label(context: LoopContext, frame_id: int) -> str:
    frame = next(item for item in context.semantic_frames if item.frame_id == frame_id)
    roles = ",".join(frame.eligible_roles)
    seed = "yes" if frame.seed_eligible else "no"
    return (
        f"[frame_id={frame.frame_id}; purpose={frame.purpose.value}; "
        f"eligible_roles={roles}; seed_candidate={seed}]"
    )


def build_qwen_request(
    context: LoopContext,
    frames: Mapping[int, Image.Image],
    prompt_template: str,
) -> RenderedQwenRequest:
    """Render configurable text while keeping labels and images interleaved."""

    expected_ids = tuple(frame.frame_id for frame in context.semantic_frames)
    supplied_ids = tuple(sorted(frames))
    if set(supplied_ids) != set(expected_ids):
        raise QwenStageError(
            f"Qwen frames must exactly match LoopContext: expected={expected_ids}, "
            f"supplied={supplied_ids}"
        )
    if prompt_template.count(_FRAME_MARKER) != 1:
        raise QwenStageError(
            f"prompt template must contain {_FRAME_MARKER!r} exactly once"
        )

    events = context.events
    replacements = {
        "task_text": context.task_text,
        "camera": context.episode.camera,
        "move_start": str(events.t_move_start),
        "close_start": str(events.t_close_start),
        "close_done": str(events.t_close_done),
        "open_start": str(events.t_open_start),
        "open_done": str(events.t_open_done),
    }
    placeholders = set(_PLACEHOLDER_PATTERN.findall(prompt_template))
    unknown = sorted(placeholders - set(replacements) - {"labeled_multimodal_frames"})
    if unknown:
        raise QwenStageError(f"unknown prompt template placeholder(s): {', '.join(unknown)}")

    partially_rendered = prompt_template
    for key, value in replacements.items():
        partially_rendered = partially_rendered.replace(f"{{{key}}}", value)
    prefix, suffix = partially_rendered.split(_FRAME_MARKER)

    labels = [_frame_label(context, frame_id) for frame_id in expected_ids]
    trace_parts = [
        f"{label}\n<image frame_id={frame_id}>"
        for label, frame_id in zip(labels, expected_ids, strict=True)
    ]
    rendered_prompt = f"{prefix}{'\n'.join(trace_parts)}{suffix}"

    content: list[dict[str, Any]] = []
    for index, (label, frame_id) in enumerate(
        zip(labels, expected_ids, strict=True)
    ):
        text = f"{prefix}{label}" if index == 0 else label
        content.append({"type": "text", "text": text})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_data_url(frames[frame_id])},
            }
        )
    if suffix:
        content.append({"type": "text", "text": suffix})
    return RenderedQwenRequest(
        rendered_prompt=rendered_prompt,
        messages=[{"role": "user", "content": content}],
        input_frame_ids=expected_ids,
    )


def _decode_response(raw_response: str) -> dict[str, Any]:
    text = raw_response.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) < 3 or lines[-1].strip() != "```":
            raise QwenStageError("Qwen response contains an incomplete JSON fence")
        if lines[0].strip() not in {"```", "```json"}:
            raise QwenStageError("Qwen response fence must contain JSON")
        text = "\n".join(lines[1:-1]).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise QwenStageError(f"Qwen response is not valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise QwenStageError("Qwen response must be a JSON object")
    fields = frozenset(payload)
    if fields != {"target", "receiver"}:
        raise QwenStageError(
            "Qwen response must contain exactly target and receiver; "
            f"got {sorted(fields)}"
        )
    return payload


def _string_list(value: Any, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise QwenStageError(f"{field} must be a list")
    cleaned: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise QwenStageError(f"{field}[{index}] must be a non-empty string")
        normalized = " ".join(item.split())
        if normalized in cleaned:
            raise QwenStageError(f"{field} must not contain duplicates")
        cleaned.append(normalized)
    return tuple(cleaned)


def _canonicalize_duplicate_candidates(
    candidate_values: dict[str, str | None],
    order: tuple[str, ...],
) -> tuple[dict[str, str | None], tuple[str, ...]]:
    groups: dict[str, list[str]] = {}
    for field in CANDIDATE_FIELDS:
        value = candidate_values[field]
        if value is not None:
            groups.setdefault(" ".join(value.split()), []).append(field)

    normalized = dict(candidate_values)
    aliases: dict[str, str] = {}
    order_rank = {field: index for index, field in enumerate(order)}
    field_rank = {field: index for index, field in enumerate(CANDIDATE_FIELDS)}
    for fields in groups.values():
        if len(fields) < 2:
            continue
        if "category_query" in fields:
            keeper = "category_query"
        else:
            keeper = min(
                fields,
                key=lambda field: (
                    order_rank.get(field, len(order)),
                    field_rank[field],
                ),
            )
        for field in fields:
            if field != keeper:
                normalized[field] = None
                aliases[field] = keeper

    normalized_order: list[str] = []
    for field in order:
        resolved = aliases.get(field, field)
        if resolved not in normalized_order:
            normalized_order.append(resolved)
    return normalized, tuple(normalized_order)


def _parse_role(
    role: str,
    payload: Any,
    context: LoopContext,
) -> RoleSemanticPlan:
    if not isinstance(payload, dict):
        raise QwenStageError(f"{role} must be a JSON object")
    fields = frozenset(payload)
    if fields != _ROLE_FIELDS:
        missing = sorted(_ROLE_FIELDS - fields)
        extra = sorted(fields - _ROLE_FIELDS)
        raise QwenStageError(
            f"{role} fields do not match schema; missing={missing}, extra={extra}"
        )
    try:
        status = SemanticStatus(payload["status"])
    except (TypeError, ValueError) as exc:
        raise QwenStageError(f"{role}.status must be ok or no_clear_seed") from exc

    seed_frame_id = payload["seed_frame_id"]
    if seed_frame_id is not None and (
        isinstance(seed_frame_id, bool) or not isinstance(seed_frame_id, int)
    ):
        raise QwenStageError(f"{role}.seed_frame_id must be an integer or null")

    reason = payload["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise QwenStageError(f"{role}.reason must be a non-empty string")
    exclude = _string_list(payload["exclude"], field=f"{role}.exclude")

    candidate_values = {field: payload[field] for field in CANDIDATE_FIELDS}
    for field, value in candidate_values.items():
        if value is not None and not isinstance(value, str):
            raise QwenStageError(f"{role}.{field} must be a string or null")
    order = _string_list(payload["recommended_order"], field=f"{role}.recommended_order")
    candidate_values, order = _canonicalize_duplicate_candidates(
        candidate_values,
        order,
    )

    if status is SemanticStatus.NO_CLEAR_SEED:
        if seed_frame_id is not None or any(
            value is not None for value in candidate_values.values()
        ) or order:
            raise QwenStageError(
                f"{role} no_clear_seed must use null seed/queries and empty order"
            )
        query_bank = None
    else:
        if seed_frame_id not in context.seed_candidates(role):  # type: ignore[arg-type]
            raise QwenStageError(
                f"{role}.seed_frame_id={seed_frame_id} is not an eligible seed; "
                f"allowed={context.seed_candidates(role)}"  # type: ignore[arg-type]
            )
        try:
            query_bank = QueryBank(
                category_query=candidate_values["category_query"],
                color_category_query=candidate_values["color_category_query"],
                shape_category_query=candidate_values["shape_category_query"],
                general_fallback_query=candidate_values["general_fallback_query"],
                recommended_order=order,
            )  # type: ignore[arg-type]
        except SemanticPlanError as exc:
            raise QwenStageError(f"invalid {role} query bank: {exc}") from exc

    return RoleSemanticPlan(
        role=role,  # type: ignore[arg-type]
        status=status,
        seed_frame_id=seed_frame_id,
        query_bank=query_bank,
        exclude=exclude,
        reason=" ".join(reason.split()),
    )


def parse_semantic_plan(
    raw_response: str,
    *,
    context: LoopContext,
    model: str,
    rendered_prompt: str,
) -> SemanticPlan:
    """Parse one joint response and enforce seed/query constraints."""

    payload = _decode_response(raw_response)
    return SemanticPlan(
        episode=context.episode,
        target=_parse_role("target", payload["target"], context),
        receiver=_parse_role("receiver", payload["receiver"], context),
        model=model,
        prompt_sha256=SemanticPlan.prompt_hash(rendered_prompt),
        input_frame_ids=tuple(frame.frame_id for frame in context.semantic_frames),
        raw_response=raw_response,
    )


def run_qwen_stage(
    context: LoopContext,
    frames: Mapping[int, Image.Image],
    config: QwenConfig,
    client: QwenClient,
    *,
    check_health: bool = True,
) -> QwenStageResult:
    """Run the complete Qwen stage without embedding prompt policy in the server."""

    template_path = Path(config.prompt_template)
    if not template_path.is_file():
        raise QwenStageError(f"Qwen prompt template is missing: {template_path}")
    prompt_template = template_path.read_text(encoding="utf-8")
    request = build_qwen_request(context, frames, prompt_template)
    try:
        health = client.health() if check_health else {}
        completion = client.complete(request.messages, max_tokens=config.max_tokens)
    except Exception as exc:
        raise QwenStageError(
            f"Qwen request failed: {exc}",
            rendered_prompt=request.rendered_prompt,
        ) from exc
    try:
        plan = parse_semantic_plan(
            completion.content,
            context=context,
            model=completion.model,
            rendered_prompt=request.rendered_prompt,
        )
    except QwenStageError as exc:
        raise QwenStageError(
            str(exc),
            rendered_prompt=request.rendered_prompt,
            raw_response=completion.content,
        ) from exc
    return QwenStageResult(
        semantic_plan=plan,
        rendered_prompt=request.rendered_prompt,
        health=health,
    )
