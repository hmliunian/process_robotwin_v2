"""Canonical text-seed-to-bbox resolver for object masks."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Protocol

import numpy as np
from PIL import Image

from ...models import (
    LoopContext,
    MaskCandidateInfo,
    MaskQCAttempt,
    MaskQCAttemptMethod,
    MaskQCStatus,
    RoleMaskQC,
    RoleSemanticPlan,
    SemanticStatus,
)
from ...models.loop_context import RoleName
from .planner import QueryCandidate, plan_role_queries
from .qc import error_report, normalize_text


@dataclass(frozen=True)
class ObjectMaskCandidate:
    candidate_id: str
    query_field: str
    query: str
    seed_frame_id: int
    mask: np.ndarray[Any, Any]
    info: MaskCandidateInfo


@dataclass(frozen=True)
class RoleAttemptExecution:
    seed_frame_id: int
    report: RoleMaskQC
    candidates: tuple[ObjectMaskCandidate, ...]
    panels: tuple[Image.Image, ...] = ()
    method: MaskQCAttemptMethod = MaskQCAttemptMethod.TEXT_QUERY
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RoleResolution:
    report: RoleMaskQC
    candidates: tuple[ObjectMaskCandidate, ...]
    panels: tuple[Image.Image, ...] = ()
    attempts: tuple[RoleAttemptExecution, ...] = ()


class TextAttemptRunner(Protocol):
    def __call__(
        self,
        seed_frame_id: int,
        query_candidates: tuple[QueryCandidate, ...],
    ) -> RoleAttemptExecution: ...


class BboxAttemptRunner(Protocol):
    def __call__(self, seed_frame_id: int) -> RoleAttemptExecution: ...


def _attempt_report(execution: RoleAttemptExecution) -> MaskQCAttempt:
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


def _finalize_resolution(
    final: RoleAttemptExecution,
    attempts: list[RoleAttemptExecution],
    *,
    reason: str | None = None,
) -> RoleResolution:
    report = final.report
    if reason is not None:
        report = replace(report, reason=normalize_text(reason))
    report = replace(
        report,
        attempts=tuple(_attempt_report(attempt) for attempt in attempts),
    )
    return RoleResolution(
        report=report,
        candidates=final.candidates,
        panels=final.panels,
        attempts=tuple(attempts),
    )


@dataclass(frozen=True)
class ObjectMaskResolver:
    """Own the global text-seed-first order and its fail-closed stop rules."""

    run_text_attempt: TextAttemptRunner
    run_bbox_attempt: BboxAttemptRunner
    query_fallback_enabled: bool
    seed_fallback_enabled: bool
    bbox_fallback_enabled: bool

    def resolve(
        self,
        context: LoopContext,
        role: RoleName,
        semantic: RoleSemanticPlan,
    ) -> RoleResolution:
        if semantic.status is SemanticStatus.NO_CLEAR_SEED:
            return RoleResolution(
                error_report(role, MaskQCStatus.REJECTED, "semantic_plan_no_clear_seed"),
                (),
            )
        assert semantic.seed_frame_id is not None
        query_candidates = plan_role_queries(
            context,
            role,
            semantic,
            query_fallback_enabled=self.query_fallback_enabled,
        )
        seed_frame_ids = [semantic.seed_frame_id]
        if self.seed_fallback_enabled:
            seed_frame_ids.extend(
                frame_id
                for frame_id in context.seed_candidates(role)
                if frame_id != semantic.seed_frame_id
            )

        executions: list[RoleAttemptExecution] = []
        for seed_frame_id in seed_frame_ids:
            execution = self.run_text_attempt(seed_frame_id, query_candidates)
            executions.append(execution)
            if execution.report.status in {MaskQCStatus.PASSED, MaskQCStatus.ERROR}:
                return _finalize_resolution(execution, executions)

        bbox_seed_frame_ids: list[int] = []
        if self.bbox_fallback_enabled:
            bbox_seed_frame_ids = list(seed_frame_ids)
            for seed_frame_id in bbox_seed_frame_ids:
                execution = self.run_bbox_attempt(seed_frame_id)
                executions.append(execution)
                if execution.report.status in {MaskQCStatus.PASSED, MaskQCStatus.ERROR}:
                    return _finalize_resolution(execution, executions)

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
        return _finalize_resolution(
            final,
            executions,
            reason=f"{final.report.reason}{suffix}",
        )


__all__ = [
    "ObjectMaskCandidate",
    "ObjectMaskResolver",
    "RoleAttemptExecution",
    "RoleResolution",
]
