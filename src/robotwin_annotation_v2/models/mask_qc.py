"""Contracts for post-SAM candidate mask quality control."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .loop_context import RoleName


class MaskQCStatus(StrEnum):
    """Outcome of semantic and mechanical mask validation."""

    NOT_RUN = "not_run"
    PASSED = "passed"
    REJECTED = "rejected"
    AMBIGUOUS = "ambiguous"
    ERROR = "error"
    NOT_APPLICABLE = "not_applicable"


class MaskQCAttemptMethod(StrEnum):
    """How one set of same-frame seed candidates was generated."""

    TEXT_QUERY = "text_query"
    BBOX_FALLBACK = "bbox_fallback"


@dataclass(frozen=True)
class MaskCandidateInfo:
    """Small JSON-safe summary of one generated candidate mask."""

    candidate_id: str
    query_field: str
    query: str
    nonempty: bool
    area_fraction: float
    component_count: int
    basic_valid: bool
    basic_reason: str | None = None
    duplicate_of: str | None = None
    seed_frame_id: int | None = None

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ValueError("candidate_id must be non-empty")
        if not self.query_field.strip() or not self.query.strip():
            raise ValueError("candidate query metadata must be non-empty")
        if not 0.0 <= self.area_fraction <= 1.0:
            raise ValueError("area_fraction must be between 0 and 1")
        if self.component_count < 0:
            raise ValueError("component_count must be non-negative")
        if self.seed_frame_id is not None and self.seed_frame_id < 0:
            raise ValueError("seed_frame_id must be non-negative")

    def to_json(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "query_field": self.query_field,
            "query": self.query,
            "nonempty": self.nonempty,
            "area_fraction": self.area_fraction,
            "component_count": self.component_count,
            "basic_valid": self.basic_valid,
            "basic_reason": self.basic_reason,
            "duplicate_of": self.duplicate_of,
            "seed_frame_id": self.seed_frame_id,
        }


@dataclass(frozen=True)
class MaskQCAttempt:
    """One auditable candidate/QC attempt at a concrete seed frame."""

    seed_frame_id: int
    status: MaskQCStatus
    selected_candidate: str | None
    selected_query_field: str | None
    selected_query: str | None
    confidence: float | None
    reason: str
    method: MaskQCAttemptMethod = MaskQCAttemptMethod.TEXT_QUERY
    candidates: tuple[MaskCandidateInfo, ...] = ()
    model: str | None = None
    raw_response: str | None = None
    rendered_prompt: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.seed_frame_id < 0:
            raise ValueError("attempt seed_frame_id must be non-negative")
        if self.status in {MaskQCStatus.NOT_RUN, MaskQCStatus.NOT_APPLICABLE}:
            raise ValueError("attempt status must describe an executed QC attempt")
        if not isinstance(self.method, MaskQCAttemptMethod):
            raise TypeError("attempt method must be a MaskQCAttemptMethod")
        if not self.reason.strip():
            raise ValueError("attempt reason must be non-empty")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("attempt confidence must be between 0 and 1")
        candidate_ids = tuple(candidate.candidate_id for candidate in self.candidates)
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("attempt candidate ids must be unique")
        if any(
            candidate.seed_frame_id is not None and candidate.seed_frame_id != self.seed_frame_id
            for candidate in self.candidates
        ):
            raise ValueError("attempt candidates must belong to its seed frame")
        if self.status is MaskQCStatus.PASSED:
            if (
                self.selected_candidate is None
                or self.selected_query_field is None
                or self.selected_query is None
            ):
                raise ValueError("passed attempt requires a selected candidate/query")
            selected = next(
                (
                    candidate
                    for candidate in self.candidates
                    if candidate.candidate_id == self.selected_candidate
                ),
                None,
            )
            if selected is None:
                raise ValueError("passed attempt must select one of its candidates")
            if (
                selected.query_field != self.selected_query_field
                or selected.query != self.selected_query
            ):
                raise ValueError("passed attempt query metadata must match its candidate")
        elif any(
            value is not None
            for value in (
                self.selected_candidate,
                self.selected_query_field,
                self.selected_query,
            )
        ):
            raise ValueError("non-passed attempt cannot contain a selected query")

    def to_json(self) -> dict[str, Any]:
        return {
            "seed_frame_id": self.seed_frame_id,
            "status": self.status.value,
            "selected_candidate": self.selected_candidate,
            "selected_query_field": self.selected_query_field,
            "selected_query": self.selected_query,
            "confidence": self.confidence,
            "reason": self.reason,
            "method": self.method.value,
            "candidates": [candidate.to_json() for candidate in self.candidates],
            "model": self.model,
            "raw_response": self.raw_response,
            "rendered_prompt": self.rendered_prompt,
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class RoleMaskQC:
    """Auditable QC decision for one role."""

    role: RoleName
    status: MaskQCStatus
    selected_candidate: str | None
    selected_query_field: str | None
    selected_query: str | None
    confidence: float | None
    reason: str
    selected_seed_frame_id: int | None = None
    candidates: tuple[MaskCandidateInfo, ...] = ()
    model: str | None = None
    raw_response: str | None = None
    rendered_prompt: str | None = None
    attempts: tuple[MaskQCAttempt, ...] = ()

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("QC reason must be non-empty")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("QC confidence must be between 0 and 1")
        if self.selected_seed_frame_id is not None and self.selected_seed_frame_id < 0:
            raise ValueError("selected_seed_frame_id must be non-negative")
        if self.status is MaskQCStatus.PASSED:
            if (
                self.selected_candidate is None
                or self.selected_query_field is None
                or self.selected_query is None
            ):
                raise ValueError("passed QC requires a selected candidate/query")
        elif any(
            value is not None
            for value in (
                self.selected_candidate,
                self.selected_query_field,
                self.selected_query,
            )
        ):
            raise ValueError("non-passed QC cannot contain a selected query")
        if self.status is not MaskQCStatus.PASSED and self.selected_seed_frame_id is not None:
            raise ValueError("non-passed QC cannot contain a selected seed frame")
        if self.selected_seed_frame_id is not None:
            selected = next(
                (
                    candidate
                    for candidate in self.candidates
                    if candidate.candidate_id == self.selected_candidate
                ),
                None,
            )
            if (
                selected is not None
                and selected.seed_frame_id is not None
                and selected.seed_frame_id != self.selected_seed_frame_id
            ):
                raise ValueError("selected candidate and selected seed frame must match")
        attempt_keys = tuple((attempt.method, attempt.seed_frame_id) for attempt in self.attempts)
        if len(attempt_keys) != len(set(attempt_keys)):
            raise ValueError("QC attempt method/seed pairs must be unique")
        if self.status is MaskQCStatus.PASSED and self.attempts:
            if self.selected_seed_frame_id is None:
                raise ValueError("passed QC with attempts requires a selected seed frame")
            matches = tuple(
                attempt
                for attempt in self.attempts
                if attempt.seed_frame_id == self.selected_seed_frame_id
                and attempt.status is MaskQCStatus.PASSED
                and attempt.selected_candidate == self.selected_candidate
                and attempt.selected_query_field == self.selected_query_field
                and attempt.selected_query == self.selected_query
            )
            if len(matches) != 1:
                raise ValueError("passed QC selection must match exactly one seed attempt")

    def to_json(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "status": self.status.value,
            "selected_candidate": self.selected_candidate,
            "selected_query_field": self.selected_query_field,
            "selected_query": self.selected_query,
            "selected_seed_frame_id": self.selected_seed_frame_id,
            "confidence": self.confidence,
            "reason": self.reason,
            "candidates": [candidate.to_json() for candidate in self.candidates],
            "model": self.model,
            "raw_response": self.raw_response,
            "rendered_prompt": self.rendered_prompt,
            "attempts": [attempt.to_json() for attempt in self.attempts],
        }


@dataclass(frozen=True)
class MaskQCResult:
    """QC reports for exactly the object roles declared by ``SemanticPlan``."""

    role_reports: tuple[RoleMaskQC, ...]
    selected_masks: dict[RoleName, Any]
    health: dict[str, Any]
    candidate_masks: dict[RoleName, dict[str, Any]] = field(default_factory=dict)
    candidate_panels: dict[RoleName, dict[str, Any]] = field(default_factory=dict)
    attempt_candidate_masks: dict[RoleName, dict[int, dict[str, Any]]] = field(default_factory=dict)
    attempt_candidate_panels: dict[RoleName, dict[int, dict[str, Any]]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        roles = tuple(report.role for report in self.role_reports)
        if not roles or roles[0] != "target" or len(set(roles)) != len(roles):
            raise ValueError("MaskQCResult requires unique roles beginning with target")
        if not set(self.selected_masks) <= set(roles):
            raise ValueError("selected mask exists for an unknown QC role")
        for role_artifacts, label in (
            (self.candidate_masks, "candidate masks"),
            (self.candidate_panels, "candidate panels"),
            (self.attempt_candidate_masks, "attempt candidate masks"),
            (self.attempt_candidate_panels, "attempt candidate panels"),
        ):
            if not set(role_artifacts) <= set(roles):
                raise ValueError(f"{label} exist for an unknown QC role")
        for report in self.role_reports:
            role = report.role
            has_selected_mask = role in self.selected_masks
            if (report.status is MaskQCStatus.PASSED) != has_selected_mask:
                raise ValueError("passed QC and selected seed masks must match")
            attempt_seeds = {attempt.seed_frame_id for attempt in report.attempts}
            for seed_artifacts, label in (
                (self.attempt_candidate_masks.get(role, {}), "attempt candidate masks"),
                (self.attempt_candidate_panels.get(role, {}), "attempt candidate panels"),
            ):
                if not set(seed_artifacts) <= attempt_seeds:
                    raise ValueError(f"{label} contain an unknown seed attempt")
                for seed_frame_id, candidates in seed_artifacts.items():
                    known_ids = {
                        candidate.candidate_id
                        for attempt in report.attempts
                        if attempt.seed_frame_id == seed_frame_id
                        for candidate in attempt.candidates
                    }
                    if not set(candidates) <= known_ids:
                        raise ValueError(f"{label} contain an unknown candidate id")

    def reports(self) -> tuple[RoleMaskQC, ...]:
        return self.role_reports

    def for_role(self, role: RoleName) -> RoleMaskQC:
        for report in self.role_reports:
            if report.role == role:
                return report
        raise KeyError(f"mask QC has no report for non-applicable role {role!r}")

    @property
    def target(self) -> RoleMaskQC:
        return self.for_role("target")

    @property
    def receiver(self) -> RoleMaskQC:
        return self.for_role("receiver")

    def to_json(self) -> dict[str, Any]:
        return {
            "format_version": "robotwin_mask_qc_v2",
            "roles": {report.role: report.to_json() for report in self.role_reports},
            "health": self.health,
        }
