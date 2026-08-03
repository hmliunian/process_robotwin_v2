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

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ValueError("candidate_id must be non-empty")
        if not self.query_field.strip() or not self.query.strip():
            raise ValueError("candidate query metadata must be non-empty")
        if not 0.0 <= self.area_fraction <= 1.0:
            raise ValueError("area_fraction must be between 0 and 1")
        if self.component_count < 0:
            raise ValueError("component_count must be non-negative")

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
    candidates: tuple[MaskCandidateInfo, ...] = ()
    model: str | None = None
    raw_response: str | None = None
    rendered_prompt: str | None = None

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("QC reason must be non-empty")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("QC confidence must be between 0 and 1")
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

    def to_json(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "status": self.status.value,
            "selected_candidate": self.selected_candidate,
            "selected_query_field": self.selected_query_field,
            "selected_query": self.selected_query,
            "confidence": self.confidence,
            "reason": self.reason,
            "candidates": [candidate.to_json() for candidate in self.candidates],
            "model": self.model,
            "raw_response": self.raw_response,
            "rendered_prompt": self.rendered_prompt,
        }


@dataclass(frozen=True)
class MaskQCResult:
    """Joint QC result and the selected seed masks used by Stage 3."""

    target: RoleMaskQC
    receiver: RoleMaskQC
    selected_masks: dict[RoleName, Any]
    health: dict[str, Any]
    candidate_masks: dict[RoleName, dict[str, Any]] = field(default_factory=dict)
    candidate_panels: dict[RoleName, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.target.role != "target" or self.receiver.role != "receiver":
            raise ValueError("MaskQCResult must contain target then receiver reports")
        for role, report in (("target", self.target), ("receiver", self.receiver)):
            has_selected_mask = role in self.selected_masks
            if (report.status is MaskQCStatus.PASSED) != has_selected_mask:
                raise ValueError("passed QC and selected seed masks must match")

    def reports(self) -> tuple[RoleMaskQC, RoleMaskQC]:
        return self.target, self.receiver

    def to_json(self) -> dict[str, Any]:
        return {
            "format_version": "robotwin_mask_qc_v1",
            "roles": {
                "target": self.target.to_json(),
                "receiver": self.receiver.to_json(),
            },
            "health": self.health,
        }
