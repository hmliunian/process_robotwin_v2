from __future__ import annotations

import numpy as np

from robotwin_annotation_v2.models import (
    EpisodeRef,
    FramePurpose,
    LoopContext,
    LoopEvents,
    MaskCandidateInfo,
    MaskQCAttemptMethod,
    MaskQCStatus,
    QueryBank,
    RoleMaskQC,
    RoleSemanticPlan,
    SemanticFrame,
    SemanticStatus,
)
from robotwin_annotation_v2.pipeline.object_mask.planner import QueryCandidate
from robotwin_annotation_v2.pipeline.object_mask.resolver import (
    ObjectMaskCandidate,
    ObjectMaskResolver,
    RoleAttemptExecution,
)


def _context() -> LoopContext:
    return LoopContext(
        episode=EpisodeRef("move_bottle_pad", 7),
        task_text="move the bottle to the pad",
        frame_count=20,
        events=LoopEvents("right", 1, 6, 8, 14, 17),
        semantic_frames=(
            SemanticFrame(0, FramePurpose.PRE_GRASP_SEED_CANDIDATE, ("target",)),
            SemanticFrame(3, FramePurpose.PRE_GRASP_SEED_CANDIDATE, ("target",)),
            SemanticFrame(5, FramePurpose.PRE_GRASP_SEED_CANDIDATE, ("target",)),
        ),
        state_source="state.parquet",
        video_source="video.mp4",
    )


def _semantic(status: SemanticStatus = SemanticStatus.OK) -> RoleSemanticPlan:
    return RoleSemanticPlan(
        role="target",
        status=status,
        seed_frame_id=None if status is SemanticStatus.NO_CLEAR_SEED else 0,
        query_bank=None if status is SemanticStatus.NO_CLEAR_SEED else QueryBank("bottle"),
        exclude=(),
        reason="test plan",
    )


def _attempt(
    seed_frame_id: int,
    status: MaskQCStatus,
    method: MaskQCAttemptMethod,
) -> RoleAttemptExecution:
    candidate_info = MaskCandidateInfo(
        candidate_id="A",
        query_field="category_query",
        query="bottle",
        nonempty=True,
        area_fraction=0.25,
        component_count=1,
        basic_valid=True,
        seed_frame_id=seed_frame_id,
    )
    candidate = ObjectMaskCandidate(
        "A",
        "category_query",
        "bottle",
        seed_frame_id,
        np.ones((2, 2), dtype=bool),
        candidate_info,
    )
    passed = status is MaskQCStatus.PASSED
    report = RoleMaskQC(
        role="target",
        status=status,
        selected_candidate="A" if passed else None,
        selected_query_field="category_query" if passed else None,
        selected_query="bottle" if passed else None,
        selected_seed_frame_id=seed_frame_id if passed else None,
        confidence=0.9 if passed else None,
        reason=status.value,
        candidates=(candidate_info,),
    )
    return RoleAttemptExecution(
        seed_frame_id,
        report,
        (candidate,),
        method=method,
        provenance={"method": method.value},
    )


def test_resolver_finishes_all_text_seeds_before_bbox() -> None:
    calls: list[tuple[str, int]] = []

    def run_text(
        seed_frame_id: int,
        _queries: tuple[QueryCandidate, ...],
    ) -> RoleAttemptExecution:
        calls.append(("text", seed_frame_id))
        return _attempt(seed_frame_id, MaskQCStatus.REJECTED, MaskQCAttemptMethod.TEXT_QUERY)

    def run_bbox(seed_frame_id: int) -> RoleAttemptExecution:
        calls.append(("bbox", seed_frame_id))
        return _attempt(seed_frame_id, MaskQCStatus.PASSED, MaskQCAttemptMethod.BBOX_FALLBACK)

    resolution = ObjectMaskResolver(
        run_text,
        run_bbox,
        query_fallback_enabled=False,
        seed_fallback_enabled=True,
        bbox_fallback_enabled=True,
    ).resolve(_context(), "target", _semantic())

    assert calls == [("text", 0), ("text", 3), ("text", 5), ("bbox", 0)]
    assert resolution.report.status is MaskQCStatus.PASSED
    assert [attempt.method for attempt in resolution.report.attempts] == [
        MaskQCAttemptMethod.TEXT_QUERY,
        MaskQCAttemptMethod.TEXT_QUERY,
        MaskQCAttemptMethod.TEXT_QUERY,
        MaskQCAttemptMethod.BBOX_FALLBACK,
    ]


def test_resolver_stops_on_text_error_and_preserves_provenance() -> None:
    calls: list[tuple[str, int]] = []

    def run_text(
        seed_frame_id: int,
        _queries: tuple[QueryCandidate, ...],
    ) -> RoleAttemptExecution:
        calls.append(("text", seed_frame_id))
        return _attempt(seed_frame_id, MaskQCStatus.ERROR, MaskQCAttemptMethod.TEXT_QUERY)

    def run_bbox(seed_frame_id: int) -> RoleAttemptExecution:
        calls.append(("bbox", seed_frame_id))
        return _attempt(seed_frame_id, MaskQCStatus.PASSED, MaskQCAttemptMethod.BBOX_FALLBACK)

    resolution = ObjectMaskResolver(
        run_text,
        run_bbox,
        query_fallback_enabled=False,
        seed_fallback_enabled=True,
        bbox_fallback_enabled=True,
    ).resolve(_context(), "target", _semantic())

    assert calls == [("text", 0)]
    assert resolution.report.status is MaskQCStatus.ERROR
    assert resolution.report.attempts[0].provenance == {"method": "text_query"}


def test_resolver_skips_attempt_ports_for_no_clear_seed() -> None:
    def unexpected(*_args: object) -> RoleAttemptExecution:
        raise AssertionError("attempt port must not run")

    resolver = ObjectMaskResolver(
        unexpected,
        unexpected,
        query_fallback_enabled=True,
        seed_fallback_enabled=True,
        bbox_fallback_enabled=True,
    )

    resolution = resolver.resolve(_context(), "target", _semantic(SemanticStatus.NO_CLEAR_SEED))

    assert resolution.report.status is MaskQCStatus.REJECTED
    assert resolution.report.reason == "semantic_plan_no_clear_seed"
    assert resolution.report.attempts == ()
