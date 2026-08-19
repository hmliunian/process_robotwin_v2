"""Stable objects passed between the three pipeline stages."""

from .loop_context import (
    EpisodeRef,
    FramePurpose,
    LoopContext,
    SemanticFrame,
)
from .mask_qc import (
    MaskCandidateInfo,
    MaskQCAttempt,
    MaskQCAttemptMethod,
    MaskQCResult,
    MaskQCStatus,
    RoleMaskQC,
)
from .mask_run import MaskRun, MaskStatus, RoleMaskResult
from .process_run import EpisodeRecord, ProcessRequest, ProcessSummary
from .semantic_plan import (
    CANDIDATE_FIELDS,
    MAX_QUERY_WORDS,
    QueryBank,
    RoleSemanticPlan,
    SemanticPlan,
    SemanticPlanError,
    SemanticStatus,
    normalize_query,
)
from .timeline import (
    EpisodeWindows,
    FrameWindow,
    LoopEvents,
    PickPlaceEvents,
    TargetOnlyEvents,
    TimelineEvents,
    derive_episode_windows,
    derive_target_hold_window,
)

__all__ = [
    "CANDIDATE_FIELDS",
    "MAX_QUERY_WORDS",
    "EpisodeRecord",
    "EpisodeRef",
    "EpisodeWindows",
    "FramePurpose",
    "FrameWindow",
    "LoopContext",
    "LoopEvents",
    "MaskCandidateInfo",
    "MaskQCAttempt",
    "MaskQCAttemptMethod",
    "MaskQCResult",
    "MaskQCStatus",
    "MaskRun",
    "MaskStatus",
    "PickPlaceEvents",
    "ProcessRequest",
    "ProcessSummary",
    "QueryBank",
    "RoleMaskQC",
    "RoleMaskResult",
    "RoleSemanticPlan",
    "SemanticFrame",
    "SemanticPlan",
    "SemanticPlanError",
    "SemanticStatus",
    "TargetOnlyEvents",
    "TimelineEvents",
    "derive_episode_windows",
    "derive_target_hold_window",
    "normalize_query",
]
