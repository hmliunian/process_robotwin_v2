"""Stable objects passed between the three pipeline stages."""

from .loop_context import (
    EpisodeRef,
    FramePurpose,
    LoopContext,
    SemanticFrame,
)
from .mask_qc import MaskCandidateInfo, MaskQCResult, MaskQCStatus, RoleMaskQC
from .mask_run import MaskRun, MaskStatus, RoleMaskResult
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
)

__all__ = [
    "CANDIDATE_FIELDS",
    "MAX_QUERY_WORDS",
    "EpisodeRef",
    "EpisodeWindows",
    "FramePurpose",
    "FrameWindow",
    "LoopContext",
    "LoopEvents",
    "MaskCandidateInfo",
    "MaskQCResult",
    "MaskQCStatus",
    "MaskRun",
    "MaskStatus",
    "PickPlaceEvents",
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
    "normalize_query",
]
