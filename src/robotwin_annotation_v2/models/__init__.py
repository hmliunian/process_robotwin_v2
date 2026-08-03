"""Stable objects passed between the three pipeline stages."""

from .loop_context import (
    EpisodeRef,
    FramePurpose,
    FrameWindow,
    LoopContext,
    LoopEvents,
    SemanticFrame,
)
from .mask_run import MaskRun, MaskStatus, RoleMaskResult
from .mask_qc import MaskCandidateInfo, MaskQCResult, MaskQCStatus, RoleMaskQC
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

__all__ = [
    "CANDIDATE_FIELDS",
    "MAX_QUERY_WORDS",
    "EpisodeRef",
    "FramePurpose",
    "FrameWindow",
    "LoopContext",
    "LoopEvents",
    "MaskRun",
    "MaskStatus",
    "MaskCandidateInfo",
    "MaskQCResult",
    "MaskQCStatus",
    "QueryBank",
    "RoleMaskResult",
    "RoleMaskQC",
    "RoleSemanticPlan",
    "SemanticFrame",
    "SemanticPlan",
    "SemanticPlanError",
    "SemanticStatus",
    "normalize_query",
]
