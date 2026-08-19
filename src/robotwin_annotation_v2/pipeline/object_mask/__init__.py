"""Pure object-mask algorithms shared by pipeline stages."""

from .planner import QueryCandidate, plan_role_queries
from .qc import MaskQCError, candidate_info, mask_iou
from .temporal_qc import (
    TemporalMaskQc,
    compose_visible_mask,
    evaluate_temporal_mask,
)

__all__ = [
    "MaskQCError",
    "QueryCandidate",
    "TemporalMaskQc",
    "candidate_info",
    "compose_visible_mask",
    "evaluate_temporal_mask",
    "mask_iou",
    "plan_role_queries",
]
