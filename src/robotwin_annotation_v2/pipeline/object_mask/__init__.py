"""Pure object-mask algorithms shared by pipeline stages."""

from .planner import QueryCandidate, plan_role_queries
from .proposals import blue_planar_region, largest_component
from .qc import MaskQCError, candidate_info, mask_iou
from .resolver import (
    ObjectMaskCandidate,
    ObjectMaskResolver,
    RoleAttemptExecution,
    RoleResolution,
)
from .temporal_qc import (
    TemporalMaskQc,
    compose_visible_mask,
    evaluate_temporal_mask,
)

__all__ = [
    "MaskQCError",
    "ObjectMaskCandidate",
    "ObjectMaskResolver",
    "QueryCandidate",
    "RoleAttemptExecution",
    "RoleResolution",
    "TemporalMaskQc",
    "blue_planar_region",
    "candidate_info",
    "compose_visible_mask",
    "evaluate_temporal_mask",
    "largest_component",
    "mask_iou",
    "plan_role_queries",
]
