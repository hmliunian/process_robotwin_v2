"""Pure object-mask algorithms shared by pipeline stages."""

from .planner import QueryCandidate, plan_role_queries
from .temporal_qc import (
    TemporalMaskQc,
    compose_visible_mask,
    evaluate_temporal_mask,
)

__all__ = [
    "QueryCandidate",
    "TemporalMaskQc",
    "compose_visible_mask",
    "evaluate_temporal_mask",
    "plan_role_queries",
]
