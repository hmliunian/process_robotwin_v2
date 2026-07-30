"""RoboTwin target/receiver annotation pipeline."""

__version__ = "0.2.0"

from .models import (
    EpisodeRef,
    FramePurpose,
    FrameWindow,
    LoopContext,
    LoopEvents,
    MaskRun,
    QueryBank,
    RoleMaskResult,
    RoleSemanticPlan,
    SemanticFrame,
    SemanticPlan,
)

__all__ = [
    "EpisodeRef",
    "FramePurpose",
    "FrameWindow",
    "LoopContext",
    "LoopEvents",
    "MaskRun",
    "QueryBank",
    "RoleMaskResult",
    "RoleSemanticPlan",
    "SemanticFrame",
    "SemanticPlan",
]
