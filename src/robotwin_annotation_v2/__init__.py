"""RoboTwin target/receiver annotation pipeline."""

from ._version import __version__
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
    "__version__",
]
