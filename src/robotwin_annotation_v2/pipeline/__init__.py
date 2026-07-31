"""The three executable pipeline stages."""

from .state_loop import (
    StateLoopError,
    build_loop_context,
    detect_arm_loops,
    detect_episode_loop,
    detect_loop_events,
    sample_semantic_frames,
)
from .qwen_stage import (
    QwenStageError,
    QwenStageResult,
    RenderedQwenRequest,
    build_qwen_request,
    parse_semantic_plan,
    run_qwen_stage,
)
from .sam_stage import (
    RoleMaskData,
    SamStageError,
    SamStageResult,
    TemporalMaskQc,
    compose_visible_mask,
    dilate_envelope,
    evaluate_temporal_mask,
    run_sam_stage,
    save_sam_artifacts,
)

__all__ = [
    "StateLoopError",
    "QwenStageError",
    "QwenStageResult",
    "RenderedQwenRequest",
    "RoleMaskData",
    "SamStageError",
    "SamStageResult",
    "TemporalMaskQc",
    "build_loop_context",
    "build_qwen_request",
    "compose_visible_mask",
    "detect_arm_loops",
    "detect_episode_loop",
    "detect_loop_events",
    "dilate_envelope",
    "evaluate_temporal_mask",
    "parse_semantic_plan",
    "run_qwen_stage",
    "run_sam_stage",
    "save_sam_artifacts",
    "sample_semantic_frames",
]
