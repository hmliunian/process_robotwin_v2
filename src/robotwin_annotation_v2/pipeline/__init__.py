"""Pipeline stages with lazy compatibility exports.

Internal code imports concrete stage modules.  These lazy exports preserve the
existing public surface without loading OpenCV-backed gripper code on every
pipeline package import.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_GROUPS = {
    ".gripper_stage": (
        "CAM_HIGH_CALIBRATION",
        "DEFAULT_GRIPPER_ROI_GEOMETRY",
        "CameraCalibration",
        "GripperRoiGeometry",
        "GripperSeedCandidate",
        "GripperSeedQCResult",
        "GripperSeedQualityGateConfig",
        "GripperStageError",
        "GripperStageResult",
        "GripperTrackResult",
        "KnownObjectTracks",
        "ObjectExclusionResult",
        "ProjectedGripperRoi",
        "apply_gripper_seed_quality_gate",
        "build_gripper_qwen_request",
        "build_gripper_seed_candidate",
        "compose_gripper_track",
        "exclude_known_objects",
        "gripper_keyframes",
        "load_qc_native_object_tracks",
        "mark_same_frame_duplicates",
        "normalized_roi_box",
        "phase_for_frame",
        "project_gripper_roi",
        "render_gripper_candidate_panel",
        "render_gripper_candidate_sheet",
        "rotation_from_rpy",
        "run_gripper_seed_qc",
        "run_gripper_stage",
    ),
    ".mask_qc": (
        "MaskQCError",
        "parse_mask_qc_response",
        "run_mask_qc_stage",
    ),
    ".object_mask.artifacts": ("save_mask_qc_artifacts",),
    "..application.sam_artifacts": ("save_sam_artifacts",),
    ".open_set_queries": ("curated_query_aliases",),
    ".qwen_stage": (
        "QwenStageError",
        "QwenStageResult",
        "RenderedQwenRequest",
        "build_qwen_request",
        "parse_semantic_plan",
        "run_qwen_stage",
    ),
    ".sam_stage": (
        "RoleMaskData",
        "SamStageError",
        "SamStageResult",
        "TemporalMaskQc",
        "compose_visible_mask",
        "dilate_envelope",
        "evaluate_temporal_mask",
        "run_sam_stage",
    ),
    ".timeline_detector": (
        "StateLoopError",
        "detect_arm_loops",
        "detect_episode_loop",
        "detect_episode_target_only",
        "detect_loop_events",
        "detect_target_only_events",
    ),
    ".state_loop": (
        "build_loop_context",
        "sample_semantic_frames",
    ),
}
_EXPORTS = {
    name: module_name
    for module_name, names in _EXPORT_GROUPS.items()
    for name in names
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load the concrete stage that owns a requested compatibility export."""

    try:
        module_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
