"""Stable domain vocabulary shared by configuration and pipeline stages."""

from .annotation_spec import (
    ANNOTATION_SPECS,
    CONTACT_PRESS_TASK_KINDS,
    AnnotationMode,
    AnnotationSpec,
    GripperBackend,
    ObjectRole,
    TargetOnlyTaskKind,
    TargetProfile,
    annotation_spec,
    target_profile_for_task_kind,
)

__all__ = [
    "ANNOTATION_SPECS",
    "CONTACT_PRESS_TASK_KINDS",
    "AnnotationMode",
    "AnnotationSpec",
    "GripperBackend",
    "ObjectRole",
    "TargetOnlyTaskKind",
    "TargetProfile",
    "annotation_spec",
    "target_profile_for_task_kind",
]
