"""Stable domain vocabulary shared by configuration and pipeline stages."""

from .annotation_spec import (
    ANNOTATION_SPECS,
    AnnotationMode,
    AnnotationSpec,
    GripperBackend,
    ObjectRole,
    annotation_spec,
)

__all__ = [
    "ANNOTATION_SPECS",
    "AnnotationMode",
    "AnnotationSpec",
    "GripperBackend",
    "ObjectRole",
    "annotation_spec",
]
