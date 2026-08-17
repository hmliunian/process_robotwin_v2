"""Data-driven semantic-role contracts for supported annotation modes.

Timeline state machines and their frame windows live in ``models.timeline``;
this module only declares which object roles apply and the default gripper
backend.  Keeping those decisions separate prevents a semantic role switch
from becoming a second, duplicated timeline implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class AnnotationMode(StrEnum):
    """Task-level annotation mode, supplied explicitly by configuration."""

    PICK_PLACE = "pick_place"
    TARGET_ONLY = "target_only"


class ObjectRole(StrEnum):
    """Semantic object channels produced by Qwen and SAM."""

    TARGET = "target"
    RECEIVER = "receiver"


class GripperBackend(StrEnum):
    """Replaceable producer for the active gripper mask."""

    URDF = "urdf"
    SAM = "sam"


@dataclass(frozen=True)
class AnnotationSpec:
    """Complete, immutable behavior switch for one annotation mode.

    ``canonical_object_roles`` remains fixed so downstream training data keeps
    the same four-channel schema.  Roles absent from ``required_object_roles``
    are published as zero masks with ``not_applicable`` provenance.
    """

    mode: AnnotationMode
    required_object_roles: tuple[ObjectRole, ...]
    default_gripper_backend: GripperBackend = GripperBackend.URDF

    def __post_init__(self) -> None:
        if not self.required_object_roles:
            raise ValueError("required_object_roles must not be empty")
        if self.required_object_roles[0] is not ObjectRole.TARGET:
            raise ValueError("target must be the first required object role")
        if len(set(self.required_object_roles)) != len(self.required_object_roles):
            raise ValueError("required_object_roles must be unique")

    @property
    def canonical_object_roles(self) -> tuple[ObjectRole, ObjectRole]:
        """Object channel order; deliberately independent of the mode."""

        return ObjectRole.TARGET, ObjectRole.RECEIVER

    @property
    def required_role_names(self) -> tuple[str, ...]:
        """String form used at JSON and model boundaries."""

        return tuple(role.value for role in self.required_object_roles)

    def requires(self, role: ObjectRole | str) -> bool:
        """Return whether ``role`` participates in semantic/QC/SAM stages."""

        return ObjectRole(role) in self.required_object_roles

ANNOTATION_SPECS: dict[AnnotationMode, AnnotationSpec] = {
    AnnotationMode.PICK_PLACE: AnnotationSpec(
        mode=AnnotationMode.PICK_PLACE,
        required_object_roles=(ObjectRole.TARGET, ObjectRole.RECEIVER),
    ),
    AnnotationMode.TARGET_ONLY: AnnotationSpec(
        mode=AnnotationMode.TARGET_ONLY,
        required_object_roles=(ObjectRole.TARGET,),
    ),
}


def annotation_spec(mode: AnnotationMode | str) -> AnnotationSpec:
    """Resolve an annotation mode without task-name inference."""

    try:
        resolved = AnnotationMode(mode)
    except ValueError as exc:
        choices = ", ".join(item.value for item in AnnotationMode)
        raise ValueError(f"unsupported annotation mode {mode!r}; choose {choices}") from exc
    return ANNOTATION_SPECS[resolved]
