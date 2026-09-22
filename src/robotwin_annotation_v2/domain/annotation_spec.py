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
    TOOL_USE = "tool_use"


class TimelineSource(StrEnum):
    """Dataset source used to establish episode event boundaries."""

    ROBOT_STATE = "robot_state"
    EPISODE_METADATA = "episode_metadata"
    VIDEO_WINDOW = "video_window"


class TargetProfile(StrEnum):
    """Target identity semantics selected independently of the timeline mode."""

    GRASP_MANIPULATION = "grasp_manipulation"
    CONTACT_PRESS = "contact_press"
    DOOR_OPEN = "door_open"
    VIDEO_OBJECT = "video_object"


class TargetOnlyTaskKind(StrEnum):
    """Manifest vocabulary used to choose target-only semantic behavior."""

    SINGLE_MOVABLE_TARGET = "single_movable_target"
    SINGLE_MOVABLE_TARGET_CONDITIONAL = "single_movable_target_conditional"
    CONTACT_ACTION_SITE = "contact_action_site"
    ARTICULATED_ACTION_SITE = "articulated_action_site"
    DOOR_OPEN_ACTION_SITE = "door_open_action_site"
    VIDEO_OBJECT = "video_object"


def target_profile_for_task_kind(
    task_kind: TargetOnlyTaskKind | str | None,
) -> TargetProfile:
    """Map manifest ``task_kind`` to semantic profile without task-name inference.

    Older target-only extracts do not declare ``task_kind``.  They retain the
    original grasp-manipulation behavior, while unknown declared values fail
    closed instead of silently selecting the wrong prompt profile.
    """

    if task_kind is None:
        return TargetProfile.GRASP_MANIPULATION
    try:
        resolved = TargetOnlyTaskKind(task_kind)
    except (TypeError, ValueError) as exc:
        choices = ", ".join(item.value for item in TargetOnlyTaskKind)
        raise ValueError(
            f"unsupported target-only task_kind {task_kind!r}; choose {choices}"
        ) from exc
    if resolved is TargetOnlyTaskKind.CONTACT_ACTION_SITE:
        return TargetProfile.CONTACT_PRESS
    if resolved is TargetOnlyTaskKind.DOOR_OPEN_ACTION_SITE:
        return TargetProfile.DOOR_OPEN
    if resolved is TargetOnlyTaskKind.VIDEO_OBJECT:
        return TargetProfile.VIDEO_OBJECT
    return TargetProfile.GRASP_MANIPULATION


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
    AnnotationMode.TOOL_USE: AnnotationSpec(
        mode=AnnotationMode.TOOL_USE,
        required_object_roles=(ObjectRole.TARGET, ObjectRole.RECEIVER),
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
