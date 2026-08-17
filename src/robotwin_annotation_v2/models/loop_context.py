"""Stage-1 context shared by pick/place and close-and-hold episodes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

from ..domain import AnnotationMode, AnnotationSpec, ObjectRole, annotation_spec
from .timeline import (
    EpisodeWindows,
    PickPlaceEvents,
    TargetOnlyEvents,
    TimelineEvents,
    derive_episode_windows,
)

RoleName = Literal["target", "receiver"]


class FramePurpose(StrEnum):
    """Why a sparse frame is submitted to Qwen."""

    PRE_GRASP_SEED_CANDIDATE = "pre_grasp_seed_candidate"
    POST_GRASP_CONTEXT = "post_grasp_context"
    PLACE_CONTEXT = "place_context"


@dataclass(frozen=True)
class EpisodeRef:
    """Stable reference to one episode and camera."""

    task: str
    episode_index: int
    camera: str = "cam_high"

    def __post_init__(self) -> None:
        if not self.task.strip():
            raise ValueError("task must be non-empty")
        if self.episode_index < 0:
            raise ValueError("episode_index must be non-negative")
        if not self.camera.strip():
            raise ValueError("camera must be non-empty")

    @property
    def episode_id(self) -> str:
        return f"{self.episode_index:06d}"

    def to_json(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "episode_index": self.episode_index,
            "episode_id": self.episode_id,
            "camera": self.camera,
        }


@dataclass(frozen=True)
class SemanticFrame:
    """One sparse RGB frame and its semantic purpose."""

    frame_id: int
    purpose: FramePurpose
    eligible_roles: tuple[RoleName, ...]

    def __post_init__(self) -> None:
        if self.frame_id < 0:
            raise ValueError("frame_id must be non-negative")
        if not self.eligible_roles:
            raise ValueError("eligible_roles must not be empty")
        if len(set(self.eligible_roles)) != len(self.eligible_roles):
            raise ValueError("eligible_roles must be unique")

    @property
    def seed_eligible(self) -> bool:
        return self.purpose is FramePurpose.PRE_GRASP_SEED_CANDIDATE

    def to_json(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "purpose": self.purpose.value,
            "eligible_roles": list(self.eligible_roles),
        }


@dataclass(frozen=True)
class LoopContext:
    """Complete, mode-validated Stage-1 output consumed by later stages.

    Event detection is mode-specific, while every downstream stage receives
    the same derived ``windows`` contract.  This is the only place where an
    annotation mode is paired with its concrete timeline type.
    """

    episode: EpisodeRef
    task_text: str
    frame_count: int
    events: TimelineEvents
    semantic_frames: tuple[SemanticFrame, ...]
    state_source: str
    video_source: str
    annotation_mode: AnnotationMode = AnnotationMode.PICK_PLACE
    windows: EpisodeWindows = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.annotation_mode, AnnotationMode):
            raise TypeError("annotation_mode must be an AnnotationMode")
        if not self.task_text.strip():
            raise ValueError("task_text must be non-empty")
        if self.frame_count < 1:
            raise ValueError("frame_count must be positive")
        expected_type = (
            PickPlaceEvents
            if self.annotation_mode is AnnotationMode.PICK_PLACE
            else TargetOnlyEvents
        )
        if not isinstance(self.events, expected_type):
            raise TypeError(
                f"annotation mode {self.annotation_mode.value} requires "
                f"{expected_type.__name__}, got {type(self.events).__name__}"
            )
        object.__setattr__(
            self,
            "windows",
            derive_episode_windows(self.events, frame_count=self.frame_count),
        )
        if not self.semantic_frames:
            raise ValueError("semantic_frames must not be empty")
        frame_ids = [frame.frame_id for frame in self.semantic_frames]
        if len(set(frame_ids)) != len(frame_ids):
            raise ValueError("semantic frame ids must be unique")
        if min(frame_ids) < 0 or max(frame_ids) >= self.frame_count:
            raise ValueError("semantic frame id is outside the episode")
        if not any(frame.seed_eligible for frame in self.semantic_frames):
            raise ValueError("at least one seed candidate is required")
        required = set(self.annotation_spec.required_role_names)
        supplied = {role for frame in self.semantic_frames for role in frame.eligible_roles}
        if not supplied <= required:
            raise ValueError(
                "semantic frames contain roles not required by annotation mode: "
                f"{sorted(supplied - required)}"
            )

    @property
    def annotation_spec(self) -> AnnotationSpec:
        return annotation_spec(self.annotation_mode)

    @property
    def timeline_kind(self) -> Literal["pick_place", "close_hold"]:
        """Stable JSON discriminator for the concrete event state machine."""

        if isinstance(self.events, PickPlaceEvents):
            return "pick_place"
        return "close_hold"

    def seed_candidates(self, role: RoleName) -> tuple[int, ...]:
        if not self.annotation_spec.requires(ObjectRole(role)):
            return ()
        return tuple(
            frame.frame_id
            for frame in self.semantic_frames
            if frame.seed_eligible and role in frame.eligible_roles
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "format_version": "robotwin_loop_context_v3",
            "annotation_mode": self.annotation_mode.value,
            "timeline_kind": self.timeline_kind,
            "required_object_roles": list(self.annotation_spec.required_role_names),
            "episode": self.episode.to_json(),
            "task_text": self.task_text,
            "frame_count": self.frame_count,
            "events": self.events.to_json(),
            "windows": self.windows.to_json(),
            "semantic_frames": [frame.to_json() for frame in self.semantic_frames],
            "sources": {
                "state": self.state_source,
                "video": self.video_source,
            },
        }
