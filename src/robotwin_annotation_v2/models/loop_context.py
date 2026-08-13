"""Stage-1 contracts for one RoboTwin pick-and-place loop."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Literal

from ..domain import AnnotationMode, AnnotationSpec, ObjectRole, annotation_spec

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
class FrameWindow:
    """Inclusive frame window ``[start, end]``."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"invalid frame window [{self.start}, {self.end}]")

    def __contains__(self, frame_id: int) -> bool:
        return self.start <= frame_id <= self.end

    def __len__(self) -> int:
        return self.end - self.start + 1

    def to_json(self) -> list[int]:
        return [self.start, self.end]


@dataclass(frozen=True)
class LoopEvents:
    """Five ordered state-derived event boundaries for one arm loop."""

    active_arm: Literal["left", "right"]
    t_move_start: int
    t_close_start: int
    t_close_done: int
    t_open_start: int
    t_open_done: int

    def __post_init__(self) -> None:
        values = (
            self.t_move_start,
            self.t_close_start,
            self.t_close_done,
            self.t_open_start,
            self.t_open_done,
        )
        if min(values) < 0:
            raise ValueError("loop event frames must be non-negative")
        if not (
            self.t_move_start <= self.t_close_start
            < self.t_close_done
            < self.t_open_start
            < self.t_open_done
        ):
            raise ValueError(f"loop events are not ordered: {values}")

    @property
    def loop_window(self) -> FrameWindow:
        return FrameWindow(self.t_move_start, self.t_open_done)

    @property
    def target_window(self) -> FrameWindow:
        return FrameWindow(self.t_move_start, self.t_close_done)

    @property
    def receiver_window(self) -> FrameWindow:
        return FrameWindow(self.t_close_done, self.t_open_done)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


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
    """Complete Stage-1 output consumed by the Qwen stage."""

    episode: EpisodeRef
    task_text: str
    frame_count: int
    events: LoopEvents
    semantic_frames: tuple[SemanticFrame, ...]
    state_source: str
    video_source: str
    annotation_mode: AnnotationMode = AnnotationMode.PICK_PLACE

    def __post_init__(self) -> None:
        if not self.task_text.strip():
            raise ValueError("task_text must be non-empty")
        if self.frame_count < 1:
            raise ValueError("frame_count must be positive")
        if self.events.t_open_done >= self.frame_count:
            raise ValueError("loop extends beyond the episode")
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
        supplied = {
            role
            for frame in self.semantic_frames
            for role in frame.eligible_roles
        }
        if not supplied <= required:
            raise ValueError(
                "semantic frames contain roles not required by annotation mode: "
                f"{sorted(supplied - required)}"
            )

    @property
    def annotation_spec(self) -> AnnotationSpec:
        return annotation_spec(self.annotation_mode)

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
            "format_version": "robotwin_loop_context_v1",
            "annotation_mode": self.annotation_mode.value,
            "required_object_roles": list(self.annotation_spec.required_role_names),
            "episode": self.episode.to_json(),
            "task_text": self.task_text,
            "frame_count": self.frame_count,
            "events": self.events.to_json(),
            "windows": {
                "loop": self.events.loop_window.to_json(),
                "target_0": self.events.target_window.to_json(),
                "receiver_0": (
                    self.events.receiver_window.to_json()
                    if self.annotation_spec.requires(ObjectRole.RECEIVER)
                    else None
                ),
            },
            "semantic_frames": [frame.to_json() for frame in self.semantic_frames],
            "sources": {
                "state": self.state_source,
                "video": self.video_source,
            },
        }
