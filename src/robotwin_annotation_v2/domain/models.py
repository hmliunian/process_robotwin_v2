"""Domain models for keyframe annotation system.

These are immutable value objects that represent core business concepts.
No dependencies on external libraries (SAM3, numpy, etc).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal


class AnnotationRole(StrEnum):
    """Role of an instance in the manipulation task."""
    TARGET = "target"
    RECEIVER = "receiver"
    GRIPPER = "gripper"


class AnchorKind(StrEnum):
    """Type of keyframe anchor point."""
    PRE_GRASP_VISIBLE = "pre_grasp_visible"              # target before grasp
    STATIC_RECEIVER_VISIBLE = "static_receiver_visible"  # receiver unoccluded
    PRE_CLOSE_OPEN = "pre_close_open"                    # gripper before close
    POST_OPEN = "post_open"                              # gripper after release


class ReviewStatus(StrEnum):
    """Review state of a keyframe request."""
    DRAFT = "draft"
    CANDIDATES_GENERATED = "candidates_generated"
    AUTO_REVIEWED = "auto_reviewed"
    NEEDS_HUMAN_REVIEW = "needs_human_review"
    APPROVED = "approved"
    REJECTED = "rejected"


class SegmentationMethod(StrEnum):
    """SAM3 prompting method."""
    TEXT_ONLY = "text_only"
    BOX_ONLY = "box_only"
    TEXT_BOX = "text_box"  # text + bbox in single request


@dataclass(frozen=True)
class EpisodeRef:
    """Reference to a RoboTwin episode."""
    coarse_task: str
    episode_id: str
    camera: str = "cam_high"

    def __str__(self) -> str:
        return f"{self.coarse_task}/episode_{self.episode_id}/{self.camera}"


@dataclass(frozen=True)
class InstanceSlot:
    """Instance to annotate (e.g., target_0, receiver_0, gripper_left)."""
    name: str
    role: AnnotationRole
    arm: Literal["left", "right"] | None = None

    def __post_init__(self) -> None:
        if self.role == AnnotationRole.GRIPPER and self.arm is None:
            raise ValueError("Gripper slot must specify arm")
        if self.role != AnnotationRole.GRIPPER and self.arm is not None:
            raise ValueError("Only gripper slots can specify arm")


@dataclass(frozen=True)
class FrameWindow:
    """Inclusive frame range [first, last]."""
    first: int
    last: int

    def __post_init__(self) -> None:
        if self.first < 0 or self.last < 0:
            raise ValueError("Frame indices must be non-negative")
        if self.first > self.last:
            raise ValueError(f"Invalid window: first={self.first} > last={self.last}")

    def __contains__(self, frame: int) -> bool:
        return self.first <= frame <= self.last

    def __len__(self) -> int:
        return self.last - self.first + 1


@dataclass(frozen=True)
class Box:
    """Normalized bounding box [0, 1]."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        if not (0 <= self.x_min <= 1 and 0 <= self.y_min <= 1 and
                0 <= self.x_max <= 1 and 0 <= self.y_max <= 1):
            raise ValueError("Box coordinates must be in [0, 1]")
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("Invalid box: min >= max")

    def area(self) -> float:
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)


@dataclass(frozen=True)
class VisualPrompt:
    """Prompt for single-frame segmentation."""
    text: str | None = None
    bbox: Box | None = None

    def __post_init__(self) -> None:
        if self.text is None and self.bbox is None:
            raise ValueError("VisualPrompt must have at least text or bbox")


@dataclass(frozen=True)
class KeyframeRequest:
    """Request to find and segment a keyframe for one instance."""
    request_id: str
    episode: EpisodeRef
    slot: InstanceSlot
    anchor_kind: AnchorKind
    allowed_window: FrameWindow
    visual_query: str
    exclusions: tuple[str, ...] = ()  # instance names to exclude
    revision: int = 1

    def next_revision(self) -> KeyframeRequest:
        """Create next revision of this request."""
        return KeyframeRequest(
            request_id=self.request_id,
            episode=self.episode,
            slot=self.slot,
            anchor_kind=self.anchor_kind,
            allowed_window=self.allowed_window,
            visual_query=self.visual_query,
            exclusions=self.exclusions,
            revision=self.revision + 1,
        )


@dataclass(frozen=True)
class MaskArtifactRef:
    """Reference to a stored mask artifact."""
    sha256: str
    relative_path: str


@dataclass(frozen=True)
class ApprovedSeed:
    """Immutable approved keyframe seed (Phase 2 input)."""
    request_id: str
    candidate_id: str
    frame_index: int
    slot: InstanceSlot
    mask_artifact: MaskArtifactRef
    approval_revision: int
    reviewer: str
    note: str = ""
