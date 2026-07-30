"""Stage-3 mask output metadata; pixel arrays stay in artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

from .loop_context import FrameWindow


class MaskStatus(StrEnum):
    OK = "ok"
    FAILED = "failed"


@dataclass(frozen=True)
class RoleMaskResult:
    role: Literal["target", "receiver"]
    status: MaskStatus
    seed_frame_id: int | None
    primary_query: str | None
    output_window: FrameWindow
    seed_mask_path: str | None
    native_track_path: str | None
    text_observation_path: str | None
    nonempty_frames: int
    failure: str | None = None

    def __post_init__(self) -> None:
        if self.nonempty_frames < 0:
            raise ValueError("nonempty_frames must be non-negative")
        if self.status is MaskStatus.OK and self.failure is not None:
            raise ValueError("successful role result cannot contain failure")
        if self.status is MaskStatus.FAILED and not self.failure:
            raise ValueError("failed role result must contain failure")

    def to_json(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "status": self.status.value,
            "seed_frame_id": self.seed_frame_id,
            "primary_query": self.primary_query,
            "output_window": self.output_window.to_json(),
            "seed_mask_path": self.seed_mask_path,
            "native_track_path": self.native_track_path,
            "text_observation_path": self.text_observation_path,
            "nonempty_frames": self.nonempty_frames,
            "failure": self.failure,
        }


@dataclass(frozen=True)
class MaskRun:
    """Complete Stage-3 result for one episode."""

    run_id: str
    episode: dict[str, Any]
    frame_count: int
    roles: tuple[RoleMaskResult, ...]
    artifact_dir: str
    format_version: str = "robotwin_mask_run_v1"

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id must be non-empty")
        if self.frame_count < 1:
            raise ValueError("frame_count must be positive")
        names = [result.role for result in self.roles]
        if names != ["target", "receiver"]:
            raise ValueError("MaskRun must contain target then receiver results")

    def to_json(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "run_id": self.run_id,
            "episode": self.episode,
            "frame_count": self.frame_count,
            "roles": [result.to_json() for result in self.roles],
            "artifact_dir": self.artifact_dir,
            "channels": {
                "target_0": 0,
                "receiver_0": 1,
                "gripper_left": "not_annotated",
                "gripper_right": "not_annotated",
            },
        }
