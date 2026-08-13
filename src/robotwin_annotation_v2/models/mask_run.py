"""Stage-3 mask output metadata; pixel arrays stay in artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

from .loop_context import FrameWindow
from .mask_qc import MaskQCStatus


class MaskStatus(StrEnum):
    OK = "ok"
    FAILED = "failed"
    QUARANTINED = "quarantined"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class RoleMaskResult:
    role: Literal["target", "receiver", "gripper_left", "gripper_right"]
    status: MaskStatus
    seed_frame_id: int | None
    primary_query: str | None
    output_window: FrameWindow | None
    seed_rgb_path: str | None
    seed_mask_path: str | None
    canonical_envelope_path: str | None
    native_track_path: str | None
    temporal_qc_path: str | None
    nonempty_frames: int
    failure: str | None = None
    qc_status: MaskQCStatus = MaskQCStatus.NOT_RUN
    qc_selected_candidate: str | None = None
    qc_reason: str | None = None

    def __post_init__(self) -> None:
        if self.nonempty_frames < 0:
            raise ValueError("nonempty_frames must be non-negative")
        if self.status in {MaskStatus.OK, MaskStatus.NOT_APPLICABLE} and self.failure is not None:
            raise ValueError("successful role result cannot contain failure")
        if self.status in {MaskStatus.FAILED, MaskStatus.QUARANTINED} and not self.failure:
            raise ValueError("failed or quarantined role result must contain failure")
        if self.status is MaskStatus.NOT_APPLICABLE and any(
            value is not None
            for value in (
                self.seed_frame_id,
                self.primary_query,
                self.output_window,
                self.seed_rgb_path,
                self.seed_mask_path,
                self.canonical_envelope_path,
                self.native_track_path,
                self.temporal_qc_path,
            )
        ):
            raise ValueError("not_applicable role cannot contain annotation artifacts")
        if self.status is MaskStatus.NOT_APPLICABLE and self.nonempty_frames != 0:
            raise ValueError("not_applicable role must be empty")
        if self.status is not MaskStatus.NOT_APPLICABLE and self.output_window is None:
            raise ValueError("applicable role requires an output window")
        if self.qc_status is not MaskQCStatus.PASSED and self.qc_selected_candidate is not None:
            raise ValueError("only passed QC may select a candidate")
        if self.qc_status not in {
            MaskQCStatus.NOT_RUN,
            MaskQCStatus.NOT_APPLICABLE,
        } and not self.qc_reason:
            raise ValueError("executed QC must include a reason")

    def to_json(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "status": self.status.value,
            "seed_frame_id": self.seed_frame_id,
            "primary_query": self.primary_query,
            "output_window": (
                None if self.output_window is None else self.output_window.to_json()
            ),
            "seed_rgb_path": self.seed_rgb_path,
            "seed_mask_path": self.seed_mask_path,
            "canonical_envelope_path": self.canonical_envelope_path,
            "native_track_path": self.native_track_path,
            "temporal_qc_path": self.temporal_qc_path,
            "nonempty_frames": self.nonempty_frames,
            "failure": self.failure,
            "qc_status": self.qc_status.value,
            "qc_selected_candidate": self.qc_selected_candidate,
            "qc_reason": self.qc_reason,
        }


@dataclass(frozen=True)
class MaskRun:
    """Complete Stage-3 result for one episode."""

    run_id: str
    episode: dict[str, Any]
    frame_count: int
    roles: tuple[RoleMaskResult, ...]
    artifact_dir: str
    format_version: str = "robotwin_mask_run_v2"

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id must be non-empty")
        if self.frame_count < 1:
            raise ValueError("frame_count must be positive")
        names = [result.role for result in self.roles]
        if names[:2] != ["target", "receiver"]:
            raise ValueError("MaskRun must begin with target then receiver results")
        extras = names[2:]
        if any(role not in {"gripper_left", "gripper_right"} for role in extras):
            raise ValueError("MaskRun optional roles must be gripper_left/right")
        if len(extras) != len(set(extras)):
            raise ValueError("MaskRun optional gripper roles must be unique")

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
