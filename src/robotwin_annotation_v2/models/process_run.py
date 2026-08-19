"""Typed records for dataset-level process requests and summaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProcessRequest:
    """Shared dataset identity and selection passed to a workflow."""

    dataset_root: Path
    output_root: Path
    task: str
    camera: str
    run_id: str | None = None
    episode_ids: tuple[int, ...] | None = None
    skip_render: bool = False


@dataclass(frozen=True)
class EpisodeRecord:
    """One public process record while preserving extensible detail fields."""

    episode_id: int | None
    status: str
    details: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> EpisodeRecord:
        if "status" not in payload:
            raise ValueError("process record must contain status")
        raw_episode = payload.get("episode")
        episode_id = None if raw_episode is None else int(raw_episode)
        details = {
            key: value
            for key, value in payload.items()
            if key not in {"episode", "status"}
        }
        return cls(
            episode_id=episode_id,
            status=str(payload["status"]),
            details=details,
        )

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if self.episode_id is not None:
            payload["episode"] = self.episode_id
        payload["status"] = self.status
        payload.update(self.details)
        return payload


@dataclass(frozen=True)
class ProcessSummary:
    """Common SAM/URDF process summary envelope.

    ``stage_mode`` and ``plan`` are optional because they belong to different
    workflow result variants.  They are omitted, rather than serialized as
    null placeholders, to preserve the existing public JSON contract.
    """

    format_version: str
    annotation_mode: str
    required_object_roles: tuple[str, ...]
    gripper_backend: str | None
    run_id: str
    dataset_root: str
    task: str
    camera: str
    discovered_episode_ids: tuple[int, ...]
    requested_episode_ids: tuple[int, ...]
    dynamic_manifest: Mapping[str, Any]
    qwen_health: Mapping[str, Any] | None
    records: tuple[EpisodeRecord, ...]
    render: Mapping[str, Any] | None
    fatal_error: str | None
    backend: Mapping[str, Any]
    passed: bool
    stage_mode: str | None = None
    plan: Mapping[str, Any] | None = None
    artifact: str | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ProcessSummary:
        """Decode a summary produced by :meth:`to_json`."""

        raw_records = payload.get("records", ())
        if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes)):
            raise TypeError("process summary records must be a sequence")
        if any(not isinstance(record, Mapping) for record in raw_records):
            raise TypeError("process summary records must be objects")
        records = tuple(
            EpisodeRecord.from_payload(record)
            for record in raw_records
        )
        raw_roles = payload.get("required_object_roles", ())
        raw_discovered = payload.get("discovered_episode_ids", ())
        raw_requested = payload.get("requested_episode_ids", ())
        raw_qwen_health = payload.get("qwen_health")
        raw_render = payload.get("render")
        raw_plan = payload.get("plan")
        return cls(
            format_version=str(payload["format_version"]),
            annotation_mode=str(payload["annotation_mode"]),
            required_object_roles=tuple(str(value) for value in raw_roles),
            gripper_backend=(
                None
                if payload.get("gripper_backend") is None
                else str(payload["gripper_backend"])
            ),
            run_id=str(payload["run_id"]),
            dataset_root=str(payload["dataset_root"]),
            task=str(payload["task"]),
            camera=str(payload["camera"]),
            discovered_episode_ids=tuple(int(value) for value in raw_discovered),
            requested_episode_ids=tuple(int(value) for value in raw_requested),
            dynamic_manifest=dict(payload["dynamic_manifest"]),
            qwen_health=(
                None if not isinstance(raw_qwen_health, Mapping) else dict(raw_qwen_health)
            ),
            records=records,
            render=None if not isinstance(raw_render, Mapping) else dict(raw_render),
            fatal_error=(
                None if payload.get("fatal_error") is None else str(payload["fatal_error"])
            ),
            backend=dict(payload["backend"]),
            passed=bool(payload["passed"]),
            stage_mode=(
                None if payload.get("stage_mode") is None else str(payload["stage_mode"])
            ),
            plan=None if not isinstance(raw_plan, Mapping) else dict(raw_plan),
            artifact=(
                None if payload.get("artifact") is None else str(payload["artifact"])
            ),
        )

    def with_artifact(self, artifact: str) -> ProcessSummary:
        return replace(self, artifact=artifact)

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format_version": self.format_version,
            "annotation_mode": self.annotation_mode,
            "required_object_roles": list(self.required_object_roles),
            "gripper_backend": self.gripper_backend,
            "run_id": self.run_id,
            "dataset_root": self.dataset_root,
            "task": self.task,
            "camera": self.camera,
            "discovered_episode_ids": list(self.discovered_episode_ids),
            "requested_episode_ids": list(self.requested_episode_ids),
            "dynamic_manifest": dict(self.dynamic_manifest),
            "qwen_health": None if self.qwen_health is None else dict(self.qwen_health),
            "records": [record.to_json() for record in self.records],
            "render": None if self.render is None else dict(self.render),
            "fatal_error": self.fatal_error,
            "backend": dict(self.backend),
        }
        if self.stage_mode is not None:
            payload["stage_mode"] = self.stage_mode
        payload["passed"] = self.passed
        if self.plan is not None:
            payload["plan"] = dict(self.plan)
        if self.artifact is not None:
            payload["artifact"] = self.artifact
        return payload


__all__ = ["EpisodeRecord", "ProcessRequest", "ProcessSummary"]
