"""Read-only adapter for the extracted RoboTwin coverage20 dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import av
import numpy as np
import pandas as pd
from PIL import Image

from ..models import EpisodeRef


class DatasetError(RuntimeError):
    """The external dataset does not satisfy the checked contract."""


@dataclass(frozen=True)
class EpisodePaths:
    parquet: Path
    video: Path
    sidecar: Path


@dataclass(frozen=True)
class EpisodeState:
    frame_count: int
    task_text: str
    gripper_states: np.ndarray
    eef_states: np.ndarray
    paths: EpisodePaths


class RoboTwinDataset:
    """Resolve metadata, state arrays and sparse RGB frames by global episode id."""

    def __init__(
        self,
        root: Path,
        *,
        task: str,
        camera: str,
        manifest_path: Path,
        manifest_data: dict[str, Any] | None = None,
    ) -> None:
        self.root = root.expanduser().resolve()
        self.task = task
        self.camera = camera
        self.manifest_path = manifest_path.expanduser().resolve()
        self._manifest_data = None if manifest_data is None else dict(manifest_data)
        self.manifest = self._load_manifest()
        self._episode_metadata: dict[int, dict[str, Any]] | None = None

    def _load_manifest(self) -> dict[str, Any]:
        if self._manifest_data is None:
            if not self.manifest_path.is_file():
                raise DatasetError(f"dataset manifest is missing: {self.manifest_path}")
            with self.manifest_path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
        else:
            payload = dict(self._manifest_data)
        if not isinstance(payload, dict):
            raise DatasetError("dataset manifest must be a JSON object")
        if payload.get("task") != self.task or payload.get("camera") != self.camera:
            raise DatasetError("dataset config does not match manifest task/camera")
        manifest_root = Path(str(payload.get("dataset_root", ""))).expanduser().resolve()
        if manifest_root != self.root:
            raise DatasetError(
                f"dataset root does not match manifest: {self.root} != {manifest_root}"
            )
        return payload

    @staticmethod
    def _chunk(episode_index: int) -> str:
        return f"chunk-{episode_index // 1000:03d}"

    def paths(self, ref: EpisodeRef) -> EpisodePaths:
        if ref.task != self.task or ref.camera != self.camera:
            raise DatasetError(f"episode ref does not match dataset: {ref}")
        episode_id = ref.episode_id
        chunk = self._chunk(ref.episode_index)
        return EpisodePaths(
            parquet=self.root / "data" / chunk / f"episode_{episode_id}.parquet",
            video=(
                self.root
                / "videos"
                / chunk
                / f"observation.images.{ref.camera}"
                / f"episode_{episode_id}.mp4"
            ),
            sidecar=self.root / "sidecars" / f"episode_{episode_id}.hdf5",
        )

    def _metadata_index(self) -> dict[int, dict[str, Any]]:
        if self._episode_metadata is not None:
            return self._episode_metadata
        path = self.root / "meta" / "episodes.jsonl"
        if not path.is_file():
            raise DatasetError(f"episode metadata is missing: {path}")
        wanted = set(int(value) for value in self.manifest["regression_episode_ids"])
        index: dict[int, dict[str, Any]] = {}
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                episode_index = int(payload["episode_index"])
                if episode_index in wanted:
                    index[episode_index] = payload
        self._episode_metadata = index
        return index

    def task_text(self, episode_index: int) -> str:
        payload = self._metadata_index().get(episode_index)
        if payload is None:
            raise DatasetError(f"episode {episode_index} is absent from episodes.jsonl")
        tasks = payload.get("tasks", [])
        if not isinstance(tasks, list) or not tasks or not isinstance(tasks[0], str):
            raise DatasetError(f"episode {episode_index} has no task text")
        return tasks[0].strip()

    def load_state(self, ref: EpisodeRef) -> EpisodeState:
        paths = self.paths(ref)
        if not paths.parquet.is_file():
            raise DatasetError(f"episode parquet is missing: {paths.parquet}")
        frame = pd.read_parquet(
            paths.parquet,
            columns=["frame_index", "episode_index", "observation.state"],
        )
        if frame.empty:
            raise DatasetError(f"episode parquet is empty: {paths.parquet}")
        frame_indices = frame["frame_index"].to_numpy(dtype=np.int64)
        if not np.array_equal(frame_indices, np.arange(len(frame_indices))):
            raise DatasetError("frame_index must be contiguous and zero-based")
        episode_indices = frame["episode_index"].to_numpy(dtype=np.int64)
        if not np.all(episode_indices == ref.episode_index):
            raise DatasetError("parquet episode_index does not match requested episode")
        state = np.stack(frame["observation.state"].to_numpy()).astype(np.float64)
        if state.shape != (len(frame), 14):
            raise DatasetError(f"observation.state must be [T,14], got {state.shape}")
        if not np.isfinite(state).all():
            raise DatasetError("observation.state contains non-finite values")
        grippers = state[:, (6, 13)]
        eef = np.stack((state[:, 0:6], state[:, 7:13]), axis=1)
        return EpisodeState(
            frame_count=len(frame),
            task_text=self.task_text(ref.episode_index),
            gripper_states=grippers,
            eef_states=eef,
            paths=paths,
        )

    def read_frames(self, ref: EpisodeRef, frame_ids: Iterable[int]) -> dict[int, Image.Image]:
        requested = tuple(sorted(set(int(value) for value in frame_ids)))
        if not requested or requested[0] < 0:
            raise ValueError("frame_ids must contain non-negative values")
        path = self.paths(ref).video
        if not path.is_file():
            raise DatasetError(f"episode video is missing: {path}")
        result: dict[int, Image.Image] = {}
        wanted = set(requested)
        with av.open(str(path)) as container:
            for frame_id, video_frame in enumerate(container.decode(video=0)):
                if frame_id in wanted:
                    result[frame_id] = video_frame.to_image().convert("RGB")
                    if len(result) == len(wanted):
                        break
        missing = sorted(wanted - set(result))
        if missing:
            raise DatasetError(f"video is missing requested frames: {missing}")
        return result

    def video_info(self, ref: EpisodeRef) -> tuple[int, tuple[int, int]]:
        path = self.paths(ref).video
        count = 0
        shape: tuple[int, int] | None = None
        with av.open(str(path)) as container:
            for video_frame in container.decode(video=0):
                if shape is None:
                    shape = (video_frame.height, video_frame.width)
                count += 1
        if shape is None:
            raise DatasetError(f"video contains no frames: {path}")
        return count, shape

    def preflight(self, episode_ids: Iterable[int]) -> dict[str, Any]:
        ids = tuple(int(value) for value in episode_ids)
        issues: list[str] = []
        metadata = self._metadata_index()
        expected_shape = tuple(int(value) for value in self.manifest["frame_shape_hw"])
        expected_surplus = int(self.manifest["raw_video_frame_surplus"])
        content: dict[str, dict[str, Any]] = {}
        for episode_index in ids:
            ref = EpisodeRef(self.task, episode_index, self.camera)
            paths = self.paths(ref)
            for kind, path in (
                ("parquet", paths.parquet),
                ("video", paths.video),
                ("sidecar", paths.sidecar),
            ):
                if not path.is_file():
                    issues.append(f"episode {episode_index}: missing {kind}: {path}")
            if episode_index not in metadata:
                issues.append(f"episode {episode_index}: missing metadata")
            if all(path.is_file() for path in (paths.parquet, paths.video)):
                try:
                    state = self.load_state(ref)
                    video_count, video_shape = self.video_info(ref)
                    surplus = video_count - state.frame_count
                    content[str(episode_index)] = {
                        "usable_frame_count": state.frame_count,
                        "raw_video_frame_count": video_count,
                        "raw_video_frame_surplus": surplus,
                        "frame_shape_hw": list(video_shape),
                    }
                    if surplus != expected_surplus:
                        issues.append(
                            f"episode {episode_index}: video surplus {surplus} "
                            f"!= {expected_surplus}"
                        )
                    if video_shape != expected_shape:
                        issues.append(
                            f"episode {episode_index}: frame shape {video_shape} "
                            f"!= {expected_shape}"
                        )
                except Exception as exc:
                    issues.append(f"episode {episode_index}: content check failed: {exc}")
        report = {
            "format_version": "robotwin_dataset_preflight_v1",
            "root": str(self.root),
            "task": self.task,
            "camera": self.camera,
            "episode_count": len(ids),
            "episode_ids": list(ids),
            "usable_frame_count_source": "parquet",
            "content": content,
            "passed": not issues,
            "issues": issues,
        }
        if issues:
            raise DatasetError("dataset preflight failed:\n" + "\n".join(issues))
        return report
