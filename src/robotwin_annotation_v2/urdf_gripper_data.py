"""Lightweight dataset contracts for deterministic URDF gripper rendering.

This module deliberately avoids the regular dataset and pipeline packages: those
packages import video/SAM dependencies that are not available in the rendering
environment.  Loop detection mirrors ``pipeline.state_loop`` using only NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np


ArmName = Literal["left", "right"]

OPEN_THRESHOLD = 0.9
CLOSED_THRESHOLD = 0.15
PARQUET_COLUMNS = (
    "frame_index",
    "episode_index",
    "observation.state",
    "observation.state.joint_absolute",
)


class UrdfGripperDataError(RuntimeError):
    """An episode does not satisfy the URDF-rendering data contract."""


@dataclass(frozen=True)
class UrdfGripperEpisodePaths:
    """Resolved coverage-dataset inputs for one camera and episode."""

    dataset_root: Path
    episode_index: int
    camera: str
    parquet: Path
    sidecar: Path
    rgb_video: Path
    depth_video: Path

    @property
    def episode_id(self) -> str:
        return format_episode_id(self.episode_index)

    def missing_files(self) -> tuple[Path, ...]:
        """Return required inputs that are absent, without mutating the dataset."""

        required = (self.parquet, self.sidecar, self.rgb_video, self.depth_video)
        return tuple(path for path in required if not path.is_file())


@dataclass(frozen=True)
class EpisodeArrays:
    """State columns whose row count defines the usable episode length."""

    episode_index: int
    observation_state: np.ndarray
    joint_absolute: np.ndarray

    @property
    def frame_count(self) -> int:
        return int(self.joint_absolute.shape[0])


@dataclass(frozen=True)
class ActiveGripperLoop:
    """One active arm and its ordered, inclusive Stage-1 event boundaries."""

    active_arm: ArmName
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
        if self.active_arm not in {"left", "right"}:
            raise ValueError(f"invalid active arm: {self.active_arm}")
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
    def start(self) -> int:
        """First frame in the inclusive active-gripper window."""

        return self.t_move_start

    @property
    def end(self) -> int:
        """Last frame in the inclusive active-gripper window."""

        return self.t_open_done

    @property
    def inclusive_window(self) -> tuple[int, int]:
        return (self.start, self.end)


@dataclass(frozen=True)
class CameraCalibrationSeries:
    """Per-frame OpenCV calibration and OpenGL pose, cropped to Parquet rows."""

    camera: str
    intrinsic_cv: np.ndarray
    extrinsic_cv: np.ndarray
    cam2world_gl: np.ndarray

    @property
    def frame_count(self) -> int:
        return int(self.intrinsic_cv.shape[0])


@dataclass(frozen=True)
class UrdfGripperEpisodeData:
    """Complete lightweight input contract consumed by the URDF renderer."""

    paths: UrdfGripperEpisodePaths
    arrays: EpisodeArrays
    loop: ActiveGripperLoop

    @property
    def frame_count(self) -> int:
        return self.arrays.frame_count

    @property
    def joint_absolute(self) -> np.ndarray:
        return self.arrays.joint_absolute

    @property
    def active_arm(self) -> ArmName:
        return self.loop.active_arm

    @property
    def active_window(self) -> tuple[int, int]:
        return self.loop.inclusive_window


def _validate_episode_index(episode_index: int) -> int:
    if isinstance(episode_index, bool) or not isinstance(episode_index, int):
        raise TypeError("episode_index must be an integer")
    if episode_index < 0:
        raise ValueError("episode_index must be non-negative")
    return episode_index


def _validate_camera(camera: str) -> str:
    if not camera or camera in {".", ".."} or "/" in camera or "\\" in camera:
        raise ValueError(f"invalid camera name: {camera!r}")
    return camera


def format_episode_id(episode_index: int) -> str:
    """Format a global episode index using the dataset's six-digit convention."""

    return f"{_validate_episode_index(episode_index):06d}"


def resolve_episode_paths(
    dataset_root: Path,
    episode_index: int,
    *,
    camera: str = "cam_high",
) -> UrdfGripperEpisodePaths:
    """Resolve Parquet, sidecar, RGB, and depth paths without checking I/O."""

    index = _validate_episode_index(episode_index)
    camera_name = _validate_camera(camera)
    root = dataset_root.expanduser().resolve()
    episode_id = format_episode_id(index)
    chunk = f"chunk-{index // 1000:03d}"
    return UrdfGripperEpisodePaths(
        dataset_root=root,
        episode_index=index,
        camera=camera_name,
        parquet=root / "data" / chunk / f"episode_{episode_id}.parquet",
        sidecar=root / "sidecars" / f"episode_{episode_id}.hdf5",
        rgb_video=(
            root
            / "videos"
            / chunk
            / f"observation.images.{camera_name}"
            / f"episode_{episode_id}.mp4"
        ),
        depth_video=(
            root
            / "sidecars"
            / "videos"
            / chunk
            / f"observation.depths.{camera_name}"
            / f"episode_{episode_id}.mkv"
        ),
    )


def _read_parquet_columns(path: Path) -> dict[str, Any]:
    """Read only required columns, preferring PyArrow without importing pandas."""

    try:
        import pyarrow.parquet as pq
    except ImportError:
        try:
            import pandas as pd
        except ImportError as exc:
            raise UrdfGripperDataError(
                "reading episode Parquet requires pyarrow or pandas"
            ) from exc
        frame = pd.read_parquet(path, columns=list(PARQUET_COLUMNS))
        return {column: frame[column].to_numpy() for column in PARQUET_COLUMNS}

    table = pq.read_table(path, columns=list(PARQUET_COLUMNS))
    return {column: table[column].to_pylist() for column in PARQUET_COLUMNS}


def _stack_state_column(values: Any, *, name: str, frame_count: int) -> np.ndarray:
    try:
        rows = [np.asarray(row, dtype=np.float64) for row in values]
        result = np.stack(rows, axis=0)
    except (TypeError, ValueError) as exc:
        raise UrdfGripperDataError(f"{name} cannot be converted to a dense array") from exc
    expected_shape = (frame_count, 14)
    if result.shape != expected_shape:
        raise UrdfGripperDataError(f"{name} must have shape {expected_shape}, got {result.shape}")
    if not np.isfinite(result).all():
        raise UrdfGripperDataError(f"{name} contains non-finite values")
    return np.ascontiguousarray(result)


def load_episode_arrays(
    parquet_path: Path,
    *,
    expected_episode_index: int | None = None,
) -> EpisodeArrays:
    """Load aligned Cartesian state and absolute Aloha joints from Parquet."""

    path = parquet_path.expanduser().resolve()
    if not path.is_file():
        raise UrdfGripperDataError(f"episode parquet is missing: {path}")
    if expected_episode_index is not None:
        expected_episode_index = _validate_episode_index(expected_episode_index)
    try:
        columns = _read_parquet_columns(path)
    except UrdfGripperDataError:
        raise
    except Exception as exc:
        raise UrdfGripperDataError(f"failed to read episode parquet {path}: {exc}") from exc

    frame_indices = np.asarray(columns["frame_index"], dtype=np.int64)
    if frame_indices.ndim != 1 or frame_indices.size == 0:
        raise UrdfGripperDataError(f"episode parquet is empty: {path}")
    frame_count = int(frame_indices.size)
    if not np.array_equal(frame_indices, np.arange(frame_count, dtype=np.int64)):
        raise UrdfGripperDataError("frame_index must be contiguous and zero-based")

    episode_indices = np.asarray(columns["episode_index"], dtype=np.int64)
    if episode_indices.shape != (frame_count,):
        raise UrdfGripperDataError("episode_index must contain one value per frame")
    unique_episode_indices = np.unique(episode_indices)
    if unique_episode_indices.size != 1:
        raise UrdfGripperDataError("episode_index must be constant within one parquet")
    episode_index = int(unique_episode_indices[0])
    if expected_episode_index is not None and episode_index != expected_episode_index:
        raise UrdfGripperDataError(
            f"parquet episode_index {episode_index} does not match requested "
            f"episode {expected_episode_index}"
        )

    observation_state = _stack_state_column(
        columns["observation.state"],
        name="observation.state",
        frame_count=frame_count,
    )
    joint_absolute = _stack_state_column(
        columns["observation.state.joint_absolute"],
        name="observation.state.joint_absolute",
        frame_count=frame_count,
    )
    return EpisodeArrays(
        episode_index=episode_index,
        observation_state=observation_state,
        joint_absolute=joint_absolute,
    )


def _median_filter(values: np.ndarray) -> np.ndarray:
    padded = np.pad(values, (1, 1), mode="edge")
    return np.median(
        np.stack((padded[:-2], padded[1:-1], padded[2:])),
        axis=0,
    )


def _first_run(condition: np.ndarray, *, start: int, length: int) -> int | None:
    for frame_id in range(max(0, start), len(condition) - length + 1):
        if bool(condition[frame_id : frame_id + length].all()):
            return frame_id
    return None


def _detect_loop_events(
    gripper_values: np.ndarray,
    eef_values: np.ndarray,
    *,
    arm: ArmName,
    stable_frames: int = 3,
    motion_floor: float = 0.002,
    rotation_scale: float = 0.05,
) -> ActiveGripperLoop:
    gripper = np.asarray(gripper_values, dtype=np.float64)
    eef = np.asarray(eef_values, dtype=np.float64)
    if gripper.ndim != 1 or gripper.size < stable_frames * 2:
        raise UrdfGripperDataError(
            "gripper values must be a sufficiently long one-dimensional array"
        )
    if eef.shape != (gripper.size, 6):
        raise UrdfGripperDataError(f"eef values must have shape {(gripper.size, 6)}")
    if stable_frames < 1:
        raise ValueError("stable_frames must be positive")

    filtered = _median_filter(gripper)
    close_candidates = np.flatnonzero(
        (filtered[1:] < OPEN_THRESHOLD) & (filtered[1:] < filtered[:-1])
    ) + 1
    if close_candidates.size == 0:
        raise UrdfGripperDataError("no close transition detected")
    close_start = int(close_candidates[0])
    close_run = _first_run(
        filtered <= CLOSED_THRESHOLD,
        start=close_start,
        length=stable_frames,
    )
    if close_run is None:
        raise UrdfGripperDataError("no stable closed transition detected")
    close_done = close_run + stable_frames - 1

    open_candidates = np.flatnonzero(
        (filtered[1:] > filtered[:-1])
        & (np.arange(1, len(filtered)) > close_done)
    ) + 1
    if open_candidates.size == 0:
        raise UrdfGripperDataError("no open transition detected")
    open_start = int(open_candidates[0])
    open_run = _first_run(
        filtered >= OPEN_THRESHOLD,
        start=open_start,
        length=stable_frames,
    )
    if open_run is None:
        raise UrdfGripperDataError("no stable reopen transition detected")
    open_done = open_run + stable_frames - 1

    delta = np.diff(eef, axis=0)
    delta[:, 3:] = (delta[:, 3:] + np.pi) % (2 * np.pi) - np.pi
    score = np.linalg.norm(delta[:, :3], axis=1)
    score += rotation_scale * np.linalg.norm(delta[:, 3:], axis=1)
    baseline = score[: max(1, min(close_start, 3))]
    median = float(np.median(baseline))
    deviation = float(np.median(np.abs(baseline - median)))
    threshold = max(median + 4.0 * deviation, motion_floor)
    motion = np.concatenate(([False], score > threshold))
    move_start = _first_run(motion, start=1, length=stable_frames)
    if move_start is None or move_start >= close_start:
        move_start = close_start

    try:
        return ActiveGripperLoop(
            active_arm=arm,
            t_move_start=move_start,
            t_close_start=close_start,
            t_close_done=close_done,
            t_open_start=open_start,
            t_open_done=open_done,
        )
    except ValueError as exc:
        raise UrdfGripperDataError(str(exc)) from exc


def _detect_arm_loops(
    gripper_values: np.ndarray,
    eef_values: np.ndarray,
    *,
    arm: ArmName,
) -> tuple[ActiveGripperLoop, ...]:
    gripper = np.asarray(gripper_values)
    eef = np.asarray(eef_values)
    events: list[ActiveGripperLoop] = []
    offset = 0
    while offset < len(gripper) - 5:
        try:
            event = _detect_loop_events(gripper[offset:], eef[offset:], arm=arm)
        except UrdfGripperDataError:
            break
        absolute = ActiveGripperLoop(
            active_arm=event.active_arm,
            t_move_start=event.t_move_start + offset,
            t_close_start=event.t_close_start + offset,
            t_close_done=event.t_close_done + offset,
            t_open_start=event.t_open_start + offset,
            t_open_done=event.t_open_done + offset,
        )
        events.append(absolute)
        offset = absolute.t_open_done + 1
    return tuple(events)


def infer_active_loop(observation_state: np.ndarray) -> ActiveGripperLoop:
    """Require exactly one complete arm loop using Stage-1's state semantics."""

    state = np.asarray(observation_state, dtype=np.float64)
    if state.ndim != 2 or state.shape[1:] != (14,):
        raise UrdfGripperDataError(f"observation.state must have shape [T,14], got {state.shape}")
    if not np.isfinite(state).all():
        raise UrdfGripperDataError("observation.state contains non-finite values")

    grippers = state[:, (6, 13)]
    eef = np.stack((state[:, 0:6], state[:, 7:13]), axis=1)
    candidates: list[ActiveGripperLoop] = []
    counts: dict[str, int] = {}
    for arm_index, arm in enumerate(("left", "right")):
        arm_events = _detect_arm_loops(
            grippers[:, arm_index],
            eef[:, arm_index],
            arm=arm,
        )
        candidates.extend(arm_events)
        counts[arm] = len(arm_events)
    if len(candidates) != 1:
        raise UrdfGripperDataError(
            f"expected exactly one active-arm loop, got {len(candidates)}; per_arm={counts}"
        )
    return candidates[0]


def load_camera_calibration(
    sidecar_path: Path,
    *,
    camera: str = "cam_high",
    frame_count: int,
) -> CameraCalibrationSeries:
    """Load OpenCV calibration lazily and discard any trailing raw frames."""

    camera_name = _validate_camera(camera)
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    path = sidecar_path.expanduser().resolve()
    if not path.is_file():
        raise UrdfGripperDataError(f"episode sidecar is missing: {path}")
    try:
        import h5py
    except ImportError as exc:
        raise UrdfGripperDataError("reading camera calibration requires h5py") from exc

    group = f"camera_params/{camera_name}"
    try:
        with h5py.File(path, "r") as handle:
            intrinsic = np.asarray(
                handle[f"{group}/intrinsic_cv"][:frame_count],
                dtype=np.float64,
            )
            extrinsic = np.asarray(
                handle[f"{group}/extrinsic_cv"][:frame_count],
                dtype=np.float64,
            )
            cam2world = np.asarray(
                handle[f"{group}/cam2world_gl"][:frame_count],
                dtype=np.float64,
            )
    except Exception as exc:
        raise UrdfGripperDataError(
            f"failed to read {camera_name} calibration from {path}: {exc}"
        ) from exc

    if intrinsic.shape != (frame_count, 3, 3):
        raise UrdfGripperDataError(
            f"intrinsic_cv must have shape {(frame_count, 3, 3)}, got {intrinsic.shape}"
        )
    if extrinsic.shape != (frame_count, 3, 4):
        raise UrdfGripperDataError(
            f"extrinsic_cv must have shape {(frame_count, 3, 4)}, got {extrinsic.shape}"
        )
    if cam2world.shape != (frame_count, 4, 4):
        raise UrdfGripperDataError(
            f"cam2world_gl must have shape {(frame_count, 4, 4)}, got {cam2world.shape}"
        )
    if not (
        np.isfinite(intrinsic).all()
        and np.isfinite(extrinsic).all()
        and np.isfinite(cam2world).all()
    ):
        raise UrdfGripperDataError("camera calibration contains non-finite values")
    return CameraCalibrationSeries(
        camera=camera_name,
        intrinsic_cv=np.ascontiguousarray(intrinsic),
        extrinsic_cv=np.ascontiguousarray(extrinsic),
        cam2world_gl=np.ascontiguousarray(cam2world),
    )


def load_urdf_gripper_episode(
    dataset_root: Path,
    episode_index: int,
    *,
    camera: str = "cam_high",
    require_media: bool = True,
) -> UrdfGripperEpisodeData:
    """Resolve and validate one episode's data needed for URDF rendering."""

    paths = resolve_episode_paths(dataset_root, episode_index, camera=camera)
    if require_media:
        missing = paths.missing_files()
        if missing:
            rendered = ", ".join(str(path) for path in missing)
            raise UrdfGripperDataError(f"episode inputs are missing: {rendered}")
    arrays = load_episode_arrays(
        paths.parquet,
        expected_episode_index=episode_index,
    )
    loop = infer_active_loop(arrays.observation_state)
    if loop.end >= arrays.frame_count:
        raise UrdfGripperDataError(
            f"active window {loop.inclusive_window} exceeds frame count {arrays.frame_count}"
        )
    return UrdfGripperEpisodeData(paths=paths, arrays=arrays, loop=loop)


__all__ = [
    "ActiveGripperLoop",
    "ArmName",
    "CameraCalibrationSeries",
    "EpisodeArrays",
    "UrdfGripperDataError",
    "UrdfGripperEpisodeData",
    "UrdfGripperEpisodePaths",
    "format_episode_id",
    "infer_active_loop",
    "load_camera_calibration",
    "load_episode_arrays",
    "load_urdf_gripper_episode",
    "resolve_episode_paths",
]
