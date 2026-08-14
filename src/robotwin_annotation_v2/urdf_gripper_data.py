"""Lightweight dataset contracts for deterministic URDF gripper rendering.

This module deliberately avoids the regular dataset and pipeline packages: those
packages import video/SAM dependencies that are not available in the rendering
environment.  Loop detection mirrors ``pipeline.state_loop`` using only NumPy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np

from .domain import AnnotationMode, annotation_spec
from .models.timeline import (
    EpisodeWindows,
    FrameWindow,
    TargetOnlyEvents,
)

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

    def to_json(self) -> dict[str, Any]:
        return {
            "active_arm": self.active_arm,
            "t_move_start": self.t_move_start,
            "t_close_start": self.t_close_start,
            "t_close_done": self.t_close_done,
            "t_open_start": self.t_open_start,
            "t_open_done": self.t_open_done,
        }


ActiveGripperEvents = ActiveGripperLoop | TargetOnlyEvents


@dataclass(frozen=True)
class AuthoritativeLoopContext:
    """Validated Stage-1 source artifact used by a derived URDF run."""

    path: Path
    task: str
    episode_index: int
    camera: str
    frame_count: int
    events: ActiveGripperEvents
    annotation_mode: AnnotationMode
    timeline_kind: str
    windows: EpisodeWindows

    @property
    def active_arm(self) -> ArmName:
        return self.events.active_arm

    @property
    def gripper_window(self) -> tuple[int, int]:
        return (self.windows.gripper.start, self.windows.gripper.end)


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
    events: ActiveGripperEvents
    gripper_window: tuple[int, int]

    @property
    def frame_count(self) -> int:
        return self.arrays.frame_count

    @property
    def joint_absolute(self) -> np.ndarray:
        return self.arrays.joint_absolute

    @property
    def active_arm(self) -> ArmName:
        return self.events.active_arm

    @property
    def active_window(self) -> tuple[int, int]:
        return self.gripper_window

    @property
    def loop(self) -> ActiveGripperEvents:
        """Compatibility alias for callers that still use the historic name."""

        return self.events


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


def _required_mapping(
    payload: Mapping[str, Any],
    key: str,
    *,
    description: str,
) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise UrdfGripperDataError(f"{description} {key} must be an object")
    return value


def _required_integer(
    payload: Mapping[str, Any],
    key: str,
    *,
    description: str,
) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise UrdfGripperDataError(f"{description} {key} must be an integer")
    return value


def _validate_recorded_window(
    windows: Mapping[str, Any],
    key: str,
    expected: tuple[int, int],
) -> None:
    raw = windows.get(key)
    if (
        not isinstance(raw, list)
        or len(raw) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in raw)
        or tuple(raw) != expected
    ):
        raise UrdfGripperDataError(
            f"source loop window {key} does not match events: {raw!r} != {list(expected)!r}"
        )


def _frame_window(value: tuple[int, int]) -> FrameWindow:
    return FrameWindow(start=value[0], end=value[1])


def _pick_place_windows(events: ActiveGripperLoop) -> EpisodeWindows:
    operation = _frame_window(events.inclusive_window)
    return EpisodeWindows(
        operation=operation,
        target=_frame_window((events.t_move_start, events.t_close_done)),
        receiver=_frame_window((events.t_close_done, events.t_open_done)),
        gripper=operation,
    )


def _target_only_windows(
    events: TargetOnlyEvents,
    *,
    frame_count: int,
) -> EpisodeWindows:
    operation = _frame_window((events.t_remove_start, frame_count - 1))
    return EpisodeWindows(
        operation=operation,
        target=_frame_window((events.t_remove_start, events.t_close_end)),
        receiver=None,
        gripper=operation,
    )


def _validate_v2_windows(
    windows: Mapping[str, Any],
    expected: EpisodeWindows,
) -> None:
    expected_keys = {"operation", "target_0", "receiver_0", "gripper"}
    if set(windows) != expected_keys:
        raise UrdfGripperDataError(
            "source loop v2 windows must contain exactly "
            f"{sorted(expected_keys)}, got {sorted(str(key) for key in windows)}"
        )
    _validate_recorded_window(
        windows,
        "operation",
        (expected.operation.start, expected.operation.end),
    )
    _validate_recorded_window(
        windows,
        "target_0",
        (expected.target.start, expected.target.end),
    )
    if expected.receiver is None:
        if windows.get("receiver_0") is not None:
            raise UrdfGripperDataError(
                "target_only source loop receiver_0 window must be null"
            )
    else:
        _validate_recorded_window(
            windows,
            "receiver_0",
            (expected.receiver.start, expected.receiver.end),
        )
    _validate_recorded_window(
        windows,
        "gripper",
        (expected.gripper.start, expected.gripper.end),
    )


def load_authoritative_loop_context(
    path: Path,
    *,
    expected_task: str,
    expected_episode_index: int,
    expected_camera: str,
) -> AuthoritativeLoopContext:
    """Load one frozen timeline and normalize its downstream windows.

    Legacy v1 artifacts are accepted only for pick/place.  New v2 artifacts
    encode either the five-event pick/place timeline or the three-event
    close-and-hold timeline without inventing release boundaries.
    """

    source = path.expanduser().resolve()
    if not source.is_file():
        raise UrdfGripperDataError(f"source loop artifact is missing: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise UrdfGripperDataError(
            f"failed to read source loop artifact {source}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise UrdfGripperDataError("source loop artifact must contain a JSON object")
    format_version = payload.get("format_version")
    if format_version not in {
        "robotwin_loop_context_v1",
        "robotwin_loop_context_v2",
    }:
        raise UrdfGripperDataError(
            f"unsupported source loop format: {format_version!r}"
        )

    episode = _required_mapping(payload, "episode", description="source loop")
    expected_index = _validate_episode_index(expected_episode_index)
    expected_identity = {
        "task": expected_task,
        "episode_index": expected_index,
        "episode_id": format_episode_id(expected_index),
        "camera": _validate_camera(expected_camera),
    }
    for key, expected in expected_identity.items():
        if episode.get(key) != expected:
            raise UrdfGripperDataError(
                f"source loop episode {key} mismatch: {episode.get(key)!r} != {expected!r}"
            )

    frame_count = _required_integer(
        payload,
        "frame_count",
        description="source loop",
    )
    if frame_count < 1:
        raise UrdfGripperDataError("source loop frame_count must be positive")
    raw_mode = payload.get("annotation_mode", AnnotationMode.PICK_PLACE.value)
    try:
        annotation_mode = AnnotationMode(raw_mode)
    except (TypeError, ValueError) as exc:
        raise UrdfGripperDataError(
            f"unsupported source loop annotation_mode: {raw_mode!r}"
        ) from exc
    spec = annotation_spec(annotation_mode)
    raw_roles = payload.get("required_object_roles")
    if raw_roles is not None and raw_roles != list(spec.required_role_names):
        raise UrdfGripperDataError(
            "source loop required_object_roles differ from annotation_mode"
        )
    if format_version == "robotwin_loop_context_v2" and raw_roles is None:
        raise UrdfGripperDataError(
            "source loop v2 must declare required_object_roles"
        )

    event_payload = _required_mapping(payload, "events", description="source loop")
    active_arm = event_payload.get("active_arm")
    if active_arm not in {"left", "right"}:
        raise UrdfGripperDataError(
            f"source loop active_arm must be left or right, got {active_arm!r}"
        )
    windows = _required_mapping(payload, "windows", description="source loop")
    if format_version == "robotwin_loop_context_v1":
        if annotation_mode is not AnnotationMode.PICK_PLACE:
            raise UrdfGripperDataError(
                "target_only close-and-hold requires robotwin_loop_context_v2"
            )
        event_keys = {
            "active_arm",
            "t_move_start",
            "t_close_start",
            "t_close_done",
            "t_open_start",
            "t_open_done",
        }
        if set(event_payload) != event_keys:
            raise UrdfGripperDataError(
                "pick_place source events must contain exactly "
                f"{sorted(event_keys)}"
            )
        event_values = {
            key: _required_integer(
                event_payload,
                key,
                description="source loop events",
            )
            for key in event_keys - {"active_arm"}
        }
        try:
            events: ActiveGripperEvents = ActiveGripperLoop(
                active_arm=active_arm,
                **event_values,
            )
        except ValueError as exc:
            raise UrdfGripperDataError(f"invalid source loop events: {exc}") from exc
        normalized_windows = _pick_place_windows(events)
        if events.end >= frame_count:
            raise UrdfGripperDataError(
                f"source loop active window {events.inclusive_window} exceeds frame count "
                f"{frame_count}"
            )
        _validate_recorded_window(windows, "loop", events.inclusive_window)
        _validate_recorded_window(
            windows,
            "target_0",
            (events.t_move_start, events.t_close_done),
        )
        _validate_recorded_window(
            windows,
            "receiver_0",
            (events.t_close_done, events.t_open_done),
        )
        raw_gripper = windows.get("gripper")
        if raw_gripper is not None:
            _validate_recorded_window(
                windows,
                "gripper",
                events.inclusive_window,
            )
        timeline_kind = "pick_place"
    else:
        timeline_kind = payload.get("timeline_kind")
        expected_kind = (
            "pick_place"
            if annotation_mode is AnnotationMode.PICK_PLACE
            else "close_hold"
        )
        if timeline_kind != expected_kind:
            raise UrdfGripperDataError(
                "source loop timeline_kind differs from annotation_mode: "
                f"{timeline_kind!r} != {expected_kind!r}"
            )
        if annotation_mode is AnnotationMode.PICK_PLACE:
            event_keys = {
                "active_arm",
                "t_move_start",
                "t_close_start",
                "t_close_done",
                "t_open_start",
                "t_open_done",
            }
            event_values = {
                key: _required_integer(
                    event_payload,
                    key,
                    description="source loop events",
                )
                for key in event_keys - {"active_arm"}
            }
            try:
                events = ActiveGripperLoop(active_arm=active_arm, **event_values)
            except ValueError as exc:
                raise UrdfGripperDataError(
                    f"invalid source loop events: {exc}"
                ) from exc
            normalized_windows = _pick_place_windows(events)
        else:
            event_keys = {
                "active_arm",
                "t_remove_start",
                "t_close_start",
                "t_close_end",
            }
            event_values = {
                key: _required_integer(
                    event_payload,
                    key,
                    description="source loop events",
                )
                for key in event_keys - {"active_arm"}
            }
            try:
                events = TargetOnlyEvents(active_arm=active_arm, **event_values)
            except ValueError as exc:
                raise UrdfGripperDataError(
                    f"invalid source loop events: {exc}"
                ) from exc
            if events.t_close_end >= frame_count:
                raise UrdfGripperDataError(
                    "target_only close_end exceeds the episode frame range"
                )
            normalized_windows = _target_only_windows(
                events,
                frame_count=frame_count,
            )
        if set(event_payload) != event_keys:
            raise UrdfGripperDataError(
                f"{annotation_mode.value} source events must contain exactly "
                f"{sorted(event_keys)}"
            )
        _validate_v2_windows(windows, normalized_windows)

    return AuthoritativeLoopContext(
        path=source,
        task=expected_task,
        episode_index=expected_index,
        camera=expected_camera,
        frame_count=frame_count,
        events=events,
        annotation_mode=annotation_mode,
        timeline_kind=str(timeline_kind),
        windows=normalized_windows,
    )


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
    authoritative_loop: ActiveGripperLoop | None = None,
    authoritative_events: ActiveGripperEvents | None = None,
    authoritative_gripper_window: tuple[int, int] | None = None,
) -> UrdfGripperEpisodeData:
    """Resolve one episode using normalized authoritative gripper activity.

    ``authoritative_loop`` remains as a pick/place-only compatibility adapter.
    New callers pass ``authoritative_events`` together with the already
    normalized ``authoritative_gripper_window`` from the frozen context.
    """

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
    if authoritative_loop is not None:
        if not isinstance(authoritative_loop, ActiveGripperLoop):
            raise TypeError("authoritative_loop must be an ActiveGripperLoop")
        if authoritative_events is not None or authoritative_gripper_window is not None:
            raise ValueError(
                "authoritative_loop cannot be combined with normalized authoritative inputs"
            )
        authoritative_events = authoritative_loop
        authoritative_gripper_window = authoritative_loop.inclusive_window
    if (authoritative_events is None) != (authoritative_gripper_window is None):
        raise ValueError(
            "authoritative_events and authoritative_gripper_window must be provided together"
        )
    if authoritative_events is None:
        events: ActiveGripperEvents = infer_active_loop(arrays.observation_state)
        gripper_window = events.inclusive_window
    else:
        if not isinstance(authoritative_events, (ActiveGripperLoop, TargetOnlyEvents)):
            raise TypeError(
                "authoritative_events must be ActiveGripperLoop or TargetOnlyEvents"
            )
        events = authoritative_events
        assert authoritative_gripper_window is not None
        if (
            not isinstance(authoritative_gripper_window, tuple)
            or len(authoritative_gripper_window) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in authoritative_gripper_window
            )
        ):
            raise TypeError(
                "authoritative_gripper_window must be a pair of integer frame ids"
            )
        gripper_window = authoritative_gripper_window
    start, end = gripper_window
    if start < 0 or end < start or end >= arrays.frame_count:
        raise UrdfGripperDataError(
            f"active window {gripper_window} exceeds frame count {arrays.frame_count}"
        )
    if isinstance(events, TargetOnlyEvents):
        if events.t_close_end >= arrays.frame_count:
            raise UrdfGripperDataError(
                "target_only close_end exceeds the episode frame range"
            )
        expected = (events.t_remove_start, arrays.frame_count - 1)
        if gripper_window != expected:
            raise UrdfGripperDataError(
                "target_only gripper window must extend from remove_start through "
                f"the final Parquet frame: {gripper_window} != {expected}"
            )
    elif gripper_window != events.inclusive_window:
        raise UrdfGripperDataError(
            "pick_place gripper window must match the complete loop: "
            f"{gripper_window} != {events.inclusive_window}"
        )
    return UrdfGripperEpisodeData(
        paths=paths,
        arrays=arrays,
        events=events,
        gripper_window=gripper_window,
    )


__all__ = [
    "ActiveGripperEvents",
    "ActiveGripperLoop",
    "ArmName",
    "AuthoritativeLoopContext",
    "CameraCalibrationSeries",
    "EpisodeArrays",
    "UrdfGripperDataError",
    "UrdfGripperEpisodeData",
    "UrdfGripperEpisodePaths",
    "format_episode_id",
    "infer_active_loop",
    "load_authoritative_loop_context",
    "load_camera_calibration",
    "load_episode_arrays",
    "load_urdf_gripper_episode",
    "resolve_episode_paths",
]
