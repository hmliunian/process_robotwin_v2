"""Convert real-robot Foxglove MCAP recordings into the RoboTwin task layout."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import struct
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal, cast

import av
import h5py
import numpy as np
import pandas as pd

NDArray = np.ndarray[Any, Any]
QualityStatus = Literal[
    "complete_pick_place",
    "incomplete_no_release",
    "non_pick_place",
]

CAMERA_TOPIC = "/camera/coracam_head_left/left_h264/video"
JOINT_TOPIC = "/observation/arm_right/arm/joint_position"
EEF_TOPIC = "/observation/arm_right/arm/end_effector_pose"
OUTPUT_CAMERA = "cam_high"
JOINT_NAMES = tuple(f"j{index}" for index in range(8))
SAFE_EPISODE_METADATA = (
    "collection_tool",
    "device.device_code",
    "episode_index",
    "start_time_us",
    "stop_time_us",
    "task.action_text",
    "task.task_code",
)


class RealMcapError(RuntimeError):
    """The source recording or requested conversion violates the checked contract."""


@dataclass(frozen=True)
class TextRecord:
    source_mcap: str
    task_text: str
    task_text_zh: str
    target: str
    receiver: str | None
    quality_status: QualityStatus
    quality_reason: str


@dataclass(frozen=True)
class SourceMetadata:
    path: Path
    values: dict[str, str]

    @property
    def start_time_us(self) -> int:
        try:
            return int(self.values["start_time_us"])
        except (KeyError, ValueError) as exc:
            raise RealMcapError(f"source metadata has no valid start_time_us: {self.path}") from exc


@dataclass(frozen=True)
class McapEpisode:
    source: SourceMetadata
    video_times_ns: NDArray
    video_payloads: tuple[bytes, ...]
    joint_times_ns: NDArray
    joint_values: NDArray
    eef_times_ns: NDArray
    eef_positions: NDArray
    eef_quaternions: NDArray


@dataclass(frozen=True)
class AlignedEpisode:
    timestamps_s: NDArray
    states: NDArray
    actions: NDArray
    joint_absolute: NDArray
    joint_actions: NDArray
    aligned_source_joints: NDArray
    aligned_positions: NDArray
    aligned_quaternions: NDArray
    aligned_rpy: NDArray
    close_frame: int
    release_frame: int


@dataclass(frozen=True)
class ConvertedEpisode:
    episode_id: int
    frame_count: int
    width: int
    height: int
    fps: float
    text: TextRecord
    source: SourceMetadata
    stats: dict[str, Any]


def _expect_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RealMcapError(f"text manifest field {field!r} must be a non-empty string")
    return value.strip()


def load_text_records(path: Path) -> tuple[TextRecord, ...]:
    """Load and validate the visual text decisions for every source MCAP."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RealMcapError(f"text manifest is missing: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise RealMcapError(f"cannot read text manifest {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("format_version") != "real_pnp_texts_v1":
        raise RealMcapError("text manifest must use format_version real_pnp_texts_v1")
    raw_records = payload.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise RealMcapError("text manifest records must be a non-empty list")

    records: list[TextRecord] = []
    seen: set[str] = set()
    allowed: set[str] = {
        "complete_pick_place",
        "incomplete_no_release",
        "non_pick_place",
    }
    for index, item in enumerate(raw_records):
        if not isinstance(item, dict):
            raise RealMcapError(f"text manifest record {index} must be an object")
        source_mcap = _expect_string(item.get("source_mcap"), field="source_mcap")
        if Path(source_mcap).name != source_mcap or not source_mcap.endswith(".mcap"):
            raise RealMcapError(f"invalid source_mcap name: {source_mcap!r}")
        if source_mcap in seen:
            raise RealMcapError(f"duplicate source_mcap in text manifest: {source_mcap}")
        seen.add(source_mcap)
        raw_status = item.get("quality_status")
        if raw_status not in allowed:
            raise RealMcapError(f"unsupported quality_status for {source_mcap}: {raw_status!r}")
        receiver_value = item.get("receiver")
        receiver = None
        if receiver_value is not None:
            receiver = _expect_string(receiver_value, field="receiver")
        status = cast(QualityStatus, raw_status)
        if status == "complete_pick_place" and receiver is None:
            raise RealMcapError(f"complete P&P record has no receiver: {source_mcap}")
        records.append(
            TextRecord(
                source_mcap=source_mcap,
                task_text=_expect_string(item.get("task_text"), field="task_text"),
                task_text_zh=_expect_string(item.get("task_text_zh"), field="task_text_zh"),
                target=_expect_string(item.get("target"), field="target"),
                receiver=receiver,
                quality_status=status,
                quality_reason=_expect_string(
                    item.get("quality_reason"),
                    field="quality_reason",
                ),
            )
        )
    return tuple(records)


def mcap_reader_factory() -> Any:
    try:
        from mcap.reader import make_reader
    except ImportError as exc:
        raise RealMcapError(
            "MCAP support is not installed; install the project with the real-mcap extra"
        ) from exc
    return make_reader


def read_source_metadata(path: Path) -> SourceMetadata:
    make_reader = mcap_reader_factory()
    try:
        with path.open("rb") as handle:
            reader = make_reader(handle)
            episode = next(item for item in reader.iter_metadata() if item.name == "episode")
    except StopIteration as exc:
        raise RealMcapError(f"MCAP has no episode metadata: {path}") from exc
    except (OSError, ValueError) as exc:
        raise RealMcapError(f"cannot read MCAP metadata {path}: {exc}") from exc
    values = {
        key: str(value) for key, value in episode.metadata.items() if key in SAFE_EPISODE_METADATA
    }
    return SourceMetadata(path=path.resolve(), values=values)


def _check_range(data: bytes, offset: int, size: int, *, context: str) -> None:
    if offset < 0 or size < 0 or offset + size > len(data):
        raise RealMcapError(f"truncated FlatBuffer while reading {context}")


def _uint16(data: bytes, offset: int, *, context: str) -> int:
    _check_range(data, offset, 2, context=context)
    return int(struct.unpack_from("<H", data, offset)[0])


def _uint32(data: bytes, offset: int, *, context: str) -> int:
    _check_range(data, offset, 4, context=context)
    return int(struct.unpack_from("<I", data, offset)[0])


def _table_root(data: bytes) -> int:
    root = _uint32(data, 0, context="root table")
    _check_range(data, root, 4, context="root table")
    return root


def _table_field(data: bytes, table: int, field_index: int) -> int | None:
    _check_range(data, table, 4, context="table")
    vtable_distance = int(struct.unpack_from("<i", data, table)[0])
    vtable = table - vtable_distance
    vtable_size = _uint16(data, vtable, context="vtable size")
    entry = vtable + 4 + field_index * 2
    if entry + 2 > vtable + vtable_size:
        return None
    relative = _uint16(data, entry, context="vtable field")
    if relative == 0:
        return None
    field = table + relative
    _check_range(data, field, 1, context="table field")
    return field


def _indirect(data: bytes, offset: int, *, context: str) -> int:
    target = offset + _uint32(data, offset, context=context)
    _check_range(data, target, 4, context=context)
    return target


def _read_string(data: bytes, field: int) -> str:
    start = _indirect(data, field, context="string")
    length = _uint32(data, start, context="string length")
    _check_range(data, start + 4, length, context="string bytes")
    try:
        return data[start + 4 : start + 4 + length].decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RealMcapError("FlatBuffer string is not UTF-8") from exc


def _read_byte_vector(data: bytes, field_index: int) -> bytes:
    root = _table_root(data)
    field = _table_field(data, root, field_index)
    if field is None:
        raise RealMcapError(f"FlatBuffer byte vector field {field_index} is absent")
    vector = _indirect(data, field, context="byte vector")
    length = _uint32(data, vector, context="byte vector length")
    _check_range(data, vector + 4, length, context="byte vector data")
    return data[vector + 4 : vector + 4 + length]


def parse_compressed_video_payload(data: bytes) -> bytes:
    """Decode the H264 payload from one foxglove.CompressedVideo message."""

    return _read_byte_vector(data, 2)


def _read_table_vector(data: bytes, table: int, field_index: int) -> tuple[int, ...]:
    field = _table_field(data, table, field_index)
    if field is None:
        raise RealMcapError(f"FlatBuffer table vector field {field_index} is absent")
    vector = _indirect(data, field, context="table vector")
    length = _uint32(data, vector, context="table vector length")
    first = vector + 4
    _check_range(data, first, length * 4, context="table vector entries")
    return tuple(
        _indirect(data, first + index * 4, context="table vector entry") for index in range(length)
    )


def _read_double(data: bytes, table: int, field_index: int, *, default: float = 0.0) -> float:
    field = _table_field(data, table, field_index)
    if field is None:
        return default
    _check_range(data, field, 8, context="double field")
    return float(struct.unpack_from("<d", data, field)[0])


def parse_named_joint_positions(data: bytes) -> dict[str, float]:
    """Decode names and positions from one foxglove.JointStates message."""

    root = _table_root(data)
    values: dict[str, float] = {}
    for joint_table in _read_table_vector(data, root, 1):
        name_field = _table_field(data, joint_table, 0)
        if name_field is None:
            raise RealMcapError("JointStates entry has no name")
        name = _read_string(data, name_field)
        if name in values:
            raise RealMcapError(f"JointStates contains duplicate joint {name!r}")
        values[name] = _read_double(data, joint_table, 1)
    return values


def parse_joint_positions(data: bytes) -> NDArray:
    """Decode one foxglove.JointStates FlatBuffer in canonical j0..j7 order."""

    values = parse_named_joint_positions(data)
    if set(values) != set(JOINT_NAMES):
        raise RealMcapError(f"JointStates names must be {list(JOINT_NAMES)}, got {sorted(values)}")
    return np.asarray([values[name] for name in JOINT_NAMES], dtype=np.float64)


def _read_nested_table(data: bytes, table: int, field_index: int, *, context: str) -> int:
    field = _table_field(data, table, field_index)
    if field is None:
        raise RealMcapError(f"FlatBuffer {context} field is absent")
    return _indirect(data, field, context=context)


def parse_eef_pose(data: bytes) -> tuple[NDArray, NDArray]:
    """Decode xyz and xyzw from one foxglove.PoseInFrame FlatBuffer."""

    root = _table_root(data)
    pose = _read_nested_table(data, root, 2, context="pose")
    position = _read_nested_table(data, pose, 0, context="position")
    orientation = _read_nested_table(data, pose, 1, context="orientation")
    xyz = np.asarray(
        [_read_double(data, position, index) for index in range(3)],
        dtype=np.float64,
    )
    xyzw = np.asarray(
        [_read_double(data, orientation, index) for index in range(4)],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(xyzw))
    if not np.isfinite(xyz).all() or not math.isfinite(norm) or norm < 1e-8:
        raise RealMcapError("PoseInFrame contains an invalid pose")
    return xyz, xyzw / norm


def read_mcap_episode(source: SourceMetadata, *, camera_topic: str = CAMERA_TOPIC) -> McapEpisode:
    """Read the one camera and state streams needed by the no-depth conversion."""

    make_reader = mcap_reader_factory()
    video_times: list[int] = []
    video_payloads: list[bytes] = []
    joint_times: list[int] = []
    joint_values: list[NDArray] = []
    eef_times: list[int] = []
    eef_positions: list[NDArray] = []
    eef_quaternions: list[NDArray] = []
    topics = [camera_topic, JOINT_TOPIC, EEF_TOPIC]
    try:
        with source.path.open("rb") as handle:
            reader = make_reader(handle)
            for schema, channel, message in reader.iter_messages(topics=topics):
                if schema is None or schema.encoding != "flatbuffer":
                    raise RealMcapError(f"topic is not FlatBuffer encoded: {channel.topic}")
                if channel.topic == camera_topic:
                    if schema.name != "foxglove.CompressedVideo":
                        raise RealMcapError(f"unexpected camera schema: {schema.name}")
                    video_times.append(int(message.log_time))
                    video_payloads.append(parse_compressed_video_payload(message.data))
                elif channel.topic == JOINT_TOPIC:
                    if schema.name != "foxglove.JointStates":
                        raise RealMcapError(f"unexpected joint schema: {schema.name}")
                    joint_times.append(int(message.log_time))
                    joint_values.append(parse_joint_positions(message.data))
                elif channel.topic == EEF_TOPIC:
                    if schema.name != "foxglove.PoseInFrame":
                        raise RealMcapError(f"unexpected EEF schema: {schema.name}")
                    position, quaternion = parse_eef_pose(message.data)
                    eef_times.append(int(message.log_time))
                    eef_positions.append(position)
                    eef_quaternions.append(quaternion)
    except OSError as exc:
        raise RealMcapError(f"cannot read MCAP {source.path}: {exc}") from exc

    if not video_payloads or not joint_values or not eef_positions:
        raise RealMcapError(
            f"MCAP is missing required streams: video={len(video_payloads)}, "
            f"joints={len(joint_values)}, eef={len(eef_positions)}: {source.path}"
        )
    result = McapEpisode(
        source=source,
        video_times_ns=np.asarray(video_times, dtype=np.int64),
        video_payloads=tuple(video_payloads),
        joint_times_ns=np.asarray(joint_times, dtype=np.int64),
        joint_values=np.stack(joint_values),
        eef_times_ns=np.asarray(eef_times, dtype=np.int64),
        eef_positions=np.stack(eef_positions),
        eef_quaternions=np.stack(eef_quaternions),
    )
    for name, timestamps in (
        ("video", result.video_times_ns),
        ("joint", result.joint_times_ns),
        ("EEF", result.eef_times_ns),
    ):
        if len(timestamps) > 1 and bool(np.any(np.diff(timestamps) <= 0)):
            raise RealMcapError(f"{name} timestamps are not strictly increasing: {source.path}")
    return result


def _interpolate_columns(source_t: NDArray, values: NDArray, target_t: NDArray) -> NDArray:
    if values.ndim != 2 or values.shape[0] != len(source_t):
        raise ValueError("interpolation values must have shape [N,D]")
    return np.column_stack(
        [np.interp(target_t, source_t, values[:, index]) for index in range(values.shape[1])]
    )


def _interpolate_quaternions(source_t: NDArray, values: NDArray, target_t: NDArray) -> NDArray:
    if values.shape != (len(source_t), 4):
        raise ValueError("quaternion values must have shape [N,4]")
    right = np.searchsorted(source_t, target_t, side="left")
    right = np.clip(right, 1, len(source_t) - 1)
    left = right - 1
    before = target_t <= source_t[0]
    after = target_t >= source_t[-1]
    denominator = source_t[right] - source_t[left]
    alpha = np.divide(
        target_t - source_t[left],
        denominator,
        out=np.zeros_like(target_t, dtype=np.float64),
        where=denominator != 0,
    )
    q0 = values[left].copy()
    q1 = values[right].copy()
    q1[np.sum(q0 * q1, axis=1) < 0] *= -1
    result = (1.0 - alpha[:, None]) * q0 + alpha[:, None] * q1
    result[before] = values[0]
    result[after] = values[-1]
    norms = np.linalg.norm(result, axis=1)
    if bool(np.any(norms < 1e-8)):
        raise RealMcapError("quaternion interpolation produced a zero quaternion")
    return cast(NDArray, result / norms[:, None])


def _quaternion_to_rpy(quaternions: NDArray) -> NDArray:
    x, y, z, w = quaternions.T
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.column_stack((roll, pitch, yaw))


def _first_stable_run(mask: NDArray, *, start: int, length: int) -> int | None:
    for index in range(max(start, 0), len(mask) - length + 1):
        if bool(mask[index : index + length].all()):
            return index
    return None


def normalize_gripper_loop(raw_values: NDArray) -> tuple[NDArray, int, int]:
    """Map physical j7 positions to the binary RoboTwin open/closed convention."""

    values = np.asarray(raw_values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 8 or not np.isfinite(values).all():
        raise RealMcapError("aligned gripper signal must be a finite 1-D array with >=8 frames")
    padded = np.pad(values, (1, 1), mode="edge")
    smooth = np.median(np.stack((padded[:-2], padded[1:-1], padded[2:])), axis=0)
    peak = float(np.quantile(smooth, 0.95))
    if peak < 0.02:
        raise RealMcapError("gripper signal has no physical open state")
    stable = 3
    open_mask = smooth >= peak * 0.70
    closed_mask = smooth <= peak * 0.65
    initial_open = _first_stable_run(open_mask, start=0, length=stable)
    if initial_open is None:
        raise RealMcapError("gripper signal has no stable initial open run")
    close_frame = _first_stable_run(
        closed_mask,
        start=initial_open + stable,
        length=stable,
    )
    if close_frame is None:
        raise RealMcapError("gripper signal has no stable close after opening")
    release_frame = _first_stable_run(
        open_mask,
        start=close_frame + stable,
        length=stable,
    )
    if release_frame is None:
        raise RealMcapError("gripper signal has no stable release after closing")
    normalized = np.ones(len(values), dtype=np.float32)
    normalized[close_frame:release_frame] = 0.0
    return normalized, close_frame, release_frame


def align_episode(episode: McapEpisode) -> AlignedEpisode:
    video_t = episode.video_times_ns.astype(np.float64)
    joint_t = episode.joint_times_ns.astype(np.float64)
    eef_t = episode.eef_times_ns.astype(np.float64)
    joints = _interpolate_columns(joint_t, episode.joint_values, video_t)
    positions = _interpolate_columns(eef_t, episode.eef_positions, video_t)
    quaternions = _interpolate_quaternions(eef_t, episode.eef_quaternions, video_t)
    rpy = _quaternion_to_rpy(quaternions)
    gripper, close_frame, release_frame = normalize_gripper_loop(joints[:, 7])

    frame_count = len(video_t)
    states = np.zeros((frame_count, 14), dtype=np.float32)
    states[:, 6] = 1.0
    states[:, 7:10] = positions.astype(np.float32)
    states[:, 10:13] = rpy.astype(np.float32)
    states[:, 13] = gripper

    joint_absolute = np.zeros((frame_count, 14), dtype=np.float32)
    joint_absolute[:, 6] = 1.0
    joint_absolute[:, 7:13] = joints[:, :6].astype(np.float32)
    joint_absolute[:, 13] = gripper

    actions = np.zeros_like(states)
    actions[:-1, :6] = states[1:, :6] - states[:-1, :6]
    actions[:-1, 7:13] = states[1:, 7:13] - states[:-1, 7:13]
    actions[:-1, 3:6] = (actions[:-1, 3:6] + np.pi) % (2 * np.pi) - np.pi
    actions[:-1, 10:13] = (actions[:-1, 10:13] + np.pi) % (2 * np.pi) - np.pi
    actions[:, 6] = np.concatenate((states[1:, 6], states[-1:, 6]))
    actions[:, 13] = np.concatenate((states[1:, 13], states[-1:, 13]))
    joint_actions = np.concatenate((joint_absolute[1:], joint_absolute[-1:]), axis=0)

    timestamps_s = ((video_t - video_t[0]) / 1_000_000_000.0).astype(np.float32)
    if not all(np.isfinite(item).all() for item in (states, actions, joint_absolute)):
        raise RealMcapError("aligned state contains non-finite values")
    return AlignedEpisode(
        timestamps_s=timestamps_s,
        states=states,
        actions=actions,
        joint_absolute=joint_absolute,
        joint_actions=joint_actions,
        aligned_source_joints=joints.astype(np.float32),
        aligned_positions=positions.astype(np.float32),
        aligned_quaternions=quaternions.astype(np.float32),
        aligned_rpy=rpy.astype(np.float32),
        close_frame=close_frame,
        release_frame=release_frame,
    )


def _video_fps(timestamps_ns: NDArray) -> float:
    if len(timestamps_ns) < 2:
        raise RealMcapError("video must contain at least two frames")
    median_delta = float(np.median(np.diff(timestamps_ns.astype(np.float64))))
    if median_delta <= 0:
        raise RealMcapError("video timestamps do not define a positive FPS")
    return 1_000_000_000.0 / median_delta


def decode_h264_frames(payloads: Sequence[bytes]) -> list[av.VideoFrame]:
    """Decode access units, preserving each source message index as frame PTS."""
    decoder = av.CodecContext.create("h264", "r")
    frames: list[av.VideoFrame] = []
    packet_index = 0
    for payload in (*payloads, b""):
        for packet in decoder.parse(payload):
            packet.pts = packet_index
            packet.dts = packet_index
            packet_index += 1
            frames.extend(decoder.decode(packet))
    frames.extend(decoder.decode(None))
    if packet_index != len(payloads) or not frames:
        raise RealMcapError("H264 requires one access unit per source message and visible frames")
    indices = [frame.pts for frame in frames if frame.pts is not None]
    if len(indices) != len(frames) or indices != sorted(set(indices)):
        raise RealMcapError("decoded H264 source frame indices are missing or unordered")
    return frames


def write_video(
    payloads: Sequence[bytes], timestamps_ns: NDArray, path: Path
) -> tuple[int, int, int, float]:
    """Decode H264 access units and write one constant-rate H264 MP4 without resizing."""

    frames = decode_h264_frames(payloads)
    if len(frames) != len(payloads) or len(frames) != len(timestamps_ns):
        raise RealMcapError(
            f"head video decode count mismatch: messages={len(payloads)}, frames={len(frames)}"
        )
    return encode_video_frames(frames, timestamps_ns, path)


def encode_video_frames(
    frames: Sequence[av.VideoFrame], timestamps_ns: NDArray, path: Path
) -> tuple[int, int, int, float]:
    """Write already decoded RGB frames without changing their resolution."""

    if not frames or len(frames) != len(timestamps_ns):
        raise RealMcapError("decoded video and timestamp counts differ")
    width = int(frames[0].width)
    height = int(frames[0].height)
    if any(frame.width != width or frame.height != height for frame in frames):
        raise RealMcapError("head video resolution changes within one episode")
    fps = _video_fps(timestamps_ns)
    rate = Fraction(round(fps * 1000), 1000).limit_denominator(1000)
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as output:
        stream = output.add_stream("libx264", rate=rate)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "veryfast"}
        time_base = Fraction(rate.denominator, rate.numerator)
        for frame_index, frame in enumerate(frames):
            frame.pts = frame_index
            frame.time_base = time_base
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    return len(frames), width, height, fps


def _frame_stats(values: NDArray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": array.min(axis=0).tolist(),
        "max": array.max(axis=0).tolist(),
        "mean": array.mean(axis=0).tolist(),
        "std": array.std(axis=0).tolist(),
        "count": [len(array)],
    }


def _write_parquet(
    path: Path,
    *,
    episode_id: int,
    global_start_index: int,
    aligned: AlignedEpisode,
) -> dict[str, Any]:
    frame_count = len(aligned.states)
    frame = pd.DataFrame(
        {
            "timestamp": aligned.timestamps_s,
            "frame_index": np.arange(frame_count, dtype=np.int64),
            "episode_index": np.full(frame_count, episode_id, dtype=np.int64),
            "index": np.arange(
                global_start_index,
                global_start_index + frame_count,
                dtype=np.int64,
            ),
            "coarse_task_index": np.zeros(frame_count, dtype=np.int64),
            "task_index": np.full(frame_count, episode_id, dtype=np.int64),
            "coarse_quality_index": np.zeros(frame_count, dtype=np.int64),
            "quality_index": np.zeros(frame_count, dtype=np.int64),
            "observation.state": list(aligned.states),
            "action": list(aligned.actions),
            "observation.state.joint_absolute": list(aligned.joint_absolute),
            "action.joint_absolute": list(aligned.joint_actions),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return {
        "observation.state": _frame_stats(aligned.states),
        "action": _frame_stats(aligned.actions),
        "observation.state.joint_absolute": _frame_stats(aligned.joint_absolute),
        "action.joint_absolute": _frame_stats(aligned.joint_actions),
    }


def _safe_metadata(source: SourceMetadata) -> dict[str, Any]:
    return {
        "source_mcap": source.path.name,
        "source_episode_metadata": dict(source.values),
        "source_action_text_trusted": False,
        "omitted_source_metadata_keys": ["config_toml"],
        "camera_topic": CAMERA_TOPIC,
        "camera": OUTPUT_CAMERA,
        "depth_available": False,
    }


def _write_sidecar(
    path: Path,
    *,
    episode: McapEpisode,
    aligned: AlignedEpisode,
    width: int,
    height: int,
    fps: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["format_version"] = "robotwin_real_mcap_sidecar_v1"
        handle.attrs["width"] = width
        handle.attrs["height"] = height
        handle.attrs["fps"] = fps
        handle.attrs["rgb_channel_order"] = "RGB"
        handle.attrs["depth_available"] = False
        handle.attrs["metadata_json"] = json.dumps(
            _safe_metadata(episode.source),
            ensure_ascii=False,
            sort_keys=True,
        )
        handle.create_dataset(
            "joint_absolute/vector",
            data=aligned.joint_absolute,
            compression="gzip",
        )
        raw = handle.create_group("raw_extra")
        video = raw.create_group("video")
        video.create_dataset("source_timestamp_ns", data=episode.video_times_ns)
        joint = raw.create_group("joint_position")
        joint.create_dataset("names", data=np.asarray(JOINT_NAMES, dtype="S2"))
        joint.create_dataset("source_timestamp_ns", data=episode.joint_times_ns)
        joint.create_dataset("source_values", data=episode.joint_values, compression="gzip")
        joint.create_dataset(
            "aligned_values",
            data=aligned.aligned_source_joints,
            compression="gzip",
        )
        joint.attrs["parquet_mapping"] = "right j0..j5 + normalized j7; source j6 sidecar-only"
        eef = raw.create_group("eef_pose")
        eef.create_dataset("source_timestamp_ns", data=episode.eef_times_ns)
        eef.create_dataset("source_position", data=episode.eef_positions, compression="gzip")
        eef.create_dataset(
            "source_quaternion_xyzw",
            data=episode.eef_quaternions,
            compression="gzip",
        )
        eef.create_dataset("aligned_position", data=aligned.aligned_positions, compression="gzip")
        eef.create_dataset(
            "aligned_quaternion_xyzw",
            data=aligned.aligned_quaternions,
            compression="gzip",
        )
        eef.create_dataset("aligned_rpy", data=aligned.aligned_rpy, compression="gzip")
        gripper = raw.create_group("gripper")
        gripper.create_dataset("normalized_open", data=aligned.states[:, 13])
        gripper.attrs["close_frame"] = aligned.close_frame
        gripper.attrs["release_frame"] = aligned.release_frame


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def _episode_metadata(converted: ConvertedEpisode) -> dict[str, Any]:
    source_episode = converted.source.values.get("episode_index")
    return {
        "episode_index": converted.episode_id,
        "tasks": [converted.text.task_text],
        "length": converted.frame_count,
        "sidecar_path": f"sidecars/episode_{converted.episode_id:06d}.hdf5",
        "source_mcap": converted.source.path.name,
        "source_episode_index": None if source_episode is None else int(source_episode),
        "source_start_time_us": converted.source.start_time_us,
        "task_text_zh": converted.text.task_text_zh,
        "target": converted.text.target,
        "receiver": converted.text.receiver,
        "quality_status": converted.text.quality_status,
        "camera": OUTPUT_CAMERA,
        "source_camera_topic": CAMERA_TOPIC,
        "depth_available": False,
    }


def _feature_info(*, height: int, width: int, fps: float) -> dict[str, Any]:
    state_names = [
        "left_x",
        "left_y",
        "left_z",
        "left_roll",
        "left_pitch",
        "left_yaw",
        "left_gripper",
        "right_x",
        "right_y",
        "right_z",
        "right_roll",
        "right_pitch",
        "right_yaw",
        "right_gripper",
    ]
    joint_names = [
        "left_j0",
        "left_j1",
        "left_j2",
        "left_j3",
        "left_j4",
        "left_j5",
        "left_gripper",
        "right_j0",
        "right_j1",
        "right_j2",
        "right_j3",
        "right_j4",
        "right_j5",
        "right_gripper",
    ]
    scalar = {"dtype": "int64", "shape": [1], "names": None}
    return {
        "observation.state": {"dtype": "float32", "shape": [14], "names": [state_names]},
        "action": {"dtype": "float32", "shape": [14], "names": [state_names]},
        f"observation.images.{OUTPUT_CAMERA}": {
            "dtype": "video",
            "shape": [height, width, 3],
            "names": ["height", "width", "rgb"],
            "info": {
                "video.height": height,
                "video.width": width,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": fps,
                "video.channels": 3,
                "has_audio": False,
            },
        },
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": dict(scalar),
        "episode_index": dict(scalar),
        "index": dict(scalar),
        "coarse_task_index": dict(scalar),
        "task_index": dict(scalar),
        "coarse_quality_index": dict(scalar),
        "quality_index": dict(scalar),
        "observation.state.joint_absolute": {
            "dtype": "float32",
            "shape": [14],
            "names": [joint_names],
        },
        "action.joint_absolute": {
            "dtype": "float32",
            "shape": [14],
            "names": [joint_names],
        },
    }


def _write_dataset_metadata(
    staging: Path,
    *,
    final_output: Path,
    converted: Sequence[ConvertedEpisode],
    excluded: Sequence[tuple[SourceMetadata, TextRecord]],
    conversion_limit: int | None,
) -> None:
    if not converted:
        raise RealMcapError("conversion selected no complete P&P episodes")
    heights = {item.height for item in converted}
    widths = {item.width for item in converted}
    if len(heights) != 1 or len(widths) != 1:
        raise RealMcapError("converted episodes do not share one video resolution")
    height = next(iter(heights))
    width = next(iter(widths))
    fps = float(np.median([item.fps for item in converted]))
    episode_ids = [item.episode_id for item in converted]
    included_sources = [
        {
            "episode_index": item.episode_id,
            "source_mcap": item.source.path.name,
            "source_episode_index": item.source.values.get("episode_index"),
            "source_start_time_us": item.source.start_time_us,
            "target": item.text.target,
            "receiver": item.text.receiver,
            "task_text": item.text.task_text,
        }
        for item in converted
    ]
    excluded_sources = [
        {
            "source_mcap": source.path.name,
            "source_episode_index": source.values.get("episode_index"),
            "source_start_time_us": source.start_time_us,
            "quality_status": text.quality_status,
            "reason": text.quality_reason,
            "task_text": text.task_text,
        }
        for source, text in excluded
    ]
    manifest = {
        "format": "robotwin_real_mcap_task_extract_v1",
        "profile": "pick_place",
        "task": "pick_and_place_real",
        "camera": OUTPUT_CAMERA,
        "source_camera_topic": CAMERA_TOPIC,
        "depth_available": False,
        "selection_policy": "quality_status == complete_pick_place",
        "dataset_root": str(final_output.resolve()),
        "episode_indices": episode_ids,
        "episode_count": len(episode_ids),
        "frame_shape_hw": [height, width],
        "raw_video_frame_surplus": 0,
        "usable_frame_count_source": "parquet",
        "smoke_episode_ids": episode_ids[:1],
        "regression_episode_ids": episode_ids,
        "conversion_limit": conversion_limit,
        "included_sources": included_sources,
        "excluded_sources": excluded_sources,
        "required_relative_files": [
            "data/chunk-*/episode_{episode_id}.parquet",
            "videos/chunk-*/observation.images.{camera}/episode_{episode_id}.mp4",
            "sidecars/episode_{episode_id}.hdf5",
        ],
    }
    (staging / "EXTRACT_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_jsonl(staging / "meta" / "episodes.jsonl", map(_episode_metadata, converted))
    write_jsonl(
        staging / "meta" / "episodes_stats.jsonl",
        ({"episode_index": item.episode_id, "stats": item.stats} for item in converted),
    )
    write_jsonl(
        staging / "meta" / "tasks.jsonl",
        (
            {
                "task_index": item.episode_id,
                "task": item.text.task_text,
                "task_zh": item.text.task_text_zh,
                "target": item.text.target,
                "receiver": item.text.receiver,
            }
            for item in converted
        ),
    )
    info = {
        "codebase_version": "robotwin_annotation_v2",
        "robot_type": "real_mobile_manipulator",
        "total_episodes": len(converted),
        "total_frames": sum(item.frame_count for item in converted),
        "total_tasks": len(converted),
        "total_videos": len(converted),
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{len(converted)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": _feature_info(height=height, width=width, fps=fps),
        "rgb_channel_order": "RGB",
        "state_action_representations": {
            "default": {
                "state": "absolute_eef_xyz_rpy_binary_gripper",
                "action": "forward_eef_delta_binary_gripper",
                "state_feature": "observation.state",
                "action_feature": "action",
            },
            "joint_absolute": {
                "state": "aligned_source_j0_to_j5_binary_gripper",
                "action": "same_representation_at_t_plus_1",
                "source_j6_policy": "preserved_in_sidecar_only",
            },
        },
        "depth_available": False,
    }
    (staging / "meta" / "info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _validate_source_set(input_root: Path, records: Sequence[TextRecord]) -> list[SourceMetadata]:
    configured = {record.source_mcap for record in records}
    actual = {path.name for path in input_root.glob("*.mcap") if path.is_file()}
    missing = sorted(configured - actual)
    unexpected = sorted(actual - configured)
    if missing or unexpected:
        raise RealMcapError(
            f"source/text manifest mismatch; missing={missing}, unexpected={unexpected}"
        )
    sources = [read_source_metadata(input_root / name) for name in configured]
    starts = [source.start_time_us for source in sources]
    if len(starts) != len(set(starts)):
        raise RealMcapError("source MCAP start_time_us values are not unique")
    return sorted(sources, key=lambda item: item.start_time_us)


def materialize_real_mcap_dataset(
    input_root: Path,
    output_root: Path,
    *,
    text_manifest: Path,
    limit: int | None = None,
) -> dict[str, Any]:
    """Atomically materialize complete P&P recordings in the task dataset layout."""

    input_root = input_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if not input_root.is_dir():
        raise RealMcapError(f"source directory does not exist: {input_root}")
    if output_root.exists():
        raise RealMcapError(f"refusing to overwrite existing output directory: {output_root}")
    if limit is not None and limit < 1:
        raise RealMcapError("limit must be positive")
    output_root.parent.mkdir(parents=True, exist_ok=True)

    records = load_text_records(text_manifest)
    record_by_name = {record.source_mcap: record for record in records}
    ordered_sources = _validate_source_set(input_root, records)
    excluded = [
        (source, record_by_name[source.path.name])
        for source in ordered_sources
        if record_by_name[source.path.name].quality_status != "complete_pick_place"
    ]
    selected = [
        source
        for source in ordered_sources
        if record_by_name[source.path.name].quality_status == "complete_pick_place"
    ]
    if limit is not None:
        selected = selected[:limit]

    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.staging-",
            dir=output_root.parent,
        )
    )
    converted: list[ConvertedEpisode] = []
    global_index = 0
    try:
        for episode_id, source in enumerate(selected):
            text = record_by_name[source.path.name]
            episode = read_mcap_episode(source)
            aligned = align_episode(episode)
            chunk = f"chunk-{episode_id // 1000:03d}"
            episode_name = f"episode_{episode_id:06d}"
            video_path = (
                staging
                / "videos"
                / chunk
                / f"observation.images.{OUTPUT_CAMERA}"
                / f"{episode_name}.mp4"
            )
            frame_count, width, height, fps = write_video(
                episode.video_payloads,
                episode.video_times_ns,
                video_path,
            )
            if frame_count != len(aligned.states):
                raise RealMcapError("video and aligned state frame counts differ")
            stats = _write_parquet(
                staging / "data" / chunk / f"{episode_name}.parquet",
                episode_id=episode_id,
                global_start_index=global_index,
                aligned=aligned,
            )
            _write_sidecar(
                staging / "sidecars" / f"{episode_name}.hdf5",
                episode=episode,
                aligned=aligned,
                width=width,
                height=height,
                fps=fps,
            )
            converted.append(
                ConvertedEpisode(
                    episode_id=episode_id,
                    frame_count=frame_count,
                    width=width,
                    height=height,
                    fps=fps,
                    text=text,
                    source=source,
                    stats=stats,
                )
            )
            global_index += frame_count
        _write_dataset_metadata(
            staging,
            final_output=output_root,
            converted=converted,
            excluded=excluded,
            conversion_limit=limit,
        )
        os.replace(staging, output_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "output_root": str(output_root),
        "episode_count": len(converted),
        "frame_count": sum(item.frame_count for item in converted),
        "excluded_source_count": len(excluded),
        "camera": OUTPUT_CAMERA,
        "depth_available": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert complete real pick-and-place MCAPs to the no-depth cam_high RoboTwin layout."
        )
    )
    parser.add_argument("input_root", type=Path, help="directory containing the source MCAP files")
    parser.add_argument("output_root", type=Path, help="new task directory to create")
    parser.add_argument(
        "--texts",
        type=Path,
        default=Path("configs/datasets/pick_and_place_real_texts.json"),
        help="reviewed per-source text and quality manifest",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="convert only the first N selected episodes (for an isolated smoke conversion)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = materialize_real_mcap_dataset(
            args.input_root,
            args.output_root,
            text_manifest=args.texts,
            limit=args.limit,
        )
    except RealMcapError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


__all__ = [
    "CAMERA_TOPIC",
    "EEF_TOPIC",
    "JOINT_TOPIC",
    "OUTPUT_CAMERA",
    "AlignedEpisode",
    "McapEpisode",
    "RealMcapError",
    "SourceMetadata",
    "TextRecord",
    "align_episode",
    "load_text_records",
    "main",
    "materialize_real_mcap_dataset",
    "mcap_reader_factory",
    "normalize_gripper_loop",
    "parse_compressed_video_payload",
    "parse_eef_pose",
    "parse_joint_positions",
    "parse_named_joint_positions",
    "read_mcap_episode",
    "read_source_metadata",
    "write_jsonl",
    "write_video",
]
