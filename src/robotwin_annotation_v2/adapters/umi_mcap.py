"""Convert UMI MCAP video and gripper cues into a target-only task extract."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import h5py
import numpy as np
import pandas as pd
import yaml

from robotwin_annotation_v2.domain import AnnotationMode
from robotwin_annotation_v2.models import TargetOnlyEvents, VideoWindowEvents
from robotwin_annotation_v2.pipeline.timeline_detector import (
    StateLoopError,
    detect_gripper_target_only_events,
)

from .real_mcap import (
    NDArray,
    RealMcapError,
    SourceMetadata,
    decode_h264_frames,
    encode_video_frames,
    mcap_reader_factory,
    parse_compressed_video_payload,
    parse_named_joint_positions,
    read_source_metadata,
    write_jsonl,
    write_video,
)

CAMERA_TOPIC = "/camera/coracam_head/left_h264/video"
GRIPPER_TOPICS = (
    "/observation/left_gripper/gripper/joint_position",
    "/observation/right_gripper/gripper/joint_position",
)
OUTPUT_CAMERA = "cam_high"
MIN_GRIPPER_TRAVEL = 10.0


@dataclass(frozen=True)
class UmiMcapEpisode:
    source: SourceMetadata
    video_times_ns: NDArray
    video_payloads: tuple[bytes, ...]
    gripper_times_ns: tuple[NDArray, NDArray]
    gripper_values: tuple[NDArray, NDArray]


@dataclass(frozen=True)
class UmiTimeline:
    timestamps_s: NDArray
    events: TargetOnlyEvents | VideoWindowEvents
    gripper_travel: tuple[float, float] | None = None
    source_frame_indices: tuple[int, ...] | None = None
    source_frame_count: int | None = None

    @property
    def metadata(self) -> dict[str, Any]:
        if isinstance(self.events, VideoWindowEvents):
            return {
                "video_events": self.events.to_json(),
                "timeline_provenance": {
                    "signal": "reviewed_video_window",
                    "active_arm_source": "task_config_review",
                    "source_topics": [CAMERA_TOPIC],
                    "source_frame_indices": list(self.source_frame_indices or ()),
                    "source_frame_count": self.source_frame_count,
                },
            }
        return {
            "target_only_events": self.events.to_json(),
            "timeline_provenance": {
                "signal": "mcap_gripper_position",
                "operation_start_policy": "episode_start",
                "source_topics": list(GRIPPER_TOPICS),
                "gripper_travel": list(self.gripper_travel or ()),
            },
        }


@dataclass(frozen=True)
class UmiTaskConfig:
    """Task semantics and an explicit, reviewed single-arm source selection."""

    mode: AnnotationMode
    task_text: str
    task_text_zh: str
    episodes: dict[str, Literal["left", "right"]]

    def __post_init__(self) -> None:
        if self.mode not in {AnnotationMode.TARGET_ONLY, AnnotationMode.TOOL_USE}:
            raise RealMcapError("UMI video tasks require target_only or tool_use mode")
        if not self.task_text.strip() or not self.task_text_zh.strip() or not self.episodes:
            raise RealMcapError("UMI task config requires task text and reviewed episodes")
        for name, arm in self.episodes.items():
            if (
                Path(name).name != name
                or not name.endswith(".mcap")
                or arm not in {"left", "right"}
            ):
                raise RealMcapError(f"invalid reviewed single-arm source: {name!r}: {arm!r}")


def load_umi_task_config(path: Path, task: str) -> UmiTaskConfig:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))["tasks"][task]
        return UmiTaskConfig(
            mode=AnnotationMode(raw["annotation_mode"]),
            task_text=raw["task_text"],
            task_text_zh=raw["task_text_zh"],
            episodes=raw["episodes"],
        )
    except (OSError, KeyError, TypeError, ValueError, AttributeError, yaml.YAMLError) as exc:
        raise RealMcapError(f"cannot load UMI task config for {task}: {exc}") from exc


@dataclass(frozen=True)
class ConvertedUmiEpisode:
    episode_id: int
    frame_count: int
    width: int
    height: int
    fps: float
    source: SourceMetadata
    timeline: UmiTimeline


def _validate_timestamps(name: str, values: NDArray, source: Path) -> None:
    if len(values) > 1 and bool(np.any(np.diff(values) <= 0)):
        raise RealMcapError(f"{name} timestamps are not strictly increasing: {source}")


def read_umi_mcap_episode(
    source: SourceMetadata, *, require_grippers: bool = True
) -> UmiMcapEpisode:
    """Read head-left H264 plus the two one-degree gripper streams."""

    video_times: list[int] = []
    video_payloads: list[bytes] = []
    gripper_times: tuple[list[int], list[int]] = ([], [])
    gripper_values: tuple[list[float], list[float]] = ([], [])
    topic_indices = {topic: index for index, topic in enumerate(GRIPPER_TOPICS)}
    try:
        with source.path.open("rb") as handle:
            reader = mcap_reader_factory()(handle)
            for schema, channel, message in reader.iter_messages(
                topics=[CAMERA_TOPIC, *GRIPPER_TOPICS]
            ):
                if schema is None or schema.encoding != "flatbuffer":
                    raise RealMcapError(f"topic is not FlatBuffer encoded: {channel.topic}")
                if channel.topic == CAMERA_TOPIC:
                    if schema.name != "foxglove.CompressedVideo":
                        raise RealMcapError(f"unexpected camera schema: {schema.name}")
                    video_times.append(int(message.log_time))
                    video_payloads.append(parse_compressed_video_payload(message.data))
                    continue
                if schema.name != "foxglove.JointStates":
                    raise RealMcapError(f"unexpected gripper schema: {schema.name}")
                values = parse_named_joint_positions(message.data)
                if set(values) != {"j0"}:
                    raise RealMcapError(
                        f"UMI gripper JointStates must contain only j0, got {sorted(values)}"
                    )
                index = topic_indices[channel.topic]
                gripper_times[index].append(int(message.log_time))
                gripper_values[index].append(values["j0"])
    except OSError as exc:
        raise RealMcapError(f"cannot read MCAP {source.path}: {exc}") from exc

    if not video_payloads or (require_grippers and any(not values for values in gripper_values)):
        raise RealMcapError(
            f"MCAP is missing required UMI streams: video={len(video_payloads)}, "
            f"left_gripper={len(gripper_values[0])}, "
            f"right_gripper={len(gripper_values[1])}: {source.path}"
        )
    episode = UmiMcapEpisode(
        source=source,
        video_times_ns=np.asarray(video_times, dtype=np.int64),
        video_payloads=tuple(video_payloads),
        gripper_times_ns=(
            np.asarray(gripper_times[0], dtype=np.int64),
            np.asarray(gripper_times[1], dtype=np.int64),
        ),
        gripper_values=(
            np.asarray(gripper_values[0], dtype=np.float64),
            np.asarray(gripper_values[1], dtype=np.float64),
        ),
    )
    _validate_timestamps("video", episode.video_times_ns, source.path)
    for arm, timestamps in zip(("left gripper", "right gripper"), episode.gripper_times_ns):
        _validate_timestamps(arm, timestamps, source.path)
    return episode


def align_umi_timeline(episode: UmiMcapEpisode) -> UmiTimeline:
    """Align aperture to RGB and derive the one real close/hold timeline."""

    video_times = episode.video_times_ns.astype(np.float64)
    aligned = np.column_stack(
        [
            np.interp(video_times, times.astype(np.float64), values)
            for times, values in zip(
                episode.gripper_times_ns,
                episode.gripper_values,
                strict=True,
            )
        ]
    )
    edge_count = max(3, len(aligned) // 10)
    opened = np.median(aligned[:edge_count], axis=0)
    closed = np.median(aligned[-edge_count:], axis=0)
    travel = np.abs(opened - closed)
    active = np.flatnonzero(travel >= MIN_GRIPPER_TRAVEL)
    if len(active) != 1:
        raise RealMcapError(
            "expected exactly one moving UMI gripper, "
            f"got travel left={travel[0]:.3f}, right={travel[1]:.3f}"
        )
    active_index = int(active[0])
    denominator = opened[active_index] - closed[active_index]
    if denominator <= 0:
        raise RealMcapError("UMI close/hold requires closing, not an opening-only episode")
    openness = np.clip(
        (aligned[:, active_index] - closed[active_index]) / denominator,
        0.0,
        1.0,
    )
    try:
        events = detect_gripper_target_only_events(
            openness,
            arm=("left", "right")[active_index],
            operation_start=0,
        )
    except StateLoopError as exc:
        raise RealMcapError(f"cannot derive UMI close/hold timeline: {exc}") from exc
    timestamps_s = ((video_times - video_times[0]) / 1_000_000_000.0).astype(np.float32)
    return UmiTimeline(
        timestamps_s=timestamps_s,
        events=events,
        gripper_travel=(float(travel[0]), float(travel[1])),
    )


def _write_frame_parquet(
    path: Path,
    *,
    episode_id: int,
    global_start: int,
    timestamps_s: NDArray,
    source_frame_indices: tuple[int, ...] | None = None,
) -> None:
    frame_count = len(timestamps_s)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "timestamp": timestamps_s,
            "frame_index": np.arange(frame_count, dtype=np.int64),
            "episode_index": np.full(frame_count, episode_id, dtype=np.int64),
            "index": np.arange(global_start, global_start + frame_count, dtype=np.int64),
            **(
                {"source_frame_index": source_frame_indices}
                if source_frame_indices is not None
                else {}
            ),
        }
    ).to_parquet(path, index=False)


def _write_sidecar(
    path: Path,
    *,
    converted: ConvertedUmiEpisode,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "source_mcap": converted.source.path.name,
        "camera": OUTPUT_CAMERA,
        "camera_topic": CAMERA_TOPIC,
        "depth_available": False,
        "robot_state_available": False,
        "timeline_source": (
            "reviewed_video_window"
            if isinstance(converted.timeline.events, VideoWindowEvents)
            else "derived_from_mcap_gripper_position"
        ),
        **converted.timeline.metadata,
    }
    with h5py.File(path, "w") as handle:
        handle.attrs["format_version"] = "robotwin_umi_object_sidecar_v1"
        handle.attrs["width"] = converted.width
        handle.attrs["height"] = converted.height
        handle.attrs["fps"] = converted.fps
        handle.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)


def _episode_metadata(
    converted: ConvertedUmiEpisode,
    *,
    task_text: str,
) -> dict[str, Any]:
    source_episode = converted.source.values.get("episode_index")
    return {
        "episode_index": converted.episode_id,
        "tasks": [task_text],
        "length": converted.frame_count,
        "source_mcap": converted.source.path.name,
        "source_episode_index": None if source_episode is None else int(source_episode),
        "source_start_time_us": converted.source.start_time_us,
        "camera": OUTPUT_CAMERA,
        "source_camera_topic": CAMERA_TOPIC,
        "depth_available": False,
        "robot_state_available": False,
        **converted.timeline.metadata,
    }


def _write_metadata(
    staging: Path,
    *,
    final_output: Path,
    task: str,
    task_text: str,
    task_text_zh: str,
    converted: Sequence[ConvertedUmiEpisode],
    limit: int | None,
    task_config: UmiTaskConfig | None = None,
) -> None:
    if not converted:
        raise RealMcapError("conversion selected no UMI episodes")
    shapes = {(item.height, item.width) for item in converted}
    if len(shapes) != 1:
        raise RealMcapError("converted episodes do not share one video resolution")
    height, width = next(iter(shapes))
    episode_ids = [item.episode_id for item in converted]
    video_window = task_config is not None
    mode = AnnotationMode.TARGET_ONLY if task_config is None else task_config.mode
    manifest = {
        "format": "robotwin_umi_video_extract_v1"
        if video_window
        else "robotwin_umi_target_only_extract_v1",
        "profile": mode.value,
        "target_profile": (
            "video_object"
            if video_window and mode is AnnotationMode.TARGET_ONLY
            else "grasp_manipulation"
        ),
        "task": task,
        **(
            {"task_kind": "video_object" if video_window else "single_movable_target"}
            if mode is AnnotationMode.TARGET_ONLY
            else {}
        ),
        "camera": OUTPUT_CAMERA,
        "source_camera_topic": CAMERA_TOPIC,
        "timeline_source": "video_window" if video_window else "episode_metadata",
        "depth_available": False,
        "robot_state_available": False,
        "dataset_root": str(final_output.resolve()),
        "episode_indices": episode_ids,
        "episode_count": len(episode_ids),
        "frame_shape_hw": [height, width],
        "raw_video_frame_surplus": 0,
        "usable_frame_count_source": "parquet",
        "smoke_episode_ids": episode_ids[:1],
        "regression_episode_ids": episode_ids,
        "conversion_limit": limit,
        "included_sources": [
            {
                "episode_index": item.episode_id,
                "source_mcap": item.source.path.name,
                "source_episode_index": item.source.values.get("episode_index"),
                **item.timeline.metadata,
            }
            for item in converted
        ],
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
    write_jsonl(
        staging / "meta" / "episodes.jsonl",
        (_episode_metadata(item, task_text=task_text) for item in converted),
    )
    write_jsonl(
        staging / "meta" / "tasks.jsonl",
        [
            {
                "task_index": 0,
                "task": task_text,
                "task_zh": task_text_zh,
                "target": None,
                "receiver": None,
            }
        ],
    )
    info = {
        "codebase_version": "robotwin_annotation_v2",
        "robot_type": "umi_mobile_manipulator",
        "total_episodes": len(converted),
        "total_frames": sum(item.frame_count for item in converted),
        "total_tasks": 1,
        "total_videos": len(converted),
        "fps": float(np.median([item.fps for item in converted])),
        "rgb_channel_order": "RGB",
        "depth_available": False,
        "robot_state_available": False,
    }
    (staging / "meta" / "info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _discover_sources(input_root: Path) -> list[SourceMetadata]:
    paths = sorted(path for path in input_root.glob("*.mcap") if path.is_file())
    if not paths:
        raise RealMcapError(f"source directory contains no MCAP files: {input_root}")
    sources = [read_source_metadata(path) for path in paths]
    starts = [source.start_time_us for source in sources]
    if len(starts) != len(set(starts)):
        raise RealMcapError("source MCAP start_time_us values are not unique")
    return sorted(sources, key=lambda item: item.start_time_us)


def materialize_umi_mcap_dataset(
    input_root: Path,
    output_root: Path,
    *,
    task: str,
    task_text: str,
    task_text_zh: str,
    limit: int | None = None,
    task_config: UmiTaskConfig | None = None,
) -> dict[str, Any]:
    """Atomically materialize a state-free UMI target-only dataset."""

    input_root = input_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if task_config is not None:
        task_text, task_text_zh = task_config.task_text, task_config.task_text_zh
    if not input_root.is_dir():
        raise RealMcapError(f"source directory does not exist: {input_root}")
    if output_root.exists():
        raise RealMcapError(f"refusing to overwrite existing output directory: {output_root}")
    if not task.strip() or task in {".", ".."} or Path(task).name != task:
        raise RealMcapError("task must be one non-empty path component")
    if not task_text.strip() or not task_text_zh.strip():
        raise RealMcapError("task text must be non-empty")
    if limit is not None and limit < 1:
        raise RealMcapError("limit must be positive")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    sources = _discover_sources(input_root)
    if task_config is not None:
        missing = set(task_config.episodes) - {source.path.name for source in sources}
        if missing:
            raise RealMcapError(f"reviewed UMI sources are missing: {sorted(missing)}")
        sources = [source for source in sources if source.path.name in task_config.episodes]
    if limit is not None:
        sources = sources[:limit]

    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.staging-", dir=output_root.parent))
    converted: list[ConvertedUmiEpisode] = []
    global_index = 0
    try:
        for episode_id, source in enumerate(sources):
            episode = (
                read_umi_mcap_episode(source)
                if task_config is None
                else read_umi_mcap_episode(source, require_grippers=False)
            )
            chunk = f"chunk-{episode_id // 1000:03d}"
            stem = f"episode_{episode_id:06d}"
            video_path = (
                staging / "videos" / chunk / f"observation.images.{OUTPUT_CAMERA}" / f"{stem}.mp4"
            )
            if task_config is None:
                timeline = align_umi_timeline(episode)
                frame_count, width, height, fps = write_video(
                    episode.video_payloads, episode.video_times_ns, video_path
                )
            else:
                frames = decode_h264_frames(episode.video_payloads)
                source_indices = tuple(cast(int, frame.pts) for frame in frames)
                timestamps = episode.video_times_ns[list(source_indices)]
                timeline = UmiTimeline(
                    timestamps_s=((timestamps - timestamps[0]) / 1e9).astype(np.float32),
                    events=VideoWindowEvents(
                        task_config.episodes[source.path.name], 0, len(frames) - 1
                    ),
                    source_frame_indices=source_indices,
                    source_frame_count=len(episode.video_payloads),
                )
                frame_count, width, height, fps = encode_video_frames(
                    frames, timestamps, video_path
                )
            if frame_count != len(timeline.timestamps_s):
                raise RealMcapError("video and derived timeline frame counts differ")
            _write_frame_parquet(
                staging / "data" / chunk / f"{stem}.parquet",
                episode_id=episode_id,
                global_start=global_index,
                timestamps_s=timeline.timestamps_s,
                source_frame_indices=timeline.source_frame_indices,
            )
            item = ConvertedUmiEpisode(
                episode_id,
                frame_count,
                width,
                height,
                fps,
                source,
                timeline,
            )
            _write_sidecar(staging / "sidecars" / f"{stem}.hdf5", converted=item)
            converted.append(item)
            global_index += frame_count
        _write_metadata(
            staging,
            final_output=output_root,
            task=task,
            task_text=task_text,
            task_text_zh=task_text_zh,
            converted=converted,
            limit=limit,
            task_config=task_config,
        )
        os.replace(staging, output_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "format_version": "robotwin_umi_conversion_summary_v1",
        "output_root": str(output_root),
        "task": task,
        "profile": AnnotationMode.TARGET_ONLY.value
        if task_config is None
        else task_config.mode.value,
        "camera": OUTPUT_CAMERA,
        "episode_count": len(converted),
        "frame_count": sum(item.frame_count for item in converted),
        "robot_state_available": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert UMI MCAPs to a state-free target-only RoboTwin task extract."
    )
    parser.add_argument("input_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--task", help="task name; defaults to the input directory name")
    parser.add_argument("--task-text", default="Pick up the object.")
    parser.add_argument("--task-text-zh", default="拿起物体。")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--task-config", type=Path, help="reviewed single-arm video task selection YAML"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    task = args.task or args.input_root.name
    try:
        summary = materialize_umi_mcap_dataset(
            args.input_root,
            args.output_root,
            task=task,
            task_text=args.task_text,
            task_text_zh=args.task_text_zh,
            limit=args.limit,
            task_config=None
            if args.task_config is None
            else load_umi_task_config(args.task_config, task),
        )
    except RealMcapError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


__all__ = [
    "CAMERA_TOPIC",
    "GRIPPER_TOPICS",
    "OUTPUT_CAMERA",
    "ConvertedUmiEpisode",
    "UmiMcapEpisode",
    "UmiTimeline",
    "align_umi_timeline",
    "main",
    "materialize_umi_mcap_dataset",
    "read_umi_mcap_episode",
]
