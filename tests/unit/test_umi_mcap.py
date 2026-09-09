from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np
import pandas as pd
import pytest

from robotwin_annotation_v2.adapters import umi_mcap
from robotwin_annotation_v2.adapters.real_mcap import RealMcapError, SourceMetadata
from robotwin_annotation_v2.adapters.umi_mcap import (
    UmiMcapEpisode,
    UmiTaskConfig,
    align_umi_timeline,
    load_umi_task_config,
    materialize_umi_mcap_dataset,
)
from robotwin_annotation_v2.domain import AnnotationMode
from robotwin_annotation_v2.models import TargetOnlyEvents


def _episode(tmp_path: Path, *, move_both: bool = False) -> UmiMcapEpisode:
    count = 20
    timestamps = np.arange(count, dtype=np.int64) * 100_000_000
    left = np.full(count, 100.0)
    right = np.asarray([100.0] * 7 + [80.0, 50.0, 20.0] + [20.0] * 10)
    if move_both:
        left = right.copy()
    return UmiMcapEpisode(
        source=SourceMetadata(tmp_path / "source.mcap", {"start_time_us": "1"}),
        video_times_ns=timestamps,
        video_payloads=tuple(b"frame" for _ in range(count)),
        gripper_times_ns=(timestamps, timestamps),
        gripper_values=(left, right),
    )


def test_align_umi_timeline_uses_only_the_active_gripper(tmp_path: Path) -> None:
    timeline = align_umi_timeline(_episode(tmp_path))

    assert timeline.events == TargetOnlyEvents("right", 0, 7, 11)
    assert timeline.timestamps_s.shape == (20,)
    assert timeline.gripper_travel == (0.0, 80.0)


def test_align_umi_timeline_rejects_two_moving_grippers(tmp_path: Path) -> None:
    with pytest.raises(RealMcapError, match="exactly one moving UMI gripper"):
        align_umi_timeline(_episode(tmp_path, move_both=True))


def test_align_umi_timeline_does_not_reverse_an_opening_episode(tmp_path: Path) -> None:
    episode = _episode(tmp_path)
    opening = replace(
        episode, gripper_values=(episode.gripper_values[0], episode.gripper_values[1][::-1])
    )
    with pytest.raises(RealMcapError, match="closing, not an opening"):
        align_umi_timeline(opening)


def test_video_only_reader_allows_missing_gripper_streams(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = SourceMetadata(tmp_path / "source.mcap", {"start_time_us": "1"})
    source.path.touch()
    message = (
        SimpleNamespace(encoding="flatbuffer", name="foxglove.CompressedVideo"),
        SimpleNamespace(topic=umi_mcap.CAMERA_TOPIC),
        SimpleNamespace(log_time=0, data=b"data"),
    )
    monkeypatch.setattr(
        umi_mcap,
        "mcap_reader_factory",
        lambda: lambda _handle: SimpleNamespace(iter_messages=lambda **_kwargs: iter([message])),
    )
    monkeypatch.setattr(umi_mcap, "parse_compressed_video_payload", lambda _data: b"video")
    assert len(umi_mcap.read_umi_mcap_episode(source, require_grippers=False).video_payloads) == 1
    with pytest.raises(RealMcapError, match="missing required"):
        umi_mcap.read_umi_mcap_episode(source)


def test_umi_materializer_refuses_to_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()

    with pytest.raises(RealMcapError, match="refusing to overwrite"):
        materialize_umi_mcap_dataset(
            source,
            output,
            task="pick_up_real",
            task_text="Pick up the object.",
            task_text_zh="拿起物体。",
        )


def test_umi_materializer_writes_no_robot_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    source_path = source_root / "source.mcap"
    source_path.touch()
    source = SourceMetadata(source_path, {"start_time_us": "1", "episode_index": "9"})
    episode = _episode(tmp_path)
    monkeypatch.setattr(umi_mcap, "read_source_metadata", lambda _path: source)
    monkeypatch.setattr(umi_mcap, "read_umi_mcap_episode", lambda _source: episode)

    def fake_write_video(
        _payloads: object,
        _timestamps: object,
        path: Path,
    ) -> tuple[int, int, int, float]:
        path.parent.mkdir(parents=True)
        path.write_bytes(b"video")
        return 20, 640, 480, 10.0

    monkeypatch.setattr(umi_mcap, "write_video", fake_write_video)
    output = tmp_path / "output"

    summary = materialize_umi_mcap_dataset(
        source_root,
        output,
        task="pick_up_real",
        task_text="Pick up the object.",
        task_text_zh="拿起物体。",
    )

    manifest = json.loads((output / "EXTRACT_MANIFEST.json").read_text())
    frame = pd.read_parquet(output / "data/chunk-000/episode_000000.parquet")
    assert summary["episode_count"] == 1
    assert manifest["timeline_source"] == "episode_metadata"
    assert manifest["robot_state_available"] is False
    assert "observation.state" not in frame.columns


@pytest.mark.parametrize("mode", [AnnotationMode.TARGET_ONLY, AnnotationMode.TOOL_USE])
def test_video_task_materializer_preserves_source_frame_indices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: AnnotationMode
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    source_path = source_root / "source.mcap"
    source_path.touch()
    source = SourceMetadata(source_path, {"start_time_us": "1"})
    episode = _episode(tmp_path)
    monkeypatch.setattr(umi_mcap, "read_source_metadata", lambda _path: source)
    monkeypatch.setattr(umi_mcap, "read_umi_mcap_episode", lambda _source, **_kwargs: episode)
    frames = [av.VideoFrame(8, 8, "rgb24") for _ in range(3)]
    for frame, index in zip(frames, (0, 2, 19), strict=True):
        frame.pts = index
    monkeypatch.setattr(umi_mcap, "decode_h264_frames", lambda _payloads: frames)

    def encode(_frames: object, _timestamps: object, path: Path) -> tuple[int, int, int, float]:
        path.parent.mkdir(parents=True)
        path.write_bytes(b"video")
        return 3, 8, 8, 10.0

    monkeypatch.setattr(umi_mcap, "encode_video_frames", encode)
    config = UmiTaskConfig(mode, "Use the held tool.", "操作工具。", {"source.mcap": "left"})
    output = tmp_path / "output"
    summary = materialize_umi_mcap_dataset(
        source_root,
        output,
        task="tool_task",
        task_text="unused",
        task_text_zh="未用",
        task_config=config,
    )
    assert summary["profile"] == mode.value
    metadata = json.loads((output / "meta/episodes.jsonl").read_text())
    assert metadata["tasks"] == [config.task_text]
    assert metadata["video_events"] == {"active_arm": "left", "t_start": 0, "t_end": 2}
    assert "target_only_events" not in metadata
    assert metadata["timeline_provenance"]["source_frame_indices"] == [0, 2, 19]
    frame_data = pd.read_parquet(output / "data/chunk-000/episode_000000.parquet")
    assert frame_data["source_frame_index"].tolist() == [0, 2, 19]
    assert frame_data["timestamp"].tolist() == pytest.approx([0, 0.2, 1.9])
    assert "observation.state" not in frame_data


def test_reviewed_single_arm_selection_excludes_uncertain_bimanual_clip() -> None:
    path = Path("configs/datasets/umi_single_arm.yaml")
    names = ("push_real", "pull_real", "turn_over_real", "scoop_real", "stir_real", "wipe_real")
    configs = {name: load_umi_task_config(path, name) for name in names}
    assert sum(len(config.episodes) for config in configs.values()) == 59
    assert "01af5625e4b77cedb925ca5ef69cd4a4.mcap" not in configs["wipe_real"].episodes
    with pytest.raises(RealMcapError, match="cannot load"):
        load_umi_task_config(path, "carry_real")
