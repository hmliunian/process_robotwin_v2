from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from robotwin_annotation_v2.adapters import umi_mcap
from robotwin_annotation_v2.adapters.real_mcap import RealMcapError, SourceMetadata
from robotwin_annotation_v2.adapters.umi_mcap import (
    UmiMcapEpisode,
    align_umi_timeline,
    materialize_umi_mcap_dataset,
)
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
