from __future__ import annotations

from pathlib import Path

import pytest

import scripts.process_dataset as process_module
from scripts.process_dataset import (
    DiscoveredEpisode,
    build_dynamic_manifest,
    discover_episodes,
)


def _touch_episode(
    root: Path,
    episode_id: int,
    *,
    camera: str = "cam_high",
    sidecar: bool = True,
    video: bool = True,
) -> None:
    chunk = f"chunk-{episode_id // 1000:03d}"
    parquet = root / "data" / chunk / f"episode_{episode_id:06d}.parquet"
    parquet.parent.mkdir(parents=True, exist_ok=True)
    parquet.touch()
    if video:
        video_path = (
            root
            / "videos"
            / chunk
            / f"observation.images.{camera}"
            / f"episode_{episode_id:06d}.mp4"
        )
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.touch()
    if sidecar:
        sidecar_path = root / "sidecars" / f"episode_{episode_id:06d}.hdf5"
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_path.touch()


def test_discover_episodes_cross_checks_video_and_sidecar(tmp_path: Path) -> None:
    _touch_episode(tmp_path, 7)
    _touch_episode(tmp_path, 1001, sidecar=False)

    result = discover_episodes(tmp_path, camera="cam_high")

    assert result.episode_ids == (7,)
    assert result.skipped == (
        {
            "episode": 1001,
            "status": "discovery_skipped",
            "missing": ["sidecar"],
            "parquet": str(
                tmp_path / "data/chunk-001/episode_001001.parquet"
            ),
        },
    )


def test_discover_episodes_rejects_invalid_chunk_name(tmp_path: Path) -> None:
    bad = tmp_path / "data/not-a-chunk"
    bad.mkdir(parents=True)
    (bad / "episode_000001.parquet").touch()

    with pytest.raises(ValueError, match="invalid chunk directory name"):
        discover_episodes(tmp_path, camera="cam_high")


def test_dynamic_manifest_contains_measured_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = DiscoveredEpisode(
        episode_id=7,
        parquet=tmp_path / "episode_000007.parquet",
        video=tmp_path / "episode_000007.mp4",
        sidecar=tmp_path / "episode_000007.hdf5",
    )
    second = DiscoveredEpisode(
        episode_id=8,
        parquet=tmp_path / "episode_000008.parquet",
        video=tmp_path / "episode_000008.mp4",
        sidecar=tmp_path / "episode_000008.hdf5",
    )
    monkeypatch.setattr(
        process_module,
        "_measure_episode",
        lambda _episode: (24, (240, 320), 1),
    )

    manifest = build_dynamic_manifest(
        tmp_path,
        task="task",
        camera="cam_high",
        episodes=(first, second),
    )

    assert manifest["format_version"] == "robotwin_dataset_manifest_dynamic_v1"
    assert manifest["frame_shape_hw"] == [240, 320]
    assert manifest["raw_video_frame_surplus"] == 1
    assert manifest["regression_episode_ids"] == [7, 8]
    assert manifest["dataset_root"] == str(tmp_path.resolve())
