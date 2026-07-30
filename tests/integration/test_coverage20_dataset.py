from __future__ import annotations

from pathlib import Path

import pytest

from robotwin_annotation_v2.adapters import RoboTwinDataset
from robotwin_annotation_v2.config import load_config
from robotwin_annotation_v2.models import EpisodeRef
from robotwin_annotation_v2.pipeline import build_loop_context


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_config(PROJECT_ROOT / "configs/pilot_move_pillbottle_pad.yaml")


def _dataset() -> RoboTwinDataset:
    if not CONFIG.dataset.root.is_dir():
        pytest.skip(f"external coverage20 dataset unavailable: {CONFIG.dataset.root}")
    return RoboTwinDataset(
        CONFIG.dataset.root,
        task=CONFIG.dataset.task,
        camera=CONFIG.dataset.camera,
        manifest_path=CONFIG.dataset.manifest,
    )


def test_coverage20_preflight() -> None:
    report = _dataset().preflight(CONFIG.dataset.regression_episode_ids)

    assert report["passed"]
    assert report["episode_count"] == 20
    assert all(
        value["raw_video_frame_surplus"] == 1
        for value in report["content"].values()
    )


def test_episode_7152_matches_verified_boundaries() -> None:
    dataset = _dataset()
    ref = EpisodeRef(CONFIG.dataset.task, 7152, CONFIG.dataset.camera)

    context = build_loop_context(dataset, ref)

    assert context.events.to_json() == {
        "active_arm": "right",
        "t_move_start": 4,
        "t_close_start": 55,
        "t_close_done": 67,
        "t_open_start": 119,
        "t_open_done": 132,
    }
    assert context.frame_count == 138
    assert context.task_text
    assert context.seed_candidates("target")
    assert context.seed_candidates("receiver")


def test_sparse_rgb_frames_decode_in_state_aligned_range() -> None:
    dataset = _dataset()
    ref = EpisodeRef(CONFIG.dataset.task, 7152, CONFIG.dataset.camera)

    frames = dataset.read_frames(ref, (0, 68, 120))

    assert tuple(frames) == (0, 68, 120)
    assert all(image.size == (320, 240) for image in frames.values())


def test_all_coverage20_episodes_have_one_ordered_loop() -> None:
    dataset = _dataset()
    for episode_index in CONFIG.dataset.regression_episode_ids:
        ref = EpisodeRef(CONFIG.dataset.task, episode_index, CONFIG.dataset.camera)
        context = build_loop_context(dataset, ref)
        events = context.events
        assert 0 <= events.t_move_start <= events.t_close_start
        assert events.t_close_start < events.t_close_done
        assert events.t_close_done < events.t_open_start < events.t_open_done
        assert events.t_open_done < context.frame_count
