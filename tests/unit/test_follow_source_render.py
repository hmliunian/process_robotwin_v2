from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import follow_source_render as follower


def _write_manifest(run_dir: Path, episode_ids: tuple[int, ...]) -> None:
    video_dir = run_dir / "rendered_videos"
    sheet_dir = video_dir / "review_sheets"
    sheet_dir.mkdir(parents=True)
    episodes = []
    for episode_id in episode_ids:
        name = f"episode_{episode_id:06d}_cam_high_overlay.mp4"
        (video_dir / name).write_bytes(b"video")
        episodes.append({"episode_index": episode_id, "output_video": name})
    (sheet_dir / "target_early.jpg").write_bytes(b"sheet")
    (video_dir / "manifest.json").write_text(
        json.dumps(
            {
                "requested_run_id": run_dir.name,
                "episodes": episodes,
                "review_sheets": ["review_sheets/target_early.jpg"],
            }
        ),
        encoding="utf-8",
    )


def test_completed_episode_ids_selects_only_renderable_records() -> None:
    summary = {
        "records": [
            {"episode": 3, "status": "completed"},
            {"episode": 4, "status": "failed"},
            {"episode": 5, "status": "sam_incomplete"},
            {"episode": 6, "status": "skipped_complete"},
        ]
    }

    assert follower._completed_episode_ids(summary) == (3, 6)


def test_completed_episode_ids_rejects_duplicates_and_empty_results() -> None:
    with pytest.raises(ValueError, match="repeats"):
        follower._completed_episode_ids(
            {
                "records": [
                    {"episode": 3, "status": "completed"},
                    {"episode": 3, "status": "skipped_complete"},
                ]
            }
        )
    with pytest.raises(ValueError, match="no completed episodes"):
        follower._completed_episode_ids({"records": [{"episode": 3, "status": "failed"}]})


def test_render_is_complete_requires_all_videos_and_review_sheets(tmp_path: Path) -> None:
    run_dir = tmp_path / "source-run"
    episode_ids = (3, 6)
    _write_manifest(run_dir, episode_ids)

    assert follower._render_is_complete(run_dir, run_dir.name, episode_ids)

    (run_dir / "rendered_videos/review_sheets/target_early.jpg").unlink()
    assert not follower._render_is_complete(run_dir, run_dir.name, episode_ids)
