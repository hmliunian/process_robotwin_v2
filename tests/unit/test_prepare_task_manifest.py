from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.prepare_task_manifest import episode_metadata, prepare


def _episode(root: Path, source: str, *, terminal_hold: bool = False) -> dict:
    gripper = [0.0] * 3 + [0.08] * 6 + [0.001] * 6
    if not terminal_hold:
        gripper += [0.08] * 6
    state = np.zeros((len(gripper), 7))
    state[:, 6] = gripper
    base = root / source
    parquet = base / "data/chunk-000/episode_000000.parquet"
    video = base / "videos/chunk-000/observation.images.head_left/episode_000000.mp4"
    for path in (parquet, video):
        path.parent.mkdir(parents=True, exist_ok=True)
    video.touch()
    pd.DataFrame(
        {
            "frame_index": range(len(state)),
            "episode_index": [0] * len(state),
            "observation.state": list(state),
        }
    ).to_parquet(parquet, index=False)
    (base / "meta").mkdir()
    names = [f"right_eef_{axis}" for axis in ("x", "y", "z", "roll", "pitch", "yaw")]
    (base / "meta/info.json").write_text(
        json.dumps({"features": {"observation.state": {"names": [*names, "right_gripper_open"]}}})
    )
    return {
        "episode_uid": f"{source}/episode_000000",
        "source_dataset": source,
        "episode_index": 0,
        "source_episode_id": source,
        "task": "open_door",
        "task_text": "Open the room door.",
        "primary_camera": "head_left",
        "split": "val",
        "frame_count": len(state),
        "state_dimension": 7,
        "files": {
            "parquet": str(parquet.relative_to(root)),
            "videos": {"head_left": str(video.relative_to(root))},
        },
    }


def test_preparation_preserves_source_ids_labels_and_original_state(tmp_path: Path) -> None:
    episodes = [_episode(tmp_path, source) for source in ("source_a", "source_b")]
    manifest = tmp_path / "selection.json"
    manifest.write_text(
        json.dumps(
            {
                "format": "remote_operation_task_manifest_v1",
                "total_episodes": 2,
                "episodes": episodes,
            }
        )
    )
    output = tmp_path / "batch"

    plan = prepare(manifest, output, task_kind="door_open_action_site")

    assert len(plan["groups"]) == 2
    for group in plan["groups"]:
        view = Path(group["input_root"])
        metadata = json.loads((view / "meta/episodes.jsonl").read_text())
        assert metadata["episode_uid"] == f"{group['source_dataset']}/episode_000000"
        assert metadata["tasks"] == ["Open the room door."]
        assert metadata["split"] == "val"
        assert metadata["target_only_events"]["active_arm"] == "right"
        path = view / "data/chunk-000/episode_000000.parquet"
        assert path.is_symlink()
        assert np.stack(pd.read_parquet(path)["observation.state"]).shape[1] == 7
        assert not list((view / "sidecars").iterdir())
    with pytest.raises(FileExistsError):
        prepare(manifest, output, task_kind="door_open_action_site")


def test_terminal_hold_keeps_missing_reopen_explicit(tmp_path: Path) -> None:
    metadata = episode_metadata(tmp_path, _episode(tmp_path, "source", terminal_hold=True))
    assert metadata["target_only_events"].get("t_reopen_start") is None
    assert metadata["timeline_derivation"]["reopen_frame"] is None


def test_missing_close_uses_video_window_without_invented_hold(tmp_path: Path) -> None:
    episode = _episode(tmp_path, "source")
    path = tmp_path / episode["files"]["parquet"]
    frame = pd.read_parquet(path)
    state = np.stack(frame["observation.state"])
    state[:, 6] = 0.08
    frame["observation.state"] = list(state)
    frame.to_parquet(path, index=False)

    metadata = episode_metadata(tmp_path, episode)

    assert "target_only_events" not in metadata
    assert metadata["video_events"] == {
        "active_arm": "right",
        "t_start": 0,
        "t_end": len(frame) - 1,
    }
    assert "no stable close" in metadata["timeline_derivation"]["state_timeline_error"]


def test_prepare_rejects_duplicate_episode_uids_before_writing(tmp_path: Path) -> None:
    episode = _episode(tmp_path, "source")
    manifest = tmp_path / "selection.json"
    manifest.write_text(
        json.dumps(
            {
                "format": "remote_operation_task_manifest_v1",
                "total_episodes": 2,
                "episodes": [episode, episode],
            }
        )
    )
    with pytest.raises(ValueError, match="duplicate episode_uid"):
        prepare(manifest, tmp_path / "batch", task_kind="door_open_action_site")
    assert not (tmp_path / "batch").exists()
