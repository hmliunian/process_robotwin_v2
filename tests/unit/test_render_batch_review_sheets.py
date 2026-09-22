from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from scripts import render_batch_review_sheets as review


def _report(tmp_path: Path) -> Path:
    path = tmp_path / "results.json"
    records = [
        {
            "task": task,
            "episode": episode,
            "run_id": "selected-v2",
            "status": status,
            "frame_count": 9,
            "objects_ready": status != "failed",
            "active_arm": "right",
            "video": f"{task}-{episode}.mp4",
        }
        for task, episode, status in (
            ("task_a", 0, "auto_complete"),
            ("task_a", 1, "failed"),
            ("task_b", 0, "needs_temporal_review"),
        )
    ]
    path.write_text(json.dumps({"episodes": records}))
    for record in records:
        (tmp_path / str(record["video"])).touch()
    raw = tmp_path / "task_a/videos/chunk-000/observation.images.cam_high/episode_000001.mp4"
    raw.parent.mkdir(parents=True)
    raw.touch()
    return path


def test_batch_keeps_final_versions_and_failed_raw_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def decode(path: Path, frame_ids: set[int]) -> dict[int, Image.Image]:
        calls.append((path, frame_ids))
        return {frame: Image.new("RGB", (640, 480), "gray") for frame in frame_ids}

    monkeypatch.setattr(review, "_decode_selected", decode)
    output = tmp_path / "sheets"
    manifest = review.render_batch_sheets(
        _report(tmp_path),
        tmp_path,
        output,
        project_root=tmp_path,
    )
    assert [sample["source_kind"] for sample in manifest["samples"]] == [
        "overlay",
        "RAW RGB",
        "overlay",
    ]
    assert all(sample["run_id"] == "selected-v2" for sample in manifest["samples"])
    assert calls[1][0].name == "episode_000001.mp4"
    assert all(frames == {2, 4, 8} for _, frames in calls)
    assert len(manifest["samples"]) == 3
    assert manifest["counts"] == {
        "auto_complete": 1,
        "failed": 1,
        "needs_temporal_review": 1,
    }
    for name in manifest["sheets"]:
        with Image.open(output / name) as image:
            assert image.size == (manifest["width"], manifest["height"])
    assert json.loads((output / "manifest.json").read_text()) == manifest


@pytest.mark.parametrize("invalid", ["duplicate", "empty", "status", "length", "ready"])
def test_batch_rejects_invalid_index(tmp_path: Path, invalid: str) -> None:
    path = _report(tmp_path)
    report = json.loads(path.read_text())
    if invalid == "duplicate":
        report["episodes"].append(report["episodes"][0])
    elif invalid == "empty":
        report["episodes"] = []
    else:
        key, value = {
            "status": ("status", "running"),
            "length": ("frame_count", 0),
            "ready": ("objects_ready", False),
        }[invalid]
        report["episodes"][0][key] = value
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        review.render_batch_sheets(path, tmp_path, tmp_path / "sheets", project_root=tmp_path)


def test_missing_final_overlay_does_not_fall_back_to_raw(tmp_path: Path) -> None:
    path = _report(tmp_path)
    (tmp_path / "task_a-0.mp4").unlink()
    with pytest.raises(FileNotFoundError):
        review.render_batch_sheets(path, tmp_path, tmp_path / "sheets", project_root=tmp_path)
