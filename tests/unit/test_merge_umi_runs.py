from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from scripts import merge_umi_runs as merge


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_batch(tmp_path: Path) -> tuple[Path, list[dict[str, Any]]]:
    source = tmp_path / "runs"
    records = []
    for episode, task, height, status in (
        (3, "pick_up", 352, "needs_temporal_review"),
        (1, "pick_up", 480, "auto_complete"),
        (5, "pick_up", 480, "failed"),
        (7, "pick_up", 352, "failed"),
        (10, "place", 480, "auto_complete"),
    ):
        group = f"{task}_640x{height}"
        name = f"episode_{episode:06d}"
        artifact = source / group / group / name / "cam_high"
        if episode != 7:
            artifact.mkdir(parents=True)
            (artifact / "masks.npz").write_bytes(b"original masks")
        overlay = source / group / "rendered_videos" / f"{name}_cam_high_overlay.mp4"
        if status != "failed":
            overlay.parent.mkdir(parents=True, exist_ok=True)
            overlay.write_bytes(b"original overlay")
        raw = tmp_path / "inputs" / group / "videos" / f"{name}.mp4"
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_bytes(b"original RGB")
        records.append(
            {
                "id": f"source_{episode}",
                "episode_index": episode,
                "episode": episode,
                "episode_uid": f"source_{episode}",
                "action_id": "A_001" if task == "pick_up" else "A_002",
                "group": group,
                "task": group,
                "run_id": f"{group}-selected",
                "status": status,
                "width": 640,
                "height": height,
                "frame_count": 9,
                "active_arm": "right",
                "objects_ready": status != "failed",
                "video": str(overlay) if status != "failed" else None,
                "artifact_dir": str(artifact),
                "source_video": str(raw),
                "error": "tracking failed" if status == "failed" else None,
            }
        )
    records.extend(
        [
            {
                "id": "screened_out",
                "episode_index": 2,
                "action_id": "A_001",
                "status": "excluded_bimanual",
                "reason": "Both arms manipulate the target.",
            },
            {
                "id": "uncertain_arm",
                "episode_index": 11,
                "action_id": "A_002",
                "status": "needs_arm_review",
                "reason": "Arm ownership is unclear.",
            },
        ]
    )
    _write_json(source / "batch_status.json", {"status": "finished", "phase": "full"})
    _write_json(source / "episode_results.json", {"phase": "full", "episodes": records})
    return source, records


@pytest.fixture
def rendered_tasks(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[dict[str, Any]]]:
    calls: dict[str, list[dict[str, Any]]] = {}

    def render(
        results_path: Path, datasets_root: Path, output_dir: Path, **kwargs: Any
    ) -> dict[str, Any]:
        records = json.loads(results_path.read_text())["episodes"]
        calls[results_path.parent.name] = records
        output_dir.mkdir(parents=True)
        sheets = ["early.jpg", "overall_index.jpg", "late.jpg"]
        for sheet in sheets:
            (output_dir / sheet).write_bytes(b"mock review sheet")
        manifest = {
            "sheets": sheets,
            "counts": dict(Counter(row["status"] for row in records)),
            "samples": [{"episode": row["episode"], "status": row["status"]} for row in records],
        }
        _write_json(output_dir / "manifest.json", manifest)
        return manifest

    monkeypatch.setattr(merge, "render_batch_sheets", render)
    return calls


def test_merge_links_both_resolutions_and_preserves_failed_and_screened_records(
    tmp_path: Path, rendered_tasks: dict[str, list[dict[str, Any]]]
) -> None:
    source, originals = _make_batch(tmp_path)
    source_files = {path: path.read_bytes() for path in source.rglob("*") if path.is_file()}
    output = tmp_path / "merged"

    result = merge.merge_runs(source, output, workers=2)

    assert result["status"] == "finished"
    assert json.loads((output / "merge_status.json").read_text())["status"] == "finished"
    assert set(rendered_tasks) == {"pick_up", "place"}
    assert [row["episode"] for row in rendered_tasks["pick_up"]] == [1, 3, 5, 7]
    assert [row["episode"] for row in rendered_tasks["place"]] == [10]
    merged = json.loads((output / "episode_results.json").read_text())["episodes"]
    by_id = {row["id"]: row for row in merged}
    assert len(by_id) == len(originals)
    for original in originals:
        row = by_id[original["id"]]
        assert row["task"] == ("pick_up" if row["action_id"] == "A_001" else "place")
        for field in ("id", "action_id", "episode_index", "status", "reason", "error"):
            if field in original:
                assert row[field] == original[field]
        if "group" not in original:
            continue
        assert row in rendered_tasks[row["task"]]
        for field in ("group", "width", "height", "run_id", "episode", "episode_uid"):
            assert row[field] == original[field]
        assert row["source_artifact_dir"] == original["artifact_dir"]
        assert row["source_overlay"] == original["video"]
        assert row["source_input_video"] == original["source_video"]
        raw = Path(row["source_video"])
        assert raw.is_symlink()
        assert raw.parent == output / row["task"] / "source_videos"
        assert raw.resolve() == Path(original["source_video"])
        episode_link = output / row["task"] / f"episode_{row['episode']:06d}"
        if row["episode"] == 7:
            assert row["artifact_dir"] is None
            assert not episode_link.exists()
            assert not episode_link.is_symlink()
        else:
            assert episode_link.is_symlink()
            assert episode_link.resolve() == Path(original["artifact_dir"]).parent
            assert Path(row["artifact_dir"]) == episode_link / "cam_high"
        if original["video"]:
            overlay = Path(row["video"])
            assert overlay.is_symlink()
            assert overlay.parent == output / row["task"] / "rendered_videos"
            assert overlay.resolve() == Path(original["video"])
        else:
            assert row["video"] is None
    assert (output / "index.html").is_file()
    for task in rendered_tasks:
        assert (output / task / "index.html").is_file()
        for name in ("early.jpg", "overall_index.jpg", "late.jpg", "manifest.json"):
            assert (output / task / "review_sheets" / name).is_file()
    assert {path: path.read_bytes() for path in source.rglob("*") if path.is_file()} == source_files


@pytest.mark.parametrize(
    "invalid",
    ["unfinished", "duplicate_id", "duplicate_episode", "dimensions", "action", "overlay"],
)
def test_invalid_source_is_rejected_before_creating_output(
    tmp_path: Path, rendered_tasks: dict[str, list[dict[str, Any]]], invalid: str
) -> None:
    source, records = _make_batch(tmp_path)
    expected_error: type[Exception] = ValueError
    if invalid == "unfinished":
        _write_json(source / "batch_status.json", {"status": "running"})
    elif invalid == "duplicate_id":
        records[-1]["id"] = records[0]["id"]
    elif invalid == "duplicate_episode":
        records[-1]["episode_index"] = records[0]["episode_index"]
    elif invalid == "dimensions":
        records[0]["height"] = 480
    elif invalid == "action":
        records[4]["action_id"] = "A_001"
    elif invalid == "overlay":
        Path(records[0]["video"]).unlink()
        expected_error = FileNotFoundError
    _write_json(source / "episode_results.json", {"episodes": records})
    output = tmp_path / "merged"

    with pytest.raises(expected_error):
        merge.merge_runs(source, output)

    assert not output.exists()
    assert not rendered_tasks


def test_existing_output_is_not_overwritten(
    tmp_path: Path, rendered_tasks: dict[str, list[dict[str, Any]]]
) -> None:
    source, _ = _make_batch(tmp_path)
    output = tmp_path / "merged"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("previous output")

    with pytest.raises(FileExistsError):
        merge.merge_runs(source, output)

    assert list(output.iterdir()) == [marker]
    assert marker.read_text() == "previous output"
    assert not rendered_tasks


def test_renderer_failure_never_marks_merge_finished(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _ = _make_batch(tmp_path)

    def fail(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("cannot decode source video")

    monkeypatch.setattr(merge, "render_batch_sheets", fail)
    output = tmp_path / "merged"

    with pytest.raises(RuntimeError, match="cannot decode source video"):
        merge.merge_runs(source, output)

    status = output / "merge_status.json"
    assert not status.exists() or json.loads(status.read_text())["status"] != "finished"
