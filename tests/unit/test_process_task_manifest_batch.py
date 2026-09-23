from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts import process_task_manifest_batch as batch


@pytest.mark.parametrize("smoke,expected", [(False, 3), (True, 2)])
def test_batch_reviews_every_selected_source_episode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, smoke: bool, expected: int
) -> None:
    groups, episodes = [], []
    for source, ids in (("source_a", [0, 1]), ("source_b", [0])):
        root = tmp_path / "inputs" / source
        for folder in ("data", "videos", "sidecars", "meta"):
            (root / folder).mkdir(parents=True)
        manifest = {
            "profile": "target_only",
            "task": "open_door",
            "camera": "head_left",
            "task_kind": "door_open_action_site",
            "episode_indices": ids,
            "timeline_source": "episode_metadata",
        }
        (root / "EXTRACT_MANIFEST.json").write_text(json.dumps(manifest))
        (root / "meta/episodes.jsonl").write_text(
            "".join(
                json.dumps({"episode_index": i, "target_only_events": {"active_arm": "right"}})
                + "\n"
                for i in ids
            )
        )
        groups.append(
            {
                "group_id": source,
                "input_root": str(root),
                "source_dataset": source,
                "episode_ids": ids,
            }
        )
        episodes.extend(
            {
                "episode_uid": f"{source}/episode_{i:06d}",
                "frame_count": 9,
                "split": "train",
                "files": {"videos": {"head_left": f"{source}-{i}.mp4"}},
            }
            for i in ids
        )
    (tmp_path / "selection.json").write_text(json.dumps({"episodes": episodes}))
    (tmp_path / "batch_plan.json").write_text(
        json.dumps(
            {
                "source_manifest": str(tmp_path / "selection.json"),
                "episode_count": 3,
                "groups": groups,
            }
        )
    )
    monkeypatch.setenv("QWEN_API_KEY", "test-key")

    def process(config: Any, **kwargs: Any) -> dict[str, Any]:
        records = [
            {"episode": i, "status": "completed" if i == 0 else "failed"}
            for i in kwargs["episode_ids"]
        ]
        for record in records:
            kwargs["episode_terminal_callback"](record["episode"], record["status"])
        return {"records": records, "dynamic_manifest": config.dataset.manifest_data}

    def render(_config: Any, **kwargs: Any) -> dict[str, Any]:
        directory = kwargs["output_dir"] / "rendered_videos"
        directory.mkdir(parents=True)
        manifest = directory / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "episodes": [
                        {"episode_index": i, "output_video": f"{i}.mp4"}
                        for i in kwargs["episode_ids"]
                    ]
                }
            )
        )
        return {"manifest": str(manifest)}

    reviews = []
    monkeypatch.setattr(batch, "process_dataset", process)
    monkeypatch.setattr(batch, "render_sam_processed", render)
    monkeypatch.setattr(
        batch, "render_batch_sheets", lambda path, *_: reviews.append(json.loads(path.read_text()))
    )
    monkeypatch.setattr(
        batch,
        "create_process_ui",
        lambda *_, **__: type(
            "UI",
            (),
            {
                "finish": lambda *_: None,
                "close": lambda *_: None,
            },
        )(),
    )

    result = batch.run_batch(tmp_path, Path("configs/process.yaml"), gpu=2, smoke=smoke)

    assert len(result["episodes"]) == expected
    assert len({r["episode_uid"] for r in result["episodes"]}) == expected
    assert reviews == [result]
    assert result["counts"].get("failed", 0) == (0 if smoke else 1)
    progress = json.loads((tmp_path / "batch_status.json").read_text())
    assert progress["status"] == "finished"
    assert len(progress["records"]) == expected
