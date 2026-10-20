from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from robotwin_annotation_v2.adapters.artifact_store import ArtifactStore
from robotwin_annotation_v2.adapters.real_mcap import write_jsonl
from robotwin_annotation_v2.application.umi_batch import _assemble_group, initialize_batch


def test_group_keeps_distinct_source_ids_texts_and_consecutive_indices(tmp_path: Path) -> None:
    items = []
    for index, (uid, text) in enumerate(
        (("source_a", "Pick up the cup"), ("source_b", "Pick up the bowl"))
    ):
        root = tmp_path / "converted" / uid
        ArtifactStore.write_json(
            root / "EXTRACT_MANIFEST.json",
            {
                "task": "pick_up_8x8",
                "frame_shape_hw": [8, 8],
                "source_camera_topic": "video",
            },
        )
        ArtifactStore.write_json(root / "meta/info.json", {"fps": 30.0})
        write_jsonl(
            root / "meta/episodes.jsonl",
            [
                {
                    "episode_index": index,
                    "source_episode_uid": uid,
                    "tasks": [text],
                    "source_mcap": f"/{uid}/episode.mcap",
                }
            ],
        )
        write_jsonl(root / "meta/tasks.jsonl", [{"task_index": 0, "task": text}])
        for name in (
            f"videos/chunk-000/episode_{index:06d}.mp4",
            f"sidecars/episode_{index:06d}.hdf5",
        ):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"cached")
        parquet = root / "data/chunk-000" / f"episode_{index:06d}.parquet"
        parquet.parent.mkdir(parents=True)
        pd.DataFrame({"index": [0, 1], "source_frame_index": [0, 3]}).to_parquet(parquet)
        items.append(
            {
                "id": uid,
                "episode_index": index,
                "root": str(root),
                "source_camera_topic": "video",
                "action_id": "A_001",
            }
        )
    group = _assemble_group("pick_up_8x8", items, tmp_path)
    output = Path(group["root"])
    metadata = [
        json.loads(line) for line in (output / "meta/episodes.jsonl").read_text().splitlines()
    ]
    assert [row["tasks"] for row in metadata] == [["Pick up the cup"], ["Pick up the bowl"]]
    assert [row["source_episode_uid"] for row in metadata] == ["source_a", "source_b"]
    second = pd.read_parquet(output / "data/chunk-000/episode_000001.parquet")
    assert second["index"].tolist() == [2, 3]
    assert second["source_frame_index"].tolist() == [0, 3]
    assert (output / "videos/chunk-000/episode_000001.mp4").is_symlink()
    assert group["frame_count"] == 4
    assert _assemble_group("pick_up_8x8", items, tmp_path) == group


def test_inventory_rejects_mistyped_excluded_episode(tmp_path: Path) -> None:
    ArtifactStore.write_text(tmp_path / "manifest.json", "[]")
    ArtifactStore.write_text(tmp_path / "prompt.txt", "screen")
    rules = tmp_path / "rules.yaml"
    rules.write_text(
        "excluded_actions: [A_003]\nscreen_prompt: prompt.txt\n"
        "excluded_episodes: {not_in_source: exclude}\n"
    )
    with pytest.raises(ValueError, match="exclusions absent"):
        initialize_batch(tmp_path, tmp_path / "batch", rules)
