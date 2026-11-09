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


def test_marker_inventory_classifies_transport_even_when_excluded(tmp_path: Path) -> None:
    import yaml

    rules = yaml.safe_load(Path("configs/datasets/umi_h200.yaml").read_text())
    assert rules["bimanual_actions"] == ["A_003"]
    assert rules["excluded_actions"] == ["A_003"]
    assert rules["excluded_episodes"] == {}
    # With every source excluded, no previews, MCAP or model are required.
    ArtifactStore.write_json(
        tmp_path / "manifest.json", [{"id": "transport", "action_id": "A_003"}]
    )
    inventory = initialize_batch(
        tmp_path, tmp_path / "batch", Path("configs/datasets/umi_h200.yaml")
    )
    assert inventory["action_arm_types"] == {"A_003": "bimanual"}
    assert inventory["excluded_action_count"] == 1
    assert inventory["episodes"] == []


@pytest.mark.parametrize("decision", ["left", "right", "bimanual", "no_action"])
def test_marker_screen_uses_task_type_and_identifies_primary_side(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, decision: str
) -> None:
    from types import SimpleNamespace
    from typing import Any

    import yaml
    from PIL import Image

    from robotwin_annotation_v2.adapters.umi_review import UmiPreview
    from robotwin_annotation_v2.application import umi_batch

    rules = yaml.safe_load(Path("configs/datasets/umi_h200.yaml").read_text())
    prompt = (Path("configs/datasets") / rules["screen_prompt"]).read_text()
    frames = []
    for time in (0, 100):
        path = tmp_path / f"{time}.png"
        Image.new("RGB", (4, 4)).save(path)
        frames.append(UmiPreview(path, f"head/right@{time}ms", time))
    monkeypatch.setattr(umi_batch, "load_umi_previews", lambda _: tuple(frames))
    calls = []

    def complete(messages: Any, **kwargs: Any) -> Any:
        calls.append(messages)
        return SimpleNamespace(
            content=json.dumps(
                {
                    "decision": decision,
                    "confidence": 0.99,
                    "left_hand": "Steadies the container",
                    "right_hand": "Operates the target",
                    "evidence_frames": [frame.label for frame in frames],
                    "reason": "Primary operating hand identified from contact evidence",
                }
            ),
            model="test-model",
            finish_reason="stop",
        )

    client = SimpleNamespace(model_id="test-model", endpoint="test", complete=complete)
    actions = ["A_003", *rules["tasks"]]
    inventory = {
        "signature": "marker-policy",
        "rules": rules,
        "prompt": prompt,
        "episodes": [
            {
                "id": action,
                "action_id": action,
                "source_dir": str(tmp_path),
                "task_text": "Operate the target",
            }
            for action in actions
        ],
    }
    counts = umi_batch.screen_batch(inventory, tmp_path / "batch", client, workers=1)
    assert counts["excluded_bimanual"] == 1
    expected = {
        "bimanual": "screen_failed",
        "no_action": "needs_arm_review",
    }.get(decision, "selected")
    assert counts[expected] == 15
    assert len(calls) == 15
    for action in actions:
        row = json.loads((tmp_path / "batch/screening" / f"{action}.json").read_text())
        assert row["arm_classification_source"] == "task_action_id"
        assert row["arm_type"] == ("bimanual" if action == "A_003" else "single")
        assert row["status"] == ("excluded_bimanual" if action == "A_003" else expected)
    # Same inputs reuse successful cached decisions; failed reviews are retried.
    if decision != "bimanual":
        assert umi_batch.screen_batch(inventory, tmp_path / "batch", client, workers=1) == counts
        assert len(calls) == 15


def test_marker_inventory_rejects_unknown_category_before_source_access(tmp_path: Path) -> None:
    import yaml

    from robotwin_annotation_v2.application.umi_batch import _task_arm_type

    rules_path = Path("configs/datasets/umi_h200.yaml")
    rules = yaml.safe_load(rules_path.read_text())
    with pytest.raises(ValueError, match="unknown task action category.*A_999"):
        _task_arm_type("A_999", rules)
    # No source paths are present: category validation must happen first.
    ArtifactStore.write_json(
        tmp_path / "manifest.json", [{"id": "unknown", "action_id": "A_999"}]
    )
    batch = tmp_path / "batch"
    with pytest.raises(ValueError, match="unknown task action category.*A_999"):
        initialize_batch(tmp_path, batch, rules_path)
    assert not (batch / "inventory.json").exists()
