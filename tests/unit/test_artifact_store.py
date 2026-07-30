from __future__ import annotations

import json
from pathlib import Path

from robotwin_annotation_v2.adapters import ArtifactStore
from robotwin_annotation_v2.models import EpisodeRef


def test_artifact_store_writes_stage_path_atomically(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    ref = EpisodeRef("move_pillbottle_pad", 7152, "cam_high")

    path = store.save_loop("test-run", ref, {"format_version": "loop"})

    assert path == (
        tmp_path
        / "test-run"
        / "move_pillbottle_pad"
        / "episode_007152"
        / "cam_high"
        / "loop.json"
    )
    assert json.loads(path.read_text(encoding="utf-8")) == {"format_version": "loop"}
    assert not list(path.parent.glob("*.tmp"))
