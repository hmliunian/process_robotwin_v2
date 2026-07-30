from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_run_entrypoint_calls_qwen_then_sam_with_same_run_id() -> None:
    module = runpy.run_path(str(PROJECT_ROOT / "scripts/run_target_receiver.py"))
    calls: list[tuple[str, int, str]] = []

    def fake_qwen(_config: Any, episode_index: int, run_id: str) -> None:
        calls.append(("qwen", episode_index, run_id))

    def fake_sam(_config: Any, episode_index: int, run_id: str) -> None:
        calls.append(("sam", episode_index, run_id))

    module["run_pipeline"].__globals__["run_qwen"] = fake_qwen
    module["run_pipeline"].__globals__["run_sam"] = fake_sam

    module["run_pipeline"]("config", 7152, "test-run")

    assert calls == [
        ("qwen", 7152, "test-run"),
        ("sam", 7152, "test-run"),
    ]
