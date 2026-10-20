from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from robotwin_annotation_v2.adapters.artifact_store import ArtifactStore
from robotwin_annotation_v2.application.umi_pipeline import UmiJob
from robotwin_annotation_v2.config import load_config
from scripts import run_umi_fast_batch as batch
from scripts.run_umi_batch import run_batch

ROOT = Path(__file__).resolve().parents[2]


def fixture_jobs(tmp_path: Path) -> tuple[dict[str, Any], list[UmiJob]]:
    config = load_config(ROOT / "configs/pilot_move_pillbottle_pad.yaml")
    config = replace(
        config,
        output_root=tmp_path / "out",
        dataset=replace(config.dataset, manifest_data={"timeline_source": "video_window"}),
    )
    jobs = []
    groups = []
    for episode, count in enumerate((10, 30, 20)):
        group = f"group-{episode % 2}"
        metadata = {
            "episode_index": episode,
            "frame_count": count,
            "id": f"source-{episode}",
            "episode_uid": f"source-{episode}",
            "status": "converted",
            "action_id": "A_001",
        }
        jobs.append(UmiJob(config, episode, group, metadata, "fixed"))
        if group not in [row["group"] for row in groups]:
            groups.append({"group": group})
    plan = {"episodes": [job.metadata for job in jobs], "groups": groups}
    ArtifactStore.write_json(tmp_path / "batch_plan.json", plan)
    return plan, jobs


def test_partition_keeps_job_profiles_and_balances_work_across_groups(tmp_path: Path) -> None:
    _, jobs = fixture_jobs(tmp_path)
    shards = batch.partition_jobs(jobs, (4, 7))
    assert {job.key for shard in shards for job in shard} == {job.key for job in jobs}
    assert [sum(job.metadata["frame_count"] for job in shard) for shard in shards] == [30, 30]
    for gpu, shard in zip((4, 7), shards, strict=True):
        assert all(job.config.sam3.gpus == (gpu,) for job in shard)
        assert all(job.config.annotation == jobs[0].config.annotation for job in shard)
    assert all(job.config.sam3.gpus == jobs[0].config.sam3.gpus for job in jobs)


def test_batch_keeps_group_ledgers_and_rejects_resume_input_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, jobs = fixture_jobs(tmp_path)
    monkeypatch.setattr(batch, "select_jobs", lambda *_, **__: jobs)
    monkeypatch.setattr(batch, "ProcessPoolExecutor", lambda **_: ThreadPoolExecutor(max_workers=2))
    reviews = []
    monkeypatch.setattr(batch, "render_batch_sheets", lambda path, *_: reviews.append(path))

    def worker(selected: list[UmiJob], output: Path, *_: Any, **__: Any) -> dict[str, Any]:
        result = {
            "episodes": [
                {
                    **job.metadata,
                    "episode": job.episode,
                    "task": job.ref.task,
                    "run_id": job.run_id,
                    "status": "needs_temporal_review",
                    "process_status": "completed",
                    "objects_ready": True,
                    "masks_ready": True,
                }
                for job in selected
            ]
        }
        ArtifactStore.write_json(output / "episode_results.json", result)
        return result

    monkeypatch.setattr(batch, "run_worker", worker)
    output = tmp_path / "out"
    result = batch.run_fast_batch(tmp_path, Path("config"), gpus=(4, 7), output_root=output)
    assert len(result["episodes"]) == 3 and all(row["objects_ready"] for row in result["episodes"])
    assert len(reviews) == 2
    for group in plan["groups"]:
        rows = json.loads((output / group["group"] / "episode_results.json").read_text())[
            "episodes"
        ]
        assert {row["run_id"] for row in rows} == {group["group"]}
    identity = (output / ".pipeline/selection.json").read_bytes()
    monkeypatch.setattr(
        batch, "select_jobs", lambda *_, **__: [replace(jobs[0], input_signature="changed")]
    )
    with pytest.raises(ValueError, match="complete selection"):
        batch.run_fast_batch(tmp_path, Path("config"), gpus=(4, 7), output_root=output)
    assert (output / ".pipeline/selection.json").read_bytes() == identity


def test_worker_failure_preserves_completed_and_unstarted_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, jobs = fixture_jobs(tmp_path)
    monkeypatch.setattr(batch, "select_jobs", lambda *_, **__: jobs)
    monkeypatch.setattr(batch, "ProcessPoolExecutor", lambda **_: ThreadPoolExecutor(max_workers=1))

    def worker(selected: list[UmiJob], output: Path, *_: Any, **__: Any) -> None:
        job = selected[0]
        ArtifactStore.write_json(
            output / "episode_results.json",
            {
                "episodes": [
                    {
                        **job.metadata,
                        "episode": job.episode,
                        "status": "needs_temporal_review",
                        "objects_ready": True,
                        "masks_ready": True,
                        "process_status": "completed",
                    }
                ]
            },
        )
        raise RuntimeError("worker lost")

    monkeypatch.setattr(batch, "run_worker", worker)
    output = tmp_path / "out"
    with pytest.raises(RuntimeError, match="worker lost"):
        batch.run_fast_batch(tmp_path, Path("config"), gpus=(4,), output_root=output)
    result = json.loads((output / "episode_results.json").read_text())
    assert len(result["episodes"]) == 3
    assert sum(row["objects_ready"] for row in result["episodes"]) == 1
    assert (
        json.loads((output / "batch_status.json").read_text())["status"] == "interrupted_or_failed"
    )


def test_existing_entry_routes_fast_mode_and_writes_review_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options = {}

    def fast(*_: Any, **kwargs: Any) -> dict[str, Any]:
        options.update(kwargs)
        return {"counts": {}, "episodes": []}

    monkeypatch.setattr(batch, "run_fast_batch", fast)
    run_batch(tmp_path, Path("config"), gpus=(2,), inference_mode="async")
    assert options["asynchronous"] and options["tracking_batch_size"] == 1
    assert (tmp_path / "runs/index.html").exists()
