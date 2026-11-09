from __future__ import annotations

import json
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from robotwin_annotation_v2.application.sam_workflow import SamEpisodeStages
from robotwin_annotation_v2.application.umi_pipeline import CURRENT_JOB, run_umi_pipeline
from tests.unit.test_umi_pipeline import fake_runtime, jobs_at, reviewed


def staged_jobs(tmp_path: Path, count: int) -> list[Any]:
    return [
        replace(
            job, config=replace(job.config, sam3=replace(job.config.sam3, seed_image_only=True))
        )
        for job in jobs_at(tmp_path, count)
    ]


def test_many_qc_producers_continue_while_writer_and_renderer_are_blocked(tmp_path: Path) -> None:
    jobs = staged_jobs(tmp_path, 12)
    owner = threading.get_ident()
    qc = threading.Barrier(8)
    writer_started = threading.Event()
    ninth_seed = threading.Event()
    render_started = threading.Event()
    eleventh_seed = threading.Event()
    saved: list[int] = []

    def seed(episode: int) -> int:
        assert threading.get_ident() == owner
        assert CURRENT_JOB.get() == jobs[episode].key
        if episode == 8:
            ninth_seed.set()
        if episode == 10:
            eleventh_seed.set()
        return episode

    def prepare(_config: Any, episode: int, _run: str, backend: Any) -> int:
        if episode == 8:
            assert writer_started.wait(5)
        if episode == 10:
            assert render_started.wait(5)
        backend.text_query_masks(episode)
        if episode < 8:
            qc.wait(5)  # Eight QC requests coexist even with one video slot.
        return episode

    def propagate(prepared: int, backend: Any) -> int:
        return backend.propagate_mask(prepared)

    def save(prepared: int, result: int) -> bool:
        assert prepared == result
        writer_started.set()
        assert ninth_seed.wait(5), "blocked writer consumed the QC producer capacity"
        saved.append(result)
        return True

    def render(job: Any, row: dict[str, Any]) -> dict[str, Any]:
        render_started.set()
        assert eleventh_seed.wait(5), "render backlog stopped inference admission"
        return reviewed(job, row)

    runtime, _ = fake_runtime(
        backend_factory=lambda **_: SimpleNamespace(
            text_query_masks=seed, propagate_mask=seed, shutdown=lambda: None
        ),
        sam_episode_stages=SamEpisodeStages(prepare, propagate, save),
    )
    result = run_umi_pipeline(
        jobs,
        output=tmp_path,
        render=render,
        runtime=runtime,
        episode_workers=8,
        video_workers=1,
        prefetch=12,
        qwen_workers=16,
        render_buffer=1,
    )
    assert result["staged"] and result["peak_episodes"] == 8
    assert result["peak_videos"] == 1 and result["peak_ready_videos"] <= 8
    assert sorted(saved) == list(range(12))
    assert all(row["objects_ready"] for row in result["episodes"])


@pytest.mark.parametrize("failed_stage", ["prepare", "propagate", "save"])
def test_staged_failure_releases_capacity_and_keeps_other_episodes(
    tmp_path: Path, failed_stage: str
) -> None:
    def step(stage: str, episode: int) -> int:
        if stage == failed_stage and episode == 0:
            raise OSError(f"failed {stage}")
        return episode

    runtime, _ = fake_runtime(
        sam_episode_stages=SamEpisodeStages(
            lambda _config, episode, _run, _backend: step("prepare", episode),
            lambda episode, _backend: step("propagate", episode),
            lambda episode, _result: step("save", episode) >= 0,
        )
    )
    result = run_umi_pipeline(
        staged_jobs(tmp_path, 5),
        output=tmp_path,
        render=reviewed,
        runtime=runtime,
        episode_workers=2,
        video_workers=1,
    )
    assert result["status"] == "finished" and result["peak_videos"] <= 1
    assert result["episodes"][0]["error_stage"] == failed_stage
    assert not result["episodes"][0]["masks_ready"]
    assert all(row["objects_ready"] for row in result["episodes"][1:])


def test_fatal_model_error_during_qc_stops_staged_propagation(tmp_path: Path) -> None:
    propagated = []

    def seed() -> None:
        raise RuntimeError("fatal CUDA")

    def prepare(_config: Any, episode: int, _run: str, backend: Any) -> int:
        try:
            backend.text_query_masks()
        except RuntimeError:
            pass  # QC may turn a backend failure into a report.
        return episode

    runtime, _ = fake_runtime(
        backend_factory=lambda **_: SimpleNamespace(text_query_masks=seed, shutdown=lambda: None),
        fatal_cuda_error=lambda exc: "fatal CUDA" in str(exc),
        sam_episode_stages=SamEpisodeStages(
            prepare,
            lambda episode, _: propagated.append(episode),
            lambda *_: True,
        ),
    )
    result = run_umi_pipeline(
        staged_jobs(tmp_path, 6),
        output=tmp_path,
        render=reviewed,
        runtime=runtime,
        episode_workers=3,
        video_workers=1,
    )
    assert result["status"] == "failed" and not propagated
    assert len(result["episodes"]) == 6
    assert not any(row["masks_ready"] for row in result["episodes"])


@pytest.mark.parametrize("interrupted_stage", ["prepare", "propagate", "save"])
def test_staged_interrupt_preserves_completed_writes(
    tmp_path: Path, interrupted_stage: str
) -> None:
    saved = threading.Event()

    def step(stage: str, episode: int) -> int:
        if episode == 1 and stage == interrupted_stage:
            assert saved.wait(5)
            raise KeyboardInterrupt
        if episode == 0 and stage == "save":
            saved.set()
        return episode

    runtime, _ = fake_runtime(
        sam_episode_stages=SamEpisodeStages(
            lambda _config, episode, _run, _backend: step("prepare", episode),
            lambda episode, _backend: step("propagate", episode),
            lambda episode, _result: step("save", episode) >= 0,
        )
    )
    with pytest.raises(KeyboardInterrupt):
        run_umi_pipeline(
            staged_jobs(tmp_path, 4),
            output=tmp_path,
            render=reviewed,
            runtime=runtime,
            episode_workers=2,
            video_workers=2,
        )
    result = json.loads((tmp_path / "episode_results.json").read_text())
    assert result["status"] == "interrupted"
    assert len(result["episodes"]) == 4
    assert result["episodes"][0]["masks_ready"]
    assert not result["episodes"][1]["masks_ready"]
