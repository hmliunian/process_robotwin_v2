from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from robotwin_annotation_v2.application.sam_task_queue import SamQueueStopped, SamTaskQueue
from robotwin_annotation_v2.application.umi_pipeline import (
    CURRENT_JOB,
    PipelineEvents,
    run_umi_pipeline,
)
from tests.unit.test_umi_pipeline import fake_runtime, jobs_at, reviewed


def test_qc_and_save_waits_release_sam_owner_for_other_episodes(tmp_path: Path) -> None:
    jobs = jobs_at(tmp_path, 3)
    owner = threading.get_ident()
    other_propagated = threading.Event()
    third_seeded = threading.Event()
    calls: list[tuple[str, int]] = []
    lifecycle = []

    def seed(episode: int) -> int:
        assert threading.get_ident() == owner
        assert CURRENT_JOB.get() == jobs[episode].key
        calls.append(("seed", episode))
        if episode == 2:
            third_seeded.set()
        return episode

    def propagate(episode: int, *, selected: int) -> int:
        assert threading.get_ident() == owner
        assert CURRENT_JOB.get() == jobs[episode].key and selected == episode
        calls.append(("propagate", episode))
        if episode == 1:
            other_propagated.set()
        return episode

    def shutdown() -> None:
        assert threading.get_ident() == owner
        lifecycle.append("shutdown")

    def factory(**_: Any) -> Any:
        assert threading.get_ident() == owner
        lifecycle.append("load")
        return SimpleNamespace(text_query_masks=seed, propagate_mask=propagate, shutdown=shutdown)

    def execute(_config: Any, episode: int, _run_id: str, backend: Any) -> bool:
        assert threading.get_ident() != owner
        selected = backend.text_query_masks(episode)
        if episode == 0:
            assert other_propagated.wait(5), "waiting for QC must not hold the model owner"
        backend.propagate_mask(episode, selected=selected)
        if episode == 0:
            assert third_seeded.wait(5), "saving must not hold the model owner"
        return True

    runtime, _ = fake_runtime(backend_factory=factory, execute_sam_episode=execute)
    events = PipelineEvents()
    result = run_umi_pipeline(
        jobs, output=tmp_path, render=reviewed, runtime=runtime, events=events, episode_workers=2
    )
    assert lifecycle == ["load", "shutdown"]
    assert calls.index(("propagate", 1)) < calls.index(("propagate", 0))
    assert result["status"] == "finished" and all(
        row["objects_ready"] for row in result["episodes"]
    )
    assert result["sam_queue"]["calls"] == 6
    assert 1 <= result["sam_queue"]["peak_pending"] <= 2
    assert result["peak_episodes"] == 2
    model_events = [event for event in events.snapshot() if event["stage"] == "sam_propagate_mask"]
    assert {event["job"] for event in model_events} == {job.key for job in jobs}


def test_full_render_queue_does_not_block_admitted_sam_calls(tmp_path: Path) -> None:
    rendering = threading.Event()
    third_seeded = threading.Event()

    def seed(episode: int) -> None:
        if episode == 2:
            third_seeded.set()

    def execute(_config: Any, episode: int, _run_id: str, backend: Any) -> bool:
        if episode == 1:
            assert rendering.wait(5)
        backend.text_query_masks(episode)
        return True

    def render(job: Any, row: dict[str, Any]) -> dict[str, Any]:
        rendering.set()
        assert third_seeded.wait(5), "render backpressure blocked the model queue"
        return reviewed(job, row)

    runtime, _ = fake_runtime(
        backend_factory=lambda **_: SimpleNamespace(text_query_masks=seed, shutdown=lambda: None),
        execute_sam_episode=execute,
    )
    result = run_umi_pipeline(
        jobs_at(tmp_path, 6),
        output=tmp_path,
        render=render,
        runtime=runtime,
        episode_workers=2,
        render_buffer=1,
    )
    assert result["peak_episodes"] <= 2 and result["peak_reviews"] == 1
    assert all(row["objects_ready"] for row in result["episodes"])


@pytest.mark.parametrize(
    "failure", [RuntimeError("device-side assert triggered"), KeyboardInterrupt()]
)
def test_owner_failure_releases_waiters_even_when_episode_catches_cuda(
    tmp_path: Path, failure: BaseException
) -> None:
    starting = threading.Barrier(2)
    calls = []
    closed = []

    def seed(episode: int) -> None:
        calls.append(episode)
        raise failure

    def execute(_config: Any, episode: int, _run_id: str, backend: Any) -> bool:
        starting.wait(timeout=5)
        try:
            backend.text_query_masks(episode)
        except RuntimeError:
            # Production QC can translate backend exceptions into failed masks.
            return False
        return True

    runtime, _ = fake_runtime(
        backend_factory=lambda **_: SimpleNamespace(
            text_query_masks=seed, shutdown=lambda: closed.append(True)
        ),
        execute_sam_episode=execute,
        fatal_cuda_error=lambda exc: "device-side assert" in str(exc),
    )
    if isinstance(failure, KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            run_umi_pipeline(
                jobs_at(tmp_path, 5),
                output=tmp_path,
                render=reviewed,
                runtime=runtime,
                episode_workers=2,
            )
    else:
        run_umi_pipeline(
            jobs_at(tmp_path, 5),
            output=tmp_path,
            render=reviewed,
            runtime=runtime,
            episode_workers=2,
        )
    result = json.loads((tmp_path / "episode_results.json").read_text())
    assert result["status"] == (
        "interrupted" if isinstance(failure, KeyboardInterrupt) else "failed"
    )
    assert len(calls) == 1 and closed == [True]
    assert len(result["episodes"]) == 5
    assert not any(row["masks_ready"] for row in result["episodes"])
    assert (
        sum(row["process_status"].startswith("not_run_after_") for row in result["episodes"]) == 3
    )


def test_episode_thread_interrupt_closes_pending_model_requests(tmp_path: Path) -> None:
    starting = threading.Barrier(2)
    closed = []

    def execute(_config: Any, episode: int, _run_id: str, backend: Any) -> bool:
        starting.wait(timeout=5)
        if episode == 0:
            raise KeyboardInterrupt
        backend.text_query_masks(episode)
        return True

    runtime, _ = fake_runtime(
        backend_factory=lambda **_: SimpleNamespace(
            text_query_masks=lambda _: None, shutdown=lambda: closed.append(True)
        ),
        execute_sam_episode=execute,
    )
    with pytest.raises(KeyboardInterrupt):
        run_umi_pipeline(
            jobs_at(tmp_path, 3),
            output=tmp_path,
            render=reviewed,
            runtime=runtime,
            episode_workers=2,
        )
    result = json.loads((tmp_path / "episode_results.json").read_text())
    assert result["status"] == "interrupted" and closed == [True]
    assert result["episodes"][0]["status"] == "interrupted_during_sam"
    assert len(result["episodes"]) == 3


def test_interrupt_retains_masks_finished_by_another_episode(tmp_path: Path) -> None:
    saved = threading.Event()

    def seed(episode: int) -> None:
        if episode == 1:
            raise KeyboardInterrupt

    def execute(_config: Any, episode: int, _run_id: str, backend: Any) -> bool:
        if episode == 1:
            assert saved.wait(5)
        backend.text_query_masks(episode)
        saved.set()
        return True

    runtime, _ = fake_runtime(
        backend_factory=lambda **_: SimpleNamespace(text_query_masks=seed, shutdown=lambda: None),
        execute_sam_episode=execute,
    )
    with pytest.raises(KeyboardInterrupt):
        run_umi_pipeline(
            jobs_at(tmp_path, 2),
            output=tmp_path,
            render=reviewed,
            runtime=runtime,
            episode_workers=2,
        )
    result = json.loads((tmp_path / "episode_results.json").read_text())
    assert result["status"] == "interrupted"
    assert result["episodes"][0]["masks_ready"]
    assert result["episodes"][0]["process_status"] == "completed"
    assert not result["episodes"][1]["masks_ready"]


def test_nonfatal_model_error_isolated_and_completed_masks_resume(tmp_path: Path) -> None:
    def seed(episode: int) -> None:
        if episode == 1:
            raise ValueError("bad candidate")

    def execute(_config: Any, episode: int, _run_id: str, backend: Any) -> bool:
        backend.text_query_masks(episode)
        return True

    runtime, _ = fake_runtime(
        backend_factory=lambda **_: SimpleNamespace(text_query_masks=seed, shutdown=lambda: None),
        execute_sam_episode=execute,
    )
    jobs = jobs_at(tmp_path, 3)
    result = run_umi_pipeline(
        jobs, output=tmp_path, render=reviewed, runtime=runtime, episode_workers=3
    )
    assert result["status"] == "finished"
    assert [row["objects_ready"] for row in result["episodes"]] == [True, False, True]
    assert result["episodes"][1]["process_status"] == "sam_failed"
    retried = []
    runtime, _ = fake_runtime(
        execute_sam_episode=lambda _c, episode, *_: retried.append(episode) or True,
        sam_episode_complete=lambda _c, _s, _r, ref: ref.episode_index != 1,
    )
    result = run_umi_pipeline(
        jobs, output=tmp_path, render=reviewed, runtime=runtime, episode_workers=3, resume=True
    )
    assert retried == [1] and all(row["objects_ready"] for row in result["episodes"])
    runtime, lifecycle = fake_runtime(sam_episode_complete=lambda *_: True)
    result = run_umi_pipeline(
        jobs, output=tmp_path, render=reviewed, runtime=runtime, episode_workers=3, resume=True
    )
    assert lifecycle == [] and result["sam_queue"]["calls"] == 0


def test_queue_close_rejects_late_calls_and_only_owner_can_service() -> None:
    queue = SamTaskQueue(
        None, capacity=2, fatal_cuda_error=lambda _: False, measure=PipelineEvents().measure
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        with pytest.raises(RuntimeError, match="only the SAM owner"):
            pool.submit(queue.run_one).result(timeout=5)
        queue.close("test shutdown")
        with pytest.raises(SamQueueStopped, match="test shutdown"):
            pool.submit(queue.text_query_masks).result(timeout=5)
    with pytest.raises(RuntimeError, match="episode workers"):
        queue.text_query_masks()
