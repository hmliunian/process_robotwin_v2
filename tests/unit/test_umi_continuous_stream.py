from __future__ import annotations

import json
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest

from robotwin_annotation_v2.application.umi_pipeline import run_umi_pipeline
from tests.unit.test_umi_pipeline import fake_runtime, jobs_at, reviewed


def test_next_batch_propagates_while_previous_render_blocks_and_model_loads_once(tmp_path):
    first = replace(jobs_at(tmp_path / "a", 1)[0], namespace="batch-a")
    second = replace(jobs_at(tmp_path / "b", 1)[0], namespace="batch-b")
    render_started = threading.Event()
    next_propagated = threading.Event()
    lifecycle = []
    owner = threading.get_ident()

    def propagate(root):
        assert threading.get_ident() == owner
        if root == second.config.dataset.root:
            next_propagated.set()
        return True

    def factory(**_):
        lifecycle.append("load")
        return SimpleNamespace(
            propagate_mask=propagate, shutdown=lambda: lifecycle.append("shutdown")
        )

    def execute(config, _ep, _run, backend):
        return backend.propagate_mask(config.dataset.root)

    runtime, _ = fake_runtime(backend_factory=factory, execute_sam_episode=execute)
    delivered = False

    def poll():
        nonlocal delivered
        if delivered:
            raise StopIteration
        if not render_started.is_set():
            return None
        delivered = True
        return second

    def render(job, row):
        if job.namespace == "batch-a":
            render_started.set()
            assert next_propagated.wait(5), "next batch waited for old render"
        return reviewed(job, row)

    terminal = []

    def ack(job, row):
        assert json.loads((job.directory / "pipeline_result.json").read_text()) == row
        terminal.append(job.key)

    report = run_umi_pipeline(
        [first],
        output=tmp_path / "worker",
        runtime=runtime,
        render=render,
        episode_workers=2,
        video_workers=1,
        prefetch=2,
        poll_job=poll,
        on_terminal=ack,
    )
    assert lifecycle == ["load", "shutdown"]
    assert set(terminal) == {first.key, second.key}
    assert report["completed_jobs"] == report["completed_outputs"] == 2
    assert report["episodes"] == []
    assert (first.directory / "timings.json").exists()


def test_empty_input_queue_keeps_backend_alive_until_explicit_end(tmp_path):
    first, second = jobs_at(tmp_path, 2)
    runtime, lifecycle = fake_runtime(execute_sam_episode=lambda *_: True)
    completed = []
    idle_polls = 0
    sent = False

    def poll():
        nonlocal idle_polls, sent
        if sent:
            raise StopIteration
        if not completed:
            return None
        assert lifecycle == ["load"]
        idle_polls += 1
        if idle_polls < 4:
            return None
        sent = True
        return second

    result = run_umi_pipeline(
        [first],
        output=tmp_path / "worker",
        runtime=runtime,
        render=reviewed,
        episode_workers=2,
        poll_job=poll,
        on_terminal=lambda job, row: completed.append(job.key),
    )
    assert idle_polls == 4 and len(completed) == 2
    assert lifecycle == ["load", "shutdown"] and result["completed_outputs"] == 2


def test_stream_rejects_configuration_change_and_closes_backend(tmp_path):
    first, second = jobs_at(tmp_path, 2)
    second = replace(
        second,
        config=replace(second.config, sam3=replace(second.config.sam3, tracking_batch_size=4)),
    )
    runtime, lifecycle = fake_runtime(execute_sam_episode=lambda *_: True)
    with pytest.raises(ValueError, match="SAM configuration"):
        run_umi_pipeline(
            [first],
            output=tmp_path / "worker",
            runtime=runtime,
            render=reviewed,
            episode_workers=2,
            poll_job=lambda: second,
            on_terminal=lambda *_: None,
        )
    assert lifecycle == ["load", "shutdown"]
    assert not (second.directory / "job.json").exists()


def test_completed_jobs_leave_bounded_live_ledger_and_timings(tmp_path):
    jobs = jobs_at(tmp_path, 40)
    runtime, _ = fake_runtime(execute_sam_episode=lambda *_: True)
    remaining = iter(jobs[1:])
    active = []

    def terminal(job, row):
        snapshot = json.loads((tmp_path / "worker/episode_results.json").read_text())
        active.append(len(snapshot["selected_job_ids"]))

    result = run_umi_pipeline(
        jobs[:1],
        output=tmp_path / "worker",
        runtime=runtime,
        render=reviewed,
        episode_workers=2,
        video_workers=1,
        prefetch=2,
        render_buffer=1,
        poll_job=lambda: next(remaining),
        on_terminal=terminal,
    )
    assert max(active) <= 6
    assert result["completed_jobs"] == 40 and result["episodes"] == []
    assert all((j.directory / "pipeline_result.json").exists() for j in jobs)
    assert all(
        e["job"] == "batch"
        for e in json.loads((tmp_path / "worker/timings.json").read_text())["events"]
    )
