from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from robotwin_annotation_v2.adapters import qwen_client
from robotwin_annotation_v2.application.sam_workflow import SamRuntime
from robotwin_annotation_v2.application.umi_pipeline import UmiJob, run_umi_pipeline
from robotwin_annotation_v2.config import AnnotationConfig, load_config
from robotwin_annotation_v2.domain import AnnotationMode, TargetProfile

ROOT = Path(__file__).resolve().parents[2]


def jobs_at(tmp_path: Path, count: int) -> list[UmiJob]:
    config = load_config(ROOT / "configs/pilot_move_pillbottle_pad.yaml")
    config = replace(
        config,
        output_root=tmp_path / "runs",
        dataset=replace(
            config.dataset,
            root=tmp_path / "inputs",
            manifest_data={"timeline_source": "video_window"},
        ),
    )
    return [
        UmiJob(config, index, f"group-{index}", input_signature="fixed") for index in range(count)
    ]


def fake_runtime(**overrides: Any) -> tuple[Any, list[str]]:
    lifecycle: list[str] = []
    owner: list[int] = []

    def backend_factory(**_: Any) -> Any:
        owner.append(threading.get_ident())
        lifecycle.append("load")
        return SimpleNamespace(shutdown=lambda: lifecycle.append("shutdown"))

    def execute(*_: Any) -> bool:
        assert threading.get_ident() == owner[0]
        return True

    values = {
        "qwen_client_factory": lambda **_: SimpleNamespace(health=dict),
        "backend_factory": backend_factory,
        "execution_errors": (RuntimeError,),
        "emit_gripper_result": lambda *_: True,
        "emit_sam_result": lambda _, result: result,
        "execute_gripper_episode": lambda *_: True,
        "execute_sam_episode": execute,
        "fatal_cuda_error": lambda _: False,
        "gripper_episode_complete": lambda *_: False,
        "sam_episode_complete": lambda *_: False,
        "run_qwen": lambda *_: None,
    }
    return SamRuntime(**(values | overrides)), lifecycle


def reviewed(_job: UmiJob, row: dict[str, Any]) -> dict[str, Any]:
    return {
        **row,
        "objects_ready": row["masks_ready"],
        "status": "needs_temporal_review" if row["masks_ready"] else "failed",
    }


def test_group_run_id_can_contain_several_episodes_and_reuses_one_backend(tmp_path: Path) -> None:
    jobs = [replace(job, run_id="shared-group") for job in jobs_at(tmp_path, 3)]
    runtime, lifecycle = fake_runtime()
    result = run_umi_pipeline(jobs, output=tmp_path, render=reviewed, runtime=runtime)
    assert lifecycle == ["load", "shutdown"]
    assert len(result["episodes"]) == 3
    assert len({job.directory for job in jobs}) == 3
    assert result["selected_job_ids"] == [job.key for job in jobs]


def test_cross_group_jobs_keep_their_annotation_profiles(tmp_path: Path) -> None:
    jobs = jobs_at(tmp_path, 2)
    config = replace(
        jobs[1].config,
        annotation=AnnotationConfig(
            mode=AnnotationMode.TARGET_ONLY, profile=TargetProfile.VIDEO_OBJECT
        ),
    )
    jobs[1] = replace(jobs[1], config=config)
    seen = {}

    def execute(config: Any, episode: int, *_: Any) -> bool:
        seen[episode] = config.annotation
        return True

    runtime, lifecycle = fake_runtime(execute_sam_episode=execute)
    run_umi_pipeline(jobs, output=tmp_path, render=reviewed, runtime=runtime)
    assert seen == {job.episode: job.config.annotation for job in jobs}
    assert seen[0] != seen[1] and lifecycle == ["load", "shutdown"]


def test_fast_options_reach_the_resident_backend_and_invalidate_old_identity(
    tmp_path: Path,
) -> None:
    jobs = jobs_at(tmp_path, 1)
    config = replace(
        jobs[0].config,
        sam3=replace(jobs[0].config.sam3, seed_image_only=True, tracking_batch_size=4),
    )
    jobs = [replace(jobs[0], config=config)]
    options = {}

    def factory(**kwargs: Any) -> Any:
        options.update(kwargs)
        return SimpleNamespace(shutdown=lambda: None)

    runtime, _ = fake_runtime(backend_factory=factory, execute_sam_episode=lambda *_: True)
    run_umi_pipeline(jobs, output=tmp_path, render=reviewed, runtime=runtime)
    assert options["seed_image_only"] is True and options["tracking_batch_size"] == 4
    changed = replace(
        jobs[0], config=replace(config, sam3=replace(config.sam3, tracking_batch_size=1))
    )
    with pytest.raises(ValueError, match="different input"):
        run_umi_pipeline([changed], output=tmp_path, render=reviewed, runtime=runtime, resume=True)


def test_prefetch_is_bounded_and_return_drains_background_reviews(tmp_path: Path) -> None:
    jobs = jobs_at(tmp_path, 5)
    planned = [threading.Event() for _ in jobs]
    sam_started = threading.Event()
    allow_sam = threading.Event()
    render_started = threading.Event()
    allow_render = threading.Event()

    def plan(_config: Any, episode: int, _run_id: str) -> None:
        planned[episode].set()
        if episode == 1:
            assert sam_started.wait(5)

    def execute(_config: Any, episode: int, *_: Any) -> bool:
        if episode == 0:
            sam_started.set()
            assert allow_sam.wait(5)
        return True

    def render(job: UmiJob, row: dict[str, Any]) -> dict[str, Any]:
        render_started.set()
        assert allow_render.wait(5)
        return reviewed(job, row)

    runtime, lifecycle = fake_runtime(run_qwen=plan, execute_sam_episode=execute)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            run_umi_pipeline,
            jobs,
            output=tmp_path,
            render=render,
            runtime=runtime,
            prefetch=2,
            render_buffer=2,
        )
        try:
            assert sam_started.wait(5)
            assert planned[2].wait(5)
            assert not planned[3].is_set()  # active episode + two outstanding plans only
            allow_sam.set()
            assert render_started.wait(5)
            assert not future.done()  # masks alone do not finish a batch
        finally:
            allow_sam.set()
            allow_render.set()
        result = future.result(timeout=5)
    assert lifecycle == ["load", "shutdown"]
    assert result["peak_plans"] == 2 and result["peak_reviews"] <= 2
    assert all(row["objects_ready"] for row in result["episodes"])


@pytest.mark.parametrize("asynchronous", [False, True])
def test_failures_keep_every_attempt_and_do_not_promote_masks_to_reviewed(
    tmp_path: Path, asynchronous: bool
) -> None:
    executed = []

    def plan(_config: Any, episode: int, _run_id: str) -> None:
        if episode == 1:
            raise ValueError("bad semantic response")

    def execute(_config: Any, episode: int, *_: Any) -> bool:
        executed.append(episode)
        return episode != 2

    def render(job: UmiJob, row: dict[str, Any]) -> dict[str, Any]:
        if job.episode == 3:
            raise OSError("encoder failed")
        return reviewed(job, row)

    runtime, lifecycle = fake_runtime(run_qwen=plan, execute_sam_episode=execute)
    result = run_umi_pipeline(
        jobs_at(tmp_path, 4),
        output=tmp_path,
        render=render,
        runtime=runtime,
        asynchronous=asynchronous,
    )
    rows = result["episodes"]
    assert set(executed) == {0, 2, 3}
    assert rows[1]["process_status"] == "planning_failed"
    assert rows[2]["process_status"] == "sam_incomplete"
    assert rows[3]["status"] == "render_failed" and rows[3]["masks_ready"]
    assert not rows[3]["objects_ready"]
    assert lifecycle == ["load", "shutdown"]
    assert json.loads((tmp_path / "episode_results.json").read_text())["status"] == "finished"


def test_explicit_resume_checks_identity_and_only_renders_completed_masks(tmp_path: Path) -> None:
    jobs = jobs_at(tmp_path, 1)
    runtime, _ = fake_runtime()
    run_umi_pipeline(jobs, output=tmp_path, render=reviewed, runtime=runtime)
    with pytest.raises(ValueError, match="explicit resume"):
        run_umi_pipeline(jobs, output=tmp_path, render=reviewed, runtime=runtime)
    with pytest.raises(ValueError, match="different input"):
        run_umi_pipeline(
            [replace(jobs[0], input_signature="changed")],
            output=tmp_path,
            render=reviewed,
            runtime=runtime,
            resume=True,
        )
    runtime, lifecycle = fake_runtime(
        sam_episode_complete=lambda *_: True,
        run_qwen=lambda *_: pytest.fail("valid masks must not request Qwen again"),
        execute_sam_episode=lambda *_: pytest.fail("valid masks must not invoke SAM again"),
    )
    result = run_umi_pipeline(jobs, output=tmp_path, render=reviewed, runtime=runtime, resume=True)
    assert result["episodes"][0]["process_status"] == "skipped_complete"
    assert lifecycle == []


@pytest.mark.parametrize("failure", [RuntimeError("health unavailable"), KeyboardInterrupt()])
def test_startup_failure_restores_gate_and_persists_terminal_state(
    tmp_path: Path, failure: BaseException
) -> None:
    previous = threading.BoundedSemaphore(3)
    original = qwen_client.install_qwen_request_gate(previous)

    def health() -> None:
        raise failure

    runtime, lifecycle = fake_runtime(
        qwen_client_factory=lambda **_: SimpleNamespace(health=health)
    )
    try:
        with pytest.raises(type(failure)):
            run_umi_pipeline(
                jobs_at(tmp_path, 3), output=tmp_path, render=reviewed, runtime=runtime
            )
        assert qwen_client._REQUEST_GATE is previous
    finally:
        qwen_client.install_qwen_request_gate(original)
    saved = json.loads((tmp_path / "episode_results.json").read_text())
    assert saved["status"] == (
        "interrupted" if isinstance(failure, KeyboardInterrupt) else "failed"
    )
    assert len(saved["episodes"]) == 3
    expected = (
        "not_run_after_interrupt"
        if isinstance(failure, KeyboardInterrupt)
        else "not_run_after_startup_failure"
    )
    for row in saved["episodes"]:
        assert row["process_status"] == expected
        assert not row["masks_ready"] and not row["objects_ready"]
        assert row["artifact_dir"] and row["camera"]
    assert lifecycle == []


def test_interrupt_during_sam_preserves_complete_manifest_and_previous_outputs(
    tmp_path: Path,
) -> None:
    executed = []

    def execute(_config: Any, episode: int, *_: Any) -> bool:
        executed.append(episode)
        if episode == 1:
            raise KeyboardInterrupt
        return True

    previous = threading.BoundedSemaphore(3)
    original = qwen_client.install_qwen_request_gate(previous)
    runtime, lifecycle = fake_runtime(execute_sam_episode=execute)
    try:
        with pytest.raises(KeyboardInterrupt):
            run_umi_pipeline(
                jobs_at(tmp_path, 3),
                output=tmp_path,
                render=reviewed,
                runtime=runtime,
                asynchronous=False,
            )
        assert qwen_client._REQUEST_GATE is previous
    finally:
        qwen_client.install_qwen_request_gate(original)
    saved = json.loads((tmp_path / "episode_results.json").read_text())
    assert saved["status"] == "interrupted"
    assert [row["process_status"] for row in saved["episodes"]] == [
        "completed",
        "interrupted_during_sam",
        "not_run_after_interrupt",
    ]
    assert saved["episodes"][0]["objects_ready"]
    assert all(row["artifact_dir"] and row["camera"] for row in saved["episodes"])
    assert not any(row["masks_ready"] for row in saved["episodes"][1:])
    assert executed == [0, 1] and lifecycle == ["load", "shutdown"]


def test_resume_subset_does_not_overwrite_complete_batch_ledger(tmp_path: Path) -> None:
    jobs = jobs_at(tmp_path, 2)
    runtime, _ = fake_runtime()
    run_umi_pipeline(jobs, output=tmp_path, render=reviewed, runtime=runtime)
    ledger = tmp_path / "episode_results.json"
    previous = ledger.read_bytes()
    with pytest.raises(ValueError, match="complete batch selection"):
        run_umi_pipeline(jobs[:1], output=tmp_path, render=reviewed, runtime=runtime, resume=True)
    assert ledger.read_bytes() == previous


def test_resume_partial_running_ledger_keeps_complete_selection(tmp_path: Path) -> None:
    jobs = [
        replace(job, run_id=f"{job.ref.task}-{job.episode:06d}-{job.ref.camera}")
        for job in jobs_at(tmp_path, 2)
    ]
    runtime, _ = fake_runtime()
    run_umi_pipeline(jobs, output=tmp_path, render=reviewed, runtime=runtime)
    ledger = tmp_path / "episode_results.json"
    saved = json.loads(ledger.read_text())
    saved.update(status="running", episodes=saved["episodes"][:1])
    ledger.write_text(json.dumps(saved))
    planned = []
    runtime, _ = fake_runtime(
        sam_episode_complete=lambda _config, _store, _run_id, ref: ref.episode_index == 0,
        run_qwen=lambda _config, episode, _run_id: planned.append(episode),
    )
    result = run_umi_pipeline(jobs, output=tmp_path, render=reviewed, runtime=runtime, resume=True)
    assert planned == [1]
    assert result["selected_job_ids"] == [job.key for job in jobs]
    assert all(row["objects_ready"] for row in result["episodes"])


def test_planners_leave_a_request_token_for_current_episode_qc(tmp_path: Path) -> None:
    release_planners = threading.Event()
    started = threading.Barrier(3)

    def plan(_config: Any, episode: int, _run_id: str) -> None:
        gate = qwen_client._REQUEST_GATE
        assert gate is not None and gate.acquire(timeout=2)
        try:
            if episode < 2:
                started.wait(timeout=3)
                assert release_planners.wait(3)
        finally:
            gate.release()

    runtime, _ = fake_runtime(run_qwen=plan)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            run_umi_pipeline,
            jobs_at(tmp_path, 3),
            output=tmp_path,
            render=reviewed,
            runtime=runtime,
            qwen_workers=3,
            prefetch=4,
        )
        try:
            started.wait(timeout=3)
            gate = qwen_client._REQUEST_GATE
            assert gate is not None and gate.acquire(blocking=False)
            gate.release()
        finally:
            release_planners.set()
        future.result(timeout=5)


def test_fatal_cuda_stops_sam_but_preserves_other_attempt_statuses(tmp_path: Path) -> None:
    executed = []

    def execute(_config: Any, episode: int, *_: Any) -> bool:
        executed.append(episode)
        raise RuntimeError("fatal cuda")

    runtime, lifecycle = fake_runtime(execute_sam_episode=execute, fatal_cuda_error=lambda _: True)
    result = run_umi_pipeline(
        jobs_at(tmp_path, 3), output=tmp_path, render=reviewed, runtime=runtime, asynchronous=False
    )
    assert executed == [0]
    assert result["status"] == "failed"
    assert [row["process_status"] for row in result["episodes"]] == [
        "sam_failed",
        "not_run_after_fatal_cuda",
        "not_run_after_fatal_cuda",
    ]
    assert lifecycle == ["load", "shutdown"]


def test_same_episode_number_in_different_tasks_keeps_distinct_outputs(tmp_path: Path) -> None:
    job = jobs_at(tmp_path, 1)[0]
    other = replace(
        job,
        run_id="other-group-0",
        config=replace(job.config, dataset=replace(job.config.dataset, task="other-task")),
    )
    runtime, _ = fake_runtime()
    result = run_umi_pipeline([job, other], output=tmp_path, render=reviewed, runtime=runtime)
    assert len({row["artifact_dir"] for row in result["episodes"]}) == 2


def test_shutdown_failure_does_not_persist_a_successful_batch_status(tmp_path: Path) -> None:
    def shutdown() -> None:
        raise RuntimeError("shutdown failed")

    runtime, _ = fake_runtime(
        backend_factory=lambda **_: SimpleNamespace(shutdown=shutdown),
        execute_sam_episode=lambda *_: True,
    )
    with pytest.raises(RuntimeError, match="shutdown failed"):
        run_umi_pipeline(jobs_at(tmp_path, 1), output=tmp_path, render=reviewed, runtime=runtime)
    saved = json.loads((tmp_path / "episode_results.json").read_text())
    assert saved["status"] == "failed"
    assert saved["episodes"][0]["masks_ready"]
