from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from robotwin_annotation_v2.application import parallel_sam_runtime as runtime_module
from robotwin_annotation_v2.application.parallel_sam_protocol import (
    SamWorkerCommand,
    SamWorkerFailure,
    SamWorkerOutcome,
    SamWorkerReady,
)
from robotwin_annotation_v2.application.sam_workflow import SamEpisodeResult
from robotwin_annotation_v2.config import ParallelConfig, PipelineConfig, load_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _parallel_config(tmp_path: Path, gpus: tuple[int, ...]) -> PipelineConfig:
    base = load_config(PROJECT_ROOT / "configs/process_qwen38_api.yaml")
    return replace(
        base,
        output_root=tmp_path,
        parallel=ParallelConfig(sam_worker_gpus=gpus, qwen_max_in_flight=2),
    )


def _record_worker_start(config: PipelineConfig, worker_id: int, launch_id: int) -> None:
    marker = config.output_root / f"worker-{worker_id}-launch-{launch_id}.started"
    marker.write_text("started", encoding="utf-8")


def _wait_for_marker(path: Path, *, timeout_seconds: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while not path.is_file():
        if time.monotonic() >= deadline:
            raise RuntimeError(f"timed out waiting for test marker: {path.name}")
        time.sleep(0.01)


def _scripted_worker_entry(
    connection: Any,
    *,
    worker_id: int,
    gpu_id: int,
    launch_id: int,
    config: PipelineConfig,
    run_id: str,
    source_only: bool,
    qwen_request_gate: Any,
) -> None:
    del source_only, qwen_request_gate
    _record_worker_start(config, worker_id, launch_id)
    if run_id == "startup-hang":
        time.sleep(30)
        return
    if run_id == "startup-fail":
        connection.send(
            SamWorkerFailure(
                worker_id=worker_id,
                gpu_id=gpu_id,
                error_type="BackendLoadError",
                error="cannot load backend",
                traceback="",
                retryable=True,
            )
        )
        connection.close()
        return
    relaunch_marker = config.output_root / "gpu-0-episode-1-dispatched"
    if run_id == "relaunch-fail-isolation" and gpu_id == 1:
        _wait_for_marker(relaunch_marker)
    connection.send(SamWorkerReady(worker_id=worker_id, gpu_id=gpu_id))
    while True:
        command = connection.recv()
        if command.kind == "shutdown":
            connection.close()
            return
        assert command.episode_id is not None
        assert command.attempt is not None
        episode_id = command.episode_id
        if run_id == "episode-hang":
            time.sleep(30)
            return
        if (
            run_id == "relaunch-fail-isolation"
            and gpu_id == 0
            and episode_id == 1
            and command.attempt == 1
        ):
            relaunch_marker.write_text("dispatched", encoding="utf-8")
            os._exit(18)
        if run_id == "hard-exit" and command.attempt == 1:
            os._exit(17)
        if run_id == "fatal-isolation" and episode_id == 1:
            connection.send(
                SamWorkerOutcome(
                    worker_id,
                    gpu_id,
                    episode_id,
                    command.attempt,
                    "failed",
                    "CUDA device assert",
                    fatal_cuda=True,
                )
            )
            connection.close()
            return
        if run_id == "ordinary-failure" and episode_id == 1:
            connection.send(
                SamWorkerOutcome(
                    worker_id,
                    gpu_id,
                    episode_id,
                    command.attempt,
                    "failed",
                    "bad mask",
                )
            )
            continue
        outcome = SamWorkerOutcome(
            worker_id,
            gpu_id,
            episode_id,
            command.attempt,
            "completed",
        )
        if run_id == "out-of-order" and episode_id == 2:
            connection.send(outcome)
            (config.output_root / "episode-2-sent").write_text("sent", encoding="utf-8")
            continue
        if run_id == "out-of-order" and episode_id == 1:
            _wait_for_marker(config.output_root / "episode-2-sent")
        connection.send(outcome)


class _RecordingReporter:
    def __init__(self) -> None:
        self.started: list[int] = []
        self.finished: list[tuple[int, str]] = []

    def episode_started(self, episode_id: int, **_kwargs: Any) -> None:
        self.started.append(episode_id)

    def episode_finished(self, episode_id: int, *, status: str, **_kwargs: Any) -> None:
        self.finished.append((episode_id, status))


def _run_scripted(
    tmp_path: Path,
    *,
    run_id: str,
    episode_ids: tuple[int, ...],
    gpus: tuple[int, ...],
) -> tuple[tuple[SamWorkerOutcome, ...], list[SamEpisodeResult], _RecordingReporter]:
    callbacks: list[SamEpisodeResult] = []
    reporter = _RecordingReporter()
    outcomes = runtime_module.run_parallel_sam_episodes(
        _parallel_config(tmp_path, gpus),
        run_id=run_id,
        episode_ids=episode_ids,
        source_only=True,
        qwen_max_in_flight=2,
        outcome_callback=callbacks.append,
        reporter=reporter,
        report_lifecycle=True,
        worker_entry=_scripted_worker_entry,
    )
    return outcomes, callbacks, reporter


def test_dynamic_dispatch_returns_input_order_and_terminalizes_once(tmp_path: Path) -> None:
    outcomes, callbacks, reporter = _run_scripted(
        tmp_path,
        run_id="out-of-order",
        episode_ids=(1, 2, 3, 4),
        gpus=(2, 5),
    )

    assert [result.episode_id for result in callbacks] != [1, 2, 3, 4]
    assert [outcome.episode_id for outcome in outcomes] == [1, 2, 3, 4]
    assert sorted(result.episode_id for result in callbacks) == [1, 2, 3, 4]
    assert sorted(reporter.started) == [1, 2, 3, 4]
    assert sorted(reporter.finished) == [
        (1, "completed"),
        (2, "completed"),
        (3, "completed"),
        (4, "completed"),
    ]


def test_starts_every_explicit_gpu_even_with_fewer_episodes(tmp_path: Path) -> None:
    outcomes, _, _ = _run_scripted(
        tmp_path,
        run_id="all-explicit-gpus",
        episode_ids=(7,),
        gpus=(1, 3, 6),
    )

    assert outcomes[0].status == "completed"
    assert sorted(path.name for path in tmp_path.glob("*.started")) == [
        "worker-0-launch-1.started",
        "worker-1-launch-1.started",
        "worker-2-launch-1.started",
    ]


def test_unexpected_worker_exit_reloads_and_retries_episode(tmp_path: Path) -> None:
    outcomes, callbacks, reporter = _run_scripted(
        tmp_path,
        run_id="hard-exit",
        episode_ids=(9,),
        gpus=(4,),
    )

    assert [(item.status, item.attempt) for item in outcomes] == [("completed", 2)]
    assert [result.episode_id for result in callbacks] == [9]
    assert reporter.started == [9]
    assert reporter.finished == [(9, "completed")]
    assert sorted(path.name for path in tmp_path.glob("*.started")) == [
        "worker-0-launch-1.started",
        "worker-0-launch-2.started",
    ]


def test_all_workers_unhealthy_marks_every_episode_not_run(tmp_path: Path) -> None:
    outcomes, callbacks, reporter = _run_scripted(
        tmp_path,
        run_id="startup-fail",
        episode_ids=(4, 8),
        gpus=(0,),
    )

    assert [item.status for item in outcomes] == [
        "not_run_no_healthy_sam_worker",
        "not_run_no_healthy_sam_worker",
    ]
    assert [item.worker_id for item in outcomes] == [None, None]
    assert [item.episode_id for item in callbacks] == [4, 8]
    assert reporter.started == []
    assert reporter.finished == [
        (4, "not_run_no_healthy_sam_worker"),
        (8, "not_run_no_healthy_sam_worker"),
    ]


def test_startup_watchdog_retires_worker_after_bounded_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_relaunch = runtime_module._relaunch_worker
    relaunch_count = 0

    def record_relaunch(*args: Any, **kwargs: Any) -> None:
        nonlocal relaunch_count
        relaunch_count += 1
        original_relaunch(*args, **kwargs)

    monkeypatch.setattr(runtime_module, "_relaunch_worker", record_relaunch)
    monkeypatch.setattr(runtime_module, "WORKER_STARTUP_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(runtime_module, "WORKER_FAILURE_EXIT_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(runtime_module, "WORKER_KILL_TIMEOUT_SECONDS", 0.05)

    outcomes, callbacks, _ = _run_scripted(
        tmp_path,
        run_id="startup-hang",
        episode_ids=(5,),
        gpus=(0,),
    )

    assert outcomes[0].status == "not_run_no_healthy_sam_worker"
    assert "WorkerTimeoutError" in str(outcomes[0].error)
    assert [result.episode_id for result in callbacks] == [5]
    assert relaunch_count == 1


def test_episode_watchdog_retries_then_terminalizes_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime_module, "EPISODE_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(runtime_module, "WORKER_FAILURE_EXIT_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(runtime_module, "WORKER_KILL_TIMEOUT_SECONDS", 0.05)

    outcomes, callbacks, reporter = _run_scripted(
        tmp_path,
        run_id="episode-hang",
        episode_ids=(6,),
        gpus=(0,),
    )

    assert [(item.status, item.attempt) for item in outcomes] == [("failed", 2)]
    assert "WorkerTimeoutError" in str(outcomes[0].error)
    assert [result.episode_id for result in callbacks] == [6]
    assert reporter.started == [6]
    assert reporter.finished == [(6, "failed")]


def test_relaunch_start_failure_retires_only_failed_gpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_relaunch = runtime_module._relaunch_worker

    def fail_gpu_zero(state: Any, *args: Any, **kwargs: Any) -> None:
        if state.gpu_id == 0:
            raise OSError("cannot relaunch gpu 0")
        original_relaunch(state, *args, **kwargs)

    monkeypatch.setattr(runtime_module, "_relaunch_worker", fail_gpu_zero)

    outcomes, callbacks, reporter = _run_scripted(
        tmp_path,
        run_id="relaunch-fail-isolation",
        episode_ids=(1, 2, 3),
        gpus=(0, 1),
    )

    assert [(item.episode_id, item.status, item.attempt) for item in outcomes] == [
        (1, "completed", 2),
        (2, "completed", 1),
        (3, "completed", 1),
    ]
    assert sorted(result.episode_id for result in callbacks) == [1, 2, 3]
    assert sorted(reporter.started) == [1, 2, 3]
    assert sorted(reporter.finished) == [
        (1, "completed"),
        (2, "completed"),
        (3, "completed"),
    ]


def test_fatal_gpu_is_retired_while_other_gpu_continues(tmp_path: Path) -> None:
    outcomes, callbacks, _ = _run_scripted(
        tmp_path,
        run_id="fatal-isolation",
        episode_ids=(1, 2, 3),
        gpus=(0, 1),
    )

    assert [(item.episode_id, item.status) for item in outcomes] == [
        (1, "failed"),
        (2, "completed"),
        (3, "completed"),
    ]
    assert outcomes[0].fatal_cuda is True
    assert sorted(result.episode_id for result in callbacks) == [1, 2, 3]


def test_ordinary_episode_failure_keeps_worker_available(tmp_path: Path) -> None:
    outcomes, _, _ = _run_scripted(
        tmp_path,
        run_id="ordinary-failure",
        episode_ids=(1, 2),
        gpus=(7,),
    )

    assert [(item.status, item.attempt) for item in outcomes] == [
        ("failed", 1),
        ("completed", 1),
    ]
    assert next(iter(tmp_path.glob("*.started"))).name == "worker-0-launch-1.started"


class _DirectConnection:
    def __init__(self, commands: tuple[SamWorkerCommand, ...]) -> None:
        self.commands = deque(commands)
        self.sent: list[object] = []
        self.closed = False

    def recv(self) -> SamWorkerCommand:
        return self.commands.popleft()

    def send(self, value: object) -> None:
        self.sent.append(value)

    def close(self) -> None:
        self.closed = True


def test_worker_loads_one_gpu_backend_once_and_shuts_it_down(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    backend = SimpleNamespace(shutdown=lambda: calls.append("shutdown"))
    fake_runtime = SimpleNamespace(
        backend_factory=lambda **kwargs: calls.append(kwargs["gpus"]) or backend,
        run_qwen=lambda *_args: calls.append("qwen"),
        execute_sam_episode=lambda *_args: calls.append("sam") or object(),
        emit_sam_result=lambda *_args: True,
        execute_gripper_episode=lambda *_args: pytest.fail("source-only ran gripper"),
        emit_gripper_result=lambda *_args: pytest.fail("source-only emitted gripper"),
        execution_errors=(RuntimeError,),
        fatal_cuda_error=lambda _exc: False,
    )
    monkeypatch.setattr(runtime_module, "load_sam_runtime", lambda: fake_runtime)
    monkeypatch.setattr(runtime_module.signal, "signal", lambda *_args: None)
    connection = _DirectConnection((SamWorkerCommand.run_episode(11), SamWorkerCommand.shutdown()))

    runtime_module.sam_worker_process_entry(
        connection,  # type: ignore[arg-type]
        worker_id=3,
        gpu_id=6,
        launch_id=1,
        config=_parallel_config(tmp_path, (6,)),
        run_id="direct-worker",
        source_only=True,
        qwen_request_gate=SimpleNamespace(acquire=lambda **_kwargs: True, release=lambda: None),
    )

    assert calls == [(6,), "qwen", "sam", "shutdown"]
    assert isinstance(connection.sent[0], SamWorkerReady)
    assert isinstance(connection.sent[1], SamWorkerOutcome)
    assert connection.sent[1].episode_id == 11  # type: ignore[union-attr]
    assert connection.closed is True


class _FakeConnection:
    def __init__(self) -> None:
        self.sent: list[object] = []
        self.closed = False

    def send(self, value: object) -> None:
        self.sent.append(value)

    def close(self) -> None:
        self.closed = True


class _FakeProcess:
    def __init__(self, *, fail_start: bool) -> None:
        self.fail_start = fail_start
        self.started = False
        self.alive = False
        self.closed = False
        self.exitcode = 0

    def start(self) -> None:
        if self.fail_start:
            raise OSError("cannot spawn")
        self.started = True
        self.alive = True

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout: float | None = None) -> None:
        del timeout
        self.alive = False

    def terminate(self) -> None:
        self.alive = False

    def kill(self) -> None:
        self.alive = False

    def close(self) -> None:
        self.closed = True


class _FailingStartContext:
    def __init__(self) -> None:
        self.connections: list[tuple[_FakeConnection, _FakeConnection]] = []
        self.processes: list[_FakeProcess] = []

    def BoundedSemaphore(self, _value: int) -> Any:
        return SimpleNamespace(acquire=lambda **_kwargs: True, release=lambda: None)

    def Pipe(self, *, duplex: bool) -> tuple[_FakeConnection, _FakeConnection]:
        assert duplex is True
        pair = (_FakeConnection(), _FakeConnection())
        self.connections.append(pair)
        return pair

    def Process(self, **_kwargs: Any) -> _FakeProcess:
        process = _FakeProcess(fail_start=len(self.processes) == 1)
        self.processes.append(process)
        return process


def test_partial_process_start_failure_cleans_started_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _FailingStartContext()
    monkeypatch.setattr(runtime_module.mp, "get_context", lambda _kind: context)

    with pytest.raises(OSError, match="cannot spawn"):
        runtime_module.run_parallel_sam_episodes(
            _parallel_config(tmp_path, (0, 1)),
            run_id="start-failure",
            episode_ids=(1,),
            source_only=True,
            qwen_max_in_flight=1,
            outcome_callback=lambda _result: None,
            reporter=None,
            report_lifecycle=False,
            worker_entry=_scripted_worker_entry,
        )

    assert context.processes[0].alive is False
    assert context.processes[0].closed is True
    assert context.connections[0][0].closed is True
    assert context.connections[0][1].closed is True
    assert context.connections[1][0].closed is True
    assert context.connections[1][1].closed is True
