"""Spawn-safe coordinator for one resident SAM backend per explicit GPU."""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import signal
import sys
import time
import traceback
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from multiprocessing.connection import Connection, wait
from multiprocessing.process import BaseProcess
from typing import Any, Literal, cast

from robotwin_annotation_v2.adapters.qwen_client import (
    FileRequestGate,
    RequestGate,
    install_qwen_request_gate,
)
from robotwin_annotation_v2.config import PipelineConfig

from .parallel_sam_protocol import (
    SamWorkerCommand,
    SamWorkerFailure,
    SamWorkerOutcome,
    SamWorkerReady,
    SamWorkerResponse,
)
from .sam_workflow import (
    SamEpisodeResult,
    capture_sam_stage_output,
    execute_sam_episode,
    load_sam_runtime,
)

WORKER_SHUTDOWN_TIMEOUT_SECONDS = 3.0
WORKER_KILL_TIMEOUT_SECONDS = 1.0
WORKER_FAILURE_EXIT_TIMEOUT_SECONDS = 2.0
WORKER_STARTUP_TIMEOUT_SECONDS = 600.0
EPISODE_TIMEOUT_SECONDS = 3600.0
MAX_EPISODE_ATTEMPTS = 2
MAX_STARTUP_ATTEMPTS = 2
MAX_IDLE_RESTARTS = 1

type WorkerPhase = Literal["starting", "idle", "busy", "retired"]
type WorkerEntry = Callable[..., None]


@dataclass
class _WorkerState:
    worker_id: int
    gpu_id: int
    process: BaseProcess
    connection: Connection
    launch_id: int = 1
    phase: WorkerPhase = "starting"
    inflight: SamWorkerCommand | None = None
    startup_failures: int = 0
    idle_restarts: int = 0
    phase_started_at: float = 0.0
    last_error: str | None = None

    def __post_init__(self) -> None:
        self.phase_started_at = time.monotonic()


def _is_retired(state: _WorkerState) -> bool:
    return state.phase == "retired"


def _send_failure(connection: Connection, failure: SamWorkerFailure) -> None:
    with contextlib.suppress(BrokenPipeError, EOFError, OSError):
        connection.send(failure)


def sam_worker_process_entry(
    connection: Connection,
    *,
    worker_id: int,
    gpu_id: int,
    launch_id: int,
    config: PipelineConfig,
    run_id: str,
    source_only: bool,
    qwen_request_gate: RequestGate,
) -> None:
    """Own one GPU-local SAM backend and execute coordinator commands serially."""

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    worker_config = replace(config, sam3=replace(config.sam3, gpus=(gpu_id,)))
    log_path = (
        worker_config.output_root
        / "logs"
        / f"{run_id}.sam-w{worker_id}-gpu{gpu_id}-r{launch_id}.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("a", encoding="utf-8", buffering=1)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = log
    sys.stderr = log
    backend: Any | None = None
    command: SamWorkerCommand | None = None
    runtime: Any | None = None
    try:
        install_qwen_request_gate(qwen_request_gate)
        runtime = load_sam_runtime()
        backend = runtime.backend_factory(
            checkpoint_path=worker_config.sam3.checkpoint,
            gpus=worker_config.sam3.gpus,
        )
        connection.send(SamWorkerReady(worker_id=worker_id, gpu_id=gpu_id))
        while True:
            command = connection.recv()
            if not isinstance(command, SamWorkerCommand):
                raise TypeError(f"invalid SAM worker command: {command!r}")
            if command.kind == "shutdown":
                break
            assert command.episode_id is not None
            assert command.attempt is not None
            result = execute_sam_episode(
                runtime,
                capture_stage_output=capture_sam_stage_output,
                dynamic=worker_config,
                episode_id=command.episode_id,
                run_id=run_id,
                backend=backend,
                source_only=source_only,
                position=command.episode_id,
                total=1,
                report_lifecycle=False,
                reporter=None,
            )
            connection.send(
                SamWorkerOutcome(
                    worker_id=worker_id,
                    gpu_id=gpu_id,
                    episode_id=result.episode_id,
                    attempt=command.attempt,
                    status=result.status,
                    error=result.error,
                    fatal_cuda=result.fatal_cuda,
                )
            )
            command = None
            if result.fatal_cuda:
                break
    except EOFError:
        pass
    except BaseException as exc:  # noqa: BLE001 - serialize the child boundary
        fatal_cuda = runtime is not None and runtime.fatal_cuda_error(exc)
        child_traceback = traceback.format_exc()
        print(child_traceback, file=log, flush=True)
        _send_failure(
            connection,
            SamWorkerFailure(
                worker_id=worker_id,
                gpu_id=gpu_id,
                episode_id=(
                    command.episode_id
                    if command is not None and command.kind == "run_episode"
                    else None
                ),
                attempt=(
                    command.attempt
                    if command is not None and command.kind == "run_episode"
                    else None
                ),
                error_type=type(exc).__name__,
                error=str(exc) or repr(exc),
                traceback=child_traceback,
                fatal_cuda=fatal_cuda,
                retryable=not fatal_cuda,
            ),
        )
    finally:
        install_qwen_request_gate(None)
        if backend is not None:
            with contextlib.suppress(BaseException):
                backend.shutdown()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log.close()
        connection.close()


def _open_worker_transport(
    context: Any,
    *,
    worker_entry: WorkerEntry,
    worker_id: int,
    gpu_id: int,
    launch_id: int,
    config: PipelineConfig,
    run_id: str,
    source_only: bool,
    qwen_request_gate: RequestGate,
) -> tuple[BaseProcess, Connection]:
    parent_connection, child_connection = context.Pipe(duplex=True)
    process = context.Process(
        target=worker_entry,
        kwargs={
            "connection": child_connection,
            "worker_id": worker_id,
            "gpu_id": gpu_id,
            "launch_id": launch_id,
            "config": config,
            "run_id": run_id,
            "source_only": source_only,
            "qwen_request_gate": qwen_request_gate,
        },
        name=f"robotwin-sam-gpu-{gpu_id}-r{launch_id}",
    )
    try:
        process.start()
    except BaseException:
        parent_connection.close()
        child_connection.close()
        with contextlib.suppress(ValueError):
            process.close()
        raise
    child_connection.close()
    return process, parent_connection


def _new_worker_state(
    context: Any,
    *,
    worker_entry: WorkerEntry,
    worker_id: int,
    gpu_id: int,
    config: PipelineConfig,
    run_id: str,
    source_only: bool,
    qwen_request_gate: RequestGate,
) -> _WorkerState:
    process, connection = _open_worker_transport(
        context,
        worker_entry=worker_entry,
        worker_id=worker_id,
        gpu_id=gpu_id,
        launch_id=1,
        config=config,
        run_id=run_id,
        source_only=source_only,
        qwen_request_gate=qwen_request_gate,
    )
    return _WorkerState(worker_id, gpu_id, process, connection)


def _close_failed_transport(state: _WorkerState) -> None:
    state.process.join(timeout=WORKER_FAILURE_EXIT_TIMEOUT_SECONDS)
    if state.process.is_alive():
        state.process.terminate()
        state.process.join(timeout=WORKER_KILL_TIMEOUT_SECONDS)
    if state.process.is_alive():
        state.process.kill()
        state.process.join(timeout=WORKER_KILL_TIMEOUT_SECONDS)
    state.connection.close()


def _relaunch_worker(
    state: _WorkerState,
    context: Any,
    *,
    worker_entry: WorkerEntry,
    config: PipelineConfig,
    run_id: str,
    source_only: bool,
    qwen_request_gate: RequestGate,
) -> None:
    _close_failed_transport(state)
    launch_id = state.launch_id + 1
    process, connection = _open_worker_transport(
        context,
        worker_entry=worker_entry,
        worker_id=state.worker_id,
        gpu_id=state.gpu_id,
        launch_id=launch_id,
        config=config,
        run_id=run_id,
        source_only=source_only,
        qwen_request_gate=qwen_request_gate,
    )
    state.process = process
    state.connection = connection
    state.launch_id = launch_id
    state.phase = "starting"
    state.inflight = None
    state.phase_started_at = time.monotonic()


def _stop_workers(states: Sequence[_WorkerState]) -> None:
    for state in states:
        if state.process.is_alive():
            with contextlib.suppress(BrokenPipeError, EOFError, OSError):
                state.connection.send(SamWorkerCommand.shutdown())
    deadline = time.monotonic() + WORKER_SHUTDOWN_TIMEOUT_SECONDS
    for state in states:
        state.process.join(timeout=max(0.0, deadline - time.monotonic()))
    alive = [state for state in states if state.process.is_alive()]
    for state in alive:
        state.process.terminate()
    deadline = time.monotonic() + WORKER_KILL_TIMEOUT_SECONDS
    for state in alive:
        state.process.join(timeout=max(0.0, deadline - time.monotonic()))
    for state in alive:
        if state.process.is_alive():
            state.process.kill()
            state.process.join(timeout=WORKER_KILL_TIMEOUT_SECONDS)
    for state in states:
        state.connection.close()
        with contextlib.suppress(ValueError):
            state.process.close()


def _worker_exit_failure(state: _WorkerState) -> SamWorkerFailure:
    command = state.inflight
    return SamWorkerFailure(
        worker_id=state.worker_id,
        gpu_id=state.gpu_id,
        episode_id=(
            command.episode_id if command is not None and command.kind == "run_episode" else None
        ),
        attempt=(
            command.attempt if command is not None and command.kind == "run_episode" else None
        ),
        error_type="WorkerExitError",
        error=f"worker exited with code {state.process.exitcode}",
        traceback="",
        retryable=True,
    )


def _worker_timeout_failure(state: _WorkerState, *, timeout_seconds: float) -> SamWorkerFailure:
    command = state.inflight
    return SamWorkerFailure(
        worker_id=state.worker_id,
        gpu_id=state.gpu_id,
        episode_id=(
            command.episode_id if command is not None and command.kind == "run_episode" else None
        ),
        attempt=(
            command.attempt if command is not None and command.kind == "run_episode" else None
        ),
        error_type="WorkerTimeoutError",
        error=f"worker {state.phase} phase exceeded {timeout_seconds:g} seconds",
        traceback="",
        retryable=True,
    )


def _failure_outcome(
    state: _WorkerState,
    command: SamWorkerCommand,
    failure: SamWorkerFailure,
) -> SamWorkerOutcome:
    assert command.episode_id is not None
    assert command.attempt is not None
    return SamWorkerOutcome(
        worker_id=state.worker_id,
        gpu_id=state.gpu_id,
        episode_id=command.episode_id,
        attempt=command.attempt,
        status="failed",
        error=f"{failure.error_type}: {failure.error}",
        fatal_cuda=failure.fatal_cuda,
    )


def run_parallel_sam_episodes(
    config: PipelineConfig,
    *,
    run_id: str,
    episode_ids: tuple[int, ...],
    source_only: bool,
    qwen_max_in_flight: int,
    outcome_callback: Callable[[SamEpisodeResult], None],
    reporter: Any | None,
    report_lifecycle: bool,
    worker_entry: WorkerEntry = sam_worker_process_entry,
) -> tuple[SamWorkerOutcome, ...]:
    """Dynamically schedule episodes across explicit GPU-bound SAM workers."""

    if not episode_ids:
        return ()
    if len(set(episode_ids)) != len(episode_ids):
        raise ValueError("parallel SAM episode ids must be unique")
    worker_gpus = config.parallel.sam_worker_gpus
    if not worker_gpus:
        raise ValueError("parallel SAM execution requires at least one worker GPU")
    if qwen_max_in_flight < 1:
        raise ValueError("qwen_max_in_flight must be positive")

    context = mp.get_context("spawn")
    qwen_request_gate = cast(
        RequestGate,
        FileRequestGate.create(
            config.output_root / "logs" / f".{run_id}.qwen-request-gate",
            capacity=qwen_max_in_flight,
        ),
    )
    states: list[_WorkerState] = []
    pending = deque(SamWorkerCommand.run_episode(episode_id) for episode_id in episode_ids)
    outcomes: dict[int, SamWorkerOutcome] = {}
    started_ids: set[int] = set()
    positions = {episode_id: position for position, episode_id in enumerate(episode_ids, 1)}

    def terminalize(outcome: SamWorkerOutcome) -> None:
        if outcome.episode_id in outcomes:
            raise RuntimeError(f"episode {outcome.episode_id} received duplicate terminal results")
        outcomes[outcome.episode_id] = outcome
        outcome_callback(
            SamEpisodeResult(
                outcome.episode_id,
                outcome.status,
                outcome.error,
                outcome.fatal_cuda,
            )
        )
        if reporter is not None and report_lifecycle:
            reporter.episode_finished(
                outcome.episode_id,
                status=outcome.status,
                detail=outcome.error,
            )

    def relaunch(state: _WorkerState) -> bool:
        try:
            _relaunch_worker(
                state,
                context,
                worker_entry=worker_entry,
                config=config,
                run_id=run_id,
                source_only=source_only,
                qwen_request_gate=qwen_request_gate,
            )
        except (OSError, RuntimeError) as exc:
            state.phase = "retired"
            state.last_error = f"worker relaunch failed: {type(exc).__name__}: {exc}"
            return False
        return True

    def validate_worker_identity(response: SamWorkerResponse, state: _WorkerState) -> None:
        if response.worker_id != state.worker_id or response.gpu_id != state.gpu_id:
            raise RuntimeError(f"SAM worker returned mismatched identity: {response}")

    def handle_failure(state: _WorkerState, failure: SamWorkerFailure) -> None:
        validate_worker_identity(failure, state)
        previous_phase = state.phase
        command = state.inflight
        state.last_error = f"{failure.error_type}: {failure.error}"
        if previous_phase == "starting":
            if failure.episode_id is not None or failure.attempt is not None:
                raise RuntimeError(f"SAM startup failure identified an episode: {failure}")
        elif previous_phase == "busy":
            if command is None or command.kind != "run_episode":
                raise RuntimeError("busy SAM worker has no episode command")
            if failure.episode_id != command.episode_id or failure.attempt != command.attempt:
                raise RuntimeError(f"SAM worker failure did not match its command: {failure}")
        elif previous_phase == "idle":
            if failure.episode_id is not None or failure.attempt is not None:
                raise RuntimeError(f"idle SAM worker failure identified an episode: {failure}")
        else:
            return

        state.phase = "retired"
        state.inflight = None
        if command is not None:
            assert command.episode_id is not None
            assert command.attempt is not None
            if failure.retryable and command.attempt < MAX_EPISODE_ATTEMPTS:
                pending.appendleft(
                    SamWorkerCommand.run_episode(
                        command.episode_id,
                        attempt=command.attempt + 1,
                    )
                )
            else:
                terminalize(_failure_outcome(state, command, failure))

        if failure.fatal_cuda or not failure.retryable:
            return
        if previous_phase == "starting":
            state.startup_failures += 1
            if state.startup_failures >= MAX_STARTUP_ATTEMPTS:
                return
        elif previous_phase == "idle":
            state.idle_restarts += 1
            if state.idle_restarts > MAX_IDLE_RESTARTS:
                return
        if len(outcomes) < len(episode_ids):
            relaunch(state)

    def handle_response(state: _WorkerState, response: object) -> None:
        if isinstance(response, SamWorkerReady):
            validate_worker_identity(response, state)
            if state.phase != "starting":
                raise RuntimeError(f"unexpected SAM worker ready message: {response}")
            state.phase = "idle"
            state.startup_failures = 0
            state.phase_started_at = time.monotonic()
            state.last_error = None
            return
        if isinstance(response, SamWorkerOutcome):
            validate_worker_identity(response, state)
            command = state.inflight
            if (
                state.phase != "busy"
                or command is None
                or command.kind != "run_episode"
                or command.episode_id != response.episode_id
                or command.attempt != response.attempt
            ):
                raise RuntimeError(f"SAM worker returned an unexpected outcome: {response}")
            state.inflight = None
            state.phase = "retired" if response.fatal_cuda else "idle"
            state.phase_started_at = time.monotonic()
            terminalize(response)
            return
        if isinstance(response, SamWorkerFailure):
            handle_failure(state, response)
            return
        raise TypeError(f"invalid SAM worker response: {response!r}")

    def receive_available(state: _WorkerState, *, ready: bool = False) -> None:
        connection = state.connection
        first = ready
        while state.connection is connection and (first or connection.poll()):
            first = False
            try:
                response = connection.recv()
            except EOFError:
                handle_failure(state, _worker_exit_failure(state))
                return
            handle_response(state, response)

    try:
        for worker_id, gpu_id in enumerate(worker_gpus):
            states.append(
                _new_worker_state(
                    context,
                    worker_entry=worker_entry,
                    worker_id=worker_id,
                    gpu_id=gpu_id,
                    config=config,
                    run_id=run_id,
                    source_only=source_only,
                    qwen_request_gate=qwen_request_gate,
                )
            )

        while len(outcomes) < len(episode_ids):
            for state in states:
                if state.phase != "idle" or not pending:
                    continue
                if not state.process.is_alive():
                    receive_available(state, ready=state.connection.poll(0.05))
                    if state.phase == "idle" and not state.process.is_alive():
                        handle_failure(state, _worker_exit_failure(state))
                    continue
                command = pending[0]
                try:
                    state.connection.send(command)
                except (BrokenPipeError, EOFError, OSError):
                    handle_failure(state, _worker_exit_failure(state))
                    continue
                pending.popleft()
                state.inflight = command
                state.phase = "busy"
                state.phase_started_at = time.monotonic()
                assert command.episode_id is not None
                if command.episode_id not in started_ids:
                    started_ids.add(command.episode_id)
                    if reporter is not None and report_lifecycle:
                        reporter.episode_started(
                            command.episode_id,
                            position=positions[command.episode_id],
                            total=len(episode_ids),
                        )

            live_connections = [state.connection for state in states if state.phase != "retired"]
            if live_connections:
                ready_connections = cast(
                    list[Connection],
                    wait(live_connections, timeout=0.2),
                )
                for connection in ready_connections:
                    state = next(item for item in states if item.connection is connection)
                    receive_available(state, ready=True)

            for state in states:
                if state.phase == "retired" or state.process.is_alive():
                    continue
                connection = state.connection
                if connection.poll(0.05):
                    receive_available(state, ready=True)
                if state.connection is connection and not _is_retired(state):
                    handle_failure(state, _worker_exit_failure(state))

            now = time.monotonic()
            for state in states:
                timeout_seconds = (
                    WORKER_STARTUP_TIMEOUT_SECONDS
                    if state.phase == "starting"
                    else EPISODE_TIMEOUT_SECONDS
                    if state.phase == "busy"
                    else None
                )
                if timeout_seconds is not None and now - state.phase_started_at > timeout_seconds:
                    handle_failure(
                        state,
                        _worker_timeout_failure(
                            state,
                            timeout_seconds=timeout_seconds,
                        ),
                    )

            if pending and all(state.phase == "retired" for state in states):
                worker_errors = "; ".join(
                    f"gpu {state.gpu_id}: {state.last_error or 'worker retired'}"
                    for state in states
                )
                while pending:
                    command = pending.popleft()
                    assert command.episode_id is not None
                    assert command.attempt is not None
                    terminalize(
                        SamWorkerOutcome(
                            worker_id=None,
                            gpu_id=None,
                            episode_id=command.episode_id,
                            attempt=command.attempt,
                            status="not_run_no_healthy_sam_worker",
                            error=f"no healthy SAM worker remains ({worker_errors})",
                        )
                    )
    finally:
        _stop_workers(states)

    return tuple(outcomes[episode_id] for episode_id in episode_ids)


__all__ = ["run_parallel_sam_episodes", "sam_worker_process_entry"]
