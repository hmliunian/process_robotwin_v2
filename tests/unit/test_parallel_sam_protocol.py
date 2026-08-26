from __future__ import annotations

import pickle

import pytest

from robotwin_annotation_v2.application.parallel_sam_protocol import (
    SamWorkerCommand,
    SamWorkerFailure,
    SamWorkerOutcome,
    SamWorkerReady,
)


def test_worker_messages_round_trip_through_pickle() -> None:
    messages = (
        SamWorkerCommand.run_episode(17, attempt=2),
        SamWorkerCommand.shutdown(),
        SamWorkerReady(worker_id=1, gpu_id=4),
        SamWorkerOutcome(
            worker_id=1,
            gpu_id=4,
            episode_id=17,
            attempt=2,
            status="completed",
        ),
        SamWorkerFailure(
            worker_id=1,
            gpu_id=4,
            episode_id=17,
            attempt=2,
            error_type="RuntimeError",
            error="worker exited",
            traceback="traceback text",
            retryable=True,
        ),
    )

    assert tuple(pickle.loads(pickle.dumps(message)) for message in messages) == messages


def test_run_and_shutdown_commands_enforce_their_payload_shape() -> None:
    with pytest.raises(ValueError, match="requires episode_id and attempt"):
        SamWorkerCommand("run_episode")
    with pytest.raises(ValueError, match="must not identify an episode"):
        SamWorkerCommand("shutdown", episode_id=1, attempt=1)
    with pytest.raises(ValueError, match="positive integer"):
        SamWorkerCommand.run_episode(1, attempt=0)
    with pytest.raises(ValueError, match="unsupported"):
        SamWorkerCommand("pause")  # type: ignore[arg-type]


def test_outcome_validates_worker_identity_and_fatal_errors() -> None:
    with pytest.raises(ValueError, match="gpu_id must be a non-negative integer"):
        SamWorkerOutcome(0, -1, 7, 1, "completed")
    with pytest.raises(ValueError, match="status must be a non-empty string"):
        SamWorkerOutcome(0, 1, 7, 1, " ")
    with pytest.raises(ValueError, match="fatal CUDA outcomes require an error"):
        SamWorkerOutcome(0, 1, 7, 1, "failed", fatal_cuda=True)
    unassigned = SamWorkerOutcome(None, None, 7, 1, "not_run")
    assert unassigned.worker_id is None


def test_failure_allows_worker_startup_and_episode_failures() -> None:
    startup = SamWorkerFailure(
        worker_id=0,
        gpu_id=1,
        error_type="BackendLoadError",
        error="checkpoint failed",
        traceback="",
    )
    assert startup.episode_id is None

    with pytest.raises(ValueError, match="both be set or both be omitted"):
        SamWorkerFailure(
            worker_id=0,
            gpu_id=1,
            episode_id=7,
            error_type="RuntimeError",
            error="failed",
            traceback="",
        )
    with pytest.raises(ValueError, match="cannot be retryable"):
        SamWorkerFailure(
            worker_id=0,
            gpu_id=1,
            error_type="CudaError",
            error="device assert",
            traceback="",
            fatal_cuda=True,
            retryable=True,
        )
