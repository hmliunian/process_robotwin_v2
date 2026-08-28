"""Pickle-safe commands and results for spawned SAM workers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

type SamWorkerCommandKind = Literal["run_episode", "shutdown"]


def _validate_identifier(value: int, *, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")


def _validate_attempt(value: int, *, field: str = "attempt") -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")


def _validate_nonempty(value: str, *, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class SamWorkerCommand:
    """A coordinator request to run one episode or stop an idle worker."""

    kind: SamWorkerCommandKind
    episode_id: int | None = None
    attempt: int | None = None

    def __post_init__(self) -> None:
        if self.kind == "run_episode":
            if self.episode_id is None or self.attempt is None:
                raise ValueError("run_episode requires episode_id and attempt")
            _validate_identifier(self.episode_id, field="episode_id")
            _validate_attempt(self.attempt)
            return
        if self.kind == "shutdown":
            if self.episode_id is not None or self.attempt is not None:
                raise ValueError("shutdown must not identify an episode")
            return
        raise ValueError(f"unsupported SAM worker command: {self.kind!r}")

    @classmethod
    def run_episode(cls, episode_id: int, *, attempt: int = 1) -> SamWorkerCommand:
        return cls("run_episode", episode_id=episode_id, attempt=attempt)

    @classmethod
    def shutdown(cls) -> SamWorkerCommand:
        return cls("shutdown")


@dataclass(frozen=True, slots=True)
class SamWorkerReady:
    """Backend-loaded handshake emitted before a worker accepts episodes."""

    worker_id: int
    gpu_id: int

    def __post_init__(self) -> None:
        _validate_identifier(self.worker_id, field="worker_id")
        _validate_identifier(self.gpu_id, field="gpu_id")


@dataclass(frozen=True, slots=True)
class SamWorkerOutcome:
    """One terminal episode result returned by a SAM worker."""

    worker_id: int | None
    gpu_id: int | None
    episode_id: int
    attempt: int
    status: str
    error: str | None = None
    fatal_cuda: bool = False

    def __post_init__(self) -> None:
        if self.worker_id is not None:
            _validate_identifier(self.worker_id, field="worker_id")
        if self.gpu_id is not None:
            _validate_identifier(self.gpu_id, field="gpu_id")
        _validate_identifier(self.episode_id, field="episode_id")
        _validate_attempt(self.attempt)
        _validate_nonempty(self.status, field="status")
        if self.error is not None:
            _validate_nonempty(self.error, field="error")
        if not isinstance(self.fatal_cuda, bool):
            raise TypeError("fatal_cuda must be a boolean")
        if self.fatal_cuda and self.error is None:
            raise ValueError("fatal CUDA outcomes require an error")


@dataclass(frozen=True, slots=True)
class SamWorkerFailure:
    """A serialized worker failure, including failures outside episode execution."""

    worker_id: int
    gpu_id: int
    error_type: str
    error: str
    traceback: str
    episode_id: int | None = None
    attempt: int | None = None
    fatal_cuda: bool = False
    retryable: bool = False

    def __post_init__(self) -> None:
        _validate_identifier(self.worker_id, field="worker_id")
        _validate_identifier(self.gpu_id, field="gpu_id")
        _validate_nonempty(self.error_type, field="error_type")
        _validate_nonempty(self.error, field="error")
        if not isinstance(self.traceback, str):
            raise TypeError("traceback must be a string")
        if (self.episode_id is None) != (self.attempt is None):
            raise ValueError("episode_id and attempt must both be set or both be omitted")
        if self.episode_id is not None and self.attempt is not None:
            _validate_identifier(self.episode_id, field="episode_id")
            _validate_attempt(self.attempt)
        if not isinstance(self.fatal_cuda, bool):
            raise TypeError("fatal_cuda must be a boolean")
        if not isinstance(self.retryable, bool):
            raise TypeError("retryable must be a boolean")
        if self.fatal_cuda and self.retryable:
            raise ValueError("fatal CUDA failures cannot be retryable")


type SamWorkerResponse = SamWorkerReady | SamWorkerOutcome | SamWorkerFailure

__all__ = [
    "SamWorkerCommand",
    "SamWorkerCommandKind",
    "SamWorkerFailure",
    "SamWorkerOutcome",
    "SamWorkerReady",
    "SamWorkerResponse",
]
