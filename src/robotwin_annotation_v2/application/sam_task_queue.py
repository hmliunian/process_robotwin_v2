"""Run queued SAM calls on the thread that owns the resident predictor."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from contextlib import AbstractContextManager
from contextvars import Context, copy_context
from dataclasses import dataclass
from queue import Empty, Queue
from typing import Any


class SamQueueStopped(RuntimeError):
    """The owner stopped accepting model work; waiting episodes must unwind."""


@dataclass
class _Call:
    method: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    future: Future[Any]
    context: Context
    submitted: float


class SamTaskQueue:
    """A blocking backend proxy for CPU workers, serviced exclusively by its owner.

    Each episode submits at most one call before waiting for its result, so the
    episode admission limit also bounds queued calls. No CUDA state leaves the
    owner; the existing backend returns CPU masks after closing each session.
    """

    _METHODS = frozenset(
        {
            "text_mask",
            "text_masks",
            "text_query_masks",
            "box_mask",
            "text_box_mask",
            "propagate_mask",
        }
    )

    def __init__(
        self,
        backend: Any,
        *,
        capacity: int,
        fatal_cuda_error: Callable[[BaseException], bool],
        measure: Callable[[str], AbstractContextManager[None]],
    ) -> None:
        if capacity < 1:
            raise ValueError("SAM queue capacity must be positive")
        self._backend = backend
        self._owner = threading.get_ident()
        self._requests: Queue[_Call] = Queue(capacity)
        self._lock = threading.Lock()
        self._stopped: str | None = None
        self._fatal_cuda_error = fatal_cuda_error
        self._measure = measure
        self.fatal_error: BaseException | None = None
        self.calls = 0
        self.peak_pending = 0
        self.queue_wait_s = 0.0
        self.model_call_s = 0.0
        self.empty_wait_s = 0.0

    def __getattr__(self, method: str) -> Callable[..., Any]:
        if method not in self._METHODS:
            raise AttributeError(method)

        def submit(*args: Any, **kwargs: Any) -> Any:
            if threading.get_ident() == self._owner:
                raise RuntimeError("SAM proxy calls must originate from episode workers")
            request = _Call(method, args, kwargs, Future(), copy_context(), time.perf_counter())
            with self._lock:
                if self._stopped is not None:
                    raise SamQueueStopped(self._stopped)
                self._requests.put_nowait(request)
                self.peak_pending = max(self.peak_pending, self._requests.qsize())
            return request.future.result()

        return submit

    def run_one(self, timeout: float = 0.05) -> bool:
        if threading.get_ident() != self._owner:
            raise RuntimeError("only the SAM owner may service model calls")
        start = time.perf_counter()
        try:
            request = self._requests.get(timeout=timeout)
        except Empty:
            self.empty_wait_s += time.perf_counter() - start
            return False
        self.empty_wait_s += time.perf_counter() - start
        self.queue_wait_s += time.perf_counter() - request.submitted
        if not request.future.set_running_or_notify_cancel():
            return False

        def invoke() -> Any:
            with self._measure(f"sam_{request.method}"):
                return getattr(self._backend, request.method)(*request.args, **request.kwargs)

        start = time.perf_counter()
        try:
            result = request.context.run(invoke)
        except BaseException as exc:
            if self._fatal_cuda_error(exc) or not isinstance(exc, (Exception, SystemExit)):
                self.fatal_error = exc
                self.close(f"SAM owner stopped: {type(exc).__name__}: {exc}")
            request.future.set_exception(exc)
            if not isinstance(exc, (Exception, SystemExit)):
                raise
        else:
            request.future.set_result(result)
        finally:
            self.calls += 1
            self.model_call_s += time.perf_counter() - start
        return True

    def close(self, reason: str = "SAM task queue closed") -> None:
        with self._lock:
            if self._stopped is not None:
                return
            self._stopped = reason
            while True:
                try:
                    request = self._requests.get_nowait()
                except Empty:
                    break
                request.future.set_exception(SamQueueStopped(reason))

    def shutdown(self) -> None:
        raise RuntimeError("backend shutdown belongs to the SAM owner")

    def snapshot(self) -> dict[str, int | float]:
        return {
            "calls": self.calls,
            "peak_pending": self.peak_pending,
            "queue_wait_s": self.queue_wait_s,
            "model_call_s": self.model_call_s,
            "empty_wait_s": self.empty_wait_s,
        }
