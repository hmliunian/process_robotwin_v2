"""Bounded UMI planning and rendering around one resident, serial SAM backend.

SAM seed QC, propagation and canonical mask publication keep their production
implementation. Only planning of other episodes and disk-based review rendering
overlap that work. A predictor is never shared between threads.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from robotwin_annotation_v2.adapters.artifact_store import ArtifactStore
from robotwin_annotation_v2.adapters.qwen_client import RequestGate, install_qwen_request_gate
from robotwin_annotation_v2.config import PipelineConfig
from robotwin_annotation_v2.models import EpisodeRef

from .sam_workflow import SamBackend, SamRuntime, load_sam_runtime, validate_sam_run_id

CURRENT_JOB: ContextVar[str] = ContextVar("umi_pipeline_job", default="batch")
CURRENT_STAGE: ContextVar[str] = ContextVar("umi_pipeline_stage", default="batch")


@dataclass(frozen=True)
class UmiJob:
    config: PipelineConfig
    episode: int
    run_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    input_signature: str = ""

    @property
    def ref(self) -> EpisodeRef:
        return EpisodeRef(self.config.dataset.task, self.episode, self.config.dataset.camera)

    @property
    def key(self) -> str:
        return f"{self.run_id}/{self.ref.task}/{self.episode:06d}/{self.ref.camera}"

    @property
    def directory(self) -> Path:
        return (
            ArtifactStore(self.config.output_root).episode_dir(self.run_id, self.ref) / ".pipeline"
        )

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "dataset_root": str(self.config.dataset.root.resolve()),
            "task": self.ref.task,
            "episode": self.episode,
            "camera": self.ref.camera,
            "annotation_mode": self.config.annotation.mode.value,
            "seed_image_only": self.config.sam3.seed_image_only,
            "tracking_batch_size": self.config.sam3.tracking_batch_size,
            "input_signature": self.input_signature,
        }

    def validate_output(self, *, resume: bool) -> None:
        path = self.directory / "job.json"
        if self.directory.parent.exists() and not resume:
            raise ValueError(f"output exists; use an explicit resume: {self.run_id}")
        if self.directory.parent.exists() and not path.is_file():
            raise ValueError(f"existing output has no job identity: {self.run_id}")
        if path.exists() and json.loads(path.read_text()) != self.identity:
            raise ValueError(f"output belongs to a different input: {self.run_id}")


@dataclass
class PipelineEvents:
    """Inclusive wall intervals; overlapping work must not be summed as makespan."""

    origin: float = field(default_factory=time.perf_counter)
    events: list[dict[str, Any]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @contextmanager
    def measure(self, stage: str, job: str | None = None) -> Iterator[None]:
        key = CURRENT_JOB.get() if job is None else job
        token = CURRENT_JOB.set(key)
        stage_token = CURRENT_STAGE.set(stage)
        start = time.perf_counter() - self.origin
        error = None
        try:
            yield
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            end = time.perf_counter() - self.origin
            with self._lock:
                self.events.append(
                    {
                        "job": key,
                        "stage": stage,
                        "start_s": start,
                        "end_s": end,
                        "wall_s": end - start,
                        "error": error,
                    }
                )
            CURRENT_JOB.reset(token)
            CURRENT_STAGE.reset(stage_token)

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self.events)


def validate_batch_selection(output: Path, jobs: Sequence[UmiJob]) -> None:
    ledger = output / "episode_results.json"
    if not ledger.exists():
        return
    previous = json.loads(ledger.read_text())
    if previous.get("selected_job_ids") != [job.key for job in jobs]:
        raise ValueError("resume requires the same complete batch selection")


def validate_umi_jobs(jobs: Sequence[UmiJob], *, resume: bool) -> None:
    """Reject incompatible jobs and unowned artifacts before publishing run state."""
    if not jobs:
        raise ValueError("batch selection contains no episodes")
    if len({job.key for job in jobs}) != len(jobs):
        raise ValueError("each batch episode requires a distinct job identity")
    first = jobs[0].config
    if len(first.sam3.gpus) != 1:
        raise ValueError("UMI pipeline currently requires exactly one SAM GPU")
    for job in jobs:
        validate_sam_run_id(job.run_id)
        if (job.config.dataset.manifest_data or {}).get("timeline_source") != "video_window":
            raise ValueError("UMI pipeline requires prepared video_window inputs")
        if job.config.sam3 != first.sam3 or any(
            getattr(job.config.qwen, name) != getattr(first.qwen, name)
            for name in ("endpoint", "model", "runtime", "api_key_env")
        ):
            raise ValueError("all jobs must share SAM configuration and Qwen service")
        job.validate_output(resume=resume)


def run_umi_pipeline(
    jobs: Sequence[UmiJob],
    *,
    output: Path,
    render: Callable[[UmiJob, dict[str, Any]], dict[str, Any]],
    asynchronous: bool = True,
    prefetch: int = 2,
    qwen_workers: int = 4,
    render_buffer: int = 2,
    resume: bool = False,
    runtime: SamRuntime[SamBackend, Any, Any] | None = None,
    events: PipelineEvents | None = None,
    request_gate: RequestGate | None = None,
) -> dict[str, Any]:
    """Run all attempts, retaining mask failures and draining bounded review work.

    ``prefetch`` bounds all outstanding plans (running + ready). Planning uses
    at most ``qwen_workers - 1`` threads, leaving capacity for QC in a single-worker
    run when the cap exceeds one. Shared multi-GPU gates enforce a global request
    limit without QC priority. Serial mode uses identical model lifetime, stages
    and output policy, providing a resident-model A/B baseline.
    """
    if not jobs or min(prefetch, qwen_workers, render_buffer) < 1:
        raise ValueError("jobs and pipeline capacities must be nonempty/positive")
    first = jobs[0].config
    if resume:
        validate_batch_selection(output, jobs)
    validate_umi_jobs(jobs, resume=resume)
    for job in jobs:
        ArtifactStore.write_json(job.directory / "job.json", job.identity)

    runtime = load_sam_runtime() if runtime is None else runtime
    events = PipelineEvents() if events is None else events
    rows: dict[str, dict[str, Any]] = {}
    written_rows: dict[str, dict[str, Any]] = {}
    backend: SamBackend | None = None
    pending_reviews: dict[Future[dict[str, Any]], UmiJob] = {}
    pending_plans: dict[Future[None], UmiJob] = {}
    fatal_error: str | None = None
    final_status = "failed"
    peak_plans = 0
    peak_reviews = 0
    active_job: UmiJob | None = None
    phase = "startup"
    startup_complete = False
    started = time.perf_counter()
    previous_gate = install_qwen_request_gate(
        request_gate if request_gate is not None else threading.BoundedSemaphore(qwen_workers)
    )

    def new_row(job: UmiJob) -> dict[str, Any]:
        return {
            **job.metadata,
            "task": job.ref.task,
            "episode": job.episode,
            "camera": job.ref.camera,
            "run_id": job.run_id,
            "artifact_dir": str(job.directory.parent),
            "objects_ready": False,
            "masks_ready": False,
        }

    def record_unstarted(status: str) -> None:
        for job in jobs:
            if job.key not in rows:
                rows[job.key] = {
                    **new_row(job),
                    "status": status,
                    "process_status": status,
                }

    def persist(status: str) -> dict[str, Any]:
        result = {
            "format": "robotwin_umi_pipeline_v1",
            "status": status,
            "mode": "async" if asynchronous else "serial",
            "gpu": first.sam3.gpus[0],
            "prefetch": prefetch,
            "qwen_max_in_flight": qwen_workers,
            "render_buffer": render_buffer,
            "mask_save": "synchronous",
            "peak_plans": peak_plans,
            "peak_reviews": peak_reviews,
            "wall_s": time.perf_counter() - started,
            "fatal_error": fatal_error,
            "selected_job_ids": [job.key for job in jobs],
            "episodes": [rows[job.key] for job in jobs if job.key in rows],
        }
        ArtifactStore.write_json(output / "episode_results.json", result)
        ArtifactStore.write_json(output / "timings.json", {"events": events.snapshot()})
        for job in jobs:
            if job.key in rows and written_rows.get(job.key) != rows[job.key]:
                ArtifactStore.write_json(job.directory / "pipeline_result.json", rows[job.key])
                written_rows[job.key] = dict(rows[job.key])
        return result

    def plan(job: UmiJob) -> None:
        with events.measure("planning", job.key):
            runtime.run_qwen(job.config, job.episode, job.run_id)

    def review(job: UmiJob, row: dict[str, Any]) -> dict[str, Any]:
        try:
            with events.measure("render", job.key):
                return render(job, dict(row))
        except Exception as exc:  # noqa: BLE001 - masks survive failed review
            return {
                **row,
                "status": "render_failed",
                "objects_ready": False,
                "render_error": f"{type(exc).__name__}: {exc}",
            }

    def collect_reviews(*, block: bool = False) -> None:
        if block and pending_reviews:
            with events.measure("render_backpressure"):
                wait(pending_reviews, return_when=FIRST_COMPLETED)
        for future in tuple(pending_reviews):
            if future.done():
                job = pending_reviews.pop(future)
                rows[job.key] = future.result()
                persist("running")

    planner_count = min(prefetch, max(1, qwen_workers - 1))
    try:
        persist("running")
        complete = {
            job.key: resume
            and runtime.sam_episode_complete(
                job.config, ArtifactStore(job.config.output_root), job.run_id, job.ref
            )
            for job in jobs
        }
        for job in jobs:
            if complete[job.key]:
                rows[job.key] = {
                    **new_row(job),
                    "process_status": "skipped_complete",
                    "masks_ready": True,
                    "status": "masks_pending_render",
                }
        if not all(complete.values()):
            with events.measure("qwen_health"):
                runtime.qwen_client_factory(config=first.qwen).health()
            with events.measure("model_load"):
                backend = runtime.backend_factory(**first.sam3.backend_options())
        startup_complete = True
        with (
            ThreadPoolExecutor(
                max_workers=planner_count, thread_name_prefix="umi-plan"
            ) as planners,
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="umi-render") as renders,
        ):
            remaining = iter(jobs)

            def submit_next() -> bool:
                nonlocal peak_plans
                job = next(remaining, None)
                if job is None:
                    return False
                if complete[job.key]:
                    future: Future[None] = Future()
                    future.set_result(None)
                else:
                    future = planners.submit(plan, job)
                pending_plans[future] = job
                peak_plans = max(peak_plans, len(pending_plans))
                return True

            for _ in range(prefetch if asynchronous else 1):
                submit_next()
            while pending_plans:
                with events.measure("plan_wait"):
                    ready, _ = wait(pending_plans, return_when=FIRST_COMPLETED)
                future = next(f for f in pending_plans if f in ready)
                job = pending_plans.pop(future)
                active_job = job
                # Refill before SAM starts, so the next plan can overlap this episode.
                if asynchronous:
                    submit_next()
                row = new_row(job)
                phase = "planning"
                try:
                    future.result()
                    phase = "sam"
                    if complete[job.key]:
                        row["process_status"] = "skipped_complete"
                        row["masks_ready"] = True
                    else:
                        assert backend is not None
                        with events.measure("sam_qc_propagation_save", job.key):
                            execution = runtime.execute_sam_episode(
                                job.config, job.episode, job.run_id, backend
                            )
                            row["masks_ready"] = runtime.emit_sam_result(job.run_id, execution)
                        row["process_status"] = (
                            "completed" if row["masks_ready"] else "sam_incomplete"
                        )
                    row["status"] = "masks_pending_render" if row["masks_ready"] else "failed"
                except (Exception, SystemExit) as exc:  # noqa: BLE001 - retain every attempt
                    row.update(
                        status="failed",
                        process_status=f"{phase}_failed",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    if runtime.fatal_cuda_error(exc):
                        fatal_error = str(row["error"])
                rows[job.key] = row
                persist("running")
                phase = "render"
                if asynchronous:
                    collect_reviews()
                    while len(pending_reviews) >= render_buffer:
                        collect_reviews(block=True)
                    pending_reviews[renders.submit(review, job, row)] = job
                    peak_reviews = max(peak_reviews, len(pending_reviews))
                else:
                    rows[job.key] = review(job, row)
                    persist("running")
                    if fatal_error is None:
                        submit_next()
                active_job = None
                if fatal_error is not None:
                    for waiting in pending_plans:
                        waiting.cancel()
                    break
            # The returned makespan includes all disk writes and review work.
            while pending_reviews:
                collect_reviews(block=True)
        if fatal_error is not None:
            record_unstarted("not_run_after_fatal_cuda")
        final_status = "finished" if fatal_error is None else "failed"
    except BaseException as exc:
        fatal_error = f"{type(exc).__name__}: {exc}"
        final_status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        # Executor shutdown has drained running work; collect completed review results.
        try:
            collect_reviews()
        finally:
            if active_job is not None and not rows.get(active_job.key, {}).get("objects_ready"):
                status = f"{final_status}_during_{phase}"
                row = rows.get(active_job.key, new_row(active_job))
                rows[active_job.key] = {
                    **row,
                    "status": status,
                    "process_status": row["process_status"] if row["masks_ready"] else status,
                    "error": fatal_error,
                }
            reason = (
                "interrupt"
                if final_status == "interrupted"
                else ("failure" if startup_complete else "startup_failure")
            )
            record_unstarted(f"not_run_after_{reason}")
        raise
    finally:
        try:
            if backend is not None:
                with events.measure("model_shutdown"):
                    backend.shutdown()
        except BaseException as exc:
            fatal_error = f"{type(exc).__name__}: {exc}"
            final_status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            raise
        finally:
            install_qwen_request_gate(previous_gate)
            final_result = persist(final_status)
    return final_result
