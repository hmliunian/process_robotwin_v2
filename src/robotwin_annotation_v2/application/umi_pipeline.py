"""Bounded UMI episode pipelines around one resident, single-owner SAM backend."""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from robotwin_annotation_v2.adapters.artifact_store import ArtifactStore
from robotwin_annotation_v2.adapters.qwen_client import (
    QWEN_PLANNING,
    PlanningRequestGate,
    RequestGate,
    install_qwen_request_gate,
)
from robotwin_annotation_v2.config import PipelineConfig
from robotwin_annotation_v2.models import EpisodeRef

from .sam_task_queue import SamTaskQueue
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
    namespace: str = ""

    @property
    def ref(self) -> EpisodeRef:
        return EpisodeRef(self.config.dataset.task, self.episode, self.config.dataset.camera)

    @property
    def key(self) -> str:
        prefix = f"{self.namespace}/" if self.namespace else ""
        return f"{prefix}{self.run_id}/{self.ref.task}/{self.episode:06d}/{self.ref.camera}"

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
            **({"namespace": self.namespace} if self.namespace else {}),
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

    def retire(self, job: str) -> list[dict[str, Any]]:
        """Move a finished stream job's measurements out of the live working set."""
        with self._lock:
            retired = [event for event in self.events if event["job"] == job]
            self.events[:] = [event for event in self.events if event["job"] != job]
            return retired

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
    episode_workers: int = 1,
    video_workers: int = 2,
    resume: bool = False,
    runtime: SamRuntime[SamBackend, Any, Any] | None = None,
    events: PipelineEvents | None = None,
    request_gate: RequestGate | None = None,
    poll_job: Callable[[], UmiJob | None] | None = None,
    on_terminal: Callable[[UmiJob, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run all attempts, retaining mask failures and draining bounded review work.

    ``prefetch`` bounds outstanding plans. Queued execution limits planning
    concurrency and uses ``episode_workers`` lightweight QC producers.
    Production stages retain sampled frames during QC; ``video_workers`` bounds
    full-video decoding, propagation and results awaiting the independent writer.
    Render backlog retains only paths, never producer slots. All SAM calls stay
    on the owner thread. Runtimes without staged hooks retain whole-episode work.
    Serial execution provides a resident-model baseline with the same outputs.
    """
    if not jobs or min(prefetch, qwen_workers, render_buffer, episode_workers, video_workers) < 1:
        raise ValueError("jobs and pipeline capacities must be nonempty/positive")
    if episode_workers > 1 and not asynchronous:
        raise ValueError("multiple episode workers require asynchronous mode")
    if poll_job is not None and (episode_workers < 2 or not asynchronous or on_terminal is None):
        raise ValueError("streaming requires async producers and a durable terminal callback")
    active_jobs = list(jobs)
    jobs = active_jobs
    first = jobs[0].config
    if resume and poll_job is None:
        validate_batch_selection(output, jobs)
    validate_umi_jobs(jobs, resume=resume)
    for job in jobs:
        ArtifactStore.write_json(job.directory / "job.json", job.identity)

    runtime = load_sam_runtime() if runtime is None else runtime
    stages = (
        runtime.sam_episode_stages if episode_workers > 1 and first.sam3.seed_image_only else None
    )
    events = PipelineEvents() if events is None else events
    rows: dict[str, dict[str, Any]] = {}
    written_rows: dict[str, dict[str, Any]] = {}
    complete: dict[str, bool] = {}
    completed_jobs = 0
    completed_outputs = 0
    seen_keys = {job.key for job in jobs}
    stream_open = poll_job is not None
    backend: SamBackend | None = None
    model_queue: SamTaskQueue | None = None
    pending_reviews: dict[Future[dict[str, Any]], UmiJob] = {}
    pending_plans: dict[Future[None], UmiJob] = {}
    fatal_error: str | None = None
    final_status = "failed"
    peak_plans = 0
    peak_reviews = 0
    peak_episodes = 0
    peak_videos = 0
    peak_ready_videos = 0
    active_job: UmiJob | None = None
    phase = "startup"
    startup_complete = False
    started = time.perf_counter()
    previous_gate = install_qwen_request_gate(
        request_gate
        if request_gate is not None
        else (
            PlanningRequestGate(qwen_workers)
            if episode_workers > 1
            else threading.BoundedSemaphore(qwen_workers)
        )
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
            "episode_workers": episode_workers,
            "video_workers": video_workers,
            "staged": stages is not None,
            "mask_save": (
                "writer"
                if stages is not None
                else "episode_worker"
                if episode_workers > 1
                else "synchronous"
            ),
            "peak_plans": peak_plans,
            "peak_reviews": peak_reviews,
            "peak_episodes": peak_episodes,
            "peak_videos": peak_videos,
            "peak_ready_videos": peak_ready_videos,
            "sam_queue": None if model_queue is None else model_queue.snapshot(),
            "wall_s": time.perf_counter() - started,
            "fatal_error": fatal_error,
            "selected_job_ids": [job.key for job in jobs],
            "episodes": [rows[job.key] for job in jobs if job.key in rows],
            **(
                {
                    "selection_scope": "active_stream",
                    "completed_jobs": completed_jobs,
                    "completed_outputs": completed_outputs,
                    "stream_open": stream_open,
                }
                if poll_job is not None
                else {}
            ),
        }
        ArtifactStore.write_json(output / "episode_results.json", result)
        ArtifactStore.write_json(output / "timings.json", {"events": events.snapshot()})
        for job in jobs:
            if job.key in rows and written_rows.get(job.key) != rows[job.key]:
                ArtifactStore.write_json(job.directory / "pipeline_result.json", rows[job.key])
                written_rows[job.key] = dict(rows[job.key])
        return result

    def plan(job: UmiJob) -> None:
        token = QWEN_PLANNING.set(True)
        try:
            with events.measure("planning", job.key):
                runtime.run_qwen(job.config, job.episode, job.run_id)
        finally:
            QWEN_PLANNING.reset(token)

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
        nonlocal completed_jobs, completed_outputs
        if block and pending_reviews:
            with events.measure("render_backpressure"):
                wait(pending_reviews, return_when=FIRST_COMPLETED)
        for future in tuple(pending_reviews):
            if future.done():
                job = pending_reviews.pop(future)
                rows[job.key] = future.result()
                persist("running")
                if poll_job is not None:
                    # The on-disk result is authoritative if the coordinator dies before ack.
                    ArtifactStore.write_json(
                        job.directory / "timings.json",
                        {
                            "events": events.retire(job.key),
                            "worker_origin": events.origin,
                        },
                    )
                    ArtifactStore.write_json(
                        job.directory / "terminal_result.json",
                        {
                            "identity": job.identity,
                            "row": rows[job.key],
                        },
                    )
                    assert on_terminal is not None
                    on_terminal(job, dict(rows[job.key]))
                    completed_jobs += 1
                    completed_outputs += bool(rows[job.key].get("objects_ready"))
                    active_jobs.remove(job)
                    rows.pop(job.key)
                    written_rows.pop(job.key, None)
                    complete.pop(job.key, None)

    def run_queued(
        planners: ThreadPoolExecutor,
        renders: ThreadPoolExecutor,
        complete: dict[str, bool],
    ) -> None:
        nonlocal model_queue, peak_plans, peak_reviews, peak_episodes, fatal_error
        nonlocal peak_videos, peak_ready_videos, stream_open
        model_queue = SamTaskQueue(
            backend,
            capacity=episode_workers + video_workers,
            fatal_cuda_error=runtime.fatal_cuda_error,
            measure=events.measure,
        )
        pending: dict[Future[Any], tuple[UmiJob, str, Any]] = {}
        ready_videos: deque[tuple[UmiJob, Any]] = deque()
        ready_reviews: deque[UmiJob] = deque()
        remaining = iter(jobs) if poll_job is None else iter(())
        initial = deque(jobs) if poll_job is not None else deque()
        admission_capacity = prefetch + episode_workers + video_workers + render_buffer

        def next_job() -> UmiJob | None:
            nonlocal stream_open
            if poll_job is None:
                return next(remaining, None)
            if initial:
                return initial.popleft()
            if not stream_open or len(active_jobs) >= admission_capacity:
                return None
            try:
                job = poll_job()
            except StopIteration:
                stream_open = False
                return None
            if job is None:
                return None
            if job.key in seen_keys:
                raise ValueError("duplicate streamed job identity")
            if job.config.sam3 != first.sam3 or any(
                getattr(job.config.qwen, name) != getattr(first.qwen, name)
                for name in ("endpoint", "model", "runtime", "api_key_env")
            ):
                raise ValueError("streamed job changed SAM configuration or Qwen service")
            validate_umi_jobs([job], resume=resume)
            ArtifactStore.write_json(job.directory / "job.json", job.identity)
            complete[job.key] = resume and runtime.sam_episode_complete(
                job.config, ArtifactStore(job.config.output_root), job.run_id, job.ref
            )
            if complete[job.key]:
                rows[job.key] = {
                    **new_row(job),
                    "process_status": "skipped_complete",
                    "masks_ready": True,
                    "status": "masks_pending_render",
                }
            active_jobs.append(job)
            seen_keys.add(job.key)
            return job

        def refill_plans() -> None:
            nonlocal peak_plans
            while len(pending_plans) < prefetch:
                if fatal_error is not None:
                    break
                job = next_job()
                if job is None:
                    break
                if complete[job.key]:
                    future: Future[None] = Future()
                    future.set_result(None)
                else:
                    future = planners.submit(plan, job)
                pending_plans[future] = job
                peak_plans = max(peak_plans, len(pending_plans))

        def execute(job: UmiJob, step: str, prepared: Any = None, result: Any = None) -> Any:
            assert model_queue is not None
            event = "sam_qc_propagation_save" if step == "execute" else f"sam_{step}"
            with events.measure(event, job.key):
                if stages is not None:
                    if step == "prepare":
                        return stages.prepare(job.config, job.episode, job.run_id, model_queue)
                    if step == "propagate":
                        return stages.propagate(prepared, model_queue)
                    execution = stages.save(prepared, result)
                else:
                    execution = runtime.execute_sam_episode(
                        job.config, job.episode, job.run_id, model_queue
                    )
                return runtime.emit_sam_result(job.run_id, execution)

        def finish(job: UmiJob, row: dict[str, Any]) -> None:
            rows[job.key] = row
            # Only paths and small result records remain here. Slow review does
            # not consume producer slots or retain decoded frames/mask volumes.
            ready_reviews.append(job)
            persist("running")

        def completed_row(job: UmiJob, masks_ready: bool) -> dict[str, Any]:
            return {
                **new_row(job),
                "masks_ready": masks_ready,
                "process_status": "completed" if masks_ready else "sam_incomplete",
                "status": "masks_pending_render" if masks_ready else "failed",
            }

        producers = ThreadPoolExecutor(max_workers=episode_workers, thread_name_prefix="umi-qc")
        videos = ThreadPoolExecutor(max_workers=video_workers, thread_name_prefix="umi-video")
        saves = ThreadPoolExecutor(max_workers=1, thread_name_prefix="umi-save")
        try:
            refill_plans()
            while (
                stream_open
                or pending_plans
                or pending
                or ready_videos
                or ready_reviews
                or pending_reviews
            ):
                collect_reviews()
                refill_plans()
                for future in tuple(pending):
                    if not future.done():
                        continue
                    job, step, prepared = pending[future]
                    try:
                        result = future.result()
                    except (Exception, SystemExit) as exc:  # noqa: BLE001 - retain each failure
                        pending.pop(future)
                        error = f"{type(exc).__name__}: {exc}"
                        finish(
                            job,
                            {
                                **new_row(job),
                                "status": "failed",
                                "process_status": "sam_failed",
                                "error_stage": step,
                                "error": error,
                            },
                        )
                        if runtime.fatal_cuda_error(exc):
                            fatal_error = error
                    else:
                        pending.pop(future)
                        if step == "prepare":
                            ready_videos.append((job, result))
                            peak_ready_videos = max(peak_ready_videos, len(ready_videos))
                        elif step == "propagate":
                            pending[saves.submit(execute, job, "save", prepared, result)] = (
                                job,
                                "save",
                                prepared,
                            )
                        else:
                            finish(job, completed_row(job, result))
                    result = prepared = None
                if model_queue.fatal_error is not None:
                    model_error = model_queue.fatal_error
                    fatal_error = f"{type(model_error).__name__}: {model_error}"
                if fatal_error is not None:
                    stream_open = False
                    model_queue.close(fatal_error)
                    for planned in pending_plans:
                        planned.cancel()
                    pending_plans.clear()
                    while ready_videos:
                        job, _ = ready_videos.popleft()
                        finish(
                            job,
                            {
                                **new_row(job),
                                "status": "failed",
                                "process_status": "not_run_after_fatal_cuda",
                                "error": fatal_error,
                            },
                        )
                while ready_reviews and len(pending_reviews) < render_buffer:
                    job = ready_reviews.popleft()
                    pending_reviews[renders.submit(review, job, rows[job.key])] = job
                    peak_reviews = max(peak_reviews, len(pending_reviews))
                # A full-volume slot is held until its writer finishes. Neither
                # a slow writer nor render can stop servicing admitted SAM calls.
                active_videos = sum(
                    step in {"propagate", "save"} for _, step, _ in pending.values()
                )
                while ready_videos and active_videos < video_workers:
                    job, prepared = ready_videos.popleft()
                    pending[videos.submit(execute, job, "propagate", prepared)] = (
                        job,
                        "propagate",
                        prepared,
                    )
                    active_videos += 1
                    peak_videos = max(peak_videos, active_videos)
                active_producers = len(ready_videos) + sum(
                    step in {"prepare", "execute"} for _, step, _ in pending.values()
                )
                for planned in tuple(pending_plans):
                    if fatal_error is not None or active_producers >= episode_workers:
                        break
                    if not planned.done():
                        continue
                    job = pending_plans.pop(planned)
                    try:
                        planned.result()
                    except (Exception, SystemExit) as exc:  # noqa: BLE001 - external planner
                        finish(
                            job,
                            {
                                **new_row(job),
                                "status": "failed",
                                "process_status": "planning_failed",
                                "error": f"{type(exc).__name__}: {exc}",
                            },
                        )
                    else:
                        if complete[job.key]:
                            finish(job, rows[job.key])
                        else:
                            step = "prepare" if stages is not None else "execute"
                            pending[producers.submit(execute, job, step)] = (job, step, None)
                            active_producers += 1
                            peak_episodes = max(peak_episodes, active_producers)
                    refill_plans()
                if pending and fatal_error is None:
                    model_queue.run_one()
                elif pending or pending_plans or pending_reviews:
                    waiting: list[Future[Any]] = [*pending, *pending_plans, *pending_reviews]
                    wait(waiting, timeout=0.05, return_when=FIRST_COMPLETED)
                elif stream_open:
                    time.sleep(0.02)
        finally:
            # Wake callers before joining any pool, including on Ctrl-C. Writers
            # already admitted are drained, so completed masks survive interruption.
            model_queue.close()
            for planned in pending_plans:
                planned.cancel()
            for pool in (producers, videos):
                pool.shutdown(wait=True, cancel_futures=True)
            saves.shutdown(wait=True)
            for future, (job, step, _) in pending.items():
                if future.cancelled():
                    continue
                try:
                    result = future.result()
                except BaseException as exc:  # noqa: BLE001 - retain interrupted worker state
                    error = f"{type(exc).__name__}: {exc}"
                else:
                    if step in {"save", "execute"}:
                        rows[job.key] = completed_row(job, result)
                        continue
                    error = f"interrupted after {step} before masks were saved"
                rows[job.key] = {
                    **new_row(job),
                    "status": "interrupted_during_sam",
                    "process_status": "interrupted_during_sam",
                    "error": error,
                }

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
        if poll_job is not None or not all(complete.values()):
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
            if episode_workers > 1:
                run_queued(planners, renders, complete)
            else:
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
