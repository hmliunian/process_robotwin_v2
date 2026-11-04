"""Experimental finite prepared-batch stream with persistent SAM workers.

Inputs are immutable prepared batches. This does not claim production shards or run screening.
One OS lock owns the store; unresolved worker leases block restart rather than resend unknown API calls.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import multiprocessing as mp
import os
import queue
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

from robotwin_annotation_v2.adapters.artifact_store import ArtifactStore
from robotwin_annotation_v2.adapters.qwen_client import FileRequestGate, ThreadLocalFileRequestGate
from robotwin_annotation_v2.application.sam_workflow import load_sam_runtime
from robotwin_annotation_v2.application.umi_job_store import UmiJobStore
from robotwin_annotation_v2.application.umi_pipeline import UmiJob, run_umi_pipeline
from scripts.run_umi_fast_batch import (
    canonical_masks_ready,
    file_hash,
    finalize_fast_batch,
    implementation_hashes,
    render_job,
    select_jobs,
)


def resident_worker(
    index: int,
    gpu: int,
    incoming: Any,
    outgoing: Any,
    output: Path,
    gate_paths: tuple[Path, ...],
    options: dict[str, int],
    parent: int,
) -> None:
    import cv2
    import torch

    torch.set_num_threads(options["cpu_threads"])
    cv2.setNumThreads(options["cpu_threads"])
    live: dict[str, UmiJob] = {}

    def poll() -> UmiJob | None:
        if os.getppid() != parent:
            raise StopIteration
        try:
            job = incoming.get_nowait()
        except queue.Empty:
            return None
        if job is None:
            raise StopIteration
        if not isinstance(job, UmiJob):
            raise TypeError("worker queue requires UmiJob records")
        job = replace(job, config=replace(job.config, sam3=replace(job.config.sam3, gpus=(gpu,))))
        live[job.key] = job
        return job

    try:
        first = None
        while first is None:
            first = poll()
            if first is None:
                time.sleep(0.05)
        key_env = first.config.qwen.api_key_env
        if key_env and not os.environ.get(key_env):
            key_file = Path(
                os.environ.get(
                    "QWEN_API_KEY_FILE",
                    Path(__file__).resolve().parents[1] / "secrets/qwen_api_key.txt",
                )
            )
            key = key_file.read_text().strip()
            if not key or any(c.isspace() for c in key):
                raise ValueError("invalid Qwen key file")
            os.environ[key_env] = key
        runtime = load_sam_runtime()
        factory = runtime.backend_factory

        def loaded(**kwargs: Any) -> Any:
            backend = factory(**kwargs)
            outgoing.put(
                ("model_loaded", index, {"pid": os.getpid(), "gpu": gpu, "at": time.time()})
            )
            return backend

        original_complete = runtime.sam_episode_complete

        def complete(config: Any, store: Any, run_id: str, ref: Any) -> bool:
            job = next(
                j
                for j in live.values()
                if j.config.output_root == config.output_root
                and j.run_id == run_id
                and j.ref == ref
            )
            return original_complete(config, store, run_id, ref) and canonical_masks_ready(job)

        def terminal(job: UmiJob, row: dict[str, Any]) -> None:
            outgoing.put(("terminal", index, (job.key, row)))
            live.pop(job.key)

        report = run_umi_pipeline(
            [first],
            output=output,
            render=render_job,
            runtime=replace(runtime, backend_factory=loaded, sam_episode_complete=complete),
            resume=True,
            poll_job=poll,
            on_terminal=terminal,
            request_gate=ThreadLocalFileRequestGate(
                gate_paths, planning_capacity=max(1, options["qwen_workers"] // 2)
            ),
            qwen_workers=options["qwen_workers"],
            episode_workers=options["episode_workers"],
            prefetch=options["prefetch"],
            video_workers=options["video_workers"],
            render_buffer=options["render_buffer"],
        )
        if report["status"] != "finished":
            raise RuntimeError(str(report.get("fatal_error")))
    except StopIteration:
        pass
    except BaseException as exc:
        outgoing.put(("error", index, f"{type(exc).__name__}: {exc}"))
        raise
    finally:
        outgoing.put(("exited", index, {"pid": os.getpid(), "at": time.time()}))


def prepare(batch: Path, config: Path, output: Path) -> tuple[dict[str, Any], list[UmiJob]]:
    plan = json.loads((batch / "batch_plan.json").read_text())
    jobs = (
        select_jobs(config, output, plan, smoke=False, only_groups=(), tracking_batch_size=1)
        if any(g["episode_ids"] for g in plan["groups"])
        else []
    )
    return plan, [replace(j, namespace=batch.name) for j in jobs]


def publish_handoff(
    batch: Path, output: Path, plan: dict[str, Any], results: list[dict[str, Any]]
) -> None:
    rows = {r["episode_index"]: dict(r) for r in plan["episodes"]}
    for row in results:
        rows[row["episode"]] = row
    result = {
        "phase": "full",
        "counts": dict(Counter(r["status"] for r in rows.values())),
        "episodes": sorted(rows.values(), key=lambda r: r["episode_index"]),
    }
    ArtifactStore.write_json(output / "episode_results.json", result)
    ArtifactStore.write_json(
        output / "batch_status.json",
        {"status": "awaiting_finalize", "inference_mode": "continuous", "counts": result["counts"]},
    )
    ArtifactStore.write_json(
        output / ".pipeline/inference_complete.json",
        {
            "version": 1,
            "batch": str(batch),
            "output": str(output),
            "plan_sha256": file_hash(batch / "batch_plan.json"),
            "selection_sha256": file_hash(output / ".pipeline/selection.json"),
            "ledger_sha256": file_hash(output / "episode_results.json"),
            "implementation": implementation_hashes(),
        },
    )


def run_continuous(
    batches: list[Path],
    config: Path,
    output: Path,
    *,
    gpus: tuple[int, ...],
    sam_workers_per_gpu: int = 2,
    qwen_workers: int = 32,
    episode_workers: int = 4,
    prefetch: int = 4,
    video_workers: int = 2,
    render_buffer: int = 2,
    cpu_threads: int = 4,
    _worker: Callable[..., None] = resident_worker,
    _prepare: Callable[..., Any] = prepare,
    _finalize: Callable[..., Any] = finalize_fast_batch,
) -> dict[str, Any]:
    if not batches or not gpus or len(set(gpus)) != len(gpus):
        raise ValueError("provide batches and distinct GPUs")
    if (
        min(sam_workers_per_gpu, qwen_workers, prefetch, video_workers, render_buffer, cpu_threads)
        < 1
        or episode_workers < 2
    ):
        raise ValueError("positive capacities and episode_workers >= 2 required")
    batches = [p.resolve() for p in batches]
    config = config.resolve()
    output = output.resolve()
    if len({p.name for p in batches}) != len(batches):
        raise ValueError("batch names must be unique")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "coordinator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _locked_run(
            batches,
            config,
            output,
            gpus * sam_workers_per_gpu,
            {
                "qwen_workers": qwen_workers,
                "episode_workers": episode_workers,
                "prefetch": prefetch,
                "video_workers": video_workers,
                "render_buffer": render_buffer,
                "cpu_threads": cpu_threads,
            },
            _worker,
            _prepare,
            _finalize,
        )


def _locked_run(
    batches: list[Path],
    config: Path,
    output: Path,
    worker_gpus: tuple[int, ...],
    options: dict[str, int],
    worker: Callable[..., None],
    loader: Callable[..., Any],
    finalizer: Callable[..., Any],
) -> dict[str, Any]:
    manifest = {
        "batches": [str(p) for p in batches],
        "config_sha256": file_hash(config),
        "worker_gpus": list(worker_gpus),
        "options": options,
        "implementation": implementation_hashes(),
    }
    manifest_path = output / "stream.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError(
            "resume requires identical input selection, configuration and implementation"
        )
    ArtifactStore.write_json(manifest_path, manifest)
    # Refuse overlapping previous workers even when their coordinator is gone.
    workers_path = output / "workers.json"
    if workers_path.exists():
        for old in json.loads(workers_path.read_text())["workers"]:
            proc = Path("/proc") / str(old["pid"])
            if proc.exists() and (proc / "stat").read_text().split()[21] == old["start_ticks"]:
                raise RuntimeError("previous worker still alive; wait for its drain before restart")
    store = UmiJobStore(output / "jobs.sqlite3")
    context = mp.get_context("spawn")
    events = context.Queue()
    capacity = sum(
        options[k] for k in ("prefetch", "episode_workers", "video_workers", "render_buffer")
    )
    queues = [context.Queue(maxsize=capacity) for _ in worker_gpus]
    processes = []
    outstanding = [0] * len(worker_gpus)
    exited = set()
    model_loads = []
    active: dict[str, tuple[Path, dict[str, Any], list[UmiJob]]] = {}
    jobs: dict[str, UmiJob] = {}
    finalizing: dict[str, Future[Any]] = {}
    finished = {}
    cursor = 0
    loading = None
    stopping = False
    last_status = 0.0
    gate = FileRequestGate.create_shared(output / "request_gate", capacity=options["qwen_workers"])
    status: dict[str, Any] = {"status": "running", "pid": os.getpid(), "started_at": time.time()}
    try:
        for i, gpu in enumerate(worker_gpus):
            p = context.Process(
                target=worker,
                args=(
                    i,
                    gpu,
                    queues[i],
                    events,
                    output / f"workers/{i}",
                    gate.token_paths,
                    options,
                    os.getpid(),
                ),
            )
            p.start()
            processes.append(p)
            ArtifactStore.write_json(
                workers_path,
                {
                    "workers": [
                        {
                            "pid": p.pid,
                            "start_ticks": (Path("/proc") / str(p.pid) / "stat")
                            .read_text()
                            .split()[21],
                        }
                        for p in processes
                    ]
                },
            )
        with (
            ThreadPoolExecutor(max_workers=1) as preparation,
            ThreadPoolExecutor(max_workers=1) as finishing,
        ):
            while True:
                if (
                    loading is None
                    and cursor < len(batches)
                    and len(active) < 2
                    and len(finalizing) < 2
                ):
                    batch = batches[cursor]
                    cursor += 1
                    loading = (
                        batch,
                        preparation.submit(
                            loader, batch, config, output / "batches" / batch.name / "runs"
                        ),
                    )
                if loading and loading[1].done():
                    batch, future = loading
                    plan, selected = future.result()
                    loading = None
                    out = output / "batches" / batch.name / "runs"
                    identity = {
                        "plan_sha256": file_hash(batch / "batch_plan.json"),
                        "prepared_sha256": file_hash(batch / "prepared_state.json")
                        if (batch / "prepared_state.json").exists()
                        else None,
                        "jobs": [{"key": j.key, "identity": j.identity} for j in selected],
                    }
                    if (
                        implementation_hashes() != manifest["implementation"]
                        or file_hash(config) != manifest["config_sha256"]
                    ):
                        raise ValueError("implementation or configuration changed while preparing")
                    selection_path = out / ".pipeline/selection.json"
                    if (
                        selection_path.exists()
                        and json.loads(selection_path.read_text()) != identity
                    ):
                        raise ValueError("existing selection identity changed")
                    store.register(batch.name, batch, identity, selected)
                    ArtifactStore.write_json(out / ".pipeline/selection.json", identity)
                    for saved in store.leases(batch.name):
                        job = next(j for j in selected if j.key == saved["key"])
                        job.validate_output(resume=True)
                        receipt = job.directory / "terminal_result.json"
                        if not receipt.exists():
                            raise RuntimeError(
                                f"unresolved lease {job.key}; inspect prior worker/API outcome before retry"
                            )
                        terminal = json.loads(receipt.read_text())
                        if terminal["identity"] != job.identity:
                            raise ValueError("terminal receipt identity changed")
                        row = terminal["row"]
                        if row.get("objects_ready") and (
                            not canonical_masks_ready(job)
                            or not Path(row.get("video", "")).is_file()
                        ):
                            raise ValueError(
                                "terminal receipt has missing canonical/overlay output"
                            )
                        store.finish(job, row, saved["worker"])
                    if store.batch_state(batch.name) == "complete":
                        finished[batch.name] = "complete"
                    else:
                        if store.batch_state(batch.name) in ("finalizing", "finalization_failed"):
                            store.set_batch_state(batch.name, "inference")
                        active[batch.name] = (batch, plan, selected)
                        jobs.update({j.key: j for j in selected})
                while True:
                    try:
                        kind, index, data = events.get_nowait()
                    except queue.Empty:
                        break
                    if kind == "terminal":
                        key, row = data
                        store.finish(jobs[key], row, index)
                        outstanding[index] -= 1
                    elif kind == "error":
                        raise RuntimeError(f"worker {index}: {data}")
                    elif kind == "model_loaded":
                        model_loads.append({"worker": index, **data})
                    elif kind == "exited":
                        exited.add(index)
                for i, p in enumerate(processes):
                    if p.exitcode is not None and (not stopping or p.exitcode != 0):
                        raise RuntimeError(f"worker {i} exited unexpectedly ({p.exitcode})")
                for name in store.finalizable():
                    if name not in active or len(finalizing) >= 2:
                        continue
                    batch, plan, selected = active.pop(name)
                    out = output / "batches" / name / "runs"
                    recorded = json.loads((out / ".pipeline/selection.json").read_text())
                    if recorded["plan_sha256"] != file_hash(batch / "batch_plan.json") or (
                        recorded["prepared_sha256"] is not None
                        and recorded["prepared_sha256"] != file_hash(batch / "prepared_state.json")
                    ):
                        raise ValueError("prepared batch changed before handoff")
                    if implementation_hashes() != manifest["implementation"]:
                        raise ValueError("implementation changed before finalization")
                    publish_handoff(batch, out, plan, store.results(name))
                    store.set_batch_state(name, "finalizing")
                    finalizing[name] = finishing.submit(finalizer, batch, output_root=out)
                    for job in selected:
                        jobs.pop(job.key)
                for name, future in list(finalizing.items()):
                    if future.done():
                        try:
                            future.result()
                        except BaseException as exc:
                            store.set_batch_state(name, "finalization_failed", str(exc))
                            raise
                        store.set_batch_state(name, "complete")
                        finished[name] = "complete"
                        del finalizing[name]
                for key in store.ready(len(worker_gpus) * capacity):
                    if key not in jobs:
                        continue
                    i = min(range(len(worker_gpus)), key=lambda n: outstanding[n])
                    if outstanding[i] >= capacity:
                        break
                    store.lease(key, i)
                    queues[i].put_nowait(jobs[key])
                    outstanding[i] += 1
                if cursor == len(batches) and loading is None and not active and not stopping:
                    for incoming in queues:
                        incoming.put_nowait(None)
                    stopping = True
                status.update(
                    counts=store.counts(),
                    active_batches=list(active),
                    finalizing=list(finalizing),
                    finished_batches=len(finished),
                    model_loads=model_loads,
                    updated_at=time.time(),
                )
                if time.monotonic() - last_status >= 2:
                    ArtifactStore.write_json(output / "status.json", status)
                    last_status = time.monotonic()
                if stopping and not finalizing and len(exited) == len(processes):
                    break
                time.sleep(0.05)
        status["result_counts"] = store.result_counts()
        status["status"] = (
            "finished_with_failures"
            if any(k != "needs_temporal_review" and v for k, v in status["result_counts"].items())
            else "complete"
        )
        ArtifactStore.write_json(output / "status.json", status)
        return status
    except BaseException as exc:
        status.update(status="needs_attention", error=f"{type(exc).__name__}: {exc}")
        ArtifactStore.write_json(output / "status.json", status)
        raise
    finally:
        # Only children created by this coordinator. Leave leases unresolved on forced stop.
        for incoming in queues:
            try:
                incoming.put_nowait(None)
            except queue.Full:
                pass
        for p in processes:
            p.join(timeout=2)
        for p in processes:
            if p.is_alive():
                p.terminate()
        for p in processes:
            p.join(timeout=5)
        for incoming in queues:
            incoming.cancel_join_thread()
            incoming.close()
        events.cancel_join_thread()
        events.close()
        store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, action="append", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    for name, default in [
        ("sam-workers-per-gpu", 2),
        ("qwen-workers", 32),
        ("episode-workers", 4),
        ("prefetch", 4),
        ("video-workers", 2),
        ("render-buffer", 2),
        ("cpu-threads", 4),
    ]:
        parser.add_argument("--" + name, type=int, default=default)
    args = vars(parser.parse_args())
    args["batches"] = args.pop("batch_dir")
    args["gpus"] = tuple(args["gpus"])
    print(json.dumps(run_continuous(**args), ensure_ascii=False))


if __name__ == "__main__":
    main()
