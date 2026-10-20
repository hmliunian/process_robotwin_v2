"""Compare ten complete UMI videos with live Qwen and identical output requirements.

Run each mode in a fresh process. ``baseline`` reconstructs serial execution with
the frozen original SAM adapter and a model reload per episode; it uses the same
pipeline core and renderer as the other modes, rather than reproducing the old CLI.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import threading
import time
from contextlib import ExitStack, suppress
from dataclasses import asdict, replace
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path
from typing import Any

from robotwin_annotation_v2.adapters.artifact_store import ArtifactStore
from robotwin_annotation_v2.adapters.qwen_client import OpenAICompatibleQwenClient
from robotwin_annotation_v2.adapters.robotwin_dataset import RoboTwinDataset
from robotwin_annotation_v2.application.dataset_input import resolve_dataset_input
from robotwin_annotation_v2.application.sam_workflow import load_sam_runtime
from robotwin_annotation_v2.application.umi_pipeline import (
    CURRENT_JOB,
    CURRENT_STAGE,
    PipelineEvents,
    UmiJob,
    run_umi_pipeline,
)
from robotwin_annotation_v2.config import DatasetBinding, ParallelConfig, bind_dataset, load_profile
from scripts.benchmark_umi_fast import monitor_gpu, sha256
from scripts.run_umi_fast_batch import render_job

ROOT = Path(__file__).resolve().parents[1]


def prepare_jobs(args: argparse.Namespace) -> tuple[list[UmiJob], dict[str, Any]]:
    selection = json.loads(args.manifest.read_text())
    cases = selection["cases"]
    if len(cases) != 10 or len({case["task"] for case in cases}) != 10:
        raise ValueError("the benchmark requires ten distinct complete-video tasks")
    hashes: dict[Path, str] = {}

    def hashed(path: Path) -> str:
        path = path.resolve()
        if path not in hashes:
            hashes[path] = sha256(path)
        return hashes[path]

    code = {
        str(path.relative_to(ROOT)): hashed(path)
        for path in sorted((ROOT / "src/robotwin_annotation_v2").rglob("*.py"))
    }
    for path in (
        Path(__file__).resolve(),
        ROOT / "scripts/run_umi_fast_batch.py",
        ROOT / "scripts/benchmark_umi_fast.py",
        ROOT / "scripts/render_batch_review_sheets.py",
    ):
        code[str(path.relative_to(ROOT))] = hashed(path)
    jobs = []
    for case in cases:
        task = case["task"]
        if not isinstance(task, str) or Path(task).name != task or task in {".", ".."}:
            raise ValueError("manifest task must be one directory name")
        target = resolve_dataset_input(args.dataset_root / task).targets[0]
        if target.task != task or target.camera != selection["camera"]:
            raise ValueError("manifest task/camera differs from prepared input")
        if case["episode"] not in target.episode_ids:
            raise ValueError("manifest episode is absent from prepared input")
        if (target.manifest_data or {}).get("timeline_source") != "video_window":
            raise ValueError("the benchmark requires prepared video_window inputs")
        profile = load_profile(args.config, mode=target.mode, target_profile=target.profile)
        checkpoint = args.checkpoint.resolve() if args.checkpoint else profile.sam3.checkpoint
        profile = replace(
            profile,
            sam3=replace(
                profile.sam3,
                checkpoint=checkpoint,
                gpus=(args.gpu,),
                seed_image_only=args.mode != "baseline",
                tracking_batch_size=0 if args.mode == "baseline" else 1,
            ),
            parallel=ParallelConfig(),
        )
        config = bind_dataset(
            profile,
            DatasetBinding(
                root=target.root,
                task=target.task,
                camera=target.camera,
                episode_ids=(case["episode"],),
                manifest_path=target.manifest_path,
                manifest_data=target.manifest_data,
            ),
        )
        config = replace(config, output_root=args.output / "runs")
        if config.qwen.runtime != "api":
            raise ValueError("the benchmark requires the configured real Qwen API")
        key_env = config.qwen.api_key_env
        if key_env and not os.environ.get(key_env):
            key_file = os.environ.get("QWEN_API_KEY_FILE")
            if not key_file:
                raise ValueError("set the configured API key environment or QWEN_API_KEY_FILE")
            key = Path(key_file).read_text().strip()
            if not key or any(char.isspace() for char in key):
                raise ValueError("invalid Qwen API key file")
            os.environ[key_env] = key
        job = UmiJob(config, case["episode"], task)
        dataset = RoboTwinDataset(
            target.root,
            task=target.task,
            camera=target.camera,
            manifest_path=config.dataset.manifest,
            manifest_data=config.dataset.manifest_data,
        )
        timeline = dataset.load_episode_timeline(job.ref)
        source = dataset._metadata_index()[case["episode"]]
        if timeline.frame_count != case["frame_count"]:
            raise ValueError("manifest frame count differs from input")
        if source.get("source_episode_uid") != case["episode_uid"]:
            raise ValueError("manifest source UID differs from input")
        if timeline.events.active_arm != case["active_arm"]:
            raise ValueError("manifest active arm differs from input")
        height, width = dataset.frame_shape(job.ref)
        files = [timeline.paths.video, timeline.source, target.manifest_path, checkpoint]
        files.extend(
            value
            for value in (*asdict(config.qwen).values(), *asdict(config.mask).values())
            if isinstance(value, Path) and value.is_file()
        )
        identity = {
            "files": {str(path): hashed(path) for path in files if path is not None},
            "qwen": asdict(config.qwen),
            "mask": asdict(config.mask),
            "annotation": asdict(config.annotation),
            "sam": asdict(config.sam3),
            "code": code,
            "baseline_adapter_sha256": hashed(args.baseline_adapter),
        }
        identity = json.loads(json.dumps(identity, default=str))
        signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        jobs.append(
            replace(
                job,
                input_signature=signature,
                metadata={
                    **case,
                    "episode_index": case["episode"],
                    "source_video": str(timeline.paths.video),
                    "width": width,
                    "height": height,
                    "input_identity": identity,
                },
            )
        )
    return jobs, selection


def instrument(events: PipelineEvents, patches: ExitStack, adapter: Any, gpu: int) -> None:
    """Record inclusive stages with the core's thread-local job attribution."""
    import torch

    from robotwin_annotation_v2.application import episode_pipeline
    from robotwin_annotation_v2.pipeline import mask_qc

    def wrap(owner: Any, name: str, stage: str, *, cuda: bool = False) -> None:
        original = getattr(owner, name)

        @wraps(original)
        def measured(*args: Any, **kwargs: Any) -> Any:
            kind = stage
            if kind == "qwen_completion":
                parent = CURRENT_STAGE.get()
                kind += "." + (
                    "semantic" if parent == "planning" else "bbox" if parent == "bbox" else "qc"
                )
            if cuda:
                torch.cuda.synchronize(gpu)
            with events.measure(kind):
                try:
                    result = original(*args, **kwargs)
                except BaseException:
                    if cuda:
                        with suppress(Exception):
                            torch.cuda.synchronize(gpu)
                    raise
                if cuda:
                    torch.cuda.synchronize(gpu)
                return result

        setattr(owner, name, measured)
        patches.callback(setattr, owner, name, original)

    wrap(OpenAICompatibleQwenClient, "complete", "qwen_completion")
    wrap(mask_qc, "_run_bbox_qc_at_seed", "bbox")
    wrap(episode_pipeline, "save_mask_qc_artifacts", "save_qc")
    wrap(episode_pipeline, "save_sam_artifacts", "save_masks")
    for name in ("text_query_masks", "box_mask", "text_box_mask"):
        wrap(adapter, name, f"sam_seed.{name}", cuda=True)
    wrap(adapter, "propagate_mask", "sam_propagate", cuda=True)


def episode_timings(job: UmiJob, events: list[dict[str, Any]]) -> dict[str, Any]:
    own = [event for event in events if event["job"] == job.key]

    def total(stage: str) -> float:
        return sum(float(event["wall_s"]) for event in own if event["stage"] == stage)

    planning = [event for event in own if event["stage"] == "planning"]
    renders = [event for event in own if event["stage"] == "render"]
    qwen = [event for event in own if event["stage"].startswith("qwen_completion.")]
    return {
        "planning_wall_s": total("planning"),
        "sam_wall_s": total("sam_qc_propagation_save"),
        "render_wall_s": total("render"),
        "plan_to_render_wall_s": (
            max(event["end_s"] for event in renders) - min(event["start_s"] for event in planning)
            if planning and renders
            else None
        ),
        "qwen_calls": len(qwen),
        "qwen_wall_s": sum(event["wall_s"] for event in qwen),
    }


def main() -> None:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-adapter", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--mode", choices=("baseline", "resident", "async"), required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/process.yaml")
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error("GPU must be nonnegative")
    args.output = args.output.resolve()
    if args.output.exists():
        parser.error("output exists; choose a fresh directory")

    import cv2
    import torch

    torch.set_num_threads(8)
    cv2.setNumThreads(8)
    torch.cuda.set_device(args.gpu)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    jobs, selection = prepare_jobs(args)
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "format": "robotwin_umi_complete_video_benchmark_v1",
        "mode": args.mode,
        "label": (
            "frozen original SAM, per-episode reload, reconstructed controlled serial core"
            if args.mode == "baseline"
            else "S1+T1 resident SAM, " + ("async" if args.mode == "async" else "serial")
        ),
        "started_utc": datetime.now(UTC).isoformat(),
        "pid": os.getpid(),
        "gpu": args.gpu,
        "gpu_name": torch.cuda.get_device_name(args.gpu),
        "torch": torch.__version__,
        "cpu_threads": 8,
        "tf32": {"cuda_matmul": True, "cudnn": True},
        "qwen_workers": 4,
        "prefetch": 2,
        "render_buffer": 2,
        "selection": selection,
        "manifest_sha256": sha256(args.manifest),
        "config_sha256": sha256(args.config),
        "baseline_adapter_sha256": sha256(args.baseline_adapter),
        "jobs": [{"key": job.key, "identity": job.identity, **job.metadata} for job in jobs],
        "preparation_s": time.perf_counter() - started,
        "timing_scope": (
            "whole wall includes runner preparation, live API, model lifetime, failures, "
            "canonical saves and all rendering; execution wall excludes preparation; "
            "events are inclusive and overlapping, not additive; Python startup is external"
        ),
        "health_policy": "once per episode in baseline, once per batch in resident/async",
        "latency_scope": (
            "plan_to_render starts when the planning thread begins, excluding startup; "
            "baseline case_wall also includes health, model load and shutdown"
        ),
        "completion_order_scope": "normal renderer returns, including failed-mask reviews",
    }
    ArtifactStore.write_json(args.output / "metadata.json", metadata)
    events = PipelineEvents(origin=started)
    model_loads = 0
    model_load_attempts = 0
    rows: dict[str, dict[str, Any]] = {}
    completion_order: list[str] = []
    status = "running"
    fatal_error: str | None = None
    execution_started = time.perf_counter()
    execution_finished: float | None = None
    stop_monitor = threading.Event()
    monitor = threading.Thread(
        target=monitor_gpu,
        args=(args.gpu, stop_monitor, args.output / "gpu.log"),
        daemon=True,
    )
    monitor.start()

    def save() -> None:
        snapshot = events.snapshot()
        result = {
            "mode": args.mode,
            "status": status,
            "fatal_error": fatal_error,
            "whole_wall_s": time.perf_counter() - started,
            "wall_s": time.perf_counter() - started,
            "execution_wall_s": (execution_finished or time.perf_counter()) - execution_started,
            "preparation_s": metadata["preparation_s"],
            "model_load_count": model_loads,
            "model_load_attempts": model_load_attempts,
            "completed": sum(bool(row["objects_ready"]) for row in rows.values()),
            "completion_order": list(completion_order),
            "selected_job_ids": [job.key for job in jobs],
            "episodes": [
                {**rows[job.key], **episode_timings(job, snapshot)}
                for job in jobs
                if job.key in rows
            ],
        }
        ArtifactStore.write_json(args.output / "timings.json", {"events": snapshot})
        ArtifactStore.write_json(args.output / "summary.json", result)

    def collect(directory: Path) -> None:
        path = directory / "episode_results.json"
        if path.is_file():
            for row in json.loads(path.read_text())["episodes"]:
                job = next(
                    job
                    for job in jobs
                    if job.run_id == row["run_id"] and job.episode == row["episode"]
                )
                rows[job.key] = row

    def render(job: UmiJob, row: dict[str, Any]) -> dict[str, Any]:
        result = render_job(job, row)
        completion_order.append(job.key)
        return result

    try:
        if args.mode == "baseline":
            spec = importlib.util.spec_from_file_location(
                "frozen_sam_baseline", args.baseline_adapter
            )
            if spec is None or spec.loader is None:
                raise ValueError("cannot import frozen baseline adapter")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            adapter: Any = module.Sam3Adapter
        else:
            from robotwin_annotation_v2.adapters.sam3_adapter import Sam3Adapter

            adapter = Sam3Adapter

        def factory(**kwargs: Any) -> Any:
            nonlocal model_loads, model_load_attempts
            model_load_attempts += 1
            with events.measure("sam_backend_load"):
                backend = adapter(**kwargs)
                torch.cuda.synchronize(args.gpu)
            model_loads += 1
            return backend

        runtime = replace(load_sam_runtime(), backend_factory=factory)
        with ExitStack() as patches:
            instrument(events, patches, adapter, args.gpu)
            groups = [[job] for job in jobs] if args.mode == "baseline" else [jobs]
            for index, group in enumerate(groups):
                directory = args.output / "pipeline"
                if args.mode == "baseline":
                    directory /= f"job-{index:02d}"
                case_started = time.perf_counter()
                token = CURRENT_JOB.set(group[0].key if args.mode == "baseline" else "batch")
                try:
                    result = run_umi_pipeline(
                        group,
                        output=directory,
                        render=render,
                        asynchronous=args.mode == "async",
                        prefetch=2,
                        qwen_workers=4,
                        render_buffer=2,
                        runtime=runtime,
                        events=events,
                    )
                    if result["status"] != "finished":
                        fatal_error = result.get("fatal_error") or result["status"]
                        status = "failed"
                except Exception as exc:  # noqa: BLE001 - keep all failed attempts
                    fatal_error = f"{type(exc).__name__}: {exc}"
                    status = "failed"
                finally:
                    CURRENT_JOB.reset(token)
                    collect(directory)
                    if args.mode == "baseline" and group[0].key in rows:
                        rows[group[0].key]["case_wall_s"] = time.perf_counter() - case_started
                    save()
                if status == "failed":
                    break
        if status == "running":
            status = "finished"
    except BaseException as exc:
        fatal_error = f"{type(exc).__name__}: {exc}"
        status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        raise
    finally:
        execution_finished = time.perf_counter()
        stop_monitor.set()
        monitor.join(timeout=10)
        for job in jobs:
            if job.key not in rows:
                rows[job.key] = {
                    **job.metadata,
                    "task": job.ref.task,
                    "episode": job.episode,
                    "camera": job.ref.camera,
                    "run_id": job.run_id,
                    "artifact_dir": str(job.directory.parent),
                    "objects_ready": False,
                    "masks_ready": False,
                    "status": "failed",
                    "process_status": "not_run_after_benchmark_failure",
                    "error": fatal_error,
                }
        save()
        metadata.update(
            status=status,
            finished_utc=datetime.now(UTC).isoformat(),
            whole_wall_s=time.perf_counter() - started,
            execution_wall_s=execution_finished - execution_started,
            model_load_count=model_loads,
            model_load_attempts=model_load_attempts,
        )
        ArtifactStore.write_json(args.output / "metadata.json", metadata)
    if status != "finished":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
