"""Resident UMI workers across groups, retaining the batch's original artifact layout."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from robotwin_annotation_v2.adapters.artifact_store import ArtifactStore
from robotwin_annotation_v2.adapters.canonical_masks import read_canonical_masks
from robotwin_annotation_v2.adapters.qwen_client import FileRequestGate, ThreadLocalFileRequestGate
from robotwin_annotation_v2.adapters.robotwin_dataset import RoboTwinDataset
from robotwin_annotation_v2.application.dataset_input import resolve_dataset_input
from robotwin_annotation_v2.application.sam_workflow import (
    PROCESS_SUMMARY_FORMAT_VERSION,
    load_sam_runtime,
    render_sam_processed,
)
from robotwin_annotation_v2.application.umi_pipeline import (
    UmiJob,
    run_umi_pipeline,
    validate_umi_jobs,
)
from robotwin_annotation_v2.config import DatasetBinding, ParallelConfig, bind_dataset, load_profile
from scripts.render_batch_review_sheets import render_batch_sheets

ROOT = Path(__file__).resolve().parents[1]


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def select_jobs(
    config_path: Path,
    output: Path,
    plan: dict[str, Any],
    *,
    smoke: bool,
    only_groups: tuple[str, ...],
    tracking_batch_size: int,
) -> list[UmiJob]:
    known = {group["group"] for group in plan["groups"]}
    if set(only_groups) - known:
        raise ValueError("unknown requested UMI groups")
    rows = {row["episode_index"]: row for row in plan["episodes"]}
    hashes: dict[Path, str] = {}

    def hashed(path: Path) -> str:
        path = path.resolve()
        if path not in hashes:
            hashes[path] = file_hash(path)
        return hashes[path]

    implementation = {
        str(path.relative_to(ROOT)): hashed(path)
        for directory in (ROOT / "src/robotwin_annotation_v2", ROOT / "scripts")
        for path in sorted(directory.rglob("*.py"))
    }
    jobs = []
    for group in plan["groups"]:
        if only_groups and group["group"] not in only_groups:
            continue
        target = resolve_dataset_input(Path(group["root"])).targets[0]
        selected = tuple(group["episode_ids"][:2] if smoke else group["episode_ids"])
        profile = load_profile(config_path, mode=target.mode, target_profile=target.profile)
        profile = replace(
            profile,
            sam3=replace(
                profile.sam3, seed_image_only=True, tracking_batch_size=tracking_batch_size
            ),
            parallel=ParallelConfig(),
        )
        config = bind_dataset(
            profile,
            DatasetBinding(
                root=target.root,
                task=target.task,
                camera=target.camera,
                episode_ids=selected,
                manifest_path=target.manifest_path,
                manifest_data=target.manifest_data,
            ),
        )
        config = replace(config, output_root=output)
        dataset = RoboTwinDataset(
            target.root,
            task=target.task,
            camera=target.camera,
            manifest_path=config.dataset.manifest,
            manifest_data=target.manifest_data,
        )
        for episode in selected:
            job = UmiJob(config, episode, group["group"])
            timeline = dataset.load_episode_timeline(job.ref)
            paths = [
                timeline.paths.video,
                timeline.source,
                target.manifest_path,
                config.sam3.checkpoint,
            ]
            paths.extend(
                value
                for value in (*asdict(config.qwen).values(), *asdict(config.mask).values())
                if isinstance(value, Path) and value.is_file()
            )
            identity = {
                "files": {str(path): hashed(path) for path in paths if path is not None},
                "qwen": asdict(config.qwen),
                "mask": asdict(config.mask),
                "annotation": asdict(config.annotation),
                "sam": {"seed_image_only": True, "tracking_batch_size": tracking_batch_size},
                "implementation": implementation,
            }
            signature = hashlib.sha256(
                json.dumps(identity, sort_keys=True, default=str).encode()
            ).hexdigest()
            row = rows[episode]
            jobs.append(
                replace(
                    job,
                    input_signature=signature,
                    metadata={
                        **row,
                        "episode_uid": row["id"],
                        "source_video": str(timeline.paths.video),
                        "frame_count": timeline.frame_count,
                        "sam_options": {
                            "seed_image_only": True,
                            "tracking_batch_size": tracking_batch_size,
                        },
                    },
                )
            )
    if not jobs:
        raise ValueError("batch selection contains no episodes")
    return jobs


def partition_jobs(jobs: list[UmiJob], gpus: tuple[int, ...]) -> list[list[UmiJob]]:
    """Balance estimated frame work without a barrier between task groups."""
    shards: list[list[UmiJob]] = [[] for _ in gpus[: len(jobs)]]
    loads = [0] * len(shards)
    for job in sorted(jobs, key=lambda item: item.metadata["frame_count"], reverse=True):
        worker = min(range(len(shards)), key=lambda index: loads[index])
        config = replace(job.config, sam3=replace(job.config.sam3, gpus=(gpus[worker],)))
        shards[worker].append(replace(job, config=config))
        loads[worker] += job.metadata["frame_count"] * len(
            job.config.annotation.spec.required_role_names
        )
    return shards


def canonical_masks_ready(job: UmiJob) -> bool:
    """Check the published archive against the selected UMI episode contract."""
    try:
        bundle = read_canonical_masks(job.directory.parent / "masks.npz")
        if bundle.frame_count != job.metadata["frame_count"] or bundle.masks[2:].any():
            return False
        if all(name in job.metadata for name in ("height", "width")) and bundle.masks.shape[2:] != (
            job.metadata["height"],
            job.metadata["width"],
        ):
            return False
        required = job.config.annotation.spec.required_role_names
        for index, role in enumerate(("target", "receiver")):
            if bundle.annotation_status[index] != (
                "valid" if role in required else "not_applicable"
            ):
                return False
            if job.config.mask.qc_enabled and bundle.qc_status[index] != (
                "passed" if role in required else "not_applicable"
            ):
                return False
            if role not in required and bundle.masks[index].any():
                return False
        return True
    except (OSError, ValueError):
        return False


def render_job(job: UmiJob, row: dict[str, Any]) -> dict[str, Any]:
    video = None
    if row["masks_ready"] and not canonical_masks_ready(job):
        row = {
            **row,
            "masks_ready": False,
            "process_status": "sam_incomplete",
            "error": "canonical masks do not match the selected episode contract",
        }
    if row["masks_ready"]:
        report = render_sam_processed(
            job.config, run_id=job.run_id, episode_ids=(job.episode,), output_dir=job.directory
        )
        manifest = Path(report["manifest"])
        entry = json.loads(manifest.read_text())["episodes"][0]
        video = str(manifest.parent / entry["output_video"])
        required_files = [Path(video), *(Path(path) for path in report["review_sheets"])]
        if not report["review_sheets"] or any(
            not path.is_file() or path.stat().st_size == 0 for path in required_files
        ):
            raise FileNotFoundError("rendering did not publish the overlay and review sheets")
    ready = bool(row["masks_ready"] and video)
    manifest_path = Path(row["artifact_dir"]) / "run_manifest.json"
    if not ready and not row.get("error") and manifest_path.exists():
        manifest_data = json.loads(manifest_path.read_text())
        row["error"] = "; ".join(
            f"{role['role']}: {role['failure']}; {role.get('qc_reason')}"
            for role in manifest_data.get("roles", [])
            if role.get("failure")
        )
    row = {
        **row,
        "video": video,
        "objects_ready": ready,
        "status": "needs_temporal_review" if ready else "failed",
    }
    ledger = job.directory / "episode_results.json"
    ArtifactStore.write_json(ledger, {"episodes": [row]})
    if not ready:
        render_batch_sheets(ledger, job.config.dataset.root.parent, job.directory / "review_sheets")
    return row


def run_worker(
    jobs: list[UmiJob],
    output: Path,
    gate_paths: tuple[Path, ...],
    *,
    asynchronous: bool,
    qwen_workers: int,
    prefetch: int,
    render_buffer: int,
    cpu_threads: int,
    resume: bool,
) -> dict[str, Any]:
    # Construction and all SAM calls happen on this process's main thread.
    import cv2
    import torch

    torch.set_num_threads(cpu_threads)
    cv2.setNumThreads(cpu_threads)
    runtime = load_sam_runtime()
    original_complete = runtime.sam_episode_complete
    # Credential preflight and pipeline startup inspect the same initial artifacts.
    initial_completion: dict[str, bool] = {}

    def complete(config: Any, store: ArtifactStore, run_id: str, ref: Any) -> bool:
        job = next(job for job in jobs if job.run_id == run_id and job.ref == ref)
        if job.key not in initial_completion:
            initial_completion[job.key] = original_complete(
                config, store, run_id, ref
            ) and canonical_masks_ready(job)
        return initial_completion[job.key]

    if not resume or not all(
        complete(job.config, ArtifactStore(job.config.output_root), job.run_id, job.ref)
        for job in jobs
    ):
        key_env = jobs[0].config.qwen.api_key_env
        if key_env and not os.environ.get(key_env):
            key_file = Path(os.environ.get("QWEN_API_KEY_FILE", ROOT / "secrets/qwen_api_key.txt"))
            key = key_file.read_text().strip()
            if not key or any(char.isspace() for char in key):
                raise ValueError("invalid Qwen API key file")
            os.environ[key_env] = key
    return run_umi_pipeline(
        jobs,
        output=output,
        render=render_job,
        asynchronous=asynchronous,
        qwen_workers=qwen_workers,
        prefetch=prefetch,
        render_buffer=render_buffer,
        resume=resume,
        runtime=replace(runtime, sam_episode_complete=complete),
        request_gate=ThreadLocalFileRequestGate(gate_paths),
    )


def run_fast_batch(
    batch_dir: Path,
    config_path: Path,
    *,
    gpus: tuple[int, ...],
    qwen_workers: int = 4,
    smoke: bool = False,
    only_groups: tuple[str, ...] = (),
    asynchronous: bool = True,
    tracking_batch_size: int = 1,
    prefetch: int = 2,
    render_buffer: int = 2,
    cpu_threads: int = 8,
    output_root: Path | None = None,
) -> dict[str, Any]:
    if not gpus or len(set(gpus)) != len(gpus) or any(gpu < 0 for gpu in gpus):
        raise ValueError("distinct nonnegative GPU IDs are required")
    if min(qwen_workers, prefetch, render_buffer, cpu_threads) < 1:
        raise ValueError("worker counts and queue capacities must be positive")
    if tracking_batch_size not in (0, 1, 4):
        raise ValueError("tracking batch must be 0, 1 or 4")
    started = time.perf_counter()
    batch_dir = batch_dir.resolve()
    output = (output_root or batch_dir / ("smoke" if smoke else "runs")).resolve()
    state = output / ".pipeline"
    selection_path = state / "selection.json"
    plan = json.loads((batch_dir / "batch_plan.json").read_text())
    jobs = select_jobs(
        config_path,
        output,
        plan,
        smoke=smoke,
        only_groups=only_groups,
        tracking_batch_size=tracking_batch_size,
    )
    shards = partition_jobs(jobs, gpus)
    selection = {
        "jobs": [{"key": job.key, "identity": job.identity} for job in jobs],
        "workers": [[job.key for job in shard] for shard in shards],
        "gpus": list(gpus[: len(shards)]),
        "scheduling": {
            "asynchronous": asynchronous,
            "prefetch": prefetch,
            "qwen_workers": qwen_workers,
            "render_buffer": render_buffer,
            "cpu_threads": cpu_threads,
        },
    }
    resume = selection_path.exists()
    if resume:
        if json.loads(selection_path.read_text()) != selection:
            raise ValueError(
                "resume requires the same complete selection, inputs and implementation"
            )
    elif output.exists() and any(output.iterdir()):
        raise ValueError("existing output has no fast-pipeline identity; choose a fresh --output")
    validate_umi_jobs(jobs, resume=resume)
    ArtifactStore.write_json(selection_path, selection)
    gate = FileRequestGate.create(state / "qwen_tokens", capacity=qwen_workers)
    rows = {row["episode_index"]: dict(row) for row in plan["episodes"]}
    selected_ids = {job.episode for job in jobs}
    for job in jobs:
        rows[job.episode].update(
            {
                **job.metadata,
                "episode": job.episode,
                "task": job.ref.task,
                "camera": job.ref.camera,
                "run_id": job.run_id,
                "artifact_dir": str(job.directory.parent),
                "status": "queued",
                "process_status": "not_run",
                "masks_ready": False,
                "objects_ready": False,
            }
        )

    def save(status: str) -> dict[str, Any]:
        for worker in range(len(shards)):
            path = state / f"worker-{worker}" / "episode_results.json"
            if path.exists():
                for row in json.loads(path.read_text())["episodes"]:
                    rows[row["episode"]] = {
                        **row,
                        "pipeline_status": row["status"],
                        "status": "needs_temporal_review" if row["objects_ready"] else "failed",
                    }
        if status == "interrupted_or_failed":
            for row in rows.values():
                if row["episode_index"] in selected_ids and row["status"] == "queued":
                    row.update(status="failed", process_status="not_run_after_worker_failure")
        result: dict[str, Any] = {
            "phase": "smoke" if smoke else "full",
            "counts": dict(Counter(row["status"] for row in rows.values())),
            "episodes": sorted(rows.values(), key=lambda row: row["episode_index"]),
        }
        ArtifactStore.write_json(output / "episode_results.json", result)
        ArtifactStore.write_json(
            output / "batch_status.json",
            {
                "status": status,
                "pid": os.getpid(),
                "worker_gpus": list(gpus),
                "wall_s": time.perf_counter() - started,
                "inference_mode": "async" if asynchronous else "resident",
                "counts": result["counts"],
            },
        )
        return result

    options: dict[str, Any] = {
        "asynchronous": asynchronous,
        "qwen_workers": qwen_workers,
        "prefetch": prefetch,
        "render_buffer": render_buffer,
        "cpu_threads": cpu_threads,
        "resume": resume,
    }
    save("running")
    try:
        with ProcessPoolExecutor(
            max_workers=len(shards), mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            futures = [
                pool.submit(run_worker, shard, state / f"worker-{i}", gate.token_paths, **options)
                for i, shard in enumerate(shards)
            ]
            pending = set(futures)
            while pending:
                done, pending = wait(pending, timeout=2, return_when=FIRST_COMPLETED)
                for future in done:
                    report = future.result()
                    if report.get("status", "finished") != "finished":
                        raise RuntimeError(
                            f"UMI worker did not finish: {report.get('fatal_error') or report['status']}"
                        )
                save("running")
        result = save("rendering_groups")
        for group in plan["groups"]:
            selected = [
                row
                for row in result["episodes"]
                if row["episode_index"] in selected_ids and row.get("run_id") == group["group"]
            ]
            if not selected:
                continue
            directory = output / group["group"]
            ledger = directory / "episode_results.json"
            ArtifactStore.write_json(ledger, {"episodes": selected})
            render_batch_sheets(ledger, batch_dir / "inputs", directory / "review_sheets")
            ArtifactStore.write_json(
                directory / "process_summary.json",
                {
                    "format": PROCESS_SUMMARY_FORMAT_VERSION,
                    "run_id": group["group"],
                    "gripper_backend": None,
                    "stage_mode": "object_source_only",
                    "inference_mode": "async" if asynchronous else "resident",
                    "records": [
                        {"episode": row["episode"], "status": row["process_status"]}
                        for row in selected
                    ],
                },
            )
        return save("finished")
    except BaseException:
        save("interrupted_or_failed")
        raise
