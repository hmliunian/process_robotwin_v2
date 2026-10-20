from __future__ import annotations

import json
import multiprocessing
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import av
import numpy as np
import pytest
from PIL import Image

from robotwin_annotation_v2.adapters.artifact_store import ArtifactStore
from robotwin_annotation_v2.adapters.canonical_masks import (
    CANONICAL_INSTANCE_NAMES,
    CANONICAL_ROLES,
    read_canonical_masks,
)
from robotwin_annotation_v2.adapters.qwen_client import FileRequestGate, ThreadLocalFileRequestGate
from robotwin_annotation_v2.application.sam_workflow import SamRuntime
from robotwin_annotation_v2.application.umi_pipeline import UmiJob, run_umi_pipeline
from robotwin_annotation_v2.config import AnnotationConfig, PipelineConfig, load_config
from robotwin_annotation_v2.domain import AnnotationMode, TargetProfile
from robotwin_annotation_v2.mask_schema import MASK_FORMAT_VERSION, default_frame_encoding
from scripts import run_umi_fast_batch as batch
from scripts.merge_umi_runs import merge_runs

ROOT = Path(__file__).resolve().parents[2]


def _write_video(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=4)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        for value in (30, 60, 90, 120):
            frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), value, np.uint8), "rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _cpu_worker(
    jobs: list[UmiJob], output: Path, gate_paths: tuple[Path, ...], **options: Any
) -> dict[str, Any]:
    """Replace only model services; run the real pipeline, renderer and disk writers."""
    backend_options: dict[str, Any] = {}

    def backend_factory(**kwargs: Any) -> Any:
        backend_options.update(kwargs)
        return SimpleNamespace(shutdown=lambda: None)

    def segment(config: PipelineConfig, episode: int, run_id: str, _backend: Any) -> bool:
        job = next(job for job in jobs if job.episode == episode)
        masks = np.zeros((4, 4, 32, 32), dtype=np.bool_)
        complete = job.metadata["test_outcome"] != "sam_incomplete"
        if complete:
            masks[0, :, 8:16, 8:16] = True
        directory = ArtifactStore(config.output_root).episode_dir(run_id, job.ref)
        ArtifactStore.write_npz(
            directory / "masks.npz",
            format_version=np.asarray(MASK_FORMAT_VERSION),
            frame_count=np.asarray(4),
            masks=masks,
            instance_names=np.asarray(CANONICAL_INSTANCE_NAMES),
            roles=np.asarray(CANONICAL_ROLES),
            annotation_status=np.asarray(
                ["valid" if complete else "invalid", *("not_applicable",) * 3]
            ),
            qc_status=np.asarray(["passed" if complete else "failed", *("not_applicable",) * 3]),
            frame_encoding=default_frame_encoding(masks),
        )
        ArtifactStore.write_json(
            directory / "run_manifest.json",
            {
                "roles": [
                    {
                        "role": "target",
                        "status": "valid" if complete else "invalid",
                        "output_window": [
                            0,
                            8 if job.metadata["test_outcome"] == "render_failure" else 3,
                        ],
                        "failure": None if complete else "seed_rejected",
                        "qc_reason": None if complete else "wrong instance",
                    }
                ]
            },
        )
        return complete

    runtime = SamRuntime(
        qwen_client_factory=lambda **_: SimpleNamespace(health=dict),
        backend_factory=backend_factory,
        execution_errors=(RuntimeError,),
        emit_gripper_result=lambda *_: False,
        emit_sam_result=lambda _, result: result,
        execute_gripper_episode=lambda *_: False,
        execute_sam_episode=segment,
        fatal_cuda_error=lambda _: False,
        gripper_episode_complete=lambda *_: False,
        sam_episode_complete=lambda *_: False,
        run_qwen=lambda *_: None,
    )
    result = run_umi_pipeline(
        jobs,
        output=output,
        render=batch.render_job,
        runtime=runtime,
        request_gate=ThreadLocalFileRequestGate(gate_paths),
        **{key: value for key, value in options.items() if key != "cpu_threads"},
    )
    ArtifactStore.write_json(
        output / "runtime.json",
        {"pid": os.getpid(), **{key: str(value) for key, value in backend_options.items()}},
    )
    return result


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="CPU overlays require ffmpeg")
def test_spawned_fast_batch_renders_and_merges_real_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    group = "pick_up_32x32"
    dataset = tmp_path / "inputs" / group
    output = tmp_path / "runs"
    manifest = {
        "dataset_root": str(dataset),
        "task": group,
        "camera": "cam_high",
        "timeline_source": "video_window",
        "regression_episode_ids": [0, 1, 2],
    }
    config = load_config(ROOT / "configs/pilot_move_pillbottle_pad.yaml")
    config = replace(
        config,
        output_root=output,
        dataset=replace(
            config.dataset,
            root=dataset,
            task=group,
            manifest=dataset / "EXTRACT_MANIFEST.json",
            manifest_data=manifest,
        ),
        annotation=AnnotationConfig(AnnotationMode.TARGET_ONLY, TargetProfile.VIDEO_OBJECT),
        sam3=replace(config.sam3, seed_image_only=True, tracking_batch_size=4),
    )
    jobs = []
    for episode, outcome in enumerate(("success", "sam_incomplete", "render_failure")):
        video = dataset / "videos/chunk-000/observation.images.cam_high"
        video /= f"episode_{episode:06d}.mp4"
        _write_video(video)
        metadata = {
            "id": f"source-{episode}",
            "episode_uid": f"source-{episode}",
            "episode_index": episode,
            "group": group,
            "action_id": "A_001",
            "frame_count": 4,
            "width": 32,
            "height": 32,
            "active_arm": "right",
            "source_video": str(video),
            "test_outcome": outcome,
            "status": "converted",
        }
        jobs.append(UmiJob(config, episode, group, metadata, "fixed-input"))
    ArtifactStore.write_text(
        dataset / "meta/episodes.jsonl",
        "".join(
            json.dumps({"episode_index": job.episode, "tasks": ["Lift the object."]}) + "\n"
            for job in jobs
        ),
    )
    ArtifactStore.write_json(
        tmp_path / "batch_plan.json",
        {"groups": [{"group": group}], "episodes": [job.metadata for job in jobs]},
    )
    monkeypatch.setattr(batch, "select_jobs", lambda *_, **__: jobs)
    monkeypatch.setattr(batch, "run_worker", _cpu_worker)

    result = batch.run_fast_batch(tmp_path, config.config_path, gpus=(0,), output_root=output)

    success, incomplete, render_failed = result["episodes"]
    assert [row["status"] for row in result["episodes"]] == [
        "needs_temporal_review",
        "failed",
        "failed",
    ]
    assert result["counts"] == {"needs_temporal_review": 1, "failed": 2}
    assert incomplete["error"] == "target: seed_rejected; wrong instance"
    assert render_failed["masks_ready"] and not render_failed["objects_ready"]
    assert render_failed["pipeline_status"] == "render_failed"
    assert "video is missing requested frames [8]" in render_failed["render_error"]
    runtime = json.loads((output / ".pipeline/worker-0/runtime.json").read_text())
    assert runtime["pid"] != os.getpid()
    assert runtime["seed_image_only"] == "True" and runtime["tracking_batch_size"] == "4"
    masks_before = {job.key: (job.directory.parent / "masks.npz").read_bytes() for job in jobs}
    for job in jobs:
        assert json.loads((job.directory / "job.json").read_text()) == job.identity
        persisted = json.loads((job.directory / "pipeline_result.json").read_text())
        assert persisted["masks_ready"] == (job.episode != 1)
        assert read_canonical_masks(job.directory.parent / "masks.npz").frame_count == 4
    with av.open(success["video"]) as container:
        assert sum(1 for _ in container.decode(video=0)) == 4
    group_sheets = json.loads((output / group / "review_sheets/manifest.json").read_text())
    assert [sample["source_kind"] for sample in group_sheets["samples"]] == [
        "overlay",
        "RAW RGB",
        "RAW RGB",
    ]

    merged = tmp_path / "merged"
    assert merge_runs(output, merged, workers=1)["status"] == "finished"
    merged_rows = json.loads((merged / "episode_results.json").read_text())["episodes"]
    assert [row["status"] for row in merged_rows] == [row["status"] for row in result["episodes"]]
    assert Path(merged_rows[0]["video"]).resolve() == Path(success["video"])
    assert (merged / "index.html").is_file()
    with Image.open(merged / "pick_up/review_sheets/overall_index.jpg") as sheet:
        assert sheet.width >= 1024
    assert masks_before == {
        job.key: (job.directory.parent / "masks.npz").read_bytes() for job in jobs
    }


def _probe_shared_tokens(paths: tuple[Path, ...], connection: Any) -> None:
    gate = ThreadLocalFileRequestGate(paths)
    held = threading.Event()
    release = threading.Event()

    def owner() -> None:
        assert gate.acquire(timeout=5)
        held.set()
        try:
            assert release.wait(10)
        finally:
            gate.release()

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(owner)
        try:
            assert held.wait(5)
            connection.send(gate.acquire(blocking=False))
            assert connection.recv() == "retry"
            acquired = gate.acquire(timeout=5)
            connection.send(acquired)
            if acquired:
                gate.release()
        finally:
            release.set()
        future.result(timeout=5)
    connection.close()


def test_request_tokens_limit_threads_across_spawned_processes(tmp_path: Path) -> None:
    gate = FileRequestGate.create(tmp_path / "tokens", capacity=2)
    assert gate.acquire(blocking=False)
    parent, child = multiprocessing.get_context("spawn").Pipe()
    process = multiprocessing.get_context("spawn").Process(
        target=_probe_shared_tokens, args=(gate.token_paths, child)
    )
    process.start()
    child.close()
    try:
        assert parent.poll(10)
        assert parent.recv() is False
        contender = FileRequestGate(gate.token_paths)
        assert not contender.acquire(blocking=False)
        gate.release()
        parent.send("retry")
        assert parent.poll(10)
        assert parent.recv() is True
        process.join(timeout=10)
        assert process.exitcode == 0
        assert contender.acquire(timeout=1)
        contender.release()
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        process.close()
        parent.close()
