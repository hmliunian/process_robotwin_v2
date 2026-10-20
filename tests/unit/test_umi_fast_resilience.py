from __future__ import annotations

import json
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from robotwin_annotation_v2.adapters.artifact_store import ArtifactStore
from robotwin_annotation_v2.adapters.canonical_masks import build_canonical_mask_bundle
from robotwin_annotation_v2.application import umi_pipeline
from robotwin_annotation_v2.application.umi_pipeline import UmiJob, run_umi_pipeline
from robotwin_annotation_v2.mask_schema import default_frame_encoding
from scripts import run_umi_fast_batch as batch
from tests.unit.test_run_umi_fast_batch import fixture_jobs
from tests.unit.test_umi_pipeline import fake_runtime, jobs_at, reviewed


def publish_masks(job: UmiJob, *, shape: tuple[int, ...] = (4, 10, 2, 3)) -> None:
    masks = np.zeros(shape, dtype=bool)
    masks[:2, :, 0, 0] = True
    bundle = build_canonical_mask_bundle(
        job.directory.parent / "masks.npz",
        frame_count=shape[1],
        masks=masks,
        annotation_status=np.asarray(("valid", "valid", "not_annotated", "not_annotated")),
        qc_status=np.asarray(("passed", "passed", "not_run", "not_run")),
        frame_encoding=default_frame_encoding(masks),
    )
    ArtifactStore.write_npz(bundle.path, **bundle.to_payload())


def test_render_backpressure_interrupt_preserves_published_masks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    render_started = threading.Event()
    release_render = threading.Event()
    second_sam_done = threading.Event()
    original_wait = umi_pipeline.wait

    def execute(_config: Any, episode: int, *_: Any) -> bool:
        if episode == 1:
            assert render_started.wait(5)
            second_sam_done.set()
        return True

    def render(job: UmiJob, row: dict[str, Any]) -> dict[str, Any]:
        render_started.set()
        assert release_render.wait(5)
        return reviewed(job, row)

    def interrupt_backpressure(*args: Any, **kwargs: Any) -> Any:
        if second_sam_done.is_set():
            release_render.set()
            raise KeyboardInterrupt
        return original_wait(*args, **kwargs)

    monkeypatch.setattr(umi_pipeline, "wait", interrupt_backpressure)
    runtime, lifecycle = fake_runtime(execute_sam_episode=execute)
    try:
        with pytest.raises(KeyboardInterrupt):
            run_umi_pipeline(
                jobs_at(tmp_path, 2),
                output=tmp_path,
                render=render,
                runtime=runtime,
                render_buffer=1,
            )
    finally:
        release_render.set()
    result = json.loads((tmp_path / "episode_results.json").read_text())
    assert result["status"] == "interrupted"
    assert result["episodes"][0]["objects_ready"]
    row = result["episodes"][1]
    assert row["masks_ready"] and not row["objects_ready"]
    assert row["process_status"] == "completed"
    assert row["status"] == "interrupted_during_render"
    assert lifecycle == ["load", "shutdown"]


def test_batch_rejects_non_umi_input_before_writing_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, jobs = fixture_jobs(tmp_path)
    invalid = replace(
        jobs[0],
        config=replace(jobs[0].config, dataset=replace(jobs[0].config.dataset, manifest_data={})),
    )
    monkeypatch.setattr(batch, "select_jobs", lambda *_, **__: [invalid])
    with pytest.raises(ValueError, match="video_window"):
        batch.run_fast_batch(tmp_path, Path("config"), gpus=(0,), output_root=tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_resume_rejects_unowned_canonical_artifacts_without_writing_state(tmp_path: Path) -> None:
    job = jobs_at(tmp_path, 1)[0]
    artifact = ArtifactStore.write_text(job.directory.parent / "masks.npz", "old artifacts")
    runtime, lifecycle = fake_runtime()
    with pytest.raises(ValueError, match="no job identity"):
        run_umi_pipeline([job], output=tmp_path, render=reviewed, runtime=runtime, resume=True)
    assert artifact.read_text() == "old artifacts"
    assert not job.directory.exists() and not (tmp_path / "episode_results.json").exists()
    assert lifecycle == []


def test_resume_startup_failure_preserves_previously_valid_masks(tmp_path: Path) -> None:
    jobs = jobs_at(tmp_path, 2)
    runtime, _ = fake_runtime()
    run_umi_pipeline(jobs, output=tmp_path, render=reviewed, runtime=runtime)

    def health() -> None:
        raise RuntimeError("Qwen unavailable")

    runtime, lifecycle = fake_runtime(
        sam_episode_complete=lambda _config, _store, _run_id, ref: ref.episode_index == 0,
        qwen_client_factory=lambda **_: SimpleNamespace(health=health),
    )
    with pytest.raises(RuntimeError, match="Qwen unavailable"):
        run_umi_pipeline(jobs, output=tmp_path, render=reviewed, runtime=runtime, resume=True)
    report = json.loads((tmp_path / "episode_results.json").read_text())
    assert report["status"] == "failed"
    assert report["episodes"][0]["masks_ready"] and not report["episodes"][0]["objects_ready"]
    assert report["episodes"][0]["process_status"] == "skipped_complete"
    assert report["episodes"][1]["process_status"] == "not_run_after_startup_failure"
    assert lifecycle == []


def test_batch_worker_failure_counts_and_rows_remain_consistent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, jobs = fixture_jobs(tmp_path)
    monkeypatch.setattr(batch, "select_jobs", lambda *_, **__: jobs)
    monkeypatch.setattr(batch, "ProcessPoolExecutor", lambda **_: ThreadPoolExecutor(max_workers=1))

    def worker(*_: Any, **__: Any) -> None:
        raise RuntimeError("worker startup failed")

    monkeypatch.setattr(batch, "run_worker", worker)
    output = tmp_path / "out"
    with pytest.raises(RuntimeError, match="worker startup failed"):
        batch.run_fast_batch(tmp_path, Path("config"), gpus=(0,), output_root=output)
    report = json.loads((output / "episode_results.json").read_text())
    assert report["counts"] == dict(Counter(row["status"] for row in report["episodes"]))
    assert report["counts"] == {"failed": len(jobs)}
    assert all(row["task"] and row["artifact_dir"] and row["run_id"] for row in report["episodes"])


def test_batch_does_not_report_finished_after_a_terminal_worker_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, jobs = fixture_jobs(tmp_path)
    monkeypatch.setattr(batch, "select_jobs", lambda *_, **__: jobs)
    monkeypatch.setattr(batch, "ProcessPoolExecutor", lambda **_: ThreadPoolExecutor(max_workers=1))
    monkeypatch.setattr(
        batch, "run_worker", lambda *_, **__: {"status": "failed", "fatal_error": "fatal CUDA"}
    )
    output = tmp_path / "out"
    with pytest.raises(RuntimeError, match="fatal CUDA"):
        batch.run_fast_batch(tmp_path, Path("config"), gpus=(0,), output_root=output)
    assert json.loads((output / "batch_status.json").read_text())["status"] != "finished"


def test_resume_worker_only_checks_each_archive_once_at_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, jobs = fixture_jobs(tmp_path)
    runtime, _ = fake_runtime(sam_episode_complete=lambda *_: True)
    checked = []
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(set_num_threads=lambda _: None))
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(setNumThreads=lambda _: None))
    monkeypatch.setattr(batch, "load_sam_runtime", lambda: runtime)
    monkeypatch.setattr(batch, "canonical_masks_ready", lambda job: checked.append(job.key) or True)

    def pipeline(selected: list[UmiJob], **kwargs: Any) -> dict[str, Any]:
        for job in selected:
            assert kwargs["runtime"].sam_episode_complete(
                job.config, ArtifactStore(job.config.output_root), job.run_id, job.ref
            )
        return {"status": "finished"}

    monkeypatch.setattr(batch, "run_umi_pipeline", pipeline)
    batch.run_worker(
        jobs,
        tmp_path / "worker",
        (),
        asynchronous=True,
        qwen_workers=4,
        prefetch=2,
        render_buffer=2,
        cpu_threads=1,
        resume=True,
    )
    assert checked == [job.key for job in jobs]


@pytest.mark.parametrize("change", ("frames", "resolution", "annotation", "qc", "gripper"))
def test_canonical_readiness_checks_episode_contract(tmp_path: Path, change: str) -> None:
    _, jobs = fixture_jobs(tmp_path)
    job = replace(jobs[0], metadata={**jobs[0].metadata, "height": 2, "width": 3})
    publish_masks(job)
    assert batch.canonical_masks_ready(job)
    if change == "frames":
        job = replace(job, metadata={**job.metadata, "frame_count": 11})
    elif change == "resolution":
        job = replace(job, metadata={**job.metadata, "width": 4})
    else:
        path = job.directory.parent / "masks.npz"
        with np.load(path, allow_pickle=False) as archive:
            payload = {key: archive[key].copy() for key in archive.files}
        if change == "annotation":
            payload["annotation_status"][0] = "failed"
        elif change == "qc":
            payload["qc_status"][0] = "rejected"
        else:
            payload["masks"][2, 0, 0, 0] = True
            payload["frame_encoding"] = default_frame_encoding(payload["masks"])
        ArtifactStore.write_npz(path, **payload)
    assert not batch.canonical_masks_ready(job)


@pytest.mark.parametrize("missing", ("overlay", "review", None))
def test_render_readiness_requires_published_overlay_and_review_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: str | None
) -> None:
    _, jobs = fixture_jobs(tmp_path)
    job = jobs[0]
    publish_masks(job)
    overlay = job.directory / "rendered_videos/episode.mp4"
    sheet = job.directory / "rendered_videos/review_sheets/sheet.jpg"
    manifest = ArtifactStore.write_json(
        overlay.parent / "manifest.json", {"episodes": [{"output_video": overlay.name}]}
    )
    if missing != "overlay":
        ArtifactStore.write_text(overlay, "rendered video")
    if missing != "review":
        ArtifactStore.write_text(sheet, "rendered sheet")
    monkeypatch.setattr(
        batch,
        "render_sam_processed",
        lambda *_, **__: {"manifest": str(manifest), "review_sheets": [str(sheet)]},
    )
    row = {"masks_ready": True, "artifact_dir": str(job.directory.parent)}
    if missing:
        with pytest.raises(FileNotFoundError, match="overlay and review sheets"):
            batch.render_job(job, row)
    else:
        result = batch.render_job(job, row)
        assert result["objects_ready"] and result["masks_ready"]


def test_invalid_canonical_masks_only_render_raw_failure_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, jobs = fixture_jobs(tmp_path)
    job = jobs[0]
    publish_masks(job, shape=(4, 1, 2, 3))
    monkeypatch.setattr(
        batch, "render_sam_processed", lambda *_, **__: pytest.fail("invalid masks")
    )
    reviews = []
    monkeypatch.setattr(batch, "render_batch_sheets", lambda *args: reviews.append(args))
    result = batch.render_job(job, {"masks_ready": True, "artifact_dir": str(job.directory.parent)})
    assert not result["objects_ready"] and not result["masks_ready"]
    assert result["process_status"] == "sam_incomplete" and result["status"] == "failed"
    assert len(reviews) == 1
