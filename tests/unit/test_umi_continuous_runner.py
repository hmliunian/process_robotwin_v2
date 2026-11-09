import json
import os
import time
from dataclasses import replace

from robotwin_annotation_v2.adapters.artifact_store import ArtifactStore
from scripts.run_umi_continuous import run_continuous
from tests.unit.test_umi_pipeline import jobs_at


def cpu_worker(index, gpu, incoming, outgoing, output, gate_paths, options, parent):
    loaded = False
    while True:
        job = incoming.get(timeout=10)
        if job is None:
            break
        if not loaded:
            outgoing.put(("model_loaded", index, {"pid": os.getpid(), "gpu": gpu}))
            loaded = True
        marker = job.config.output_root.parents[2] / "finalizing-a"
        if job.namespace == "b":
            deadline = time.time() + 8
            while not marker.exists():
                assert time.time() < deadline, "next batch was never overlapped with finalizer"
                time.sleep(0.02)
            marker.with_name("b-started").touch()
        row = {
            **job.metadata,
            "episode": job.episode,
            "status": "needs_temporal_review",
            "objects_ready": True,
        }
        ArtifactStore.write_json(job.directory / "job.json", job.identity)
        ArtifactStore.write_json(job.directory / "pipeline_result.json", row)
        outgoing.put(("terminal", index, (job.key, row)))
    outgoing.put(("exited", index, {}))


def load_test_batch(batch, config, output):
    job = jobs_at(batch, 1)[0]
    job = replace(
        job,
        namespace=batch.name,
        config=replace(job.config, output_root=output),
        metadata={"id": batch.name, "episode_index": 0},
    )
    return json.loads((batch / "batch_plan.json").read_text()), [job]


def cpu_finalizer(batch, output_root):
    if batch.name == "a":
        marker = output_root.parents[2] / "finalizing-a"
        marker.touch()
        deadline = time.time() + 8
        while not marker.with_name("b-started").exists():
            assert time.time() < deadline, "finalization blocked dispatch of next batch"
            time.sleep(0.02)


def test_spawned_worker_spans_batches_and_cpu_finalizer_overlaps(tmp_path):
    batches = []
    for name in ("a", "b"):
        b = tmp_path / name
        b.mkdir()
        batches.append(b)
        (b / "batch_plan.json").write_text(
            json.dumps({"groups": [], "episodes": [{"episode_index": 0, "id": name}]})
        )
    config = tmp_path / "config.yaml"
    config.write_text("test")
    kwargs = {
        "gpus": (0,),
        "sam_workers_per_gpu": 1,
        "_worker": cpu_worker,
        "_prepare": load_test_batch,
        "_finalize": cpu_finalizer,
    }
    result = run_continuous(batches, config, tmp_path / "output", **kwargs)
    assert result["status"] == "complete" and result["counts"] == {"terminal": 2}
    assert len(result["model_loads"]) == 1 and result["finished_batches"] == 2
    again = run_continuous(batches, config, tmp_path / "output", **kwargs)
    assert again["status"] == "complete" and again["model_loads"] == []


def crash_after_publish(index, gpu, incoming, outgoing, output, gate_paths, options, parent):
    job = incoming.get(timeout=10)
    row = {**job.metadata, "episode": job.episode, "status": "failed", "objects_ready": False}
    ArtifactStore.write_json(job.directory / "job.json", job.identity)
    ArtifactStore.write_json(job.directory / "pipeline_result.json", row)
    if job.namespace == "durable":
        ArtifactStore.write_json(
            job.directory / "terminal_result.json", {"identity": job.identity, "row": row}
        )
    # Simulate process exit between durable publication and coordinator acknowledgement.


def no_finalize(batch, output_root):
    pass


def test_restart_reconciles_only_explicit_terminal_receipt(tmp_path):
    import pytest

    for name in ("durable", "partial"):
        batch = tmp_path / name
        batch.mkdir()
        (batch / "batch_plan.json").write_text(
            json.dumps({"groups": [], "episodes": [{"episode_index": 0, "id": name}]})
        )
        config = tmp_path / "config.yaml"
        config.write_text("test")
        output = tmp_path / (name + "-output")
        options = {
            "gpus": (0,),
            "sam_workers_per_gpu": 1,
            "_prepare": load_test_batch,
            "_finalize": no_finalize,
        }
        with pytest.raises(RuntimeError, match="exited unexpectedly"):
            run_continuous([batch], config, output, _worker=crash_after_publish, **options)
        if name == "durable":
            result = run_continuous([batch], config, output, _worker=cpu_worker, **options)
            assert result["status"] == "finished_with_failures"
            assert result["counts"] == {"terminal": 1} and not result["model_loads"]
        else:
            with pytest.raises(RuntimeError, match="unresolved lease"):
                run_continuous([batch], config, output, _worker=cpu_worker, **options)


def load_many(batch, config, output):
    jobs = [
        replace(
            j,
            namespace=batch.name,
            config=replace(j.config, output_root=output),
            metadata={"id": f"{batch.name}-{j.episode}", "episode_index": j.episode},
        )
        for j in jobs_at(batch, 16)
    ]
    return json.loads((batch / "batch_plan.json").read_text()), jobs


def test_sixteen_workers_reuse_across_batches_without_duplicate_jobs(tmp_path):
    batches = []
    for name in ("a", "b"):
        batch = tmp_path / name
        batch.mkdir()
        batches.append(batch)
        (batch / "batch_plan.json").write_text(
            json.dumps(
                {
                    "groups": [],
                    "episodes": [{"episode_index": i, "id": f"{name}-{i}"} for i in range(16)],
                }
            )
        )
    config = tmp_path / "config.yaml"
    config.write_text("test")
    result = run_continuous(
        batches,
        config,
        tmp_path / "output",
        gpus=tuple(range(8)),
        sam_workers_per_gpu=2,
        _worker=cpu_worker,
        _prepare=load_many,
        _finalize=cpu_finalizer,
    )
    assert result["counts"] == {"terminal": 32}
    assert len(result["model_loads"]) == 16
    assert len({m["pid"] for m in result["model_loads"]}) == 16
