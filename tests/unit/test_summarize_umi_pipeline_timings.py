from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from robotwin_annotation_v2.adapters.artifact_store import ArtifactStore
from scripts.summarize_umi_pipeline_timings import (
    overlap_seconds,
    parse_process_time,
    summarize,
    summarize_mode,
    union_seconds,
)


def event(
    job: str, stage: str, start: float, end: float, *, error: str | None = None
) -> dict[str, Any]:
    return {"job": job, "stage": stage, "start_s": start, "end_s": end, "error": error}


def row(episode: int, *, ready: bool = True, process_status: str = "completed") -> dict[str, Any]:
    return {
        "task": "task",
        "episode": episode,
        "camera": "cam_high",
        "run_id": "run",
        "frame_count": 10,
        "status": "needs_temporal_review" if ready else "failed",
        "process_status": process_status,
        "objects_ready": ready,
        "masks_ready": ready,
    }


def write_mode(
    root: Path,
    mode: str,
    rows: list[dict[str, Any]],
    events: list[dict[str, Any]],
    *,
    selection: list[str] | None = None,
) -> None:
    ArtifactStore.write_json(
        root / mode / "summary.json",
        {
            "mode": mode,
            "status": "finished",
            "whole_wall_s": 60,
            "execution_wall_s": 50,
            "preparation_s": 9,
            "model_load_count": 1,
            "model_load_attempts": 1,
            "selected_job_ids": selection
            or [f"run/task/{item['episode']:06d}/cam_high" for item in rows],
            "episodes": rows,
        },
    )
    ArtifactStore.write_json(root / mode / "timings.json", {"events": events})


def test_interval_union_and_intersection_do_not_double_count_nested_work() -> None:
    assert union_seconds([(1, 5), (2, 3), (5, 6), (9, 10)]) == 6
    assert overlap_seconds([(1, 8), (2, 3)], [(0, 5), (4, 10)]) == 7
    assert overlap_seconds([], [(1, 2)]) == 0
    with pytest.raises(ValueError, match="ordered"):
        union_seconds([(3, 1)])


@pytest.mark.parametrize(
    ("raw", "seconds"),
    (
        ("682.36user 34.81system 7:15.12elapsed 164%CPU", 435.12),
        ("Elapsed (wall clock) time (h:mm:ss or m:ss): 1:02:03.40", 3723.4),
        ("real 12.34\nuser 3.0\nsys 1.2", 12.34),
        ("elapsed_seconds=5.25", 5.25),
        ("42.5", 42.5),
    ),
)
def test_process_wall_time_supports_gnu_and_posix_formats(
    tmp_path: Path, raw: str, seconds: float
) -> None:
    path = tmp_path / "mode.time"
    path.write_text(raw)
    result = parse_process_time(path)
    assert result["status"] == "parsed" and result["wall_s"] == pytest.approx(seconds)


def test_process_wall_time_distinguishes_missing_pending_and_invalid(tmp_path: Path) -> None:
    path = tmp_path / "mode.time"
    assert parse_process_time(path)["status"] == "missing"
    path.touch()
    assert parse_process_time(path)["status"] == "pending"
    path.write_text("unexpected diagnostics")
    assert parse_process_time(path)["status"] == "unrecognized"


def test_mode_totals_keep_parent_overlap_and_batch_load_attribution_separate(
    tmp_path: Path,
) -> None:
    first, second = "run/task/000000/cam_high", "run/task/000001/cam_high"
    events = [
        event("batch", "qwen_health", 0, 1),
        event("batch", "model_load", 1, 3),
        event("batch", "sam_backend_load", 1.2, 2.9),
        event(first, "planning", 3, 4),
        event(first, "qwen_completion.semantic", 3.2, 3.9),
        event(first, "sam_qc_propagation_save", 4, 14),
        event(first, "sam_seed.text_query_masks", 4.1, 5.1),
        event(first, "qwen_completion.qc", 5.1, 6.1),
        event(first, "sam_propagate", 6.1, 13.1),
        event(second, "planning", 5, 9),
        event(second, "qwen_completion.semantic", 5.5, 8.5),
        event(second, "sam_qc_propagation_save", 14, 18),
        event(first, "render", 12, 16),
        event(second, "render", 18, 20),
    ]
    write_mode(tmp_path, "async", [row(0), row(1)], events)
    report, episodes = summarize_mode(tmp_path, "async")
    assert report["overlap"] == {
        "sam_parent_union_s": 14,
        "planning_with_sam_s": 4,
        "render_with_sam_s": 4,
        "planning_or_render_with_sam_s": 8,
    }
    assert report["stages"]["model_load"]["calls"] == 1
    assert report["stages"]["model_load"]["service_wall_s"] == 2
    assert report["batch_stages"]["model_load"]["calls"] == 1
    assert episodes[0]["model_load_calls"] == 0
    assert episodes[0]["planning_service_s"] == 1
    assert episodes[0]["sam_service_s"] == 10
    assert episodes[0]["render_service_s"] == 4
    assert episodes[0]["plan_to_render_wall_s"] == 13
    assert episodes[0]["qwen_calls"] == 2
    assert episodes[0]["baseline_case_wall_s"] is None
    assert report["throughput"]["whole"]["ready_per_minute"] == 2
    assert report["unmatched_event_jobs"] == []


def test_failed_attempts_and_unstarted_jobs_keep_full_wall_denominators(tmp_path: Path) -> None:
    failed = "run/task/000001/cam_high"
    rows = [
        {**row(0), "case_wall_s": 20},
        {**row(1, ready=False, process_status="planning_failed"), "case_wall_s": 10},
        row(2, ready=False, process_status="not_run_after_fatal_cuda"),
    ]
    events = [
        event(failed, "qwen_health", 20, 21),
        event(failed, "planning", 22, 25, error="bad response"),
        event(failed, "qwen_completion.semantic", 22, 25, error="bad response"),
        event(failed, "render", 25, 27),
    ]
    write_mode(tmp_path, "baseline", rows, events)
    report, episodes = summarize_mode(tmp_path, "baseline")
    assert [report[key] for key in ("selected", "attempted", "ready", "failed", "not_run")] == [
        3,
        2,
        1,
        1,
        1,
    ]
    assert report["throughput"]["whole"] == {"attempts_per_minute": 2, "ready_per_minute": 1}
    assert report["failure_cost"]["service_wall_s"] == 5
    assert report["failure_cost"]["interval_union_s"] == 5
    assert report["failure_cost"]["baseline_failed_case_wall_s"] == 10
    assert report["stages"]["qwen"]["error_calls"] == 1
    assert episodes[1]["plan_to_render_wall_s"] == 5
    assert episodes[1]["baseline_case_wall_s"] == 10


def test_summary_preserves_missing_modes_and_unreported_selected_jobs(tmp_path: Path) -> None:
    selected = [f"run/task/{index:06d}/cam_high" for index in range(3)]
    write_mode(tmp_path, "baseline", [row(0)], [], selection=selected)
    ArtifactStore.write_text(tmp_path / "baseline.time", "real 90.0\n")
    report = summarize(tmp_path)
    assert report["same_selection"] is None
    baseline = report["modes"]["baseline"]
    assert baseline["unreported"] == 2 and baseline["not_run"] == 0
    assert baseline["process_time"]["wall_s"] == 90
    assert baseline["throughput"]["process"]["ready_per_minute"] == pytest.approx(2 / 3)
    assert report["modes"]["resident"]["status"] == "unavailable"
    assert json.loads((tmp_path / "analysis/timing_summary.json").read_text()) == report
    with (tmp_path / "analysis/per_episode_timings.csv").open() as handle:
        records = list(csv.DictReader(handle))
    assert len(records) == 1 and records[0]["mode"] == "baseline"
    assert records[0]["baseline_case_wall_s"] == ""


def test_selection_changes_are_explicit_and_duplicate_rows_are_rejected(tmp_path: Path) -> None:
    write_mode(tmp_path, "baseline", [row(0)], [])
    write_mode(tmp_path, "resident", [row(1)], [])
    assert summarize(tmp_path)["same_selection"] is False
    write_mode(tmp_path, "async", [row(0), row(0)], [])
    with pytest.raises(ValueError, match="duplicate episode identity"):
        summarize_mode(tmp_path, "async")


def test_repeat_mode_keeps_first_trial_and_its_failure_cost_separate(tmp_path: Path) -> None:
    job = "run/task/000000/cam_high"
    failed_events = [
        event(job, "planning", 0, 1),
        event(job, "sam_qc_propagation_save", 1, 4, error="transport failed"),
        event(job, "render", 4, 5),
    ]
    write_mode(tmp_path, "async", [row(0, ready=False, process_status="sam_failed")], failed_events)
    first = summarize(tmp_path)
    write_mode(tmp_path, "async_repeat", [row(0)], [event(job, "planning", 0, 1)])
    report = summarize(tmp_path, modes=("baseline", "resident", "async", "async_repeat"))
    assert report["modes"]["async"] == first["modes"]["async"]
    assert report["modes"]["async"]["failed"] == 1
    assert report["modes"]["async"]["failure_cost"]["service_wall_s"] == 5
    assert report["modes"]["async_repeat"]["ready"] == 1
    assert report["modes"]["async_repeat"]["failure_cost"]["service_wall_s"] == 0
    with (tmp_path / "analysis/per_episode_timings.csv").open() as handle:
        records = list(csv.DictReader(handle))
    assert [(item["mode"], item["outcome"]) for item in records] == [
        ("async", "failed"),
        ("async_repeat", "ready"),
    ]
