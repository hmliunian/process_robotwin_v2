"""Summarize full-video UMI pipeline timings offline, preserving failure costs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from robotwin_annotation_v2.adapters.artifact_store import ArtifactStore

MODES = ("baseline", "resident", "async")
PARENT_STAGES = {"planning", "sam_qc_propagation_save", "render"}
type Interval = tuple[float, float]


def merged(intervals: Iterable[Interval]) -> list[Interval]:
    result: list[Interval] = []
    for start, end in sorted(intervals):
        if not math.isfinite(start) or not math.isfinite(end) or end < start:
            raise ValueError("event intervals must have finite, ordered endpoints")
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def union_seconds(intervals: Iterable[Interval]) -> float:
    return sum(end - start for start, end in merged(intervals))


def overlap_seconds(left: Iterable[Interval], right: Iterable[Interval]) -> float:
    first, second = merged(left), merged(right)
    i = j = 0
    overlap = 0.0
    while i < len(first) and j < len(second):
        overlap += max(0.0, min(first[i][1], second[j][1]) - max(first[i][0], second[j][0]))
        if first[i][1] <= second[j][1]:
            i += 1
        else:
            j += 1
    return overlap


def intervals(events: Iterable[dict[str, Any]]) -> list[Interval]:
    return [(float(event["start_s"]), float(event["end_s"])) for event in events]


def event_statistics(events: list[dict[str, Any]]) -> dict[str, int | float]:
    spans = intervals(events)
    return {
        "calls": len(events),
        "error_calls": sum(bool(event.get("error")) for event in events),
        "service_wall_s": sum(end - start for start, end in spans),
        "interval_union_s": union_seconds(spans),
    }


def stage_groups(events: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups = {
        name: [event for event in events if event["stage"] == stage]
        for name, stage in (
            ("planning", "planning"),
            ("sam", "sam_qc_propagation_save"),
            ("render", "render"),
            ("health", "qwen_health"),
            ("model_load", "model_load"),
            ("model_shutdown", "model_shutdown"),
            ("propagation", "sam_propagate"),
            ("save_masks", "save_masks"),
            ("save_qc", "save_qc"),
            ("render_backpressure", "render_backpressure"),
        )
    }
    # sam_backend_load is nested inside model_load; count the enclosing call once.
    if not groups["model_load"]:
        groups["model_load"] = [event for event in events if event["stage"] == "sam_backend_load"]
    for name, prefix in (("seed", "sam_seed."), ("qwen", "qwen_completion.")):
        groups[name] = [event for event in events if event["stage"].startswith(prefix)]
    for category in ("semantic", "qc", "bbox"):
        groups[f"qwen_{category}"] = [
            event for event in events if event["stage"] == f"qwen_completion.{category}"
        ]
    return groups


def parse_process_time(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "status": "missing", "wall_s": None}
    raw = path.read_text().strip()
    patterns = (
        r"([0-9:.]+)elapsed",
        r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*([0-9:.]+)",
        r"(?:^|\n)real\s+([0-9.]+)(?:\s|$)",
        r"(?:process_wall_s|elapsed_s|elapsed_seconds)\s*[:=]\s*([0-9.]+)",
        r"^([0-9.]+)$",
    )
    for pattern in patterns:
        match = re.search(pattern, raw)
        if match:
            parts = match[1].split(":")
            if len(parts) <= 3:
                wall_s = sum(
                    float(value) * 60**index for index, value in enumerate(reversed(parts))
                )
                if math.isfinite(wall_s) and wall_s >= 0:
                    return {"path": str(path), "status": "parsed", "wall_s": wall_s}
    return {"path": str(path), "status": "unrecognized" if raw else "pending", "wall_s": None}


def job_key(row: dict[str, Any]) -> str:
    return f"{row['run_id']}/{row['task']}/{int(row['episode']):06d}/{row['camera']}"


def outcome(row: dict[str, Any]) -> str:
    if row["objects_ready"]:
        return "ready"
    if str(row.get("process_status", "")).startswith("not_run"):
        return "not_run"
    status = str(row.get("status", ""))
    if status in {"queued", "running", "masks_pending_render"}:
        return "pending"
    return "failed"


def episode_timing(mode: str, row: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    key = job_key(row)
    own = [event for event in events if event["job"] == key]
    groups = stage_groups(own)
    result = {
        "mode": mode,
        "job": key,
        "task": row["task"],
        "episode": row["episode"],
        "camera": row["camera"],
        "frame_count": row.get("frame_count"),
        "status": row["status"],
        "process_status": row.get("process_status"),
        "outcome": outcome(row),
        "objects_ready": bool(row["objects_ready"]),
        "masks_ready": bool(row.get("masks_ready")),
        "baseline_case_wall_s": row.get("case_wall_s") if mode == "baseline" else None,
        "plan_to_render_wall_s": (
            max(float(event["end_s"]) for event in groups["render"])
            - min(float(event["start_s"]) for event in groups["planning"])
            if groups["planning"] and groups["render"]
            else None
        ),
    }
    for name, selected in groups.items():
        stats = event_statistics(selected)
        result[f"{name}_calls"] = stats["calls"]
        result[f"{name}_service_s"] = stats["service_wall_s"]
    return result


def summarize_mode(root: Path, mode: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    directory = root / mode
    summary_path, events_path = directory / "summary.json", directory / "timings.json"
    if not summary_path.is_file() or not events_path.is_file():
        return {
            "status": "unavailable",
            "missing": [str(path) for path in (summary_path, events_path) if not path.is_file()],
        }, []
    summary = json.loads(summary_path.read_text())
    events = json.loads(events_path.read_text())["events"]
    groups = stage_groups(events)
    episodes = [episode_timing(mode, row, events) for row in summary["episodes"]]
    keys = [row["job"] for row in episodes]
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate episode identity: {mode}")
    selection = summary["selected_job_ids"]
    if len(set(selection)) != len(selection) or set(keys) - set(selection):
        raise ValueError(f"episode rows do not match selected identities: {mode}")
    counts = Counter(row["outcome"] for row in episodes)
    counts["unreported"] = len(selection) - len(episodes)
    attempts = counts["ready"] + counts["failed"] + counts["pending"]
    whole = summary.get("whole_wall_s", summary.get("wall_s"))
    process = parse_process_time(root / f"{mode}.time")
    walls = {
        "whole": whole,
        "execution": summary.get("execution_wall_s"),
        "process": process["wall_s"],
    }
    throughput = {
        name: {
            "attempts_per_minute": 60 * attempts / wall if wall and wall > 0 else None,
            "ready_per_minute": 60 * counts["ready"] / wall if wall and wall > 0 else None,
        }
        for name, wall in walls.items()
    }
    sam = intervals(groups["sam"])
    planning, rendering = intervals(groups["planning"]), intervals(groups["render"])
    failed = {row["job"] for row in episodes if row["outcome"] == "failed"}
    failed_services = [
        event for event in events if event["job"] in failed and event["stage"] in PARENT_STAGES
    ]
    batch_events = [event for event in events if event["job"] == "batch"]
    result = {
        "status": summary["status"],
        "fatal_error": summary.get("fatal_error"),
        "selected_job_ids": selection,
        "selected": len(selection),
        "attempted": attempts,
        "ready": counts["ready"],
        "failed": counts["failed"],
        "not_run": counts["not_run"],
        "pending": counts["pending"],
        "unreported": counts["unreported"],
        "whole_wall_s": whole,
        "execution_wall_s": walls["execution"],
        "preparation_s": summary.get("preparation_s"),
        "process_time": process,
        "model_load_count": summary.get("model_load_count"),
        "model_load_attempts": summary.get("model_load_attempts"),
        "unmatched_event_jobs": sorted(
            {event["job"] for event in events} - set(selection) - {"batch"}
        ),
        "throughput": throughput,
        "stages": {name: event_statistics(selected) for name, selected in groups.items()},
        "batch_stages": {
            name: event_statistics(selected)
            for name, selected in stage_groups(batch_events).items()
        },
        "overlap": {
            "sam_parent_union_s": union_seconds(sam),
            "planning_with_sam_s": overlap_seconds(planning, sam),
            "render_with_sam_s": overlap_seconds(rendering, sam),
            "planning_or_render_with_sam_s": overlap_seconds([*planning, *rendering], sam),
        },
        "failure_cost": {
            **event_statistics(failed_services),
            "baseline_failed_case_wall_s": (
                sum(
                    float(row["baseline_case_wall_s"] or 0.0)
                    for row in episodes
                    if row["job"] in failed
                )
                if mode == "baseline"
                else None
            ),
            "baseline_not_run_case_wall_s": (
                sum(
                    float(row["baseline_case_wall_s"] or 0.0)
                    for row in episodes
                    if row["outcome"] == "not_run"
                )
                if mode == "baseline"
                else None
            ),
        },
    }
    return result, episodes


def summarize(
    root: Path, output: Path | None = None, *, modes: tuple[str, ...] = MODES
) -> dict[str, Any]:
    output = root / "analysis" if output is None else output
    mode_results = {}
    episodes = []
    for mode in modes:
        result, records = summarize_mode(root, mode)
        mode_results[mode] = result
        episodes.extend(records)
    selections = [
        result["selected_job_ids"]
        for result in mode_results.values()
        if "selected_job_ids" in result
    ]
    report = {
        "format": "robotwin_umi_pipeline_timing_summary_v1",
        "root": str(root.resolve()),
        "same_selection": all(value == selections[0] for value in selections)
        if len(selections) > 1
        else None,
        "scope": {
            "service_time": "Inclusive parent/child stage times cannot be added across groups.",
            "overlap": "Union of planning/render parents intersected with SAM parents; nested calls excluded.",
            "latency": "Planning thread start to render end; excludes earlier queuing and model startup.",
            "baseline_case": "Only baseline has case wall including health, model load and shutdown.",
            "health": "Baseline probes per episode; resident/async probe once per batch.",
            "failure_cost": "Failed episodes' planning/SAM/render service union; shared startup stays in batch_stages.",
            "throughput": "All wall denominators retain failed-attempt and shutdown/render-drain costs.",
        },
        "modes": mode_results,
    }
    ArtifactStore.write_json(output / "timing_summary.json", report)
    path = output / "per_episode_timings.csv"
    with path.open("w", newline="") as handle:
        columns = list(episodes[0]) if episodes else ["mode", "job", "outcome"]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(episodes)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--modes", nargs="+", default=MODES)
    args = parser.parse_args()
    report = summarize(args.root, args.output, modes=tuple(args.modes))
    print(json.dumps({mode: value["status"] for mode, value in report["modes"].items()}))


if __name__ == "__main__":
    main()
