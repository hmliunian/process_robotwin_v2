"""Merge finished UMI resolution groups into task folders without copying source data."""

from __future__ import annotations

import argparse
import html
import json
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from robotwin_annotation_v2.adapters.artifact_store import ArtifactStore
from robotwin_annotation_v2.adapters.rendering import file_sha256
from scripts.render_batch_review_sheets import MOMENTS, STATUS_STYLE, render_batch_sheets

SCREENING_STATUSES = {
    "excluded_bimanual",
    "excluded_no_action",
    "excluded_review",
    "needs_arm_review",
}


def _load_rows(source: Path) -> list[dict[str, Any]]:
    status = json.loads((source / "batch_status.json").read_text())
    if status["status"] != "finished":
        raise ValueError("source batch must be finished before merging")
    report = json.loads((source / "episode_results.json").read_text())
    rows = report["episodes"]
    actions: dict[str, str] = {}
    for key in ("id", "episode_index"):
        if len({row[key] for row in rows}) != len(rows):
            raise ValueError(f"duplicate {key} in source results")
    for row in rows:
        if row["status"] in SCREENING_STATUSES:
            continue
        if row["status"] not in STATUS_STYLE:
            raise ValueError(f"nonterminal episode: {row['id']}")
        match = re.fullmatch(r"([A-Za-z0-9_-]+)_(\d+)x(\d+)", row["group"])
        if not match or tuple(map(int, match.groups()[1:])) != (row["width"], row["height"]):
            raise ValueError(f"invalid resolution group: {row['group']}")
        task = match[1]
        if actions.setdefault(row["action_id"], task) != task:
            raise ValueError(f"conflicting tasks for {row['action_id']}")
        ready = row["status"] != "failed"
        if bool(row["objects_ready"]) != ready or row["episode"] != row["episode_index"]:
            raise ValueError(f"inconsistent episode: {row['id']}")
        if row["frame_count"] < 1:
            raise ValueError(f"empty episode: {row['id']}")
        required = [row["source_video"]]
        if ready:
            if not row.get("video") or not row.get("artifact_dir"):
                raise ValueError(f"missing complete output: {row['id']}")
            required += [row["video"], str(Path(row["artifact_dir"]) / "masks.npz")]
        elif row.get("video"):
            required.append(row["video"])
        for value in required:
            path = Path(value)
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"missing or empty source file: {path}")
    if not actions or any(row["action_id"] not in actions for row in rows):
        raise ValueError("cannot map every action to a processed task")
    for row in rows:
        row["source_task"] = row.get("task")
        row["task"] = actions[row["action_id"]]
    return sorted(rows, key=lambda row: row["episode_index"])


def _link(source: Path, target: Path) -> str:
    source = source.resolve(strict=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source, target_is_directory=source.is_dir())
    return str(target)


def _anchor(path: Path | str, base: Path, label: str) -> str:
    relative = os.path.relpath(path, base)
    return f'<a href="{html.escape(relative, quote=True)}">{html.escape(label)}</a>'


def _page(title: str, content: str) -> str:
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{html.escape(title)}</title>"
        "<style>body{font:15px system-ui;margin:28px;color:#172b3a}"
        "table{border-collapse:collapse;width:100%}td,th{padding:9px;border-bottom:1px solid #ddd;"
        "text-align:left;vertical-align:top}a{color:#1765a6}"
        "img{max-width:100%;height:auto}code{overflow-wrap:anywhere}</style>"
        f"<h1>{html.escape(title)}</h1>"
        "<p>Original resolutions and statuses are preserved. REVIEW outputs still need visual "
        "review; FAILED clips use raw RGB in contact sheets.</p>"
        f"{content}</html>"
    )


def _merge_task(task: str, rows: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    task_dir = output / task
    task_dir.mkdir()
    processed = [row for row in rows if row["status"] in STATUS_STYLE]
    for row in processed:
        row["review_label"] = f"{row['width']}x{row['height']}"
        episode = f"episode_{row['episode']:06d}"
        artifact = Path(row["artifact_dir"]) if row.get("artifact_dir") else None
        row["source_artifact_dir"] = row.get("artifact_dir")
        row["artifact_dir"] = None
        if artifact and artifact.is_dir():
            _link(artifact.parent, task_dir / episode)
            row["artifact_dir"] = str(task_dir / episode / artifact.name)
        row["source_input_video"] = row["source_video"]
        row["source_video"] = _link(
            Path(row["source_video"]), task_dir / "source_videos" / f"{episode}.mp4"
        )
        row["source_overlay"] = row.get("video")
        if row.get("video"):
            row["video"] = _link(
                Path(row["video"]), task_dir / "rendered_videos" / Path(row["video"]).name
            )
    result_path = task_dir / "episode_results.json"
    ArtifactStore.write_json(
        result_path,
        {
            "task": task,
            "counts": dict(Counter(r["status"] for r in processed)),
            "episodes": processed,
        },
    )
    render_batch_sheets(result_path, output, task_dir / "review_sheets")
    _link(task_dir / "review_sheets", output / "review_sheets" / task)
    sheet_links = " | ".join(
        _anchor(task_dir / "review_sheets" / f"{name}.jpg", task_dir, name) for name in MOMENTS
    )
    table = []
    for row in rows:
        links = [
            _anchor(row[key], task_dir, label)
            for key, label in (
                ("video", "overlay"),
                ("source_video", "RGB"),
                ("artifact_dir", "artifacts"),
            )
            if row.get(key)
        ]
        size = f"{row['width']}x{row['height']}" if "width" in row else "—"
        cells = [
            row["episode_index"],
            row["id"],
            size,
            row["status"],
            row.get("error") or row.get("reason") or "",
        ]
        table.append(
            "<tr>"
            + "".join(f"<td>{html.escape(str(value))}</td>" for value in cells)
            + f"<td>{' | '.join(links)}</td></tr>"
        )
    ArtifactStore.write_text(
        task_dir / "index.html",
        _page(
            task,
            f'<p><a href="../index.html">All tasks</a> | {sheet_links}</p>'
            "<table><tr><th>Episode</th><th>Source ID</th><th>Resolution</th><th>Status</th>"
            "<th>Reason</th><th>Files</th></tr>" + "".join(table) + "</table>",
        ),
    )
    return {
        "task": task,
        "action_id": rows[0]["action_id"],
        "source_groups": sorted({r["group"] for r in processed}),
        "candidate_count": len(rows),
        "processed_count": len(processed),
        "counts": dict(Counter(row["status"] for row in rows)),
    }


def merge_runs(source: Path, output: Path, *, workers: int = 4) -> dict[str, Any]:
    """Create a fresh merged view and render each task independently on CPU."""
    if workers < 1:
        raise ValueError("workers must be positive")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    source, output = source.resolve(strict=True), output.resolve()
    if source in output.parents:
        raise ValueError("merged output must be outside source runs")
    source_index = source / "episode_results.json"
    fingerprint = file_sha256(source_index)
    rows = _load_rows(source)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["task"]].append(row)
    output.mkdir(parents=True)
    progress: dict[str, Any] = {
        "status": "running",
        "source_runs": str(source),
        "source_results_sha256": fingerprint,
        "task_count": len(groups),
        "counts": dict(Counter(row["status"] for row in rows)),
        "tasks": [],
    }

    def save() -> None:
        progress["updated_at"] = datetime.now(UTC).isoformat()
        ArtifactStore.write_json(output / "merge_status.json", progress)

    save()
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_merge_task, task, records, output)
                for task, records in sorted(groups.items())
            ]
            for future in as_completed(futures):
                task_summary = future.result()
                progress["tasks"].append(task_summary)
                save()
                print(json.dumps(task_summary), flush=True)
        if file_sha256(source_index) != fingerprint:
            raise ValueError("source results changed during merge")
        progress["tasks"].sort(key=lambda task: task["action_id"])
        ArtifactStore.write_json(
            output / "episode_results.json",
            {
                "phase": "merged",
                "source_runs": str(source),
                "counts": progress["counts"],
                "episodes": rows,
            },
        )
        table, gallery = [], []
        for task in progress["tasks"]:
            name = task["task"]
            sheets = " | ".join(
                _anchor(output / name / "review_sheets" / f"{moment}.jpg", output, moment)
                for moment in MOMENTS
            )
            counts = task["counts"]
            table.append(
                f"<tr><td>{task['action_id']}</td><td>"
                + _anchor(output / name / "index.html", output, name)
                + f"</td><td>{task['processed_count']}</td>"
                f"<td>{counts.get('needs_temporal_review', 0) + counts.get('auto_complete', 0)}</td>"
                f"<td>{counts.get('failed', 0)}</td>"
                f"<td>{task['candidate_count'] - task['processed_count']}</td>"
                f"<td>{sheets}</td></tr>"
            )
            gallery.append(
                f"<h2>{html.escape(name)}</h2><p>"
                + " | ".join(
                    _anchor(
                        output / "review_sheets" / name / f"{moment}.jpg",
                        output / "review_sheets",
                        moment,
                    )
                    for moment in MOMENTS
                )
                + f'</p><a href="{name}/overall_index.jpg">'
                f'<img loading="lazy" src="{name}/overall_index.jpg" alt="{name}"></a>'
            )
        ArtifactStore.write_text(
            output / "index.html",
            _page(
                "UMI tasks — merged resolutions",
                '<p><a href="review_sheets/index.html">Review sheet gallery</a> | '
                '<a href="episode_results.json">All episode records</a></p>'
                "<table><tr><th>Action</th><th>Task</th><th>Processed</th><th>Outputs</th>"
                "<th>Failed</th><th>Screened out / pending review</th><th>Review sheets</th></tr>"
                + "".join(table)
                + "</table>",
            ),
        )
        ArtifactStore.write_text(
            output / "review_sheets/index.html",
            _page(
                "Merged task review sheets",
                '<p><a href="../index.html">All tasks</a></p>' + "".join(gallery),
            ),
        )
        ArtifactStore.write_text(
            output / "README.md",
            "# Merged UMI tasks\n\nOpen `index.html` for tasks and `review_sheets/index.html` "
            "for the combined review gallery.\n\n"
            "Each task contains episode directory links, `rendered_videos/`, `source_videos/`, "
            "a processed-only `episode_results.json`, and three merged review sheets. "
            "The root episode index and task HTML pages also retain screening exclusions "
            "and pending arm reviews.\n\n"
            "Source episode IDs, resolutions, statuses and run IDs are retained. "
            "The source data is referenced by symlinks and must remain available; "
            "linked manifests retain their original task names and paths. "
            "No masks or source videos were resized or modified. "
            "Failed episodes may contain partial masks and are not successful outputs.\n",
        )
        progress["status"] = "finished"
        save()
    except Exception as exc:
        progress.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save()
        raise
    return progress


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    merge_runs(args.source, args.output, workers=args.workers)


if __name__ == "__main__":
    main()
