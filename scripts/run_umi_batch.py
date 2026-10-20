"""Run prepared UMI groups through the existing GPU worker scheduler and publish a ledger."""

from __future__ import annotations

import argparse
import html
import json
import os
from collections import Counter
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from robotwin_annotation_v2.adapters.artifact_store import ArtifactStore
from robotwin_annotation_v2.application.dataset_input import resolve_dataset_input
from robotwin_annotation_v2.application.dataset_runtime import process_dataset
from robotwin_annotation_v2.config import DatasetBinding, ParallelConfig, bind_dataset, load_profile
from robotwin_annotation_v2.models import EpisodeRef
from robotwin_annotation_v2.terminal_ui import create_process_ui
from scripts.render_batch_review_sheets import render_batch_sheets

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPLETE = {"completed", "skipped_complete"}


def _read(path: Path) -> Any:
    return json.loads(path.read_text())


def run_batch(
    batch_dir: Path,
    config_path: Path,
    *,
    gpus: tuple[int, ...],
    qwen_workers: int = 4,
    smoke: bool = False,
    only_groups: tuple[str, ...] = (),
    inference_mode: str = "legacy",
    tracking_batch_size: int = 1,
    prefetch: int = 2,
    render_buffer: int = 2,
    cpu_threads: int = 8,
    output_root: Path | None = None,
) -> dict[str, Any]:
    if inference_mode not in {"legacy", "resident", "async"}:
        raise ValueError("inference mode must be legacy, resident or async")
    if inference_mode != "legacy":
        from scripts.run_umi_fast_batch import run_fast_batch

        result = run_fast_batch(
            batch_dir,
            config_path,
            gpus=gpus,
            qwen_workers=qwen_workers,
            smoke=smoke,
            only_groups=only_groups,
            asynchronous=inference_mode == "async",
            tracking_batch_size=tracking_batch_size,
            prefetch=prefetch,
            render_buffer=render_buffer,
            cpu_threads=cpu_threads,
            output_root=output_root,
        )
        output = (output_root or batch_dir / ("smoke" if smoke else "runs")).resolve()
        write_index(output, result)
        return result
    batch_dir = batch_dir.resolve()
    plan = _read(batch_dir / "batch_plan.json")
    output = (output_root or batch_dir / ("smoke" if smoke else "runs")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "episode_results.json"
    results = {r["episode_index"]: r for r in plan["episodes"]}
    if result_path.exists():
        for row in _read(result_path)["episodes"]:
            if row.get("process_status"):
                results[row["episode_index"]] = row
    progress = {
        "phase": "smoke" if smoke else "full",
        "pid": os.getpid(),
        "worker_gpus": list(gpus),
        "status": "running",
    }

    def save() -> dict[str, Any]:
        counts = dict(Counter(r["status"] for r in results.values()))
        index = {
            "phase": progress["phase"],
            "counts": counts,
            "episodes": sorted(results.values(), key=lambda row: row["episode_index"]),
        }
        ArtifactStore.write_json(result_path, index)
        ArtifactStore.write_json(
            output / "batch_status.json",
            {
                **progress,
                "updated_at": datetime.now(UTC).isoformat(),
                "counts": counts,
            },
        )
        return index

    save()
    known_groups = {g["group"] for g in plan["groups"]}
    if set(only_groups) - known_groups:
        raise ValueError("unknown requested UMI groups")
    for group in plan["groups"]:
        run_id = group["group"]
        if only_groups and run_id not in only_groups:
            continue
        target = resolve_dataset_input(Path(group["root"])).targets[0]
        selected = tuple(group["episode_ids"][:2] if smoke else group["episode_ids"])
        profile = load_profile(config_path, mode=target.mode, target_profile=target.profile)
        profile = replace(profile, parallel=ParallelConfig(gpus[: len(selected)], qwen_workers))
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
        if config.qwen.api_key_env and not os.environ.get(config.qwen.api_key_env):
            key = (PROJECT_ROOT / "secrets/qwen_api_key.txt").read_text().strip()
            if not key or any(c.isspace() for c in key):
                raise ValueError("invalid Qwen API key file")
            os.environ[config.qwen.api_key_env] = key
        progress["current_group"] = run_id
        save()

        def terminal(episode_id: int, status: str) -> None:
            results[episode_id].update(
                process_status=status,
                status="masks_pending_render" if status in COMPLETE else "failed",
            )
            save()

        reporter = create_process_ui("plain", verbose=False)
        try:
            summary = process_dataset(
                config,
                dataset_root=target.root,
                task=target.task,
                camera=target.camera,
                output_root=output,
                run_id=run_id,
                episode_ids=selected,
                object_source_only=True,
                reporter=reporter,
                episode_terminal_callback=terminal,
            )
            reporter.finish(summary)
        except Exception as exc:  # noqa: BLE001 - retain failed groups, continue other actions
            summary = {
                "run_id": run_id,
                "error": f"{type(exc).__name__}: {exc}",
                "records": [
                    {"episode": e, "status": "failed", "error": str(exc)} for e in selected
                ],
            }
            ArtifactStore.write_json(output / run_id / "batch_failure.json", summary)
        finally:
            reporter.close()
        records = {r["episode"]: r for r in summary["records"] if "episode" in r}
        render = summary.get("render")
        videos = {}
        if render and Path(render["manifest"]).exists():
            render_path = Path(render["manifest"])
            videos = {
                r["episode_index"]: str(render_path.parent / r["output_video"])
                for r in _read(render_path)["episodes"]
            }
        store = ArtifactStore(output)
        for episode_id in selected:
            row = results[episode_id]
            record = records.get(episode_id, {"status": "not_run"})
            ref = EpisodeRef(target.task, episode_id, target.camera)
            artifact_dir = store.episode_dir(run_id, ref)
            ready = record["status"] in COMPLETE and (artifact_dir / "masks.npz").is_file()
            failures = []
            run_manifest = artifact_dir / "run_manifest.json"
            if not ready and run_manifest.exists():
                failures = [
                    f"{role['role']}: {role['failure']}; {role.get('qc_reason')}"
                    for role in _read(run_manifest).get("roles", [])
                    if role.get("failure")
                ]
            video = videos.get(episode_id)
            chunk = f"chunk-{episode_id // 1000:03d}"
            row.update(
                episode=episode_id,
                episode_uid=row["id"],
                task=target.task,
                run_id=run_id,
                status="needs_temporal_review" if ready and video else "failed",
                process_status=record["status"],
                objects_ready=bool(ready and video),
                error=record.get("error")
                or summary.get("error")
                or "; ".join(failures)
                or ("canonical masks or overlay unavailable" if not ready or not video else None),
                video=video,
                artifact_dir=str(artifact_dir),
                source_video=str(
                    target.root
                    / "videos"
                    / chunk
                    / f"observation.images.{target.camera}"
                    / f"episode_{episode_id:06d}.mp4"
                ),
                review_note="Model-screened single arm; full video masks require visual review.",
            )
        save()
        # Each group is small enough for review sheets; failed episodes retain their RGB evidence.
        group_result = output / run_id / "episode_results.json"
        ArtifactStore.write_json(group_result, {"episodes": [results[e] for e in selected]})
        try:
            render_batch_sheets(
                group_result, batch_dir / "inputs", output / run_id / "review_sheets"
            )
        except Exception as exc:  # noqa: BLE001 - review rendering must not erase mask results
            ArtifactStore.write_json(
                output / run_id / "review_sheet_failure.json", {"error": str(exc)}
            )
        print(
            json.dumps(
                {"group": run_id, "counts": dict(Counter(results[e]["status"] for e in selected))}
            ),
            flush=True,
        )
    progress["status"] = "finished"
    index = save()
    write_index(output, index)
    return index


def write_index(output: Path, index: dict[str, Any]) -> None:
    rows = []
    for row in index["episodes"]:
        video = row.get("video")
        link = (
            ""
            if not video
            else f'<a href="{html.escape(os.path.relpath(video, output))}">overlay</a>'
        )
        rows.append(
            f"<tr><td>{row['action_id']}</td><td>{row['id']}</td>"
            f"<td>{row['status']}</td><td>{link}</td>"
            f"<td>{html.escape(str(row.get('error') or row.get('reason') or ''))}</td></tr>"
        )
    ArtifactStore.write_text(
        output / "index.html",
        '<!doctype html><meta charset="utf-8">'
        "<title>UMI batch review</title><h1>UMI batch review</h1>"
        f"<p>{html.escape(json.dumps(index['counts']))}</p>"
        "<p>Model screening and masks require visual review.</p>"
        '<table border="1"><tr><th>Action</th><th>Source ID</th>'
        "<th>Status</th><th>Video</th><th>Reason</th></tr>" + "".join(rows) + "</table>",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_dir", type=Path)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/process.yaml")
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    parser.add_argument("--qwen-workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--groups", nargs="*", default=[])
    parser.add_argument(
        "--inference-mode", choices=("legacy", "resident", "async"), default="legacy"
    )
    parser.add_argument("--tracking-batch-size", type=int, choices=(0, 1, 4), default=1)
    parser.add_argument("--prefetch", type=int, default=2)
    parser.add_argument("--render-buffer", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_batch(
        args.batch_dir,
        args.config,
        gpus=tuple(args.gpus),
        qwen_workers=args.qwen_workers,
        smoke=args.smoke,
        only_groups=tuple(args.groups),
        inference_mode=args.inference_mode,
        tracking_batch_size=args.tracking_batch_size,
        prefetch=args.prefetch,
        render_buffer=args.render_buffer,
        cpu_threads=args.cpu_threads,
        output_root=args.output,
    )


if __name__ == "__main__":
    main()
