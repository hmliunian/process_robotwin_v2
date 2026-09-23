"""Run prepared task-manifest groups and publish a complete cross-source review index."""

from __future__ import annotations

import argparse
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
from robotwin_annotation_v2.application.sam_workflow import (
    build_sam_dynamic_config,
    render_sam_processed,
)
from robotwin_annotation_v2.config import DatasetBinding, bind_dataset, load_profile
from robotwin_annotation_v2.terminal_ui import create_process_ui
from scripts.render_batch_review_sheets import render_batch_sheets

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPLETE = {"completed", "skipped_complete"}


def run_batch(batch_dir: Path, config_path: Path, *, gpu: int, smoke: bool) -> dict[str, Any]:
    batch_dir = batch_dir.resolve()
    plan = json.loads((batch_dir / "batch_plan.json").read_text())
    manifest = json.loads((batch_dir / Path(plan["source_manifest"]).name).read_text())
    source_root = Path(plan["source_manifest"]).parent
    episodes = {e["episode_uid"]: e for e in manifest["episodes"]}
    output = batch_dir / ("smoke" if smoke else "runs")
    phase_dir = output if smoke else batch_dir
    index: dict[str, Any] = {"phase": "smoke" if smoke else "full", "episodes": [], "groups": []}
    status_path = batch_dir / "batch_status.json"
    progress: dict[str, Any] = {
        "phase": index["phase"],
        "pid": os.getpid(),
        "gpu": gpu,
        "status": "running",
        "total": len(plan["groups"]) if smoke else plan["episode_count"],
        "records": [],
    }

    def save_progress() -> None:
        progress["updated_at"] = datetime.now(UTC).isoformat()
        progress["counts"] = dict(Counter(r["status"] for r in progress["records"]))
        ArtifactStore.write_json(status_path, progress)

    save_progress()
    for group in plan["groups"]:
        target = resolve_dataset_input(Path(group["input_root"])).targets[0]
        selected = tuple(group["episode_ids"][:1] if smoke else group["episode_ids"])
        run_id = group["group_id"]
        run_dir = output / run_id
        if run_dir.exists():
            raise FileExistsError(f"fresh run already exists: {run_dir}")
        profile = load_profile(config_path, mode=target.mode, target_profile=target.profile)
        profile = replace(profile, sam3=replace(profile.sam3, gpus=(gpu,)))
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
        save_progress()

        def terminal(episode_id: int, status: str, source: str = group["source_dataset"]) -> None:
            progress["records"].append(
                {
                    "episode_uid": f"{source}/episode_{episode_id:06d}",
                    "status": status,
                }
            )
            save_progress()

        reporter = create_process_ui("plain", verbose=False)
        summary: dict[str, Any]
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
                skip_render=True,
                reporter=reporter,
                episode_terminal_callback=terminal,
            )
            reporter.finish(summary)
        except Exception as exc:  # noqa: BLE001 - retain failed groups in the final review
            summary = {
                "run_id": run_id,
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
                "records": [
                    {"episode": e, "status": "failed", "error": str(exc)} for e in selected
                ],
            }
            ArtifactStore.write_json(run_dir / "process_summary.json", summary)
            recorded = {r["episode_uid"] for r in progress["records"]}
            for episode_id in selected:
                if f"{group['source_dataset']}/episode_{episode_id:06d}" not in recorded:
                    terminal(episode_id, "failed")
            print(json.dumps(summary), flush=True)
        finally:
            reporter.close()

        records = {r["episode"]: r for r in summary["records"] if "episode" in r}
        successful = tuple(e for e in selected if records.get(e, {}).get("status") in COMPLETE)
        videos = {}
        if successful:
            dynamic = build_sam_dynamic_config(
                config,
                root=target.root,
                task=target.task,
                camera=target.camera,
                manifest=summary["dynamic_manifest"],
                output_root=output,
            )
            rendered = render_sam_processed(
                dynamic,
                run_id=run_id,
                episode_ids=successful,
                output_dir=run_dir,
            )
            render_index = json.loads(Path(rendered["manifest"]).read_text())
            videos = {
                r["episode_index"]: str(run_dir / "rendered_videos" / r["output_video"])
                for r in render_index["episodes"]
            }
            summary["render"] = rendered
            ArtifactStore.write_json(run_dir / "process_summary.json", summary)
        index["groups"].append({"run_id": run_id, "summary": str(run_dir / "process_summary.json")})
        metadata = {
            row["episode_index"]: row
            for row in (
                json.loads(line)
                for line in (target.root / "meta/episodes.jsonl").read_text().splitlines()
            )
        }
        for episode_id in selected:
            uid = f"{group['source_dataset']}/episode_{episode_id:06d}"
            episode = episodes[uid]
            record = records.get(episode_id, {"status": "not_run"})
            ready = record["status"] in COMPLETE
            episode_dir = run_dir / target.task / f"episode_{episode_id:06d}" / target.camera
            temporal_path = episode_dir / "target_0/temporal_qc.json"
            temporal = json.loads(temporal_path.read_text()) if temporal_path.exists() else {}
            run_manifest = episode_dir / "run_manifest.json"
            failures = []
            if not ready and run_manifest.exists():
                for role in json.loads(run_manifest.read_text()).get("roles", []):
                    if role.get("failure"):
                        failures.append(f"{role['role']}: {role['failure']}; {role.get('qc_reason')}")
            needs_review = (
                temporal.get("status") not in {None, "pass"}
                or target.manifest_data.get("timeline_source") == "video_window"
            )
            index["episodes"].append(
                {
                    "episode_uid": uid,
                    "source_dataset": group["source_dataset"],
                    "task": target.task,
                    "episode": episode_id,
                    "camera": target.camera,
                    "run_id": run_id,
                    "status": "failed"
                    if not ready
                    else "needs_temporal_review"
                    if needs_review
                    else "auto_complete",
                    "objects_ready": ready,
                    "process_status": record["status"],
                    "error": record.get("error") or "; ".join(failures) or None,
                    "temporal_status": temporal.get("status"),
                    "timeline_source": target.manifest_data.get("timeline_source"),
                    "timeline_note": metadata[episode_id].get("timeline_derivation", {}).get(
                        "state_timeline_error"
                    ),
                    "frame_count": episode["frame_count"],
                    "active_arm": metadata[episode_id].get(
                        "target_only_events", metadata[episode_id].get("video_events", {})
                    )["active_arm"],
                    "split": episode["split"],
                    "video": videos.get(episode_id),
                    "artifact_dir": str(episode_dir),
                    "source_video": str(source_root / episode["files"]["videos"][target.camera]),
                }
            )
        ArtifactStore.write_json(phase_dir / "episode_results.json", index)

    expected = len(plan["groups"]) if smoke else plan["episode_count"]
    if len(index["episodes"]) != expected:
        raise ValueError("final episode count does not match selection")
    counts = dict(Counter(r["status"] for r in index["episodes"]))
    index["counts"] = counts
    index_path = ArtifactStore.write_json(phase_dir / "episode_results.json", index)
    render_batch_sheets(index_path, source_root, phase_dir / "review_sheets")
    progress.update(
        status="finished",
        result_counts=counts,
        index=str(index_path),
        review_sheets=str(phase_dir / "review_sheets"),
    )
    save_progress()
    print(
        json.dumps({"phase": index["phase"], "counts": counts, "index": str(index_path)}),
        flush=True,
    )
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_dir", type=Path)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/process.yaml")
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    try:
        run_batch(args.batch_dir, args.config, gpu=args.gpu, smoke=args.smoke)
    except Exception as exc:
        path = args.batch_dir / "batch_status.json"
        progress = json.loads(path.read_text()) if path.exists() else {}
        progress.update(status="crashed", error=f"{type(exc).__name__}: {exc}")
        ArtifactStore.write_json(path, progress)
        raise


if __name__ == "__main__":
    main()
