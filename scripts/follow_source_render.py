#!/usr/bin/env python3
"""Render completed source runs, waiting for unfinished runs to publish summaries."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from robotwin_annotation_v2.application.sam_workflow import (
    build_sam_dynamic_config,
    read_json_object,
    render_sam_processed,
)
from robotwin_annotation_v2.config import load_config

RENDERABLE_STATUSES = frozenset({"completed", "skipped_complete"})


def _simple_run_id(value: str) -> str:
    if (
        not value
        or value != value.strip()
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or ".." in value
    ):
        raise argparse.ArgumentTypeError("run ids must be simple directory names")
    return value


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _emit(run_id: str, status: str, **details: Any) -> None:
    print(
        json.dumps(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "run_id": run_id,
                "status": status,
                **details,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def _completed_episode_ids(summary: Mapping[str, Any]) -> tuple[int, ...]:
    records = summary.get("records")
    if not isinstance(records, list):
        raise TypeError("process summary records must be a list")
    episode_ids: list[int] = []
    for record in records:
        if not isinstance(record, Mapping) or record.get("status") not in RENDERABLE_STATUSES:
            continue
        episode_id = record.get("episode")
        if isinstance(episode_id, bool) or not isinstance(episode_id, int):
            raise TypeError("renderable process records must contain integer episode ids")
        episode_ids.append(episode_id)
    if not episode_ids:
        raise ValueError("process summary has no completed episodes to render")
    if len(episode_ids) != len(set(episode_ids)):
        raise ValueError("process summary repeats completed episode ids")
    return tuple(episode_ids)


def _read_optional_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _render_is_complete(run_dir: Path, run_id: str, episode_ids: tuple[int, ...]) -> bool:
    video_dir = run_dir / "rendered_videos"
    manifest = _read_optional_object(video_dir / "manifest.json")
    if manifest is None or manifest.get("requested_run_id") != run_id:
        return False
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != len(episode_ids):
        return False
    rendered_ids: list[int] = []
    for record in episodes:
        if not isinstance(record, Mapping):
            return False
        episode_id = record.get("episode_index")
        output_video = record.get("output_video")
        if (
            isinstance(episode_id, bool)
            or not isinstance(episode_id, int)
            or not isinstance(output_video, str)
            or not (video_dir / output_video).is_file()
        ):
            return False
        rendered_ids.append(episode_id)
    sheets = manifest.get("review_sheets")
    return (
        tuple(rendered_ids) == episode_ids
        and isinstance(sheets, list)
        and bool(sheets)
        and all(isinstance(path, str) and (video_dir / path).is_file() for path in sheets)
    )


def _render_when_ready(
    output_root: Path,
    config_path: Path,
    run_id: str,
    poll_seconds: float,
    force: bool,
) -> dict[str, Any]:
    run_dir = output_root / run_id
    summary_path = run_dir / "process_summary.json"
    if not summary_path.is_file():
        _emit(run_id, "waiting_for_process_summary")
    while not summary_path.is_file():
        time.sleep(poll_seconds)

    contract = read_json_object(
        run_dir / "source_run_contract.json",
        description="source run contract",
    )
    summary = read_json_object(summary_path, description="process summary")
    for field in ("run_id", "task", "camera", "dataset_root", "annotation_mode"):
        if contract.get(field) != summary.get(field):
            raise ValueError(f"source contract and process summary disagree on {field}")
    if contract.get("run_id") != run_id:
        raise ValueError("source contract run_id does not match its directory")
    dynamic_manifest = contract.get("dynamic_manifest")
    if not isinstance(dynamic_manifest, dict):
        raise TypeError("source run contract dynamic_manifest must be an object")

    episode_ids = _completed_episode_ids(summary)
    if not force and _render_is_complete(run_dir, run_id, episode_ids):
        return {"run_id": run_id, "status": "skipped_complete", "episode_count": len(episode_ids)}

    config = load_config(config_path)
    if config.annotation.mode.value != contract.get("annotation_mode"):
        raise ValueError("render config annotation mode does not match the source run")
    dynamic = build_sam_dynamic_config(
        config,
        root=Path(str(contract["dataset_root"])),
        task=str(contract["task"]),
        camera=str(contract["camera"]),
        manifest=dynamic_manifest,
        output_root=output_root,
    )
    _emit(run_id, "rendering", episode_count=len(episode_ids))
    result = render_sam_processed(
        dynamic,
        run_id=run_id,
        episode_ids=episode_ids,
        output_dir=run_dir,
    )
    return {"run_id": run_id, "status": "completed", **result}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=_positive_float, default=30.0)
    parser.add_argument("--max-parallel", type=_positive_int, default=2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("run_ids", nargs="+", type=_simple_run_id)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output_root = args.output_root.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    failures: list[str] = []
    with ProcessPoolExecutor(max_workers=args.max_parallel) as executor:
        futures = {
            executor.submit(
                _render_when_ready,
                output_root,
                config_path,
                run_id,
                args.poll_seconds,
                args.force,
            ): run_id
            for run_id in args.run_ids
        }
        for future in as_completed(futures):
            run_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001 - isolate independent source runs
                failures.append(run_id)
                _emit(run_id, "failed", error=f"{type(exc).__name__}: {exc}")
            else:
                _emit(run_id, str(result["status"]), result=result)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
