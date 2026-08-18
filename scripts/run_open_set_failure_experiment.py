#!/usr/bin/env python3
"""Run the open-set query/seed fallback only on the known failed episodes."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from robotwin_annotation_v2.application.dataset_runtime import process_dataset
from robotwin_annotation_v2.config import PipelineConfig, load_config
from robotwin_annotation_v2.terminal_ui import UI_MODES, create_process_ui

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "open_set_mask_fallback_failures.yaml"
EXPERIMENT_FORMAT = "robotwin_open_set_failure_experiment_v1"


@dataclass(frozen=True)
class FailureExperiment:
    pipeline: PipelineConfig
    dataset_root: Path
    run_id_prefix: str
    task_episode_ids: tuple[tuple[str, tuple[int, ...]], ...]
    smoke_episode_ids: tuple[int, ...]
    expected_failure_count: int

    @property
    def episode_owner(self) -> dict[int, str]:
        return {
            episode_id: task
            for task, episode_ids in self.task_episode_ids
            for episode_id in episode_ids
        }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--task",
        dest="tasks",
        action="append",
        help="Run one declared task; repeat to select multiple tasks.",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--episode-ids",
        type=int,
        nargs="+",
        help="Run a subset; every id must be in the declared 52 failures.",
    )
    selection.add_argument(
        "--smoke",
        action="store_true",
        help="Run the six representative query/seed diagnostic failures.",
    )
    parser.add_argument(
        "--sam-gpu",
        type=int,
        help="Override the single SAM GPU from the experiment config.",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--run-id-prefix")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute episodes already complete under the selected run ids.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate config and input files without contacting Qwen or loading SAM.",
    )
    parser.add_argument("--ui", choices=UI_MODES, default="plain")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return value


def _integer_tuple(value: Any, *, label: str) -> tuple[int, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, int) or isinstance(item, bool) for item in value
    ):
        raise ValueError(f"{label} must be a list of integers")
    output = tuple(value)
    if not output or any(item < 0 for item in output):
        raise ValueError(f"{label} must contain non-negative episode ids")
    if len(output) != len(set(output)):
        raise ValueError(f"{label} contains duplicate episode ids")
    return output


def _load_experiment(path: Path) -> FailureExperiment:
    resolved = path.expanduser().resolve()
    pipeline = load_config(resolved)
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    root = _mapping(raw, label="config")
    experiment = _mapping(root.get("experiment"), label="experiment")
    if experiment.get("format_version") != EXPERIMENT_FORMAT:
        raise ValueError(f"experiment.format_version must be {EXPERIMENT_FORMAT!r}")

    dataset_root_value = experiment.get("dataset_root")
    if not isinstance(dataset_root_value, str) or not dataset_root_value.strip():
        raise ValueError("experiment.dataset_root must be a non-empty path")
    dataset_root_path = Path(dataset_root_value).expanduser()
    dataset_root = (
        dataset_root_path
        if dataset_root_path.is_absolute()
        else resolved.parent / dataset_root_path
    ).resolve()

    run_id_prefix = experiment.get("run_id_prefix")
    if not isinstance(run_id_prefix, str) or not run_id_prefix.strip():
        raise ValueError("experiment.run_id_prefix must be a non-empty string")
    if (
        run_id_prefix != run_id_prefix.strip()
        or "/" in run_id_prefix
        or "\\" in run_id_prefix
        or ".." in run_id_prefix
    ):
        raise ValueError("experiment.run_id_prefix must be a simple directory-name prefix")

    tasks_raw = _mapping(experiment.get("tasks"), label="experiment.tasks")
    task_episode_ids = tuple(
        (
            str(task),
            _integer_tuple(episode_ids, label=f"experiment.tasks.{task}"),
        )
        for task, episode_ids in tasks_raw.items()
    )
    if not task_episode_ids or any(not task.strip() for task, _ in task_episode_ids):
        raise ValueError("experiment.tasks must contain named task entries")

    all_ids = tuple(
        episode_id for _task, episode_ids in task_episode_ids for episode_id in episode_ids
    )
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("episode ids must be unique across experiment.tasks")
    expected = experiment.get("expected_failure_count")
    if isinstance(expected, bool) or not isinstance(expected, int) or expected < 1:
        raise ValueError("experiment.expected_failure_count must be a positive integer")
    if len(all_ids) != expected:
        raise ValueError(f"experiment declares {len(all_ids)} failures, expected {expected}")

    smoke_ids = _integer_tuple(
        experiment.get("smoke_episode_ids"),
        label="experiment.smoke_episode_ids",
    )
    unknown_smoke = sorted(set(smoke_ids) - set(all_ids))
    if unknown_smoke:
        raise ValueError(f"smoke episode ids are outside the failure set: {unknown_smoke}")

    if pipeline.qwen.endpoint != "http://127.0.0.1:18086/v1/chat/completions":
        raise ValueError("qwen.endpoint must use the local port 18086 service")
    if pipeline.qwen.timeout_seconds != 600:
        raise ValueError("qwen.timeout_seconds must be 600 for this experiment")
    if not pipeline.mask.qc_enabled:
        raise ValueError("mask.qc_enabled must be true for this experiment")
    if pipeline.mask.qc_max_candidates != 8:
        raise ValueError("mask.qc_max_candidates must be 8 for this experiment")
    if not pipeline.mask.qc_query_fallback_enabled:
        raise ValueError("mask.qc_query_fallback_enabled must be true")
    if not pipeline.mask.qc_seed_fallback_enabled:
        raise ValueError("mask.qc_seed_fallback_enabled must be true")

    return FailureExperiment(
        pipeline=pipeline,
        dataset_root=dataset_root,
        run_id_prefix=run_id_prefix,
        task_episode_ids=task_episode_ids,
        smoke_episode_ids=smoke_ids,
        expected_failure_count=expected,
    )


def _select_episodes(
    experiment: FailureExperiment,
    *,
    tasks: Sequence[str] | None,
    episode_ids: Sequence[int] | None,
    smoke: bool,
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    declared_tasks = dict(experiment.task_episode_ids)
    selected_task_names = tuple(dict.fromkeys(tasks or declared_tasks))
    unknown_tasks = sorted(set(selected_task_names) - set(declared_tasks))
    if unknown_tasks:
        raise ValueError(f"tasks are outside the experiment config: {unknown_tasks}")

    owner = experiment.episode_owner
    requested_ids = (
        experiment.smoke_episode_ids
        if smoke
        else tuple(dict.fromkeys(int(value) for value in episode_ids))
        if episode_ids is not None
        else None
    )
    if requested_ids is not None:
        unknown_ids = sorted(set(requested_ids) - set(owner))
        if unknown_ids:
            raise ValueError(f"episode ids are outside the declared failures: {unknown_ids}")
        conflicting = sorted(
            episode_id
            for episode_id in requested_ids
            if owner[episode_id] not in selected_task_names
        )
        if conflicting:
            raise ValueError(f"episode ids do not belong to the selected tasks: {conflicting}")

    selected = []
    requested_set = None if requested_ids is None else set(requested_ids)
    for task, declared_ids in experiment.task_episode_ids:
        if task not in selected_task_names:
            continue
        values = (
            declared_ids
            if requested_set is None
            else tuple(value for value in declared_ids if value in requested_set)
        )
        if values:
            selected.append((task, values))
    if not selected:
        raise ValueError("episode selection is empty")
    return tuple(selected)


def _episode_paths(dataset_root: Path, camera: str, episode_id: int) -> tuple[Path, ...]:
    chunk = f"chunk-{episode_id // 1000:03d}"
    return (
        dataset_root / "data" / chunk / f"episode_{episode_id:06d}.parquet",
        dataset_root
        / "videos"
        / chunk
        / f"observation.images.{camera}"
        / f"episode_{episode_id:06d}.mp4",
        dataset_root / "sidecars" / f"episode_{episode_id:06d}.hdf5",
    )


def _validate_inputs(
    experiment: FailureExperiment,
    selected: tuple[tuple[str, tuple[int, ...]], ...],
) -> dict[str, Any]:
    pipeline = experiment.pipeline
    required_config_paths = [
        pipeline.dataset.manifest,
        pipeline.qwen.prompt_template,
        pipeline.sam3.checkpoint,
        pipeline.mask.qc_prompt_template,
    ]
    if pipeline.mask.qc_bbox_fallback_enabled:
        required_config_paths.append(pipeline.mask.qc_bbox_prompt_template)
    missing = [str(path) for path in required_config_paths if path is None or not path.is_file()]
    for task, episode_ids in selected:
        task_root = experiment.dataset_root / task
        for episode_id in episode_ids:
            missing.extend(
                str(path)
                for path in _episode_paths(task_root, pipeline.dataset.camera, episode_id)
                if not path.is_file()
            )
    if missing:
        preview = "\n".join(f"  - {path}" for path in missing[:20])
        suffix = "" if len(missing) <= 20 else f"\n  ... and {len(missing) - 20} more"
        raise FileNotFoundError(f"experiment inputs are missing:\n{preview}{suffix}")

    selected_count = sum(len(values) for _task, values in selected)
    return {
        "status": "validated",
        "config": str(pipeline.config_path),
        "dataset_root": str(experiment.dataset_root),
        "output_root": str(pipeline.output_root),
        "qwen_endpoint": pipeline.qwen.endpoint,
        "qwen_timeout_seconds": pipeline.qwen.timeout_seconds,
        "sam_gpu": pipeline.sam3.gpus[0],
        "qc_max_candidates": pipeline.mask.qc_max_candidates,
        "qc_query_fallback_enabled": pipeline.mask.qc_query_fallback_enabled,
        "qc_seed_fallback_enabled": pipeline.mask.qc_seed_fallback_enabled,
        "qc_bbox_fallback_enabled": pipeline.mask.qc_bbox_fallback_enabled,
        "qc_bbox_prompt_template": (
            None
            if pipeline.mask.qc_bbox_prompt_template is None
            else str(pipeline.mask.qc_bbox_prompt_template)
        ),
        "selected_task_count": len(selected),
        "selected_episode_count": selected_count,
        "selected": {task: list(values) for task, values in selected},
    }


def _write_batch_summary(output_root: Path, run_id_prefix: str, payload: dict[str, Any]) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / f"{run_id_prefix}-batch-summary.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _run(
    experiment: FailureExperiment,
    selected: tuple[tuple[str, tuple[int, ...]], ...],
    args: argparse.Namespace,
) -> dict[str, Any]:
    # ``main`` applies the optional CLI override before validation, so the
    # validated experiment is the single source of truth here.
    sam_gpu = experiment.pipeline.sam3.gpus[0]
    output_root = (
        experiment.pipeline.output_root
        if args.output_root is None
        else args.output_root.expanduser().resolve()
    )
    run_id_prefix = experiment.run_id_prefix if args.run_id_prefix is None else args.run_id_prefix
    if not run_id_prefix or run_id_prefix != run_id_prefix.strip():
        raise ValueError("--run-id-prefix must be a non-empty trimmed string")
    if "/" in run_id_prefix or "\\" in run_id_prefix or ".." in run_id_prefix:
        raise ValueError("--run-id-prefix must be a simple directory-name prefix")

    pipeline = replace(
        experiment.pipeline,
        sam3=replace(experiment.pipeline.sam3, gpus=(sam_gpu,)),
        output_root=output_root,
    )
    requested_count = sum(len(values) for _task, values in selected)
    aggregate: dict[str, Any] = {
        "format_version": "robotwin_open_set_failure_batch_summary_v1",
        "config": str(pipeline.config_path),
        "run_id_prefix": run_id_prefix,
        "dataset_root": str(experiment.dataset_root),
        "output_root": str(output_root),
        "qwen_endpoint": pipeline.qwen.endpoint,
        "sam_gpu": sam_gpu,
        "requested_episode_count": requested_count,
        "tasks": {},
        "status_counts": {},
        "passed": False,
    }
    status_counts: Counter[str] = Counter()

    for task, episode_ids in selected:
        run_id = f"{run_id_prefix}-{task}-object-source"
        reporter = create_process_ui(args.ui, verbose=args.verbose)
        try:
            summary = process_dataset(
                pipeline,
                dataset_root=experiment.dataset_root / task,
                task=task,
                camera=pipeline.dataset.camera,
                output_root=output_root,
                run_id=run_id,
                episode_ids=episode_ids,
                force=args.force,
                skip_render=True,
                object_source_only=True,
                incremental_source=True,
                reporter=reporter,
            )
        except Exception as exc:
            reporter.failed(exc)
            aggregate["tasks"][task] = {
                "run_id": run_id,
                "requested_episode_ids": list(episode_ids),
                "error": f"{type(exc).__name__}: {exc}",
                "passed": False,
            }
            aggregate["status_counts"] = dict(sorted(status_counts.items()))
            artifact = _write_batch_summary(output_root, run_id_prefix, aggregate)
            raise RuntimeError(f"task {task} failed; partial summary: {artifact}") from exc
        else:
            reporter.finish(summary)
        finally:
            reporter.close()

        selected_set = set(episode_ids)
        selected_records = [
            record for record in summary.get("records", []) if record.get("episode") in selected_set
        ]
        status_counts.update(str(record.get("status", "unknown")) for record in selected_records)
        aggregate["tasks"][task] = {
            "run_id": run_id,
            "requested_episode_ids": list(episode_ids),
            "records": selected_records,
            "passed": bool(summary.get("passed")),
            "process_summary": summary.get("artifact"),
        }
        aggregate["status_counts"] = dict(sorted(status_counts.items()))
        _write_batch_summary(output_root, run_id_prefix, aggregate)

    terminal_count = sum(status_counts.values())
    aggregate["passed"] = terminal_count == requested_count and all(
        task_summary.get("passed") for task_summary in aggregate["tasks"].values()
    )
    aggregate["batch_summary"] = str(_write_batch_summary(output_root, run_id_prefix, aggregate))
    return aggregate


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    try:
        experiment = _load_experiment(args.config)
        selected = _select_episodes(
            experiment,
            tasks=args.tasks,
            episode_ids=args.episode_ids,
            smoke=args.smoke,
        )
        if args.sam_gpu is not None:
            experiment = replace(
                experiment,
                pipeline=replace(
                    experiment.pipeline,
                    sam3=replace(experiment.pipeline.sam3, gpus=(args.sam_gpu,)),
                ),
            )
        validation = _validate_inputs(experiment, selected)
        if args.validate_only:
            print(json.dumps(validation, ensure_ascii=False, indent=2))
            return
        summary = _run(experiment, selected, args)
    except (FileNotFoundError, TypeError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"error: {exc}") from exc

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
