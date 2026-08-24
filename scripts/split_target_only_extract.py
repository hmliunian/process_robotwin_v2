#!/usr/bin/env python3
"""Materialize a task subset from an existing target-only extract."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import uuid
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from robotwin_annotation_v2.domain import CONTACT_PRESS_TASK_KINDS

DEFAULT_SOURCE_ROOT = Path("/DATA/disk8/xuran/add_mask_robotwin/dataset/target_only_20_v2")
DEFAULT_OUTPUT_ROOT = Path(
    "/DATA/disk8/xuran/add_mask_robotwin/dataset/target_only_20_v2_contact_press"
)
CONTACT_PRESS_KINDS = frozenset(kind.value for kind in CONTACT_PRESS_TASK_KINDS)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _selected_tasks(
    selection: Mapping[str, Any],
    tasks: Iterable[str] | None,
) -> tuple[dict[str, Any], ...]:
    records = tuple(selection["tasks"])
    by_task = {str(record["task"]): record for record in records}
    if tasks is None:
        selected = tuple(
            dict(record) for record in records if record.get("task_kind") in CONTACT_PRESS_KINDS
        )
    else:
        requested = tuple(dict.fromkeys(tasks))
        if not requested:
            raise ValueError("at least one task is required")
        missing = sorted(set(requested) - set(by_task))
        if missing:
            raise ValueError(f"tasks are absent from source selection: {missing}")
        selected = tuple(dict(by_task[task]) for task in requested)
    if not selected:
        raise ValueError("source selection contains no contact_press tasks")
    invalid = [
        str(record["task"])
        for record in selected
        if record.get("task_kind") not in CONTACT_PRESS_KINDS
    ]
    if invalid:
        raise ValueError(f"selected tasks are not contact_press tasks: {invalid}")
    return selected


def build_subset_manifests(
    source_root: Path,
    output_root: Path,
    source_selection: Mapping[str, Any],
    source_collection: Mapping[str, Any],
    tasks: Iterable[str] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build root manifests for a contact-press subset without touching files."""

    selected = _selected_tasks(source_selection, tasks)
    source_by_task = {str(record["task"]): record for record in source_collection["datasets"]}
    missing_collection = sorted(
        {str(record["task"]) for record in selected} - set(source_by_task)
    )
    if missing_collection:
        raise ValueError(f"tasks are absent from source collection: {missing_collection}")

    subset_selection = dict(source_selection)
    subset_selection["output_dataset_root"] = str(output_root)
    subset_selection["tasks"] = list(selected)
    subset_selection["selection_kind"] = "contact_press_subset"
    subset_selection["derived_from"] = str(source_root / "SELECTION_MANIFEST.json")
    scope = dict(source_selection.get("scope", {}))
    scope.update(
        {
            "task_count": len(selected),
            "episode_count": sum(int(record["episode_count"]) for record in selected),
            "single_arm_target_only_task_slice_count": len(selected),
            "movable_target_task_slice_count": 0,
            "articulated_action_site_task_slice_count": sum(
                record["task_kind"] == "articulated_action_site" for record in selected
            ),
            "contact_action_site_task_slice_count": sum(
                record["task_kind"] == "contact_action_site" for record in selected
            ),
        }
    )
    subset_selection["scope"] = scope
    subset_selection["reuse_task_slice_count"] = sum(
        record.get("materialization") == "reuse_existing_extract" for record in selected
    )
    subset_selection["reuse_episode_count"] = sum(
        int(record["episode_count"])
        for record in selected
        if record.get("materialization") == "reuse_existing_extract"
    )
    subset_selection["source_copy_task_slice_count"] = len(selected) - subset_selection[
        "reuse_task_slice_count"
    ]
    subset_selection["source_copy_episode_count"] = sum(
        int(record["episode_count"])
        for record in selected
        if record.get("materialization") != "reuse_existing_extract"
    )

    datasets: list[dict[str, Any]] = []
    for record in selected:
        task = str(record["task"])
        dataset = dict(source_by_task[task])
        dataset["dataset_root"] = str(output_root / task)
        dataset["extract_manifest"] = str(output_root / task / "EXTRACT_MANIFEST.json")
        datasets.append(dataset)
    subset_collection = dict(source_collection)
    subset_collection.update(
        {
            "selection_manifest": str(output_root / "SELECTION_MANIFEST.json"),
            "task_count": len(datasets),
            "episode_count": sum(int(record["episode_count"]) for record in datasets),
            "datasets": datasets,
            "selection_kind": "contact_press_subset",
            "derived_from": str(source_root / "EXTRACT_MANIFEST.json"),
        }
    )
    return subset_selection, subset_collection


def _verify_task(root: Path, selection_record: Mapping[str, Any]) -> None:
    task = str(selection_record["task"])
    manifest_path = root / "EXTRACT_MANIFEST.json"
    manifest = _read_json(manifest_path)
    if manifest.get("task") != task or manifest.get("task_kind") != selection_record.get("task_kind"):
        raise ValueError(f"{task}: task manifest identity mismatch")
    if manifest.get("episode_indices") != selection_record.get("episode_indices"):
        raise ValueError(f"{task}: episode selection mismatch")
    for record in manifest.get("copied_files", []):
        path = root / str(record["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(record["bytes"]):
            raise OSError(f"{task}: byte-size mismatch: {path}")
        if _sha256(path) != str(record["sha256"]):
            raise OSError(f"{task}: checksum mismatch: {path}")


def materialize(
    source_root: Path,
    output_root: Path,
    *,
    tasks: Sequence[str] | None = None,
) -> dict[str, int]:
    source_root = source_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_root}")
    source_selection = _read_json(source_root / "SELECTION_MANIFEST.json")
    source_collection = _read_json(source_root / "EXTRACT_MANIFEST.json")
    subset_selection, subset_collection = build_subset_manifests(
        source_root,
        output_root,
        source_selection,
        source_collection,
        tasks,
    )
    staging_root = output_root.parent / f".{output_root.name}.staging-{uuid.uuid4().hex[:8]}"
    staging_root.mkdir(parents=True)
    try:
        for record in subset_selection["tasks"]:
            task = str(record["task"])
            shutil.copytree(source_root / task, staging_root / task)
            task_manifest = _read_json(staging_root / task / "EXTRACT_MANIFEST.json")
            task_manifest["selection_manifest"] = str(output_root / "SELECTION_MANIFEST.json")
            _write_json(staging_root / task / "EXTRACT_MANIFEST.json", task_manifest)
            _verify_task(staging_root / task, record)
        _write_json(staging_root / "SELECTION_MANIFEST.json", subset_selection)
        _write_json(staging_root / "EXTRACT_MANIFEST.json", subset_collection)
        actual_tasks = {path.name for path in staging_root.iterdir() if path.is_dir()}
        expected_tasks = {str(record["task"]) for record in subset_selection["tasks"]}
        if actual_tasks != expected_tasks:
            raise ValueError(f"subset task directories differ: {sorted(actual_tasks)}")
        staging_root.rename(output_root)
    except Exception:
        print(f"Materialization failed; staging data retained at {staging_root}", flush=True)
        raise
    return {
        "task_count": len(subset_selection["tasks"]),
        "episode_count": int(subset_selection["scope"]["episode_count"]),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--task", dest="tasks", action="append")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = materialize(
        args.source_root,
        args.output_root,
        tasks=None if args.tasks is None else tuple(args.tasks),
    )
    print(json.dumps({"status": "completed", **summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
