#!/usr/bin/env python3
"""Plan and materialize the versioned target-only 20-per-task RoboTwin extract."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import uuid
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robotwin_annotation_v2.adapters.robotwin_dataset import EpisodePaths, EpisodeState
from robotwin_annotation_v2.pipeline import (
    StateLoopError,
    detect_episode_loop,
    detect_episode_target_only,
)
from robotwin_annotation_v2.pipeline.state_loop import _close_transition

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = Path("/DATA/disk8/xuran/robotwin2.0_full_aloha_rgbd_320_rgb_correct_v1")
DEFAULT_OUTPUT_ROOT = Path("/DATA/disk8/xuran/add_mask_robotwin/dataset/target_only_20_v2")
DEFAULT_SELECTION = PROJECT_ROOT / "configs/datasets/target_only_20_v2_selection.json"
DEFAULT_PROFILE_REUSE_ROOT = Path("/DATA/disk8/xuran/add_mask_robotwin/dataset/profile_compat_20")
DEFAULT_ADJUST_REUSE_ROOT = Path("/DATA/disk8/xuran/add_mask_robotwin/dataset/target_only_20")

MOVABLE_TASKS = (
    "adjust_bottle",
    "beat_block_hammer",
    "dump_bin_bigbin",
    "grab_roller",
    "handover_block",
    "handover_mic",
    "hanging_mug",
    "lift_pot",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "move_stapler_pad",
    "place_a2b_left",
    "place_a2b_right",
    "place_container_plate",
    "place_empty_cup",
    "place_fan",
    "place_mouse_pad",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    "put_object_cabinet",
    "rotate_qrcode",
    "shake_bottle",
    "shake_bottle_horizontally",
    "stamp_seal",
)
CONDITIONAL_MOVABLE_TASK = "place_bread_basket"
ARTICULATED_TASKS = ("open_laptop", "open_microwave", "turn_switch")
ALL_TASKS = (*MOVABLE_TASKS, CONDITIONAL_MOVABLE_TASK, *ARTICULATED_TASKS)
INDEX_COLUMNS = (
    "coarse_task_index",
    "task_index",
    "coarse_quality_index",
    "quality_index",
)


@dataclass(frozen=True)
class Candidate:
    task: str
    episode_index: int
    domain: str
    arm: str
    timeline_shape: str
    source_variant: str

    def to_json(self) -> dict[str, Any]:
        return {
            "episode_index": self.episode_index,
            "domain": self.domain,
            "arm": self.arm,
            "timeline_shape": self.timeline_shape,
            "source_dataset_variant": self.source_variant,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--selection-output", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--profile-reuse-root", type=Path, default=DEFAULT_PROFILE_REUSE_ROOT)
    parser.add_argument("--adjust-reuse-root", type=Path, default=DEFAULT_ADJUST_REUSE_ROOT)
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--materialize",
        action="store_true",
        help="Copy the planned data after writing or verifying the selection manifest.",
    )
    action.add_argument(
        "--validate-only",
        action="store_true",
        help="Recompute the published extract's counts, metadata, and file checksums.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _metadata_by_task(source_root: Path) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with (source_root / "meta/episodes.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            task = str(row["full_structured_tasks"][0])
            if task in ALL_TASKS:
                result[task].append(row)
    missing = sorted(set(ALL_TASKS) - set(result))
    if missing:
        raise ValueError(f"source metadata is missing tasks: {missing}")
    return result


def _episode_relative_files(episode_index: int) -> tuple[Path, ...]:
    chunk = episode_index // 1000
    stem = f"episode_{episode_index:06d}"
    return (
        Path(f"data/chunk-{chunk:03d}/{stem}.parquet"),
        Path(f"sidecars/{stem}.hdf5"),
        Path(f"videos/chunk-{chunk:03d}/observation.images.cam_high/{stem}.mp4"),
        Path(f"sidecars/videos/chunk-{chunk:03d}/observation.depths.cam_high/{stem}.mkv"),
    )


def _load_state(source_root: Path, row: dict[str, Any]) -> EpisodeState:
    episode_index = int(row["episode_index"])
    parquet = source_root / _episode_relative_files(episode_index)[0]
    frame = pd.read_parquet(parquet, columns=["observation.state"])
    if frame.empty:
        raise ValueError(f"empty state parquet: {parquet}")
    state = np.stack(frame["observation.state"].to_numpy()).astype(np.float64)
    if state.shape != (len(frame), 14):
        raise ValueError(f"unexpected state shape {state.shape}: {parquet}")
    return EpisodeState(
        frame_count=len(frame),
        task_text=str(row["tasks"][0]),
        gripper_states=state[:, (6, 13)],
        eef_states=np.stack((state[:, 0:6], state[:, 7:13]), axis=1),
        paths=EpisodePaths(parquet=parquet, video=Path(), sidecar=Path()),
    )


def _first_close_arm(state: EpisodeState) -> str:
    starts: dict[str, int] = {}
    for arm_index, arm in enumerate(("left", "right")):
        try:
            _filtered, close_start, _close_end = _close_transition(
                state.gripper_states[:, arm_index], stable_frames=3
            )
        except StateLoopError:
            continue
        starts[arm] = close_start
    if not starts:
        return "none"
    earliest = min(starts.values())
    winners = [arm for arm, frame_id in starts.items() if frame_id == earliest]
    return winners[0] if len(winners) == 1 else "both"


def _candidate_from_row(
    source_root: Path,
    row: dict[str, Any],
    *,
    require_single_loop: bool,
) -> Candidate | None:
    if not bool(row["geometry_valid"]):
        return None
    episode_index = int(row["episode_index"])
    missing = [
        relative_path.as_posix()
        for relative_path in _episode_relative_files(episode_index)
        if not (source_root / relative_path).is_file()
    ]
    if missing:
        return None
    state = _load_state(source_root, row)
    loop_arm: str | None = None
    target_only_arm: str | None = None
    try:
        loop_arm = detect_episode_loop(state).active_arm
    except StateLoopError:
        pass
    if require_single_loop and loop_arm is None:
        return None
    try:
        target_only_arm = detect_episode_target_only(state).active_arm
    except StateLoopError:
        pass
    return Candidate(
        task=str(row["full_structured_tasks"][0]),
        episode_index=episode_index,
        domain=str(row["full_structured_tasks"][3]),
        arm=loop_arm or target_only_arm or _first_close_arm(state),
        timeline_shape=(
            "pick_place"
            if loop_arm is not None
            else "close_hold"
            if target_only_arm is not None
            else "other"
        ),
        source_variant=str(row["source_dataset_variant"]),
    )


def _spread(items: Sequence[Candidate], count: int) -> list[Candidate]:
    ordered = sorted(items, key=lambda item: item.episode_index)
    if count < 0 or count > len(ordered):
        raise ValueError(f"cannot select {count} items from a pool of {len(ordered)}")
    if count == 0:
        return []
    positions = np.linspace(0, len(ordered) - 1, num=count).round().astype(int)
    if len({int(value) for value in positions}) != count:
        raise AssertionError("spread selection unexpectedly produced duplicate positions")
    return [ordered[int(position)] for position in positions]


def _arm_allocations(
    candidates_by_domain: dict[str, list[Candidate]],
    quotas: dict[str, int],
) -> dict[str, int] | None:
    """Choose left counts per domain, preferring 10/10 overall then domain balance."""

    domains = tuple(quotas)
    if len(domains) != 2:
        raise ValueError("arm allocation expects two domains")
    first, second = domains
    capacities = {
        domain: {
            arm: sum(item.arm == arm for item in candidates_by_domain[domain])
            for arm in ("left", "right")
        }
        for domain in domains
    }
    if any(
        capacities[domain]["left"] + capacities[domain]["right"] < quotas[domain]
        for domain in domains
    ):
        return None

    total = sum(quotas.values())
    target_left = total // 2
    options: list[tuple[tuple[float, float, int], dict[str, int]]] = []
    for first_left in range(quotas[first] + 1):
        second_left = target_left - first_left
        if not 0 <= second_left <= quotas[second]:
            continue
        allocation = {first: first_left, second: second_left}
        feasible = all(
            allocation[domain] <= capacities[domain]["left"]
            and quotas[domain] - allocation[domain] <= capacities[domain]["right"]
            for domain in domains
        )
        if not feasible:
            continue
        domain_imbalance = sum(abs(allocation[domain] - quotas[domain] / 2) for domain in domains)
        options.append(((0.0, domain_imbalance, first_left), allocation))
    if options:
        return min(options, key=lambda item: item[0])[1]

    # If 10/10 overall is impossible, choose the closest feasible total split.
    for first_left in range(quotas[first] + 1):
        for second_left in range(quotas[second] + 1):
            allocation = {first: first_left, second: second_left}
            feasible = all(
                allocation[domain] <= capacities[domain]["left"]
                and quotas[domain] - allocation[domain] <= capacities[domain]["right"]
                for domain in domains
            )
            if not feasible:
                continue
            total_left = first_left + second_left
            domain_imbalance = sum(
                abs(allocation[domain] - quotas[domain] / 2) for domain in domains
            )
            options.append(
                (
                    (abs(total_left - target_left), domain_imbalance, first_left),
                    allocation,
                )
            )
    return None if not options else min(options, key=lambda item: item[0])[1]


def _select_candidates(
    candidates: Sequence[Candidate],
    *,
    clean_count: int,
    randomized_count: int,
) -> list[Candidate]:
    quotas = {"clean": clean_count, "randomized": randomized_count}
    candidates_by_domain = {
        domain: [item for item in candidates if item.domain == domain] for domain in quotas
    }
    for domain, quota in quotas.items():
        if len(candidates_by_domain[domain]) < quota:
            raise ValueError(
                f"{candidates[0].task}: {domain} has {len(candidates_by_domain[domain])} "
                f"eligible episodes, needs {quota}"
            )
    allocation = _arm_allocations(candidates_by_domain, quotas)
    selected: list[Candidate] = []
    for domain, quota in quotas.items():
        if allocation is None:
            selected.extend(_spread(candidates_by_domain[domain], quota))
            continue
        left_count = allocation[domain]
        right_count = quota - left_count
        selected.extend(
            _spread(
                [item for item in candidates_by_domain[domain] if item.arm == "left"],
                left_count,
            )
        )
        selected.extend(
            _spread(
                [item for item in candidates_by_domain[domain] if item.arm == "right"],
                right_count,
            )
        )
    selected.sort(key=lambda item: item.episode_index)
    if len(selected) != clean_count + randomized_count:
        raise AssertionError("selection count differs from requested count")
    return selected


def _reuse_records(profile_root: Path, adjust_root: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for root in (profile_root, adjust_root):
        manifest_path = root / "EXTRACT_MANIFEST.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = _read_json(manifest_path)
        for record in manifest["datasets"]:
            task = str(record["task"])
            if task not in ALL_TASKS:
                continue
            if task in records:
                raise ValueError(f"duplicate reuse task: {task}")
            records[task] = dict(record)
    return records


def _task_kind(task: str) -> str:
    if task in ARTICULATED_TASKS:
        return "articulated_action_site"
    if task == CONDITIONAL_MOVABLE_TASK:
        return "single_movable_target_conditional"
    return "single_movable_target"


def build_selection(
    source_root: Path,
    profile_reuse_root: Path,
    adjust_reuse_root: Path,
) -> dict[str, Any]:
    metadata = _metadata_by_task(source_root)
    metadata_index = {int(row["episode_index"]): row for rows in metadata.values() for row in rows}
    reuse = _reuse_records(profile_reuse_root, adjust_reuse_root)
    tasks: list[dict[str, Any]] = []
    for task in ALL_TASKS:
        reused = reuse.get(task)
        if reused is not None:
            episode_ids = [int(value) for value in reused["episode_indices"]]
            if len(episode_ids) != 20 or len(set(episode_ids)) != 20:
                raise ValueError(f"reuse selection for {task} must contain 20 unique episodes")
            selected = []
            for episode_index in episode_ids:
                row = metadata_index[episode_index]
                candidate = _candidate_from_row(
                    source_root,
                    row,
                    require_single_loop=task == CONDITIONAL_MOVABLE_TASK,
                )
                if candidate is None:
                    raise ValueError(f"reuse episode is no longer eligible: {task}/{episode_index}")
                selected.append(candidate)
            selected.sort(key=lambda item: item.episode_index)
            materialization = "reuse_existing_extract"
            reuse_root = str(Path(reused["dataset_root"]).resolve())
        else:
            candidates = [
                candidate
                for row in metadata[task]
                if (
                    candidate := _candidate_from_row(
                        source_root,
                        row,
                        require_single_loop=task == CONDITIONAL_MOVABLE_TASK,
                    )
                )
                is not None
            ]
            clean_count = 9 if task == CONDITIONAL_MOVABLE_TASK else 10
            selected = _select_candidates(
                candidates,
                clean_count=clean_count,
                randomized_count=20 - clean_count,
            )
            materialization = "copy_from_source_dataset"
            reuse_root = None

        domain_counts = {
            domain: sum(item.domain == domain for item in selected)
            for domain in ("clean", "randomized")
        }
        arm_counts = {
            arm: sum(item.arm == arm for item in selected)
            for arm in ("left", "right", "both", "none")
        }
        tasks.append(
            {
                "task": task,
                "task_kind": _task_kind(task),
                "episode_count": len(selected),
                "episode_indices": [item.episode_index for item in selected],
                "domain_counts": domain_counts,
                "arm_counts": {key: value for key, value in arm_counts.items() if value},
                "candidate_pool_count": sum(
                    1
                    for row in metadata[task]
                    if bool(row["geometry_valid"])
                    and all(
                        (source_root / relative_path).is_file()
                        for relative_path in _episode_relative_files(int(row["episode_index"]))
                    )
                ),
                "single_loop_candidate_pool_count": (
                    144 if task == CONDITIONAL_MOVABLE_TASK else None
                ),
                "materialization": materialization,
                "reuse_dataset_root": reuse_root,
                "episodes": [item.to_json() for item in selected],
            }
        )

    reused_count = sum(item["materialization"] == "reuse_existing_extract" for item in tasks)
    source_count = len(tasks) - reused_count
    return {
        "format": "robotwin_target_only_20_selection_v2",
        "source_dataset_root": str(source_root),
        "output_dataset_root": str(DEFAULT_OUTPUT_ROOT),
        "camera": "cam_high",
        "content_policy": (
            "global episode ids; geometry-valid byte-identical parquet, HDF5, cam_high RGB "
            "and cam_high depth; per-task filtered metadata"
        ),
        "scope": {
            "strict_single_movable_source_episode_count": 14_994,
            "strict_single_movable_source_fraction": 0.5452,
            "strict_task_slice_count": 28,
            "articulated_action_site_task_slice_count": 3,
            "task_count": len(tasks),
            "episode_count": sum(int(item["episode_count"]) for item in tasks),
        },
        "sampling_policy": {
            "default_domain_counts": {"clean": 10, "randomized": 10},
            "place_bread_basket_domain_counts": {"clean": 9, "randomized": 11},
            "arm_policy": (
                "balance left/right overall when state-derived arm pools permit; otherwise "
                "spread deterministically across global episode ids"
            ),
            "spread_policy": "inclusive evenly spaced order statistics by global episode id",
        },
        "reuse_task_slice_count": reused_count,
        "reuse_episode_count": reused_count * 20,
        "source_copy_task_slice_count": source_count,
        "source_copy_episode_count": source_count * 20,
        "tasks": tasks,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _copy_verified(source: Path, destination: Path) -> dict[str, Any]:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    source_hash = _sha256(source)
    destination_hash = _sha256(destination)
    if source_hash != destination_hash:
        raise OSError(f"checksum mismatch after copy: {source} -> {destination}")
    return {
        "path": destination.as_posix(),
        "bytes": destination.stat().st_size,
        "sha256": destination_hash,
        "copied_from": source.as_posix(),
    }


def _filter_jsonl(
    source: Path,
    destination: Path,
    *,
    key: str,
    accepted: set[int],
) -> int:
    rows: list[str] = []
    with source.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line and int(json.loads(line)[key]) in accepted:
                rows.append(line)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return len(rows)


def _generated_file_record(path: Path, *, relative_to: Path, source: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "filtered_from": source.as_posix(),
    }


def _materialize_task(
    *,
    source_root: Path,
    staging_root: Path,
    final_root: Path,
    selection_path: Path,
    task_record: dict[str, Any],
) -> dict[str, Any]:
    task = str(task_record["task"])
    task_root = staging_root / task
    episode_ids = [int(value) for value in task_record["episode_indices"]]
    accepted = set(episode_ids)
    reuse_root_value = task_record.get("reuse_dataset_root")
    episode_source_root = Path(str(reuse_root_value)).resolve() if reuse_root_value else source_root
    copied_files: list[dict[str, Any]] = []
    for episode_index in episode_ids:
        for relative_path in _episode_relative_files(episode_index):
            record = _copy_verified(
                episode_source_root / relative_path,
                task_root / relative_path,
            )
            record["path"] = relative_path.as_posix()
            copied_files.append(record)

    for source in sorted((source_root / "meta").glob("*.json")):
        relative_path = source.relative_to(source_root)
        record = _copy_verified(source, task_root / relative_path)
        record["path"] = relative_path.as_posix()
        copied_files.append(record)
    structure_doc = source_root / "DATASET_STRUCTURE.md"
    if structure_doc.is_file():
        relative_path = structure_doc.relative_to(source_root)
        record = _copy_verified(structure_doc, task_root / relative_path)
        record["path"] = relative_path.as_posix()
        copied_files.append(record)

    metadata_counts: dict[str, int] = {}
    for name in ("episodes.jsonl", "episodes_stats.jsonl", "geometry_valid.jsonl"):
        count = _filter_jsonl(
            source_root / "meta" / name,
            task_root / "meta" / name,
            key="episode_index",
            accepted=accepted,
        )
        if count != len(episode_ids):
            raise ValueError(f"{task}/{name}: filtered {count}/{len(episode_ids)} rows")
        metadata_counts[name] = count
        copied_files.append(
            _generated_file_record(
                task_root / "meta" / name,
                relative_to=task_root,
                source=source_root / "meta" / name,
            )
        )

    accepted_task_indices: set[int] = set()
    for episode_index in episode_ids:
        parquet = task_root / _episode_relative_files(episode_index)[0]
        frame = pd.read_parquet(parquet, columns=list(INDEX_COLUMNS))
        for column in INDEX_COLUMNS:
            accepted_task_indices.update(int(value) for value in frame[column].unique())
    task_rows = _filter_jsonl(
        source_root / "meta/tasks.jsonl",
        task_root / "meta/tasks.jsonl",
        key="task_index",
        accepted=accepted_task_indices,
    )
    if task_rows != len(accepted_task_indices):
        raise ValueError(
            f"{task}/tasks.jsonl: filtered {task_rows}/{len(accepted_task_indices)} rows"
        )
    copied_files.append(
        _generated_file_record(
            task_root / "meta/tasks.jsonl",
            relative_to=task_root,
            source=source_root / "meta/tasks.jsonl",
        )
    )

    manifest = {
        "format": "robotwin_target_only_sparse_extract_v2",
        "profile": "target_only",
        "task": task,
        "task_kind": task_record["task_kind"],
        "camera": "cam_high",
        "source_dataset_root": str(source_root),
        "materialized_episode_files_from": str(episode_source_root),
        "selection_manifest": str(final_root / "SELECTION_MANIFEST.json"),
        "selection_source_manifest": str(selection_path),
        "content_policy": (
            "original global episode ids and byte-identical selected cam_high RGB-D "
            "episode files; filtered episode/task JSONL rows"
        ),
        "episode_indices": episode_ids,
        "episode_count": len(episode_ids),
        "episode_file_count": len(episode_ids) * 4,
        "episode_specific_metadata_rows": metadata_counts,
        "referenced_task_rows": task_rows,
        "copied_files": copied_files,
    }
    _write_json(task_root / "EXTRACT_MANIFEST.json", manifest)
    return {
        "profile": "target_only",
        "task": task,
        "task_kind": task_record["task_kind"],
        "dataset_root": str(final_root / task),
        "episode_count": len(episode_ids),
        "episode_indices": episode_ids,
        "domain_counts": task_record["domain_counts"],
        "arm_counts": task_record["arm_counts"],
        "materialization": task_record["materialization"],
        "materialized_episode_files_from": str(episode_source_root),
        "episode_file_count": len(episode_ids) * 4,
        "total_bytes": sum(int(record["bytes"]) for record in copied_files),
        "extract_manifest": str(final_root / task / "EXTRACT_MANIFEST.json"),
    }


def _validate_staging(staging_root: Path, selection: dict[str, Any]) -> None:
    if len(selection["tasks"]) != 31:
        raise ValueError("selection must contain exactly 31 task slices")
    total = 0
    for task_record in selection["tasks"]:
        task = str(task_record["task"])
        episode_ids = [int(value) for value in task_record["episode_indices"]]
        if len(episode_ids) != 20 or len(set(episode_ids)) != 20:
            raise ValueError(f"{task}: expected 20 unique episode ids")
        task_root = staging_root / task
        manifest = _read_json(task_root / "EXTRACT_MANIFEST.json")
        if manifest["episode_indices"] != episode_ids:
            raise ValueError(f"{task}: extract manifest differs from selection")
        for episode_index in episode_ids:
            missing = [
                relative_path.as_posix()
                for relative_path in _episode_relative_files(episode_index)
                if not (task_root / relative_path).is_file()
            ]
            if missing:
                raise FileNotFoundError(f"{task}/{episode_index}: missing {missing}")
        total += len(episode_ids)
    if total != 620:
        raise ValueError(f"selection materialized {total} episodes, expected 620")


def validate_output(output_root: Path, selection: dict[str, Any]) -> dict[str, int]:
    _validate_staging(output_root, selection)
    published_selection = _read_json(output_root / "SELECTION_MANIFEST.json")
    if published_selection != selection:
        raise ValueError("published selection manifest differs from the source plan")
    collection = _read_json(output_root / "EXTRACT_MANIFEST.json")
    if int(collection["task_count"]) != 31 or int(collection["episode_count"]) != 620:
        raise ValueError("collection manifest count mismatch")
    collection_tasks = {str(item["task"]): item for item in collection["datasets"]}
    if set(collection_tasks) != {str(item["task"]) for item in selection["tasks"]}:
        raise ValueError("collection manifest task set differs from selection")

    verified_files = 0
    verified_bytes = 0
    for task_record in selection["tasks"]:
        task = str(task_record["task"])
        task_root = output_root / task
        extract = _read_json(task_root / "EXTRACT_MANIFEST.json")
        if str(extract["task"]) != task:
            raise ValueError(f"{task}: extract manifest task mismatch")
        episode_ids = [int(value) for value in task_record["episode_indices"]]
        metadata_rows: list[dict[str, Any]] = []
        with (task_root / "meta/episodes.jsonl").open(encoding="utf-8") as handle:
            metadata_rows = [json.loads(line) for line in handle if line.strip()]
        if [int(row["episode_index"]) for row in metadata_rows] != episode_ids:
            raise ValueError(f"{task}: filtered episode metadata differs from selection")
        if any(
            str(row["full_structured_tasks"][0]) != task or not bool(row["geometry_valid"])
            for row in metadata_rows
        ):
            raise ValueError(f"{task}: metadata contains a wrong task or invalid geometry")
        for record in extract["copied_files"]:
            path = task_root / str(record["path"])
            if not path.is_file():
                raise FileNotFoundError(path)
            size = path.stat().st_size
            if size != int(record["bytes"]):
                raise OSError(f"{task}: byte-size mismatch: {path}")
            if _sha256(path) != str(record["sha256"]):
                raise OSError(f"{task}: checksum mismatch: {path}")
            verified_files += 1
            verified_bytes += size
        dataset_root = Path(str(collection_tasks[task]["dataset_root"]))
        if dataset_root != task_root:
            raise ValueError(f"{task}: collection dataset_root is stale: {dataset_root}")
    return {"verified_files": verified_files, "verified_bytes": verified_bytes}


def materialize(
    source_root: Path,
    output_root: Path,
    selection_path: Path,
    selection: dict[str, Any],
) -> None:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = output_root.parent / f".{output_root.name}.staging-{uuid.uuid4().hex[:8]}"
    if staging_root.exists():
        raise FileExistsError(staging_root)
    staging_root.mkdir()
    print(f"Materializing into staging directory: {staging_root}", flush=True)
    datasets = []
    try:
        for task_record in selection["tasks"]:
            task = str(task_record["task"])
            datasets.append(
                _materialize_task(
                    source_root=source_root,
                    staging_root=staging_root,
                    final_root=output_root,
                    selection_path=selection_path,
                    task_record=task_record,
                )
            )
            print(f"materialized {task}: 20 episodes", flush=True)
        shutil.copy2(selection_path, staging_root / "SELECTION_MANIFEST.json")
        collection = {
            "format": "robotwin_target_only_extract_collection_v2",
            "source_dataset_root": str(source_root),
            "selection_manifest": str(output_root / "SELECTION_MANIFEST.json"),
            "layout": "<output_root>/<task>/RoboTwin sparse cam_high RGB-D dataset",
            "profile": "target_only",
            "task_count": len(datasets),
            "episode_count": sum(int(item["episode_count"]) for item in datasets),
            "datasets": datasets,
        }
        _write_json(staging_root / "EXTRACT_MANIFEST.json", collection)
        _validate_staging(staging_root, selection)
        staging_root.rename(output_root)
    except Exception:
        print(f"Materialization failed; staging data retained at {staging_root}", flush=True)
        raise
    print(f"Materialized 31 task slices / 620 episodes at {output_root}")


def main() -> None:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    selection_path = args.selection_output.expanduser().resolve()
    selection = build_selection(
        source_root,
        args.profile_reuse_root.expanduser().resolve(),
        args.adjust_reuse_root.expanduser().resolve(),
    )
    selection["output_dataset_root"] = str(output_root)
    if selection_path.exists():
        existing = _read_json(selection_path)
        if existing != selection:
            raise ValueError(
                f"selection drift detected; refusing to overwrite existing plan: {selection_path}"
            )
        print(f"Verified existing selection manifest: {selection_path}")
    else:
        _write_json(selection_path, selection)
        print(f"Wrote selection manifest: {selection_path}")
    print(
        f"Plan: {selection['scope']['task_count']} tasks / "
        f"{selection['scope']['episode_count']} episodes; "
        f"reuse {selection['reuse_episode_count']}, copy from source "
        f"{selection['source_copy_episode_count']}"
    )
    if args.materialize:
        materialize(source_root, output_root, selection_path, selection)
        validation = validate_output(output_root, selection)
        print(
            f"Validated {validation['verified_files']} files / "
            f"{validation['verified_bytes'] / 2**20:.1f} MiB"
        )
    elif args.validate_only:
        validation = validate_output(output_root, selection)
        print(
            f"Validated {validation['verified_files']} files / "
            f"{validation['verified_bytes'] / 2**20:.1f} MiB at {output_root}"
        )


if __name__ == "__main__":
    main()
