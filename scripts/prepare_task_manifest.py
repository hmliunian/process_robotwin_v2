"""Bind a remote-operation task index to read-only object-mask input views."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robotwin_annotation_v2.adapters.artifact_store import ArtifactStore
from robotwin_annotation_v2.adapters.real_mcap import RealMcapError, normalize_gripper_loop
from robotwin_annotation_v2.adapters.robotwin_dataset import EpisodePaths, EpisodeState
from robotwin_annotation_v2.domain import TargetOnlyTaskKind
from robotwin_annotation_v2.pipeline.timeline_detector import (
    detect_episode_target_only,
    detect_target_only_events,
)


def source_file(root: Path, value: str) -> Path:
    path = root / value
    if Path(value).is_absolute() or ".." in Path(value).parts:
        raise ValueError(f"manifest file path must be relative: {value}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def episode_metadata(root: Path, episode: dict[str, Any]) -> dict[str, Any]:
    """Derive events from real state, retaining raw files and corrected task labels."""

    parquet = source_file(root, episode["files"]["parquet"])
    frame = pd.read_parquet(parquet, columns=["frame_index", "episode_index", "observation.state"])
    count = len(frame)
    if count != episode["frame_count"] or not np.array_equal(
        frame["frame_index"], np.arange(count)
    ):
        raise ValueError(f"invalid frame count/index: {parquet}")
    if not (frame["episode_index"] == episode["episode_index"]).all():
        raise ValueError(f"episode index mismatch: {parquet}")
    state = np.stack(frame["observation.state"]).astype(np.float64)
    if state.shape != (count, episode["state_dimension"]) or not np.isfinite(state).all():
        raise ValueError(f"invalid state array: {parquet}")
    metadata: dict[str, Any] = {
        "episode_index": episode["episode_index"],
        "episode_uid": episode["episode_uid"],
        "tasks": [episode["task_text"]],
        "length": count,
        "split": episode["split"],
        "source_episode_id": episode["source_episode_id"],
    }
    provenance: dict[str, Any] = {
        "source_parquet": episode["files"]["parquet"],
        "source_parquet_sha256": hashlib.sha256(parquet.read_bytes()).hexdigest(),
        "detector": "detect_episode_target_only",
    }
    if state.shape[1] == 7:
        info = json.loads((root / episode["source_dataset"] / "meta/info.json").read_text())
        names = info["features"]["observation.state"]["names"]
        expected = [f"right_eef_{axis}" for axis in ("x", "y", "z", "roll", "pitch", "yaw")]
        if names != [*expected, "right_gripper_open"]:
            raise ValueError(f"unsupported single-arm state layout: {names}")
        try:
            gripper, close, reopen = normalize_gripper_loop(state[:, 6], allow_terminal_hold=True)
        except RealMcapError as exc:
            metadata["video_events"] = {"active_arm": "right", "t_start": 0, "t_end": count - 1}
            provenance.update(detector="video_window", state_timeline_error=str(exc))
            metadata["timeline_derivation"] = provenance
            return metadata
        events = detect_target_only_events(gripper, state[:, :6], arm="right")
        provenance.update(
            detector="normalize_gripper_loop -> detect_target_only_events",
            normalization="existing real_mcap 3-frame median; q95 open*0.70, closed*0.65",
            close_frame=close,
            reopen_frame=None if reopen == count else reopen,
        )
    elif state.shape[1] == 14:
        events = detect_episode_target_only(
            EpisodeState(
                count,
                episode["task_text"],
                state[:, (6, 13)],
                np.stack((state[:, :6], state[:, 7:13]), axis=1),
                EpisodePaths(parquet, parquet, parquet),
            )
        )
    else:
        raise ValueError(f"unsupported state dimension: {state.shape[1]}")
    metadata.update(target_only_events=events.to_json(), timeline_derivation=provenance)
    return metadata


def prepare(manifest_path: Path, output_dir: Path, *, task_kind: str) -> dict[str, Any]:
    """Create new per-source/task views without changing or copying input videos."""

    manifest_path = manifest_path.resolve()
    root = manifest_path.parent
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    episodes = payload["episodes"]
    if payload.get("format") != "remote_operation_task_manifest_v1" or not episodes:
        raise ValueError("expected a non-empty remote_operation_task_manifest_v1")
    if len(episodes) != payload["total_episodes"]:
        raise ValueError("manifest total_episodes mismatch")
    ids = [e["episode_uid"] for e in episodes]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate episode_uid")
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    metadata = {}
    for episode in episodes:
        source, task, camera = (episode[k] for k in ("source_dataset", "task", "primary_camera"))
        for name in (source, task, camera):
            if not name or Path(name).name != name or name in {".", ".."}:
                raise ValueError(f"invalid dataset identifier: {name}")
        if episode["episode_uid"] != f"{source}/episode_{episode['episode_index']:06d}":
            raise ValueError("episode_uid does not match source and episode index")
        source_file(root, episode["files"]["videos"][camera])
        metadata[episode["episode_uid"]] = episode_metadata(root, episode)
        timeline = (
            "video_window"
            if "video_events" in metadata[episode["episode_uid"]]
            else "episode_metadata"
        )
        groups[source, task, camera, timeline].append(episode)

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(manifest_path, output_dir / manifest_path.name)
    csv_path = manifest_path.with_suffix(".csv")
    if csv_path.is_file():
        shutil.copy2(csv_path, output_dir / csv_path.name)
    plan: dict[str, Any] = {
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "episode_count": len(episodes),
        "frame_count": sum(e["frame_count"] for e in episodes),
        "stage_mode": "object_source_only",
        "groups": [],
    }
    for (source, task, camera, timeline), selected in sorted(groups.items()):
        group_id = f"{source}--{task}"
        if timeline == "video_window":
            group_id += "--video-window"
        view = output_dir / "inputs" / group_id
        for directory in ("data", "videos", "sidecars", "meta"):
            (view / directory).mkdir(parents=True, exist_ok=True)
        episode_ids = [e["episode_index"] for e in selected]
        if len(set(episode_ids)) != len(episode_ids):
            raise ValueError(f"duplicate source episode index: {group_id}")
        for episode in selected:
            chunk = f"chunk-{episode['episode_index'] // 1000:03d}"
            stem = f"episode_{episode['episode_index']:06d}"
            files = [
                (episode["files"]["parquet"], view / "data" / chunk / f"{stem}.parquet"),
                (
                    episode["files"]["videos"][camera],
                    view / "videos" / chunk / f"observation.images.{camera}" / f"{stem}.mp4",
                ),
            ]
            for relative, destination in files:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.symlink_to(source_file(root, relative))
        (view / "meta/episodes.jsonl").write_text(
            "".join(
                json.dumps(metadata[e["episode_uid"]], ensure_ascii=False) + "\n" for e in selected
            ),
            encoding="utf-8",
        )
        binding = {
            "profile": "target_only",
            "task": task,
            "task_kind": task_kind,
            "camera": camera,
            "dataset_root": str(view),
            "source_dataset_root": str(root / source),
            "episode_indices": episode_ids,
            "regression_episode_ids": episode_ids,
            "timeline_source": timeline,
            "object_source_only": True,
            "source_manifest_sha256": plan["source_manifest_sha256"],
        }
        ArtifactStore.write_json(view / "EXTRACT_MANIFEST.json", binding)
        plan["groups"].append(
            {
                "group_id": group_id,
                "input_root": str(view),
                "task": task,
                "camera": camera,
                "source_dataset": source,
                "episode_ids": episode_ids,
            }
        )
    ArtifactStore.write_json(output_dir / "batch_plan.json", plan)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--task-kind", required=True, choices=[k.value for k in TargetOnlyTaskKind])
    args = parser.parse_args()
    plan = prepare(args.manifest, args.output_dir, task_kind=args.task_kind)
    print(json.dumps({"episode_count": plan["episode_count"], "output_dir": str(args.output_dir)}))


if __name__ == "__main__":
    main()
