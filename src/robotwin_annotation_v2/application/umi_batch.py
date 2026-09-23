"""Resumable UMI inventory, single-arm screening, and native-resolution task preparation."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd
import yaml

from robotwin_annotation_v2.adapters.artifact_store import ArtifactStore
from robotwin_annotation_v2.adapters.qwen_client import OpenAICompatibleQwenClient
from robotwin_annotation_v2.adapters.real_mcap import read_source_metadata, write_jsonl
from robotwin_annotation_v2.adapters.umi_mcap import (
    UmiTaskConfig,
    convert_umi_episode,
    write_umi_metadata,
)
from robotwin_annotation_v2.adapters.umi_review import (
    load_umi_previews,
    preview_identity,
    screen_umi_episode,
)
from robotwin_annotation_v2.domain import AnnotationMode


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def initialize_batch(source_root: Path, batch_dir: Path, rules_path: Path) -> dict[str, Any]:
    """Freeze original IDs, source fingerprints, semantics and screening prompt."""

    source_root, batch_dir = source_root.resolve(), batch_dir.resolve()
    rules = yaml.safe_load(rules_path.read_text())
    prompt = (rules_path.parent / rules["screen_prompt"]).read_text()
    manifest = _read(source_root / "manifest.json")
    records = [r for r in manifest if r["action_id"] not in rules["excluded_actions"]]
    missing = set(rules["excluded_episodes"]) - {r["id"] for r in records}
    if missing:
        raise ValueError(f"configured exclusions absent from source inventory: {sorted(missing)}")
    episodes = []
    for index, record in enumerate(sorted(records, key=lambda r: (r["action_id"], r["id"]))):
        task = rules["tasks"][record["action_id"]]
        source_dir = (source_root / record["relative_path"]).resolve()
        if not source_dir.is_relative_to(source_root):
            raise ValueError("source path escapes dataset root")
        stat = (source_dir / "episode.mcap").stat()
        task_data = _read(source_dir / "task.json")
        steps = task_data["item_info"]["steps"]
        action_text = " ".join(s["action"].get("rendered_en", "") for s in steps).strip()
        action_text = action_text or record["action_text"]
        episodes.append(
            {
                **record,
                "episode_index": index,
                "source_dir": str(source_dir),
                "source_size": stat.st_size,
                "source_mtime_ns": stat.st_mtime_ns,
                "task_text": f"{action_text}\n{task['objects']}",
            }
        )
    if len({e["id"] for e in episodes}) != len(episodes):
        raise ValueError("duplicate UMI source IDs")
    inventory = {
        "version": 1,
        "source_root": str(source_root),
        "rules": rules,
        "prompt": prompt,
        "episodes": episodes,
        "excluded_action_count": len(manifest) - len(records),
    }
    inventory["signature"] = _digest(inventory)
    path = batch_dir / "inventory.json"
    if path.exists() and _read(path)["signature"] != inventory["signature"]:
        raise ValueError("batch inputs or rules changed; use a new batch directory")
    if not path.exists():
        ArtifactStore.write_json(path, inventory)
    return inventory


def screen_batch(
    inventory: dict[str, Any],
    batch_dir: Path,
    client: OpenAICompatibleQwenClient,
    *,
    workers: int = 4,
    limit_per_action: int | None = None,
) -> dict[str, int]:
    """Cache each decision independently; API or evidence failures never select an arm."""

    selected = []
    per_action: Counter[str] = Counter()
    for episode in inventory["episodes"]:
        action = episode["action_id"]
        if limit_per_action is None or per_action[action] < limit_per_action:
            selected.append(episode)
            per_action[action] += 1
    rules = inventory["rules"]

    def review(episode: dict[str, Any]) -> dict[str, Any]:
        path = batch_dir / "screening" / f"{episode['id']}.json"
        base = {"id": episode["id"], "action_id": episode["action_id"]}
        exclusion = rules["excluded_episodes"].get(episode["id"])
        if exclusion:
            result = {
                **base,
                "status": "excluded_review",
                "reason": exclusion,
                "review_source": "assistant_multiframe_review",
            }
            ArtifactStore.write_json(path, result)
            return result
        try:
            previews = load_umi_previews(Path(episode["source_dir"]))
            evidence = preview_identity(previews)
            identity = _digest(
                {
                    "inventory": inventory["signature"],
                    "previews": evidence,
                    "model": client.model_id,
                    "endpoint": client.endpoint,
                }
            )
            if path.exists():
                saved = _read(path)
                if saved.get("input_signature") == identity and saved["status"] != "screen_failed":
                    return dict(saved)
            answer = screen_umi_episode(
                client,
                previews=previews,
                prompt=inventory["prompt"],
                task_text=episode["task_text"],
            )
            decision = answer["decision"]
            status = "needs_arm_review"
            if decision in {"bimanual", "no_action"}:
                status = f"excluded_{decision}"
            elif (
                decision in {"left", "right"}
                and answer["confidence"] >= rules["min_arm_confidence"]
            ):
                status = "selected"
            result = {
                **base,
                **answer,
                "status": status,
                "input_signature": identity,
                "previews": evidence,
            }
        except Exception as exc:  # noqa: BLE001 - preserve per-source failures for reruns
            result = {**base, "status": "screen_failed", "error": f"{type(exc).__name__}: {exc}"}
        ArtifactStore.write_json(path, result)
        return result

    counts: Counter[str] = Counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = [pool.submit(review, episode) for episode in selected]
        for future in as_completed(pending):
            result = future.result()
            counts[result["status"]] += 1
            progress = {
                "phase": "screening",
                "pid": os.getpid(),
                "completed": sum(counts.values()),
                "total": len(selected),
                "counts": dict(counts),
                "last": result["id"],
            }
            ArtifactStore.write_json(batch_dir / "screening_status.json", progress)
            print(json.dumps(progress, ensure_ascii=False), flush=True)
    return dict(counts)


def _convert_selected(
    episode: dict[str, Any], selection: dict[str, Any], task: dict[str, Any], batch_dir: Path
) -> dict[str, Any]:
    source_path = Path(episode["source_dir"]) / "episode.mcap"
    stat = source_path.stat()
    if (stat.st_size, stat.st_mtime_ns) != (episode["source_size"], episode["source_mtime_ns"]):
        raise ValueError("source MCAP changed since inventory was created")
    identity = _digest({"source": episode, "selection": selection, "converter_version": 1})
    output = batch_dir / "converted" / episode["id"]
    summary_path = output / "conversion.json"
    if summary_path.exists():
        saved = _read(summary_path)
        if saved["input_signature"] != identity:
            raise ValueError("conversion inputs changed; use a new batch directory")
        return dict(saved)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{episode['id']}-", dir=output.parent))
    try:
        arm = cast(Literal["left", "right"], selection["decision"])
        item = convert_umi_episode(
            read_source_metadata(source_path),
            staging,
            episode_id=episode["episode_index"],
            active_arm=arm,
            active_arm_source=selection["review_source"],
        )
        group = f"{task['name']}_{item.width}x{item.height}"
        config = UmiTaskConfig(
            AnnotationMode(task["annotation_mode"]),
            episode["task_text"],
            episode["action_text"],
            {"episode.mcap": arm},
        )
        write_umi_metadata(
            staging,
            final_output=output,
            task=group,
            task_text=config.task_text,
            task_text_zh=config.task_text_zh,
            converted=[item],
            limit=None,
            task_config=config,
        )
        metadata = _read_lines(staging / "meta/episodes.jsonl")[0]
        metadata.update(
            source_episode_uid=episode["id"],
            source_mcap=str(source_path),
            arm_screening=str(batch_dir / "screening" / f"{episode['id']}.json"),
        )
        write_jsonl(staging / "meta/episodes.jsonl", [metadata])
        result = {
            "id": episode["id"],
            "episode_index": episode["episode_index"],
            "action_id": episode["action_id"],
            "status": "prepared",
            "group": group,
            "input_signature": identity,
            "root": str(output),
            "frame_count": item.frame_count,
            "width": item.width,
            "height": item.height,
            "fps": item.fps,
            "active_arm": arm,
            "source_camera_topic": item.timeline.source_camera_topic,
        }
        ArtifactStore.write_json(staging / "conversion.json", result)
        os.replace(staging, output)
        return result
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _read_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _assemble_group(group: str, items: list[dict[str, Any]], batch_dir: Path) -> dict[str, Any]:
    """Link cached videos and sidecars; write consecutive frame indices and per-clip tasks."""

    output = batch_dir / "inputs" / group
    output.mkdir(parents=True, exist_ok=True)
    items.sort(key=lambda item: item["episode_index"])
    first = Path(items[0]["root"])
    manifest = _read(first / "EXTRACT_MANIFEST.json")
    metadata, tasks, included = [], [], []
    frame_start = 0
    for task_index, item in enumerate(items):
        root = Path(item["root"])
        for folder in ("videos", "sidecars"):
            for source in (root / folder).rglob("*"):
                if not source.is_file():
                    continue
                destination = output / source.relative_to(root)
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.is_symlink() and destination.resolve() == source.resolve():
                    continue
                destination.symlink_to(source)
        parquet = next((root / "data").rglob("*.parquet"))
        frame_data = pd.read_parquet(parquet)
        frame_data["index"] = range(frame_start, frame_start + len(frame_data))
        frame_data["task_index"] = task_index
        destination = output / parquet.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame_data.to_parquet(destination, index=False)
        frame_start += len(frame_data)
        metadata.extend(_read_lines(root / "meta/episodes.jsonl"))
        task = _read_lines(root / "meta/tasks.jsonl")[0]
        tasks.append({**task, "task_index": task_index})
        included.append(
            {
                "episode_index": item["episode_index"],
                "source_episode_uid": item["id"],
                "source_mcap": metadata[-1]["source_mcap"],
            }
        )
    ids = [item["episode_index"] for item in items]
    topics = sorted({item["source_camera_topic"] for item in items})
    manifest.pop("source_camera_topic", None)
    manifest.update(
        dataset_root=str(output),
        episode_indices=ids,
        episode_count=len(ids),
        smoke_episode_ids=ids[:2],
        regression_episode_ids=ids,
        included_sources=included,
        source_camera_topics=topics,
    )
    ArtifactStore.write_json(output / "EXTRACT_MANIFEST.json", manifest)
    write_jsonl(output / "meta/episodes.jsonl", metadata)
    write_jsonl(output / "meta/tasks.jsonl", tasks)
    info = _read(first / "meta/info.json")
    info.update(
        total_episodes=len(ids),
        total_frames=frame_start,
        total_tasks=len(tasks),
        total_videos=len(ids),
    )
    ArtifactStore.write_json(output / "meta/info.json", info)
    return {
        "group": group,
        "root": str(output),
        "action_id": items[0]["action_id"],
        "episode_ids": ids,
        "frame_count": frame_start,
    }


def prepare_batch(
    inventory: dict[str, Any], batch_dir: Path, *, workers: int = 4
) -> dict[str, Any]:
    """Convert only confidently screened single-arm clips and retain all other statuses."""

    selected, records = [], []
    for episode in inventory["episodes"]:
        path = batch_dir / "screening" / f"{episode['id']}.json"
        selection = _read(path) if path.exists() else {"status": "not_screened"}
        if selection["status"] == "selected":
            selected.append((episode, selection))
        else:
            records.append(
                {
                    "id": episode["id"],
                    "episode_index": episode["episode_index"],
                    "action_id": episode["action_id"],
                    **selection,
                }
            )
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _convert_selected,
                episode,
                selection,
                inventory["rules"]["tasks"][episode["action_id"]],
                batch_dir,
            ): episode
            for episode, selection in selected
        }
        for future in as_completed(futures):
            episode = futures[future]
            try:
                result = future.result()
                groups[result["group"]].append(result)
            except Exception as exc:  # noqa: BLE001 - isolate corrupt sources
                result = {
                    "id": episode["id"],
                    "episode_index": episode["episode_index"],
                    "action_id": episode["action_id"],
                    "status": "prepare_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            records.append(result)
            print(
                json.dumps(
                    {
                        "phase": "prepare",
                        "completed": len(records),
                        "total": len(inventory["episodes"]),
                        "last": result,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    plan = {
        "inventory_signature": inventory["signature"],
        "episodes": sorted(records, key=lambda r: r["episode_index"]),
        "groups": [
            _assemble_group(group, items, batch_dir) for group, items in sorted(groups.items())
        ],
        "counts": dict(Counter(r["status"] for r in records)),
    }
    ArtifactStore.write_json(batch_dir / "batch_plan.json", plan)
    return plan
