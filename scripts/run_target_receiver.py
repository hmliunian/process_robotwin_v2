#!/usr/bin/env python3
"""Run or inspect the three-stage target/receiver pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from robotwin_annotation_v2.adapters import ArtifactStore, RoboTwinDataset
from robotwin_annotation_v2.config import PipelineConfig, load_config
from robotwin_annotation_v2.models import EpisodeRef
from robotwin_annotation_v2.pipeline import build_loop_context


def _dataset(config: PipelineConfig) -> RoboTwinDataset:
    return RoboTwinDataset(
        config.dataset.root,
        task=config.dataset.task,
        camera=config.dataset.camera,
        manifest_path=config.dataset.manifest,
    )


def _print(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _episode_ref(config: PipelineConfig, episode_index: int) -> EpisodeRef:
    if episode_index not in config.dataset.regression_episode_ids:
        raise ValueError(f"episode {episode_index} is not in the configured regression manifest")
    return EpisodeRef(config.dataset.task, episode_index, config.dataset.camera)


def run_preflight(config: PipelineConfig) -> None:
    dataset = _dataset(config)
    manifest_ids = tuple(int(value) for value in dataset.manifest["regression_episode_ids"])
    if manifest_ids != config.dataset.regression_episode_ids:
        raise ValueError("config regression_episode_ids differ from the dataset manifest")
    report = dataset.preflight(config.dataset.regression_episode_ids)
    _print(report)


def run_loop(config: PipelineConfig, episode_index: int, run_id: str | None) -> None:
    dataset = _dataset(config)
    ref = _episode_ref(config, episode_index)
    context = build_loop_context(dataset, ref)
    store = ArtifactStore(config.output_root)
    selected_run_id = run_id or store.new_run_id()
    path = store.save_loop(selected_run_id, ref, context.to_json())
    _print(
        {
            "run_id": selected_run_id,
            "stage": "loop",
            "artifact": str(path),
            "events": context.events.to_json(),
            "semantic_frame_ids": [frame.frame_id for frame in context.semantic_frames],
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "loop"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument(
            "--config",
            type=Path,
            default=Path("configs/pilot_move_pillbottle_pad.yaml"),
        )
        if command == "loop":
            subparser.add_argument("--episode", type=int, required=True)
            subparser.add_argument("--run-id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.command == "preflight":
        run_preflight(config)
    elif args.command == "loop":
        run_loop(config, args.episode, args.run_id)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
