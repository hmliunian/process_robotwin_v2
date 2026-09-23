"""Screen and process a nested UMI dataset, excluding configured action categories."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from robotwin_annotation_v2.adapters.qwen_client import OpenAICompatibleQwenClient
from robotwin_annotation_v2.application.umi_batch import (
    initialize_batch,
    prepare_batch,
    screen_batch,
)
from robotwin_annotation_v2.config import load_profile
from robotwin_annotation_v2.domain import AnnotationMode, TargetProfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("batch_dir", type=Path)
    parser.add_argument(
        "--rules", type=Path, default=PROJECT_ROOT / "configs/datasets/umi_h200.yaml"
    )
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/process.yaml")
    parser.add_argument("--phase", choices=("screen", "prepare", "all"), default="screen")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit-per-action", type=int)
    parser.add_argument("--gpus", type=int, nargs="+")
    args = parser.parse_args()
    if args.workers < 1 or (args.limit_per_action is not None and args.limit_per_action < 1):
        parser.error("workers and limit-per-action must be positive")
    if args.phase == "all" and (not args.gpus or args.limit_per_action is not None):
        parser.error("all requires --gpus and does not accept --limit-per-action")
    inventory = initialize_batch(args.source_root, args.batch_dir, args.rules)
    if args.phase == "prepare":
        prepare_batch(inventory, args.batch_dir.resolve(), workers=args.workers)
        return
    profile = load_profile(
        args.config, mode=AnnotationMode.TARGET_ONLY, target_profile=TargetProfile.VIDEO_OBJECT
    )
    if profile.qwen.api_key_env and not os.environ.get(profile.qwen.api_key_env):
        key = (PROJECT_ROOT / "secrets/qwen_api_key.txt").read_text().strip()
        if not key or any(c.isspace() for c in key):
            raise ValueError("invalid Qwen API key file")
        os.environ[profile.qwen.api_key_env] = key
    client = OpenAICompatibleQwenClient.from_config(profile.qwen)
    client.health()
    screen_batch(
        inventory,
        args.batch_dir.resolve(),
        client,
        workers=args.workers,
        limit_per_action=args.limit_per_action,
    )
    if args.phase == "all":
        from scripts.run_umi_batch import run_batch

        prepare_batch(inventory, args.batch_dir.resolve(), workers=args.workers)
        run_batch(args.batch_dir, args.config, gpus=tuple(args.gpus), qwen_workers=args.workers)


if __name__ == "__main__":
    main()
