#!/usr/bin/env python3
"""Run or inspect the three-stage target/receiver pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from robotwin_annotation_v2.adapters import (
    ArtifactStore,
    OpenAICompatibleQwenClient,
    RoboTwinDataset,
    Sam3Adapter,
    Sam3Error,
    sam3_video_resource,
)
from robotwin_annotation_v2.config import PipelineConfig, load_config
from robotwin_annotation_v2.models import EpisodeRef, LoopContext, MaskStatus, SemanticPlan
from robotwin_annotation_v2.pipeline import (
    QwenStageError,
    SamStageError,
    build_loop_context,
    parse_semantic_plan,
    run_qwen_stage,
    run_sam_stage,
    save_sam_artifacts,
)


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


def run_qwen(config: PipelineConfig, episode_index: int, run_id: str | None) -> None:
    dataset = _dataset(config)
    ref = _episode_ref(config, episode_index)
    context = build_loop_context(dataset, ref)
    frames = dataset.read_frames(
        ref,
        (frame.frame_id for frame in context.semantic_frames),
    )
    client = OpenAICompatibleQwenClient(
        endpoint=config.qwen.endpoint,
        model=config.qwen.model,
        timeout_seconds=config.qwen.timeout_seconds,
    )
    store = ArtifactStore(config.output_root)
    selected_run_id = run_id or store.new_run_id()
    loop_path = store.save_loop(selected_run_id, ref, context.to_json())
    try:
        result = run_qwen_stage(context, frames, config.qwen, client)
    except QwenStageError as exc:
        paths = store.save_qwen_failure(
            selected_run_id,
            ref,
            error=str(exc),
            rendered_prompt=exc.rendered_prompt,
            raw_response=exc.raw_response,
        )
        _print(
            {
                "run_id": selected_run_id,
                "stage": "qwen_failed",
                "loop_artifact": str(loop_path),
                "artifacts": {name: str(path) for name, path in paths.items()},
                "error": str(exc),
            }
        )
        raise SystemExit(2) from exc

    plan = result.semantic_plan
    paths = store.save_semantic_plan(
        selected_run_id,
        ref,
        plan.to_json(),
        rendered_prompt=result.rendered_prompt,
        raw_response=plan.raw_response,
    )
    _print(
        {
            "run_id": selected_run_id,
            "stage": "qwen",
            "loop_artifact": str(loop_path),
            "artifacts": {name: str(path) for name, path in paths.items()},
            "server": result.health,
            "target": {
                "seed_frame_id": plan.target.seed_frame_id,
                "primary_query": plan.target.primary_query,
            },
            "receiver": {
                "seed_frame_id": plan.receiver.seed_frame_id,
                "primary_query": plan.receiver.primary_query,
            },
        }
    )


def _load_saved_semantic_plan(
    store: ArtifactStore,
    run_id: str,
    context: LoopContext,
) -> SemanticPlan:
    episode_dir = store.episode_dir(run_id, context.episode)
    loop_path = episode_dir / "loop.json"
    semantic_path = episode_dir / "semantic_plan.json"
    prompt_path = episode_dir / "qwen_rendered_prompt.txt"
    raw_path = episode_dir / "qwen_raw_response.txt"
    for path in (loop_path, semantic_path, prompt_path, raw_path):
        if not path.is_file():
            raise ValueError(f"required Stage-2 artifact is missing: {path}")
    saved_loop = json.loads(loop_path.read_text(encoding="utf-8"))
    if saved_loop != context.to_json():
        raise ValueError("saved loop.json differs from the current Stage-1 result")
    saved_plan = json.loads(semantic_path.read_text(encoding="utf-8"))
    rendered_prompt = prompt_path.read_text(encoding="utf-8")
    raw_response = raw_path.read_text(encoding="utf-8")
    plan = parse_semantic_plan(
        raw_response,
        context=context,
        model=str(saved_plan.get("model", "")),
        rendered_prompt=rendered_prompt,
    )
    if saved_plan != plan.to_json():
        raise ValueError("saved semantic_plan.json fails provenance validation")
    return plan


def run_sam(config: PipelineConfig, episode_index: int, run_id: str) -> None:
    dataset = _dataset(config)
    ref = _episode_ref(config, episode_index)
    context = build_loop_context(dataset, ref)
    store = ArtifactStore(config.output_root)
    plan = _load_saved_semantic_plan(store, run_id, context)
    shape_values = tuple(int(value) for value in dataset.manifest["frame_shape_hw"])
    if len(shape_values) != 2:
        raise ValueError(f"invalid dataset frame shape: {shape_values}")
    frame_shape = (shape_values[0], shape_values[1])

    backend: Sam3Adapter | None = None
    try:
        backend = Sam3Adapter(
            checkpoint_path=config.sam3.checkpoint,
            gpus=config.sam3.gpus,
        )
        with sam3_video_resource(
            Path(context.video_source),
            minimum_frame_count=context.frame_count,
        ) as resource_path:
            result = run_sam_stage(
                context,
                plan,
                backend,
                resource_path,
                frame_shape=frame_shape,
                mask_config=config.mask,
            )
        seed_frame_ids = {
            value
            for value in (plan.target.seed_frame_id, plan.receiver.seed_frame_id)
            if value is not None
        }
        seed_images = dataset.read_frames(ref, seed_frame_ids) if seed_frame_ids else {}
        mask_run = save_sam_artifacts(
            store,
            run_id,
            context,
            plan,
            result,
            seed_images=seed_images,
        )
    except (Sam3Error, SamStageError, RuntimeError, ValueError, OSError) as exc:
        failure_path = store.save_sam_failure(run_id, ref, error=str(exc))
        _print(
            {
                "run_id": run_id,
                "stage": "sam_failed",
                "artifact": str(failure_path),
                "error": str(exc),
            }
        )
        raise SystemExit(3) from exc
    finally:
        if backend is not None:
            backend.shutdown()

    _print(
        {
            "run_id": run_id,
            "stage": (
                "sam"
                if all(role.status is MaskStatus.OK for role in mask_run.roles)
                else "sam_incomplete"
            ),
            "artifact": str(Path(mask_run.artifact_dir) / "run_manifest.json"),
            "roles": [role.to_json() for role in mask_run.roles],
        }
    )
    if any(role.status is not MaskStatus.OK for role in mask_run.roles):
        raise SystemExit(4)


def run_pipeline(config: PipelineConfig, episode_index: int, run_id: str | None) -> None:
    selected_run_id = run_id or ArtifactStore.new_run_id()
    run_qwen(config, episode_index, selected_run_id)
    run_sam(config, episode_index, selected_run_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "loop", "qwen", "sam", "run"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument(
            "--config",
            type=Path,
            default=Path("configs/pilot_move_pillbottle_pad.yaml"),
        )
        if command in {"loop", "qwen", "sam", "run"}:
            subparser.add_argument("--episode", type=int, required=True)
            subparser.add_argument("--run-id", required=command == "sam")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.command == "preflight":
        run_preflight(config)
    elif args.command == "loop":
        run_loop(config, args.episode, args.run_id)
    elif args.command == "qwen":
        run_qwen(config, args.episode, args.run_id)
    elif args.command == "sam":
        run_sam(config, args.episode, args.run_id)
    elif args.command == "run":
        run_pipeline(config, args.episode, args.run_id)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
