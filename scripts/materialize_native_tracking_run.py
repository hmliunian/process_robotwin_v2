#!/usr/bin/env python3
"""Create a versioned mask run from saved SAM3 native tracks without rerunning SAM3."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image

from robotwin_annotation_v2.adapters import ArtifactStore, RoboTwinDataset
from robotwin_annotation_v2.config import PipelineConfig, load_config
from robotwin_annotation_v2.models import (
    EpisodeRef,
    FrameWindow,
    LoopContext,
    MaskStatus,
    SemanticPlan,
)
from robotwin_annotation_v2.pipeline import (
    RoleMaskData,
    SamStageResult,
    build_loop_context,
    compose_visible_mask,
    dilate_envelope,
    evaluate_temporal_mask,
    parse_semantic_plan,
    save_sam_artifacts,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RoleName = Literal["target", "receiver"]
ROLE_NAMES: tuple[RoleName, ...] = ("target", "receiver")
ROLE_DIRECTORIES = ("target_0", "receiver_0")
STAGE_FILES = (
    "loop.json",
    "semantic_plan.json",
    "qwen_rendered_prompt.txt",
    "qwen_raw_response.txt",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "pilot_move_pillbottle_pad.yaml",
    )
    parser.add_argument(
        "--selection-manifest",
        type=Path,
        default=(
            PROJECT_ROOT
            / "artifacts"
            / "rendered_videos"
            / "coverage20_best_current"
            / "manifest.json"
        ),
        help="Render manifest that pins the source run for every episode",
    )
    parser.add_argument("--runs-root", type=Path, help="Defaults to output.root in config")
    parser.add_argument("--output-run-id", required=True)
    parser.add_argument("--episode-ids", type=int, nargs="*")
    parser.add_argument(
        "--identity-review",
        type=Path,
        help="Optional dataset-specific review JSON; rejected roles are quarantined",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset(config: PipelineConfig) -> RoboTwinDataset:
    return RoboTwinDataset(
        config.dataset.root,
        task=config.dataset.task,
        camera=config.dataset.camera,
        manifest_path=config.dataset.manifest,
    )


def _load_mask(path: Path | None, shape: tuple[int, int]) -> np.ndarray | None:
    if path is None or not path.is_file():
        return None
    with Image.open(path) as image:
        mask = np.asarray(image.convert("L")) > 0
    if mask.shape != shape:
        raise ValueError(f"mask shape mismatch: expected={shape}, actual={mask.shape}: {path}")
    return mask


def _load_track(path: Path | None, expected_shape: tuple[int, int, int]) -> np.ndarray:
    if path is None or not path.is_file():
        return np.zeros(expected_shape, dtype=bool)
    with np.load(path, allow_pickle=False) as archive:
        track = np.asarray(archive["masks"], dtype=bool)
    if track.shape != expected_shape:
        raise ValueError(
            f"native track shape mismatch: expected={expected_shape}, actual={track.shape}: {path}"
        )
    return track


def _identity_record(
    review: dict[str, Any],
    episode_index: int,
    role: str,
) -> dict[str, Any] | None:
    record = review.get("episodes", {}).get(str(episode_index), {}).get(role)
    if record is None:
        return None
    if not isinstance(record, dict) or record.get("decision") not in {"accept", "review", "reject"}:
        raise ValueError(f"invalid identity review for episode={episode_index}, role={role}")
    if not isinstance(record.get("reason"), str) or not record["reason"].strip():
        raise ValueError(f"identity review needs a reason for episode={episode_index}, role={role}")
    return record


def _load_semantic_plan(source_dir: Path, context: LoopContext) -> SemanticPlan:
    saved_path = source_dir / "semantic_plan.json"
    prompt_path = source_dir / "qwen_rendered_prompt.txt"
    raw_path = source_dir / "qwen_raw_response.txt"
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    plan = parse_semantic_plan(
        raw_path.read_text(encoding="utf-8"),
        context=context,
        model=str(saved["model"]),
        rendered_prompt=prompt_path.read_text(encoding="utf-8"),
    )
    if plan.to_json() != saved:
        raise ValueError(f"saved semantic plan fails provenance validation: {saved_path}")
    return plan


def _role_data(
    *,
    role: RoleName,
    source_dir: Path,
    source_record: dict[str, Any],
    frame_count: int,
    frame_shape: tuple[int, int],
    config: PipelineConfig,
    identity_record: dict[str, Any] | None,
) -> RoleMaskData:
    start, end = (int(value) for value in source_record["output_window"])
    window = FrameWindow(start, end)
    seed_relative = source_record.get("seed_mask_path")
    seed_path = None if not seed_relative else source_dir / seed_relative
    seed = _load_mask(seed_path, frame_shape)
    envelope_relative = source_record.get("canonical_envelope_path")
    envelope_path = None if not envelope_relative else source_dir / envelope_relative
    envelope = _load_mask(envelope_path, frame_shape)
    if envelope is None and seed is not None:
        padding = (
            config.mask.target_envelope_padding_px
            if role == "target"
            else config.mask.receiver_envelope_padding_px
        )
        envelope = dilate_envelope(seed, padding)
    native_relative = source_record.get("native_track_path")
    native_path = None if not native_relative else source_dir / native_relative
    native = _load_track(native_path, (frame_count, *frame_shape))
    visible = compose_visible_mask(native, window)
    temporal_qc = (
        None
        if seed is None or not seed.any()
        else evaluate_temporal_mask(
            visible,
            window,
            config.mask,
            reference_mask=seed,
        )
    )

    if seed is None or not seed.any():
        status = MaskStatus.FAILED
        failure = "empty_text_seed"
        visible[:] = False
    elif not visible.any():
        status = MaskStatus.FAILED
        failure = "native_track_empty_in_output_window"
        visible[:] = False
    elif temporal_qc is not None and temporal_qc.status == "quarantine":
        status = MaskStatus.QUARANTINED
        failure = "temporal_qc_quarantine:" + ",".join(temporal_qc.issues)
        visible[:] = False
    elif identity_record is not None and identity_record["decision"] == "reject":
        status = MaskStatus.QUARANTINED
        failure = "identity_qc_reject:" + " ".join(identity_record["reason"].split())
        visible[:] = False
    else:
        status = MaskStatus.OK
        failure = None

    return RoleMaskData(
        role=role,
        status=status,
        seed_frame_id=source_record.get("seed_frame_id"),
        primary_query=source_record.get("primary_query"),
        output_window=window,
        seed_mask=seed,
        canonical_envelope=envelope,
        native_track=native,
        visible_mask=visible,
        temporal_qc=temporal_qc,
        failure=failure,
    )


def materialize(
    *,
    config: PipelineConfig,
    selection_manifest: Path,
    runs_root: Path,
    output_run_id: str,
    episode_ids: tuple[int, ...],
    identity_review: dict[str, Any],
) -> dict[str, Any]:
    store = ArtifactStore(runs_root)
    output_run_dir = store.run_dir(output_run_id)
    if output_run_dir.exists():
        raise FileExistsError(f"output run already exists: {output_run_dir}")

    selection = json.loads(selection_manifest.read_text(encoding="utf-8"))
    selected_by_episode = {
        int(record["episode_index"]): record for record in selection["episodes"]
    }
    missing = sorted(set(episode_ids) - set(selected_by_episode))
    if missing:
        raise ValueError(f"selection manifest is missing episodes: {missing}")
    dataset = _dataset(config)
    frame_shape_values = tuple(int(value) for value in dataset.manifest["frame_shape_hw"])
    if len(frame_shape_values) != 2:
        raise ValueError(f"invalid dataset frame shape: {frame_shape_values}")
    frame_shape = (frame_shape_values[0], frame_shape_values[1])

    records: list[dict[str, Any]] = []
    for position, episode_index in enumerate(episode_ids, start=1):
        selection_record = selected_by_episode[episode_index]
        source_masks = Path(selection_record["source_masks"])
        source_dir = source_masks.parent
        source_manifest_path = source_dir / "run_manifest.json"
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        ref = EpisodeRef(config.dataset.task, episode_index, config.dataset.camera)
        context = build_loop_context(dataset, ref)
        saved_loop = json.loads((source_dir / "loop.json").read_text(encoding="utf-8"))
        if saved_loop != context.to_json():
            raise ValueError(f"saved loop differs from current Stage 1: {source_dir}")
        semantic_plan = _load_semantic_plan(source_dir, context)

        role_values: list[RoleMaskData] = []
        seed_images: dict[int, Image.Image] = {}
        for role, role_directory, source_role in zip(
            ROLE_NAMES,
            ROLE_DIRECTORIES,
            source_manifest["roles"],
            strict=True,
        ):
            if source_role["role"] != role:
                raise ValueError(f"source role order mismatch: {source_manifest_path}")
            review_record = _identity_record(identity_review, episode_index, role)
            role_values.append(
                _role_data(
                    role=role,
                    source_dir=source_dir,
                    source_record=source_role,
                    frame_count=context.frame_count,
                    frame_shape=frame_shape,
                    config=config,
                    identity_record=review_record,
                )
            )
            seed_frame = source_role.get("seed_frame_id")
            seed_rgb_relative = source_role.get("seed_rgb_path")
            if seed_frame is not None and seed_rgb_relative:
                with Image.open(source_dir / seed_rgb_relative) as image:
                    seed_images[int(seed_frame)] = image.convert("RGB").copy()
            elif role_values[-1].seed_mask is not None:
                raise ValueError(f"source seed RGB is missing: {source_dir / role_directory}")

        result = SamStageResult(
            frame_count=context.frame_count,
            frame_shape=frame_shape,
            target=role_values[0],
            receiver=role_values[1],
        )
        mask_run = save_sam_artifacts(
            store,
            output_run_id,
            context,
            semantic_plan,
            result,
            seed_images=seed_images,
        )
        output_dir = Path(mask_run.artifact_dir)
        for filename in STAGE_FILES:
            shutil.copy2(source_dir / filename, output_dir / filename)
        output_manifest_path = output_dir / "run_manifest.json"
        output_manifest = json.loads(output_manifest_path.read_text(encoding="utf-8"))
        output_manifest["lineage"] = {
            "operation": "materialize_saved_sam3_native_tracks",
            "source_run_id": str(selection_record["run_id"]),
            "source_manifest": str(source_manifest_path),
            "source_masks": str(source_masks),
            "source_masks_sha256": _sha256(source_masks),
            "identity_review": {
                role: _identity_record(identity_review, episode_index, role)
                for role in ROLE_NAMES
            },
        }
        ArtifactStore.write_json(output_manifest_path, output_manifest)
        records.append(
            {
                "episode_index": episode_index,
                "source_run_id": str(selection_record["run_id"]),
                "output_manifest": str(output_manifest_path),
                "role_status": {
                    value.role: value.status.value for value in role_values
                },
            }
        )
        print(
            json.dumps(
                {
                    "progress": f"{position}/{len(episode_ids)}",
                    **records[-1],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    run_manifest = {
        "format_version": "robotwin_native_tracking_materialization_v1",
        "created_at": datetime.now(UTC).isoformat(),
        "output_run_id": output_run_id,
        "selection_manifest": str(selection_manifest),
        "identity_review_format": identity_review.get("format_version"),
        "episodes": records,
    }
    ArtifactStore.write_json(output_run_dir / "materialization_manifest.json", run_manifest)
    return run_manifest


def main() -> None:
    args = _parse_args()
    config = load_config(args.config)
    configured = set(config.dataset.regression_episode_ids)
    episode_ids = tuple(args.episode_ids or config.dataset.regression_episode_ids)
    unknown = sorted(set(episode_ids) - configured)
    if unknown:
        raise ValueError(f"episodes are outside the configured coverage20 set: {unknown}")
    identity_review = (
        {}
        if args.identity_review is None
        else json.loads(args.identity_review.expanduser().resolve().read_text(encoding="utf-8"))
    )
    report = materialize(
        config=config,
        selection_manifest=args.selection_manifest.expanduser().resolve(),
        runs_root=(args.runs_root or config.output_root).expanduser().resolve(),
        output_run_id=args.output_run_id,
        episode_ids=episode_ids,
        identity_review=identity_review,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
