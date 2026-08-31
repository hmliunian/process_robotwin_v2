#!/usr/bin/env python3
"""Create a versioned mask run from saved SAM3 native tracks without rerunning SAM3."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from PIL import Image

from robotwin_annotation_v2.adapters import ArtifactStore, RoboTwinDataset
from robotwin_annotation_v2.application.provenance import (
    prompt_bundle_for_config,
    prompt_bundle_from_manifest,
    target_profile_from_config,
    target_profile_from_manifest,
    validate_profile_provenance_pair,
)
from robotwin_annotation_v2.config import PipelineConfig, load_config
from robotwin_annotation_v2.domain import AnnotationMode, TargetProfile
from robotwin_annotation_v2.models import (
    EpisodeRef,
    FrameWindow,
    LoopContext,
    MaskStatus,
    SemanticPlan,
)
from robotwin_annotation_v2.models.semantic_plan import canonical_target_profile
from robotwin_annotation_v2.pipeline import (
    RoleMaskData,
    SamStageResult,
    build_loop_context,
    canonicalize_legacy_semantic_plan,
    compose_visible_mask,
    dilate_envelope,
    evaluate_temporal_mask,
    parse_semantic_plan,
    save_sam_artifacts,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RoleName = Literal["target", "receiver"]
ROLE_NAMES: tuple[RoleName, ...] = ("target", "receiver")


def _validate_source_prompt_provenance(
    source_manifest: Mapping[str, Any],
    *,
    target_profile: str | None,
    prompt_bundle: Mapping[str, Any] | None,
) -> None:
    """Validate a source run's prompt contract against the current config.

    Runs created before prompt bundles were introduced are intentionally kept
    readable.  Once a source manifest carries a bundle, however, reusing its
    native tracks under a different prompt/profile would relabel the semantic
    result, so require an exact content/profile match and fail closed.
    """

    source_bundle = prompt_bundle_from_manifest(source_manifest, strict=True)
    if source_bundle is None:
        return
    validate_profile_provenance_pair(
        source_manifest,
        {
            "target_profile": target_profile,
            "prompt_bundle": prompt_bundle,
        },
        label="source and configured prompt provenance",
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


def _load_mask(
    path: Path | None,
    shape: tuple[int, int],
) -> np.ndarray[Any, Any] | None:
    if path is None or not path.is_file():
        return None
    with Image.open(path) as image:
        mask = np.asarray(image.convert("L")) > 0
    if mask.shape != shape:
        raise ValueError(f"mask shape mismatch: expected={shape}, actual={mask.shape}: {path}")
    return mask


def _load_track(
    path: Path | None,
    expected_shape: tuple[int, int, int],
) -> np.ndarray[Any, Any]:
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


def _load_semantic_plan(
    source_dir: Path,
    context: LoopContext,
    *,
    target_profile: str | None = None,
) -> SemanticPlan:
    saved_path = source_dir / "semantic_plan.json"
    prompt_path = source_dir / "qwen_rendered_prompt.txt"
    raw_path = source_dir / "qwen_raw_response.txt"
    saved = json.loads(saved_path.read_text(encoding="utf-8"))
    configured_profile = canonical_target_profile(target_profile)
    saved_profile_field_present = "target_profile" in saved
    raw_saved_profile = saved.get("target_profile")
    saved_profile = canonical_target_profile(raw_saved_profile)
    if (
        configured_profile is not None
        and saved_profile_field_present
        and raw_saved_profile is not None
        and saved_profile is None
    ):
        raise ValueError(
            "saved semantic_plan.json target_profile differs from current config"
        )
    if saved_profile is not None and saved_profile is not configured_profile:
        raise ValueError(
            "saved semantic_plan.json target_profile differs from current config"
        )
    plan = cast(
        SemanticPlan,
        parse_semantic_plan(
            raw_path.read_text(encoding="utf-8"),
            context=context,
            model=str(saved["model"]),
            rendered_prompt=prompt_path.read_text(encoding="utf-8"),
            target_profile=configured_profile,
        ),
    )
    comparable_saved = saved
    if configured_profile is TargetProfile.DOOR_OPEN and (
        not saved_profile_field_present or saved_profile is None
    ):
        comparable_saved = canonicalize_legacy_semantic_plan(
            saved,
            target_profile=configured_profile,
        )
    elif saved_profile_field_present and saved_profile is None:
        comparable_saved = {
            key: value for key, value in saved.items() if key != "target_profile"
        }
    if plan.to_json() != comparable_saved:
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
    target_profile = target_profile_from_config(config)
    prompt_bundle = prompt_bundle_for_config(config)

    records: list[dict[str, Any]] = []
    for position, episode_index in enumerate(episode_ids, start=1):
        selection_record = selected_by_episode[episode_index]
        source_masks = Path(selection_record["source_masks"])
        source_dir = source_masks.parent
        source_manifest_path = source_dir / "run_manifest.json"
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        source_profile = target_profile_from_manifest(source_manifest)
        if source_profile is not None:
            configured_source_profile = canonical_target_profile(source_profile)
            configured_profile = canonical_target_profile(target_profile)
            if configured_source_profile is not configured_profile:
                raise ValueError(
                    "source run target_profile differs from current config"
                )
        # Validate a present source bundle before reusing its stage metadata;
        # legacy source manifests may legitimately omit the bundle.
        _validate_source_prompt_provenance(
            source_manifest,
            target_profile=target_profile,
            prompt_bundle=prompt_bundle,
        )
        ref = EpisodeRef(config.dataset.task, episode_index, config.dataset.camera)
        context = build_loop_context(
            dataset,
            ref,
            annotation_mode=AnnotationMode(config.annotation.mode),
        )
        saved_loop = json.loads((source_dir / "loop.json").read_text(encoding="utf-8"))
        if saved_loop != context.to_json():
            raise ValueError(f"saved loop differs from current Stage 1: {source_dir}")
        semantic_plan = _load_semantic_plan(
            source_dir,
            context,
            target_profile=target_profile,
        )

        source_roles = source_manifest.get("roles")
        if not isinstance(source_roles, list):
            raise TypeError(f"source manifest roles must be a list: {source_manifest_path}")
        source_roles_by_name: dict[str, dict[str, Any]] = {}
        for source_role in source_roles:
            if not isinstance(source_role, dict) or source_role.get("role") not in {
                "target",
                "receiver",
            }:
                continue
            role_name = str(source_role["role"])
            if role_name in source_roles_by_name:
                raise ValueError(f"duplicate source role {role_name}: {source_manifest_path}")
            source_roles_by_name[role_name] = source_role

        role_values: list[RoleMaskData] = []
        seed_images: dict[int, Image.Image] = {}
        required_roles = context.annotation_spec.required_role_names
        for role in required_roles:
            if role not in source_roles_by_name:
                raise ValueError(
                    f"source manifest is missing required role {role}: {source_manifest_path}"
                )
            source_role = source_roles_by_name[role]
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
                raise ValueError(
                    f"source seed RGB is missing: {source_dir / role}_0"
                )

        result = SamStageResult(
            frame_count=context.frame_count,
            frame_shape=frame_shape,
            role_masks=tuple(role_values),
        )
        mask_run = save_sam_artifacts(
            store,
            output_run_id,
            context,
            semantic_plan,
            result,
            seed_images=seed_images,
            target_profile=target_profile,
            prompt_bundle=prompt_bundle,
        )
        output_dir = Path(mask_run.artifact_dir)
        # Re-publish the validated Stage-1/2 envelopes instead of copying the
        # source semantic-plan JSON verbatim.  Profile-aware parsing may
        # canonicalize legacy handle wording (and add target_profile), so the
        # saved JSON must match the plan consumed by this materialization.
        store.save_loop(output_run_id, ref, context.to_json())
        store.save_semantic_plan(
            output_run_id,
            ref,
            semantic_plan.to_json(),
            rendered_prompt=(source_dir / "qwen_rendered_prompt.txt").read_text(
                encoding="utf-8"
            ),
            raw_response=(source_dir / "qwen_raw_response.txt").read_text(
                encoding="utf-8"
            ),
        )
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
