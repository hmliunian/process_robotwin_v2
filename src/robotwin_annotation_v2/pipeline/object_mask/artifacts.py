"""Artifact writer for object-mask QC attempts and diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ...adapters.artifact_store import ArtifactStore
from ...models import LoopContext, MaskQCResult


def save_mask_qc_artifacts(
    store: ArtifactStore,
    run_id: str,
    context: LoopContext,
    result: MaskQCResult,
    *,
    candidate_masks: Mapping[str, Mapping[str, np.ndarray[Any, Any]]] | None = None,
) -> Path:
    """Persist QC decisions and optional candidate masks for later review."""

    episode_dir = store.episode_dir(run_id, context.episode)
    reports = result.to_json()
    reports["episode"] = context.episode.to_json()
    masks_to_save = result.candidate_masks if candidate_masks is None else candidate_masks
    mask_paths: dict[str, dict[str, str]] = {}
    if masks_to_save:
        for role, masks in masks_to_save.items():
            mask_paths[role] = {}
            for candidate_id, mask in masks.items():
                path = store.write_png(
                    episode_dir / role / "qc_candidates" / f"candidate_{candidate_id}.mask.png",
                    np.asarray(mask, dtype=bool),
                )
                mask_paths[role][candidate_id] = str(path.relative_to(episode_dir))
    panel_paths: dict[str, dict[str, str]] = {}
    for role, panels in result.candidate_panels.items():
        panel_paths[role] = {}
        for candidate_id, panel in panels.items():
            path = store.write_png(
                episode_dir / role / "qc_candidates" / f"candidate_{candidate_id}.overlay.png",
                np.asarray(panel.convert("RGB")),
                rgb=True,
            )
            panel_paths[role][candidate_id] = str(path.relative_to(episode_dir))
    attempt_paths: dict[str, dict[str, dict[str, Any]]] = {}
    for report in result.role_reports:
        role = report.role
        mask_attempts = result.attempt_candidate_masks.get(role, {})
        panel_attempts = result.attempt_candidate_panels.get(role, {})
        frame_ids = sorted(set(mask_attempts) | set(panel_attempts))
        if not frame_ids:
            continue
        attempt_paths[role] = {}
        for seed_frame_id in frame_ids:
            frame_key = f"frame_{seed_frame_id:06d}"
            frame_dir = episode_dir / role / "qc_candidates" / frame_key
            frame_mask_paths: dict[str, str] = {}
            for candidate_id, mask in mask_attempts.get(seed_frame_id, {}).items():
                path = store.write_png(
                    frame_dir / f"candidate_{candidate_id}.mask.png",
                    np.asarray(mask, dtype=bool),
                )
                frame_mask_paths[candidate_id] = str(path.relative_to(episode_dir))
            frame_panel_paths: dict[str, str] = {}
            for candidate_id, panel in panel_attempts.get(seed_frame_id, {}).items():
                path = store.write_png(
                    frame_dir / f"candidate_{candidate_id}.overlay.png",
                    np.asarray(panel.convert("RGB")),
                    rgb=True,
                )
                frame_panel_paths[candidate_id] = str(path.relative_to(episode_dir))
            attempt_paths[role][frame_key] = {
                "seed_frame_id": seed_frame_id,
                "candidate_masks": frame_mask_paths,
                "candidate_panels": frame_panel_paths,
            }
    reports["artifacts"] = {
        "candidate_masks": mask_paths,
        "candidate_panels": panel_paths,
        "attempts": attempt_paths,
    }
    return store.write_json(episode_dir / "mask_qc.json", reports)


__all__ = ["save_mask_qc_artifacts"]
