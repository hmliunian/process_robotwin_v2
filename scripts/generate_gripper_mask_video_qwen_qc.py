#!/usr/bin/env python3
"""Generate coverage20 gripper videos with Qwen-selected propagation seeds."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from robotwin_annotation_v2.adapters import (
    OpenAICompatibleQwenClient,
    RoboTwinDataset,
    Sam3Adapter,
    sam3_video_resource,
)
from robotwin_annotation_v2.config import load_config
from robotwin_annotation_v2.experiments import (
    build_gripper_seed_candidate,
    compose_gripper_track,
    gripper_keyframes,
    load_qc_native_object_tracks,
    mark_same_frame_duplicates,
    normalized_roi_box,
    project_gripper_roi,
    render_gripper_candidate_panel,
    render_gripper_candidate_sheet,
    run_gripper_seed_qc,
)
from robotwin_annotation_v2.models import EpisodeRef
from robotwin_annotation_v2.models.mask_qc import MaskQCStatus
from robotwin_annotation_v2.pipeline import build_loop_context

from generate_gripper_mask_video_preview import (
    _build_roi_track,
    _render_video,
    _sha256,
    _track_summary,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QC_PROMPT = PROJECT_ROOT / "configs" / "prompts" / "gripper_seed_candidate_qc.txt"
DEFAULT_OBJECT_RUN_ROOT = Path(
    "/DATA/disk8/xuran/add_mask_robotwin/process_data_v2/artifacts/runs/"
    "coverage20-qc-contact-v5-native"
)


class SeedRejected(RuntimeError):
    """No candidate exists, so even the mandatory seed fallback is impossible."""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "pilot_move_pillbottle_pad.yaml",
    )
    parser.add_argument(
        "--object-mask-run-root",
        type=Path,
        default=DEFAULT_OBJECT_RUN_ROOT,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode-ids", type=int, nargs="*")
    parser.add_argument("--sam3-gpu", type=int, default=2)
    parser.add_argument("--gripper-text", default="black robot gripper")
    parser.add_argument("--qc-prompt-template", type=Path, default=DEFAULT_QC_PROMPT)
    parser.add_argument("--qc-max-tokens", type=int, default=220)
    parser.add_argument("--qc-max-attempts", type=int, default=2)
    parser.add_argument("--qc-min-confidence", type=float, default=0.70)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--temp-root", type=Path, default=Path("/tmp"))
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="fast")
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _complete_episode(path: Path) -> bool:
    return (
        (path / "manifest.json").is_file()
        and (path / "episode_gripper_review.mp4").is_file()
        and (path / "gripper_masks.npz").is_file()
    )


def _episode_context_frames(
    keyframes: tuple[int, ...],
    *,
    events: Any,
    frame_images: dict[int, Any],
) -> dict[int, Any]:
    values = (
        keyframes[0],
        events.t_close_done,
        (events.t_close_done + events.t_open_start) // 2,
        events.t_open_done,
    )
    return {frame_id: frame_images[frame_id] for frame_id in dict.fromkeys(values)}


def _build_candidates(
    adapter: Sam3Adapter,
    resource: Path,
    *,
    frame_images: dict[int, Any],
    keyframes: tuple[int, ...],
    rois: tuple[Any, ...],
    roi_track: np.ndarray,
    target_track: np.ndarray,
    receiver_track: np.ndarray,
    events: Any,
    frame_count: int,
    frame_shape: tuple[int, int],
    gripper_text: str,
) -> tuple[Any, ...]:
    candidates: list[Any] = []
    for frame_id in keyframes:
        roi = rois[frame_id]
        box = normalized_roi_box(roi, frame_shape)
        if box is None:
            continue
        prompt_variants = (
            ("box_only", None),
            ("text_box", gripper_text),
        )
        for prompt_mode, prompt_text in prompt_variants:
            if prompt_text is None:
                raw = adapter.box_mask(
                    resource,
                    box,
                    frame_id=frame_id,
                    frame_count=frame_count,
                    frame_shape=frame_shape,
                )
            else:
                raw = adapter.text_box_mask(
                    resource,
                    prompt_text,
                    box,
                    frame_id=frame_id,
                    frame_count=frame_count,
                    frame_shape=frame_shape,
                )
            candidate_id = chr(ord("A") + len(candidates))
            candidates.append(
                build_gripper_seed_candidate(
                    candidate_id=candidate_id,
                    frame_id=frame_id,
                    events=events,
                    prompt_mode=prompt_mode,
                    prompt_text=prompt_text,
                    raw_mask=raw,
                    roi_mask=roi_track[frame_id],
                    target_mask=target_track[frame_id],
                    receiver_mask=receiver_track[frame_id],
                    rgb=np.asarray(frame_images[frame_id], dtype=np.uint8),
                    tcp_pixel_xy=roi.tcp_pixel_xy,
                )
            )
    return mark_same_frame_duplicates(candidates)


def _save_candidates(
    output_dir: Path,
    candidates: tuple[Any, ...],
    panels: dict[str, Any],
) -> dict[str, Any]:
    panel_dir = output_dir / "seed_candidates"
    panel_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for candidate in candidates:
        arrays[f"{candidate.candidate_id}_raw"] = candidate.raw_mask
        arrays[f"{candidate.candidate_id}_roi"] = candidate.roi_mask
        arrays[f"{candidate.candidate_id}_cropped"] = candidate.cropped_mask
        arrays[f"{candidate.candidate_id}_clean"] = candidate.clean_mask
        arrays[f"{candidate.candidate_id}_target_removed"] = candidate.target_removed
        arrays[f"{candidate.candidate_id}_receiver_removed"] = candidate.receiver_removed
        panels[candidate.candidate_id].save(
            panel_dir / f"candidate_{candidate.candidate_id}.png",
            format="PNG",
        )
    archive_path = output_dir / "seed_candidates.npz"
    np.savez_compressed(archive_path, **arrays)
    sheet = render_gripper_candidate_sheet(candidates, panels)
    sheet_path = output_dir / "seed_candidates.jpg"
    sheet.save(sheet_path, format="JPEG", quality=84, optimize=True)
    return {
        "archive": {
            "path": archive_path.name,
            "sha256": _sha256(archive_path),
            "bytes": archive_path.stat().st_size,
        },
        "sheet": {
            "path": sheet_path.name,
            "sha256": _sha256(sheet_path),
            "bytes": sheet_path.stat().st_size,
        },
        "panels": {
            candidate_id: f"seed_candidates/candidate_{candidate_id}.png"
            for candidate_id in panels
        },
    }


def _run_episode(
    *,
    config: Any,
    dataset: RoboTwinDataset,
    adapter: Sam3Adapter,
    qwen: OpenAICompatibleQwenClient,
    object_root: Path,
    episode_id: int,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.perf_counter()
    ref = EpisodeRef(config.dataset.task, episode_id, config.dataset.camera)
    state = dataset.load_state(ref)
    context = build_loop_context(dataset, ref)
    frame_shape = tuple(int(value) for value in dataset.manifest["frame_shape_hw"])
    if len(frame_shape) != 2:
        raise ValueError(f"invalid frame shape: {frame_shape}")
    expected_shape = (state.frame_count, *frame_shape)
    objects = load_qc_native_object_tracks(
        object_root,
        ref,
        expected_shape=expected_shape,
    )
    active_window = (context.events.t_move_start, context.events.t_open_done)
    arm_index = 0 if context.events.active_arm == "left" else 1
    rois_by_frame = {
        frame_id: project_gripper_roi(
            state.eef_states[frame_id, arm_index],
            state.gripper_states[frame_id, arm_index],
        )
        for frame_id in range(*active_window)
    }
    # range(start, end) above omits the inclusive endpoint; keep the contract explicit.
    rois_by_frame[active_window[1]] = project_gripper_roi(
        state.eef_states[active_window[1], arm_index],
        state.gripper_states[active_window[1], arm_index],
    )
    roi_track, rois = _build_roi_track(
        state.eef_states,
        state.gripper_states,
        context.events,
        frame_shape=frame_shape,
    )
    keyframes = gripper_keyframes(rois_by_frame, context.events, frame_shape=frame_shape)
    context_frame_ids = tuple(
        dict.fromkeys(
            (
                *keyframes,
                context.events.t_close_done,
                (context.events.t_close_done + context.events.t_open_start) // 2,
                context.events.t_open_done,
            )
        )
    )
    frame_images = dataset.read_frames(ref, context_frame_ids)
    raw_surplus = int(dataset.manifest["raw_video_frame_surplus"])
    with sam3_video_resource(
        state.paths.video,
        minimum_frame_count=state.frame_count + raw_surplus,
        temp_root=args.temp_root,
    ) as resource:
        object_propagation_started = time.perf_counter()
        target_tracking_window = (
            min(objects.target_seed_frame, active_window[0]),
            max(objects.target_seed_frame, active_window[1]),
        )
        receiver_tracking_window = (
            min(objects.receiver_seed_frame, active_window[0]),
            max(objects.receiver_seed_frame, active_window[1]),
        )
        target_track = adapter.propagate_mask(
            resource,
            objects.target_seed_mask,
            seed_frame=objects.target_seed_frame,
            frame_count=state.frame_count,
            frame_shape=frame_shape,
            tracking_window=target_tracking_window,
        )
        receiver_track = adapter.propagate_mask(
            resource,
            objects.receiver_seed_mask,
            seed_frame=objects.receiver_seed_frame,
            frame_count=state.frame_count,
            frame_shape=frame_shape,
            tracking_window=receiver_tracking_window,
        )
        object_propagation_seconds = time.perf_counter() - object_propagation_started
        candidates = _build_candidates(
            adapter,
            resource,
            frame_images=frame_images,
            keyframes=keyframes,
            rois=rois,
            roi_track=roi_track,
            target_track=target_track,
            receiver_track=receiver_track,
            events=context.events,
            frame_count=state.frame_count,
            frame_shape=frame_shape,
            gripper_text=args.gripper_text,
        )
        panels = {
            candidate.candidate_id: render_gripper_candidate_panel(
                frame_images[candidate.frame_id],
                candidate,
                rois[candidate.frame_id],
            )
            for candidate in candidates
        }
        context_images = _episode_context_frames(
            keyframes,
            events=context.events,
            frame_images=frame_images,
        )
        qc = run_gripper_seed_qc(
            context,
            candidates,
            panels,
            context_images,
            prompt_template_path=args.qc_prompt_template,
            client=qwen,
            max_tokens=args.qc_max_tokens,
            max_attempts=args.qc_max_attempts,
            minimum_confidence=args.qc_min_confidence,
        )
        if qc.status is not MaskQCStatus.PASSED or qc.selected is None:
            raise SeedRejected(
                f"episode {episode_id} gripper seed QC {qc.status.value}: {qc.reason}"
            )
        selected = qc.selected
        propagation_started = time.perf_counter()
        native_track = adapter.propagate_mask(
            resource,
            selected.clean_mask,
            seed_frame=selected.frame_id,
            frame_count=state.frame_count,
            frame_shape=frame_shape,
            tracking_window=active_window,
        )
        propagation_seconds = time.perf_counter() - propagation_started

    result = compose_gripper_track(
        native_track,
        roi_track,
        target_track,
        receiver_track,
        active_window=active_window,
    )
    # The batch driver creates an isolated temporary episode directory before
    # inference so a failed render cannot look like a complete artifact.
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_artifacts = _save_candidates(output_dir, candidates, panels)
    qc_path = output_dir / "seed_qc.json"
    _write_json(qc_path, qc.to_json())
    if qc.rendered_prompt is not None:
        (output_dir / "qwen_rendered_prompt.txt").write_text(
            qc.rendered_prompt,
            encoding="utf-8",
        )
    if qc.raw_response is not None:
        (output_dir / "qwen_raw_response.txt").write_text(qc.raw_response, encoding="utf-8")

    archive_path = output_dir / "gripper_masks.npz"
    np.savez_compressed(
        archive_path,
        format_version=np.asarray("robotwin_gripper_mask_video_qwen_qc_v1"),
        episode_index=np.asarray(episode_id, dtype=np.int64),
        seed_frame=np.asarray(selected.frame_id, dtype=np.int64),
        selected_candidate=np.asarray(selected.candidate_id),
        active_window=np.asarray(active_window, dtype=np.int64),
        seed_mask=selected.clean_mask,
        native_track=native_track,
        roi_track=result.roi_mask,
        candidate_track=result.candidate_mask,
        target_track=target_track,
        receiver_track=receiver_track,
        target_removed=result.target_removed,
        receiver_removed=result.receiver_removed,
        gripper_track=result.gripper_mask,
    )
    video_path = output_dir / "episode_gripper_review.mp4"
    contact_sheet_path = output_dir / "episode_contact_sheet.jpg"
    render_started = time.perf_counter()
    render = _render_video(
        state.paths.video,
        video_path,
        contact_sheet_path,
        episode_id=episode_id,
        frame_count=state.frame_count,
        events=context.events,
        rois=rois,
        native_track=native_track,
        result=result,
        crf=args.crf,
        preset=args.preset,
    )
    render_seconds = time.perf_counter() - render_started
    manifest = {
        "format": "robotwin_gripper_mask_video_qwen_qc_v1",
        "status": "review_required",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "episode": {
            "task": ref.task,
            "episode_index": episode_id,
            "camera": ref.camera,
            "frame_count": state.frame_count,
            "active_arm": context.events.active_arm,
            "events": context.events.to_json(),
            "active_window": list(active_window),
            "source_video": str(state.paths.video),
            "source_video_sha256": _sha256(state.paths.video),
        },
        "contract": {
            "visible_only": True,
            "amodal_completion": False,
            "morphology": "none",
            "operation": "qwen_selected_seed -> native_track & pose_roi & ~(target | receiver)",
            "outside_active_window": "empty",
        },
        "known_objects": {
            **objects.provenance,
            "usage": (
                "QC-selected target/receiver seed masks repropagated with native SAM3 "
                "through the full gripper active window"
            ),
            "saved_role_tracks": "provenance_only_not_used_for_gripper_exclusion",
            "full_window_tracks": {
                "target": _track_summary(target_track, *active_window),
                "receiver": _track_summary(receiver_track, *active_window),
            },
        },
        "seed": {
            "selected_candidate": selected.candidate_id,
            "frame": selected.frame_id,
            "prompt_mode": selected.prompt_mode,
            "prompt_text": selected.prompt_text,
            "selection_source": "forced_fallback" if qc.forced_fallback else "qwen",
            "clean_pixels": selected.clean_pixels,
            "qc": qc.to_json(),
            "candidate_artifacts": candidate_artifacts,
        },
        "sam3": {
            "checkpoint": str(config.sam3.checkpoint),
            "checkpoint_sha256": _sha256(config.sam3.checkpoint),
            "visible_gpu_index": args.sam3_gpu,
            "propagation": "sam3_native_mask_forward_backward",
        },
        "tracks": {
            "native": _track_summary(native_track, *active_window),
            "pose_cropped_candidate": _track_summary(result.candidate_mask, *active_window),
            "final_gripper": _track_summary(result.gripper_mask, *active_window),
            "target_removed": _track_summary(result.target_removed, *active_window),
            "receiver_removed": _track_summary(result.receiver_removed, *active_window),
        },
        "timing": {
            "object_full_window_propagation_seconds": object_propagation_seconds,
            "native_propagation_seconds": propagation_seconds,
            "native_propagation_fps": (
                (active_window[1] - active_window[0] + 1) / propagation_seconds
            ),
            "render_seconds": render_seconds,
            "total_seconds": time.perf_counter() - started,
        },
        "artifacts": {
            "mask_archive": {
                "path": archive_path.name,
                "sha256": _sha256(archive_path),
                "bytes": archive_path.stat().st_size,
            },
            "review_video": {
                "path": video_path.name,
                "sha256": _sha256(video_path),
                "bytes": video_path.stat().st_size,
            },
            "contact_sheet": {
                "path": contact_sheet_path.name,
                "sha256": _sha256(contact_sheet_path),
                "bytes": contact_sheet_path.stat().st_size,
            },
            "seed_qc": {
                "path": qc_path.name,
                "sha256": _sha256(qc_path),
                "bytes": qc_path.stat().st_size,
            },
        },
        "implementation": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": _sha256(Path(__file__).resolve()),
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    return {
        "episode": episode_id,
        "status": manifest["status"],
        "output_dir": str(output_dir),
        "video": str(video_path),
        "selected_candidate": selected.candidate_id,
        "seed_frame": selected.frame_id,
        "selection_source": "forced_fallback" if qc.forced_fallback else "qwen",
        "native_propagation_fps": manifest["timing"]["native_propagation_fps"],
        "final_nonempty_frames": manifest["tracks"]["final_gripper"]["nonempty_frames"],
    }


def main() -> None:
    args = _parse_args()
    if not 0 <= args.crf <= 51:
        raise ValueError("--crf must be between 0 and 51")
    config = load_config(args.config)
    episode_ids = tuple(args.episode_ids or config.dataset.regression_episode_ids)
    unknown = sorted(set(episode_ids) - set(config.dataset.regression_episode_ids))
    if unknown:
        raise ValueError(f"episode ids are outside coverage20: {unknown}")
    output_root = args.output_dir.expanduser().resolve()
    if output_root.exists() and not args.resume and any(output_root.iterdir()):
        raise FileExistsError(
            f"output directory is non-empty; pass --resume to continue: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    object_root = args.object_mask_run_root.expanduser().resolve()
    dataset = RoboTwinDataset(
        config.dataset.root,
        task=config.dataset.task,
        camera=config.dataset.camera,
        manifest_path=config.dataset.manifest,
    )
    qwen = OpenAICompatibleQwenClient(
        endpoint=config.qwen.endpoint,
        model=config.qwen.model,
        timeout_seconds=config.qwen.timeout_seconds,
    )
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    manifest_path = output_root / "batch_manifest.json"
    if args.resume and manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        records.extend(previous.get("episodes", []))
        failures.extend(previous.get("failures", []))
        completed_ids = {
            int(record["episode"])
            for record in records
            if "episode" in record
            and _complete_episode(output_root / f"episode_{int(record['episode']):06d}")
        }
        failures = [
            failure
            for failure in failures
            if int(failure.get("episode", -1)) not in completed_ids
        ]
    existing_ids = {int(record["episode"]) for record in records if "episode" in record}
    backend = Sam3Adapter(checkpoint_path=config.sam3.checkpoint, gpus=(args.sam3_gpu,))
    try:
        for episode_id in episode_ids:
            final_dir = output_root / f"episode_{episode_id:06d}"
            if episode_id in existing_ids and _complete_episode(final_dir):
                continue
            if final_dir.exists():
                raise FileExistsError(
                    f"episode output exists but is incomplete; inspect before rerunning: {final_dir}"
                )
            temporary = Path(tempfile.mkdtemp(prefix=f".episode_{episode_id:06d}-", dir=output_root))
            try:
                record = _run_episode(
                    config=config,
                    dataset=dataset,
                    adapter=backend,
                    qwen=qwen,
                    object_root=object_root,
                    episode_id=episode_id,
                    output_dir=temporary,
                    args=args,
                )
                temporary.rename(final_dir)
                record["output_dir"] = str(final_dir)
                record["video"] = str(final_dir / "episode_gripper_review.mp4")
                records = [item for item in records if int(item.get("episode", -1)) != episode_id]
                records.append(record)
                failures = [
                    item for item in failures if int(item.get("episode", -1)) != episode_id
                ]
            except Exception as exc:
                shutil.rmtree(temporary, ignore_errors=True)
                failure = {
                    "episode": episode_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                failures = [item for item in failures if int(item.get("episode", -1)) != episode_id]
                failures.append(failure)
            _write_json(
                manifest_path,
                {
                    "format": "robotwin_gripper_mask_video_qwen_qc_batch_v1",
                    "status": "running",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "object_mask_run_root": str(object_root),
                    "episode_ids": list(episode_ids),
                    "episodes": sorted(records, key=lambda item: int(item["episode"])),
                    "failures": sorted(failures, key=lambda item: int(item["episode"])),
                },
            )
            print(json.dumps(records[-1] if records and records[-1].get("episode") == episode_id else failures[-1], ensure_ascii=False))
    finally:
        backend.shutdown()
    status = "completed" if not failures and len(records) == len(episode_ids) else "partial_failure"
    _write_json(
        manifest_path,
        {
            "format": "robotwin_gripper_mask_video_qwen_qc_batch_v1",
            "status": status,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "object_mask_run_root": str(object_root),
            "episode_ids": list(episode_ids),
            "episodes": sorted(records, key=lambda item: int(item["episode"])),
            "failures": sorted(failures, key=lambda item: int(item["episode"])),
        },
    )
    print(
        json.dumps(
            {
                "status": status,
                "episode_count": len(records),
                "failure_count": len(failures),
                "manifest": str(manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
