#!/usr/bin/env python3
"""Replay archived contact/press mask candidates with a replacement QC prompt.

The replay is intentionally narrower than Stage 3: it never loads SAM and never
generates a new mask.  Candidate masks, seed frames, timeline context, and attempt
order all come from an existing ``mask_qc.json`` artifact.  Only the visual-QC
prompt and its Qwen response are new.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal, cast

import av
import numpy as np
from av.error import FFmpegError
from PIL import Image

from robotwin_annotation_v2.adapters.qwen_client import OpenAICompatibleQwenClient
from robotwin_annotation_v2.config import MaskConfig, load_config
from robotwin_annotation_v2.domain import AnnotationMode
from robotwin_annotation_v2.models import (
    EpisodeRef,
    FramePurpose,
    LoopContext,
    MaskCandidateInfo,
    MaskQCAttemptMethod,
    SemanticFrame,
    TargetOnlyEvents,
)
from robotwin_annotation_v2.models.loop_context import RoleName
from robotwin_annotation_v2.pipeline import mask_qc as mask_qc_pipeline
from robotwin_annotation_v2.pipeline.object_mask.resolver import ObjectMaskCandidate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "pilot_contact_press_target_only.yaml"
REPLAY_FORMAT = "robotwin_contact_press_mask_qc_replay_v1"


class ReplayError(RuntimeError):
    """An archived replay input does not satisfy the fixed-candidate contract."""


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReplayError(f"{label} must be an object")
    return value


def _list(value: Any, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ReplayError(f"{label} must be a list")
    return value


def _integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReplayError(f"{label} must be an integer")
    return value


def _number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReplayError(f"{label} must be a number")
    return float(value)


def _text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReplayError(f"{label} must be a non-empty string")
    return value


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ReplayError(f"{label} is missing: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplayError(f"cannot read {label} {path}: {exc}") from exc
    return dict(_mapping(payload, label=label))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_loop_context(episode_dir: Path) -> tuple[LoopContext, Path]:
    path = episode_dir / "loop.json"
    payload = _read_json(path, label="archived loop")
    if payload.get("annotation_mode") != AnnotationMode.TARGET_ONLY.value:
        raise ReplayError(f"archived loop is not target_only: {path}")

    episode = _mapping(payload.get("episode"), label="loop.episode")
    task = _text(episode.get("task"), label="loop.episode.task")
    episode_index = _integer(
        episode.get("episode_index"),
        label="loop.episode.episode_index",
    )
    camera = _text(episode.get("camera"), label="loop.episode.camera")

    raw_events = _mapping(payload.get("events"), label="loop.events")
    active_arm_value = raw_events.get("active_arm")
    if active_arm_value not in {"left", "right"}:
        raise ReplayError("loop.events.active_arm must be left or right")
    active_arm = cast(Literal["left", "right"], active_arm_value)
    reopen_value = raw_events.get("t_reopen_start")
    reopen = (
        None
        if reopen_value is None
        else _integer(reopen_value, label="loop.events.t_reopen_start")
    )
    events = TargetOnlyEvents(
        active_arm=active_arm,
        t_remove_start=_integer(
            raw_events.get("t_remove_start"),
            label="loop.events.t_remove_start",
        ),
        t_close_start=_integer(
            raw_events.get("t_close_start"),
            label="loop.events.t_close_start",
        ),
        t_close_end=_integer(
            raw_events.get("t_close_end"),
            label="loop.events.t_close_end",
        ),
        t_reopen_start=reopen,
    )

    semantic_frames: list[SemanticFrame] = []
    for index, raw_frame in enumerate(
        _list(payload.get("semantic_frames"), label="loop.semantic_frames")
    ):
        frame = _mapping(raw_frame, label=f"loop.semantic_frames[{index}]")
        raw_roles = _list(
            frame.get("eligible_roles"),
            label=f"loop.semantic_frames[{index}].eligible_roles",
        )
        roles: list[RoleName] = []
        for raw_role in raw_roles:
            if raw_role not in {"target", "receiver"}:
                raise ReplayError(f"unsupported archived semantic role: {raw_role!r}")
            roles.append(cast(RoleName, raw_role))
        try:
            purpose = FramePurpose(frame.get("purpose"))
        except (TypeError, ValueError) as exc:
            raise ReplayError(
                f"invalid loop.semantic_frames[{index}].purpose"
            ) from exc
        semantic_frames.append(
            SemanticFrame(
                frame_id=_integer(
                    frame.get("frame_id"),
                    label=f"loop.semantic_frames[{index}].frame_id",
                ),
                purpose=purpose,
                eligible_roles=tuple(roles),
            )
        )

    sources = _mapping(payload.get("sources"), label="loop.sources")
    context = LoopContext(
        episode=EpisodeRef(task, episode_index, camera),
        task_text=_text(payload.get("task_text"), label="loop.task_text"),
        frame_count=_integer(payload.get("frame_count"), label="loop.frame_count"),
        events=events,
        semantic_frames=tuple(semantic_frames),
        state_source=_text(sources.get("state"), label="loop.sources.state"),
        video_source=_text(sources.get("video"), label="loop.sources.video"),
        annotation_mode=AnnotationMode.TARGET_ONLY,
    )
    return context, path


def _decode_frames(video_path: Path, frame_ids: Sequence[int]) -> dict[int, Image.Image]:
    wanted = set(frame_ids)
    if not wanted:
        raise ReplayError("replay frame selection is empty")
    if not video_path.is_file():
        raise ReplayError(f"archived source video is missing: {video_path}")
    frames: dict[int, Image.Image] = {}
    try:
        with av.open(str(video_path)) as container:
            for frame_id, video_frame in enumerate(container.decode(video=0)):
                if frame_id in wanted:
                    frames[frame_id] = cast(Any, video_frame).to_image().convert("RGB")
                    if len(frames) == len(wanted):
                        break
    except (OSError, FFmpegError) as exc:
        raise ReplayError(f"cannot decode archived source video {video_path}: {exc}") from exc
    missing = sorted(wanted - set(frames))
    if missing:
        raise ReplayError(f"archived source video is missing replay frames: {missing}")
    return frames


def _resolve_source_episode(
    value: Any,
    *,
    artifact_root: Path,
) -> Path:
    raw_path = Path(_text(value, label="audit source episode artifact"))
    resolved = raw_path.expanduser().resolve() if raw_path.is_absolute() else artifact_root / raw_path
    resolved = resolved.resolve()
    if not resolved.is_dir():
        raise ReplayError(f"archived source episode directory is missing: {resolved}")
    return resolved


def _select_audit_records(
    audit: Mapping[str, Any],
    *,
    artifact_root: Path,
    tasks: frozenset[str] | None,
    episode_ids: frozenset[int] | None,
) -> tuple[tuple[str, int, Path], ...]:
    selected: list[tuple[str, int, Path]] = []
    seen: set[tuple[str, int]] = set()
    for index, raw_record in enumerate(_list(audit.get("episodes"), label="audit.episodes")):
        record = _mapping(raw_record, label=f"audit.episodes[{index}]")
        if record.get("stage") != "mask_qc_rejected":
            continue
        task = _text(record.get("task"), label=f"audit.episodes[{index}].task")
        episode = _integer(
            record.get("episode"),
            label=f"audit.episodes[{index}].episode",
        )
        if tasks is not None and task not in tasks:
            continue
        if episode_ids is not None and episode not in episode_ids:
            continue
        key = (task, episode)
        if key in seen:
            raise ReplayError(f"audit repeats episode {task}/{episode}")
        seen.add(key)
        artifacts = _mapping(
            record.get("artifacts"),
            label=f"audit.episodes[{index}].artifacts",
        )
        selected.append(
            (
                task,
                episode,
                _resolve_source_episode(
                    artifacts.get("source_episode"),
                    artifact_root=artifact_root,
                ),
            )
        )
    if not selected:
        raise ReplayError("audit selection contains no mask_qc_rejected episodes")
    return tuple(selected)


def _audit_success_summary(
    audit: Mapping[str, Any],
    *,
    tasks: frozenset[str] | None,
) -> tuple[int, int]:
    episode_count = 0
    success_count = 0
    for index, raw_record in enumerate(_list(audit.get("episodes"), label="audit.episodes")):
        record = _mapping(raw_record, label=f"audit.episodes[{index}]")
        task = _text(record.get("task"), label=f"audit.episodes[{index}].task")
        if tasks is not None and task not in tasks:
            continue
        episode_count += 1
        if record.get("final_status", record.get("stage")) == "completed":
            success_count += 1
    if not episode_count:
        raise ReplayError("audit success-rate selection contains no episodes")
    return episode_count, success_count


def _candidate_info(raw: Mapping[str, Any], *, seed_frame_id: int) -> MaskCandidateInfo:
    candidate_id = _text(raw.get("candidate_id"), label="candidate.candidate_id")
    raw_seed = raw.get("seed_frame_id")
    candidate_seed = (
        seed_frame_id
        if raw_seed is None
        else _integer(raw_seed, label=f"candidate {candidate_id}.seed_frame_id")
    )
    if candidate_seed != seed_frame_id:
        raise ReplayError(
            f"candidate {candidate_id} seed {candidate_seed} differs from attempt {seed_frame_id}"
        )
    nonempty = raw.get("nonempty")
    basic_valid = raw.get("basic_valid")
    if not isinstance(nonempty, bool) or not isinstance(basic_valid, bool):
        raise ReplayError(f"candidate {candidate_id} validity fields must be booleans")
    duplicate_of = raw.get("duplicate_of")
    basic_reason = raw.get("basic_reason")
    if duplicate_of is not None and not isinstance(duplicate_of, str):
        raise ReplayError(f"candidate {candidate_id}.duplicate_of must be a string or null")
    if basic_reason is not None and not isinstance(basic_reason, str):
        raise ReplayError(f"candidate {candidate_id}.basic_reason must be a string or null")
    return MaskCandidateInfo(
        candidate_id=candidate_id,
        query_field=_text(raw.get("query_field"), label=f"candidate {candidate_id}.query_field"),
        query=_text(raw.get("query"), label=f"candidate {candidate_id}.query"),
        nonempty=nonempty,
        area_fraction=_number(
            raw.get("area_fraction"),
            label=f"candidate {candidate_id}.area_fraction",
        ),
        component_count=_integer(
            raw.get("component_count"),
            label=f"candidate {candidate_id}.component_count",
        ),
        basic_valid=basic_valid,
        basic_reason=basic_reason,
        duplicate_of=duplicate_of,
        seed_frame_id=candidate_seed,
    )


def _load_attempt_candidates(
    episode_dir: Path,
    raw_attempt: Mapping[str, Any],
    artifact_attempts: Mapping[str, Any],
) -> tuple[tuple[ObjectMaskCandidate, ...], list[dict[str, Any]]]:
    seed_frame_id = _integer(raw_attempt.get("seed_frame_id"), label="attempt.seed_frame_id")
    frame_key = f"frame_{seed_frame_id:06d}"
    frame_artifacts = _mapping(
        artifact_attempts.get(frame_key),
        label=f"mask_qc.artifacts.attempts.target.{frame_key}",
    )
    mask_paths = _mapping(
        frame_artifacts.get("candidate_masks"),
        label=f"mask_qc artifact masks at {frame_key}",
    )

    candidates: list[ObjectMaskCandidate] = []
    provenance: list[dict[str, Any]] = []
    for index, raw_candidate in enumerate(
        _list(raw_attempt.get("candidates"), label="attempt.candidates")
    ):
        candidate_payload = _mapping(raw_candidate, label=f"attempt.candidates[{index}]")
        info = _candidate_info(candidate_payload, seed_frame_id=seed_frame_id)
        relative_value = mask_paths.get(info.candidate_id)
        relative = Path(
            _text(
                relative_value,
                label=f"candidate {info.candidate_id} archived mask path",
            )
        )
        path = (episode_dir / relative).resolve()
        if not path.is_relative_to(episode_dir.resolve()):
            raise ReplayError(f"candidate mask escapes the episode directory: {relative}")
        if not path.is_file():
            raise ReplayError(f"archived candidate mask is missing: {path}")
        try:
            with Image.open(path) as image:
                mask = np.asarray(image.convert("L"), dtype=np.uint8) > 0
        except OSError as exc:
            raise ReplayError(f"cannot read archived candidate mask {path}: {exc}") from exc
        candidates.append(
            ObjectMaskCandidate(
                candidate_id=info.candidate_id,
                query_field=info.query_field,
                query=info.query,
                seed_frame_id=seed_frame_id,
                mask=mask,
                info=info,
            )
        )
        provenance.append(
            {
                "candidate_id": info.candidate_id,
                "path": str(path),
                "sha256": _sha256(path),
            }
        )
    return tuple(candidates), provenance


def _report_view(report: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "status",
        "selected_candidate",
        "selected_query_field",
        "selected_query",
        "selected_seed_frame_id",
        "confidence",
        "reason",
        "model",
        "raw_response",
        "rendered_prompt",
    )
    return {key: report.get(key) for key in keys if key in report}


def _new_episode_report(attempts: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], int]:
    chosen: tuple[int, Mapping[str, Any]] | None = None
    meaningful: list[tuple[int, Mapping[str, Any]]] = []
    for attempt in attempts:
        index = cast(int, attempt["attempt_index"])
        new_report = attempt.get("new")
        effective = (
            _mapping(new_report, label="replayed attempt report")
            if new_report is not None
            else _mapping(attempt["old"], label="archived attempt report")
        )
        if attempt["has_basic_valid_candidate"]:
            meaningful.append((index, effective))
        if chosen is None and effective.get("status") in {"passed", "error"}:
            chosen = (index, effective)
    if chosen is None:
        if meaningful:
            chosen = meaningful[-1]
        elif attempts:
            final = attempts[-1]
            effective = final.get("new") or final["old"]
            chosen = (
                cast(int, final["attempt_index"]),
                _mapping(effective, label="final attempt report"),
            )
        else:
            raise ReplayError("archived target QC report contains no attempts")
    index, report = chosen
    output = _report_view(report)
    output["selected_attempt_index"] = index
    if output.get("status") not in {"passed", "error"}:
        text_seeds = [
            str(attempt["seed_frame_id"])
            for attempt in attempts
            if attempt["method"] == MaskQCAttemptMethod.TEXT_QUERY.value
        ]
        bbox_seeds = [
            str(attempt["seed_frame_id"])
            for attempt in attempts
            if attempt["method"] == MaskQCAttemptMethod.BBOX_FALLBACK.value
        ]
        reason = str(output.get("reason") or "fixed candidates rejected")
        if text_seeds:
            reason += f"; text seed frames: {','.join(text_seeds)}"
        if bbox_seeds:
            reason += f"; bbox fallback seed frames: {','.join(bbox_seeds)}"
        output["reason"] = reason
    return output, index


def _replay_episode(
    episode_dir: Path,
    *,
    expected_task: str,
    expected_episode: int,
    mask_config: MaskConfig,
    client: mask_qc_pipeline.MaskQCClient,
) -> dict[str, Any]:
    context, loop_path = _load_loop_context(episode_dir)
    if (context.episode.task, context.episode.episode_index) != (
        expected_task,
        expected_episode,
    ):
        raise ReplayError(
            "audit/source loop identity mismatch: "
            f"{expected_task}/{expected_episode} != "
            f"{context.episode.task}/{context.episode.episode_index}"
        )
    qc_path = episode_dir / "mask_qc.json"
    qc_payload = _read_json(qc_path, label="archived mask QC")
    roles = _mapping(qc_payload.get("roles"), label="mask_qc.roles")
    old_target = _mapping(roles.get("target"), label="mask_qc.roles.target")
    raw_attempts = _list(old_target.get("attempts"), label="mask_qc target attempts")
    artifacts = _mapping(qc_payload.get("artifacts"), label="mask_qc.artifacts")
    artifact_roles = _mapping(artifacts.get("attempts"), label="mask_qc.artifacts.attempts")
    artifact_attempts = _mapping(
        artifact_roles.get("target"),
        label="mask_qc.artifacts.attempts.target",
    )

    seed_ids = {
        _integer(
            _mapping(attempt, label=f"mask_qc target attempts[{index}]").get(
                "seed_frame_id"
            ),
            label=f"mask_qc target attempts[{index}].seed_frame_id",
        )
        for index, attempt in enumerate(raw_attempts)
    }
    frame_ids = seed_ids | {frame.frame_id for frame in context.semantic_frames}
    frames = _decode_frames(Path(context.video_source), sorted(frame_ids))

    attempt_results: list[dict[str, Any]] = []
    for index, raw_attempt_value in enumerate(raw_attempts):
        raw_attempt = _mapping(raw_attempt_value, label=f"mask_qc target attempts[{index}]")
        seed_frame_id = _integer(
            raw_attempt.get("seed_frame_id"),
            label=f"mask_qc target attempts[{index}].seed_frame_id",
        )
        try:
            method = MaskQCAttemptMethod(raw_attempt.get("method"))
        except (TypeError, ValueError) as exc:
            raise ReplayError(f"invalid archived attempt method at index {index}") from exc
        candidates, candidate_artifacts = _load_attempt_candidates(
            episode_dir,
            raw_attempt,
            artifact_attempts,
        )
        has_valid = any(candidate.info.basic_valid for candidate in candidates)
        new_report: dict[str, Any] | None = None
        skip_reason: str | None = None
        if has_valid:
            execution = mask_qc_pipeline._evaluate_candidate_visual_qc(
                context,
                "target",
                seed_frame_id=seed_frame_id,
                candidates=candidates,
                seed_image=frames[seed_frame_id],
                context_images=frames,
                mask_config=mask_config,
                client=client,
                method=method,
                provenance={"fixed_candidate_replay": True},
            )
            new_report = execution.report.to_json()
        else:
            skip_reason = "no archived mechanically valid candidate; prompt was not invoked"
        attempt_results.append(
            {
                "attempt_index": index,
                "method": method.value,
                "seed_frame_id": seed_frame_id,
                "has_basic_valid_candidate": has_valid,
                "replayed": new_report is not None,
                "skip_reason": skip_reason,
                "candidate_artifacts": candidate_artifacts,
                "old": dict(raw_attempt),
                "new": new_report,
            }
        )

    new_target, selected_attempt = _new_episode_report(attempt_results)
    old_view = _report_view(old_target)
    transition = f"{old_view.get('status')}->{new_target.get('status')}"
    return {
        "task": expected_task,
        "episode": expected_episode,
        "source_episode_dir": str(episode_dir),
        "inputs": {
            "loop": {"path": str(loop_path), "sha256": _sha256(loop_path)},
            "mask_qc": {"path": str(qc_path), "sha256": _sha256(qc_path)},
            "video": str(Path(context.video_source)),
        },
        "old": old_view,
        "new": new_target,
        "selected_new_attempt_index": selected_attempt,
        "transition": transition,
        "changed": (
            old_view.get("status") != new_target.get("status")
            or old_view.get("selected_candidate") != new_target.get("selected_candidate")
            or old_view.get("selected_seed_frame_id")
            != new_target.get("selected_seed_frame_id")
        ),
        "attempts": attempt_results,
    }


def replay_archive(
    *,
    audit_path: Path,
    artifact_root: Path,
    config_path: Path,
    qc_prompt_path: Path,
    mask_config: MaskConfig,
    client: mask_qc_pipeline.MaskQCClient,
    tasks: frozenset[str] | None = None,
    episode_ids: frozenset[int] | None = None,
) -> dict[str, Any]:
    """Replay every selected archived attempt and summarize episode transitions."""

    resolved_audit = audit_path.expanduser().resolve()
    resolved_artifact_root = artifact_root.expanduser().resolve()
    resolved_prompt = qc_prompt_path.expanduser().resolve()
    if not resolved_prompt.is_file():
        raise ReplayError(f"replacement QC prompt is missing: {resolved_prompt}")
    audit = _read_json(resolved_audit, label="failure audit")
    selected = _select_audit_records(
        audit,
        artifact_root=resolved_artifact_root,
        tasks=tasks,
        episode_ids=episode_ids,
    )
    baseline_episode_count, baseline_success_count = _audit_success_summary(
        audit,
        tasks=tasks,
    )
    health = client.health()
    episodes = [
        _replay_episode(
            episode_dir,
            expected_task=task,
            expected_episode=episode,
            mask_config=mask_config,
            client=client,
        )
        for task, episode, episode_dir in selected
    ]
    transitions = Counter(str(episode["transition"]) for episode in episodes)
    replayed_attempts = sum(
        int(attempt["replayed"])
        for episode in episodes
        for attempt in cast(list[dict[str, Any]], episode["attempts"])
    )
    flipped = [
        {"task": episode["task"], "episode": episode["episode"]}
        for episode in episodes
        if episode["transition"] == "rejected->passed"
    ]
    projected_success_count = baseline_success_count + len(flipped)
    projected_success_rate = projected_success_count / baseline_episode_count
    baseline_success_rate = baseline_success_count / baseline_episode_count
    return {
        "format_version": REPLAY_FORMAT,
        "audit": {"path": str(resolved_audit), "sha256": _sha256(resolved_audit)},
        "artifact_root": str(resolved_artifact_root),
        "config": str(config_path.expanduser().resolve()),
        "replacement_qc_prompt": {
            "path": str(resolved_prompt),
            "sha256": _sha256(resolved_prompt),
        },
        "qwen_health": health,
        "selection": {
            "stage": "mask_qc_rejected",
            "tasks": None if tasks is None else sorted(tasks),
            "episode_ids": None if episode_ids is None else sorted(episode_ids),
        },
        "summary": {
            "baseline_episode_count": baseline_episode_count,
            "baseline_success_count": baseline_success_count,
            "baseline_success_rate": baseline_success_rate,
            "projected_success_count": projected_success_count,
            "projected_success_rate": projected_success_rate,
            "projected_delta_percentage_points": (
                (projected_success_rate - baseline_success_rate) * 100.0
            ),
            "episode_count": len(episodes),
            "archived_attempt_count": sum(len(episode["attempts"]) for episode in episodes),
            "replayed_visual_attempt_count": replayed_attempts,
            "transition_counts": dict(sorted(transitions.items())),
            "old_rejected_new_passed_count": len(flipped),
            "old_rejected_new_passed": flipped,
        },
        "episodes": episodes,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Root used to resolve relative source_episode paths stored in the audit.",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--qc-prompt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", dest="tasks", action="append")
    parser.add_argument("--episode-ids", type=int, nargs="+")
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _write_json(path: Path, payload: Mapping[str, Any], *, force: bool) -> None:
    resolved = path.expanduser().resolve()
    if resolved.exists() and not force:
        raise FileExistsError(f"replay output already exists; use --force: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".{resolved.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(resolved)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = load_config(args.config)
    prompt = args.qc_prompt.expanduser().resolve()
    mask_config = replace(
        config.mask,
        qc_enabled=True,
        qc_prompt_template=prompt,
        qc_max_tokens=(
            config.mask.qc_max_tokens
            if args.max_tokens is None
            else args.max_tokens
        ),
    )
    client = OpenAICompatibleQwenClient(
        endpoint=config.qwen.endpoint,
        model=config.qwen.model,
        timeout_seconds=config.qwen.timeout_seconds,
    )
    report = replay_archive(
        audit_path=args.audit,
        artifact_root=args.artifact_root,
        config_path=args.config,
        qc_prompt_path=prompt,
        mask_config=mask_config,
        client=client,
        tasks=None if args.tasks is None else frozenset(args.tasks),
        episode_ids=(
            None if args.episode_ids is None else frozenset(args.episode_ids)
        ),
    )
    _write_json(args.output, report, force=bool(args.force))
    print(
        json.dumps(
            {
                "status": "completed",
                "output": str(args.output.expanduser().resolve()),
                **cast(dict[str, Any], report["summary"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
