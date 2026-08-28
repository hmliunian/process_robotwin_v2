#!/usr/bin/env python3
"""Generate a static HTML report and global review sheets for one rendered run."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from robotwin_annotation_v2.adapters.artifact_store import ArtifactStore
from robotwin_annotation_v2.adapters.rendering import build_sheets

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = PROJECT_ROOT / "configs" / "report_templates" / "rendered_run.html"
REPORT_FORMAT = "robotwin_rendered_run_report_v1"
REQUIRED_ROLES = ("target", "receiver")
SHEET_ORDER = ("target_early", "target_late", "receiver_early", "receiver_late")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "rendered_videos",
        type=Path,
        help="Directory containing manifest.json and overlay MP4 files",
    )
    parser.add_argument("--columns", type=int, default=4, help="Review-sheet columns")
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, help="Output HTML (default: <dir>/report.html)")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _dataset_episode_metadata(dataset_root: Path) -> dict[int, dict[str, Any]]:
    metadata_path = dataset_root / "meta" / "episodes.jsonl"
    if not metadata_path.is_file():
        return {}
    records: dict[int, dict[str, Any]] = {}
    for line in metadata_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict) and "episode_index" in value:
            records[int(value["episode_index"])] = value
    return records


def _local_episode_dir(
    run_dir: Path,
    task: str,
    camera: str,
    episode_index: int,
    rendered: dict[str, Any] | None,
) -> Path:
    if rendered is not None:
        source_masks = Path(str(rendered.get("source_masks", "")))
        source_dir = source_masks.parent
        if (source_dir / "run_manifest.json").is_file():
            return source_dir
    return run_dir / task / f"episode_{episode_index:06d}" / camera


def _attempts(role_qc: dict[str, Any]) -> list[dict[str, Any]]:
    value = role_qc.get("attempts", [])
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        return [item for item in value.values() if isinstance(item, dict)]
    return []


def _role_resolution(
    role: str,
    rendered: dict[str, Any] | None,
    role_qc: dict[str, Any] | None,
    run_role: dict[str, Any] | None,
) -> dict[str, Any]:
    instance = f"{role}_0"
    annotation_status = None
    qc_status = None
    nonempty_frames = 0
    if rendered is not None:
        annotation_status = rendered.get("annotation_status", {}).get(instance)
        qc_status = rendered.get("qc_status", {}).get(instance)
        nonempty_frames = int(rendered.get("nonempty_frames", {}).get(instance, 0))
    if run_role is not None:
        annotation_status = annotation_status or run_role.get("status")
        qc_status = qc_status or run_role.get("qc_status")

    attempts = _attempts(role_qc or {})
    text_attempts = [item for item in attempts if item.get("method") == "text_query"]
    bbox_attempts = [item for item in attempts if item.get("method") == "bbox_fallback"]
    selected_field = (role_qc or {}).get("selected_query_field")
    selected_method = "bbox_fallback" if selected_field == "bbox_fallback" else "text_query"
    selected_attempt_index = next(
        (
            index
            for index, item in enumerate(text_attempts)
            if item.get("status") == "passed" and item.get("selected_query_field") == selected_field
        ),
        None,
    )
    if qc_status != "passed":
        resolution = "unresolved"
    elif selected_method == "bbox_fallback":
        resolution = "bbox"
    elif selected_attempt_index is not None and selected_attempt_index > 0:
        resolution = "alternate_seed"
    else:
        resolution = "direct_text"

    attempt_summary = [
        {
            "method": str(item.get("method", "unknown")),
            "seed": item.get("seed_frame_id"),
            "status": str(item.get("status", "unknown")),
            "selectedField": item.get("selected_query_field"),
        }
        for item in attempts
    ]
    return {
        "role": role,
        "annotationStatus": annotation_status or "missing",
        "qcStatus": qc_status or (role_qc or {}).get("status") or "missing",
        "runStatus": (run_role or {}).get("status", "missing"),
        "failure": (run_role or {}).get("failure"),
        "qcReason": (run_role or {}).get("qc_reason") or (role_qc or {}).get("reason"),
        "nonemptyFrames": nonempty_frames,
        "resolution": resolution,
        "enteredS3": bool(bbox_attempts),
        "selectedCandidate": (role_qc or {}).get("selected_candidate"),
        "selectedQueryField": selected_field,
        "selectedQuery": (role_qc or {}).get("selected_query"),
        "selectedSeed": (role_qc or {}).get("selected_seed_frame_id"),
        "attemptCount": len(attempts),
        "textAttemptCount": len(text_attempts),
        "bboxAttemptCount": len(bbox_attempts),
        "attempts": attempt_summary,
    }


def _episode_status(has_video: bool, roles: dict[str, dict[str, Any]]) -> str:
    if not has_video:
        return "missing"
    annotations = {record["annotationStatus"] for record in roles.values()}
    qc = {record["qcStatus"] for record in roles.values()}
    if "error" in qc:
        return "error"
    if "quarantined" in annotations:
        return "quarantined"
    if annotations != {"valid"}:
        return "incomplete"
    return "passed"


def _episode_route(has_video: bool, roles: dict[str, dict[str, Any]]) -> str:
    if not has_video:
        return "missing"
    if any(role["enteredS3"] for role in roles.values()):
        return "bbox"
    if any(role["resolution"] == "alternate_seed" for role in roles.values()):
        return "alternate_seed"
    return "direct_text"


def _failure_detail(episode_dir: Path) -> str | None:
    failure_path = episode_dir / "qwen_failure.json"
    if not failure_path.is_file():
        return None
    value = _read_json(failure_path)
    return str(value.get("error")) if value.get("error") else None


def _role_records(run_manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records = run_manifest.get("roles", [])
    if not isinstance(records, list):
        return {}
    return {
        str(item["role"]): item
        for item in records
        if isinstance(item, dict) and item.get("role") in REQUIRED_ROLES
    }


def _task_text(metadata: dict[str, Any], rendered: dict[str, Any] | None) -> str:
    if rendered is not None and rendered.get("task_text"):
        return str(rendered["task_text"])
    tasks = metadata.get("tasks", [])
    if isinstance(tasks, list) and tasks:
        return str(tasks[0])
    return "Task text unavailable"


def collect_report(render_dir: Path) -> dict[str, Any]:
    """Collect manifest, process, QC, and provenance data into one JSON-safe model."""

    render_dir = render_dir.resolve()
    manifest_path = render_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"render manifest not found: {manifest_path}")
    render_manifest = _read_json(manifest_path)
    run_dir = render_dir.parent
    process_path = run_dir / "process_summary.json"
    process = _read_json(process_path) if process_path.is_file() else {}
    task = str(render_manifest.get("task") or process.get("task") or "unknown")
    camera = str(render_manifest.get("camera") or process.get("camera") or "unknown")
    run_id = str(render_manifest.get("requested_run_id") or process.get("run_id") or run_dir.name)
    dataset_root = Path(
        str(render_manifest.get("dataset_root") or process.get("dataset_root") or "")
    )
    dataset_metadata = _dataset_episode_metadata(dataset_root)
    rendered_by_id = {
        int(item["episode_index"]): item
        for item in render_manifest.get("episodes", [])
        if isinstance(item, dict)
    }
    requested_ids = [int(item) for item in process.get("requested_episode_ids", [])]
    if not requested_ids:
        requested_ids = sorted(rendered_by_id)
    process_records = {
        int(item["episode"]): item
        for item in process.get("records", [])
        if isinstance(item, dict) and "episode" in item
    }

    episodes: list[dict[str, Any]] = []
    prompt_versions: Counter[str] = Counter()
    for episode_index in requested_ids:
        rendered = rendered_by_id.get(episode_index)
        metadata = dataset_metadata.get(episode_index, {})
        episode_dir = _local_episode_dir(run_dir, task, camera, episode_index, rendered)
        run_manifest_path = episode_dir / "run_manifest.json"
        mask_qc_path = episode_dir / "mask_qc.json"
        semantic_path = episode_dir / "semantic_plan.json"
        run_manifest = _read_json(run_manifest_path) if run_manifest_path.is_file() else {}
        mask_qc = _read_json(mask_qc_path) if mask_qc_path.is_file() else {}
        semantic = _read_json(semantic_path) if semantic_path.is_file() else {}
        if semantic.get("prompt_version"):
            prompt_versions[str(semantic["prompt_version"])] += 1
        run_roles = _role_records(run_manifest)
        qc_roles = mask_qc.get("roles", {})
        if not isinstance(qc_roles, dict):
            qc_roles = {}
        roles = {
            role: _role_resolution(
                role,
                rendered,
                qc_roles.get(role) if isinstance(qc_roles.get(role), dict) else None,
                run_roles.get(role),
            )
            for role in REQUIRED_ROLES
        }
        has_video = rendered is not None and (render_dir / str(rendered["output_video"])).is_file()
        process_record = process_records.get(episode_index, {})
        status = _episode_status(has_video, roles)
        failure = _failure_detail(episode_dir)
        episodes.append(
            {
                "index": episode_index,
                "label": f"EP {episode_index:06d}",
                "taskText": _task_text(metadata, rendered),
                "taskTextZh": metadata.get("task_text_zh"),
                "targetName": metadata.get("target"),
                "receiverName": metadata.get("receiver"),
                "processStatus": process_record.get("status", "unknown"),
                "processError": process_record.get("error") or failure,
                "status": status,
                "route": _episode_route(has_video, roles),
                "hasVideo": has_video,
                "video": rendered.get("output_video") if rendered is not None else None,
                "durationSeconds": float(rendered.get("duration_seconds", 0.0))
                if rendered is not None
                else 0.0,
                "frameCount": int(rendered.get("frame_count", 0))
                if rendered is not None
                else int(metadata.get("length", 0)),
                "width": int(rendered.get("width", 0)) if rendered is not None else None,
                "height": int(rendered.get("height", 0)) if rendered is not None else None,
                "frameRate": rendered.get("frame_rate") if rendered is not None else None,
                "roles": roles,
            }
        )

    rendered_episodes = [episode for episode in episodes if episode["hasVideo"]]
    role_values = [
        episode["roles"][role] for episode in rendered_episodes for role in REQUIRED_ROLES
    ]
    process_counts = Counter(episode["processStatus"] for episode in episodes)
    status_counts = Counter(episode["status"] for episode in episodes)
    resolution_counts = Counter(role["resolution"] for role in role_values)
    qc_by_role = {
        role: Counter(episode["roles"][role]["qcStatus"] for episode in rendered_episodes)
        for role in REQUIRED_ROLES
    }
    annotation_by_role = {
        role: Counter(episode["roles"][role]["annotationStatus"] for episode in rendered_episodes)
        for role in REQUIRED_ROLES
    }

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        grouped[episode["taskText"]].append(episode)
    groups = []
    for task_text, members in grouped.items():
        rendered_members = [item for item in members if item["hasVideo"]]
        groups.append(
            {
                "taskText": task_text,
                "taskTextZh": next(
                    (item["taskTextZh"] for item in members if item["taskTextZh"]), None
                ),
                "episodeIds": [item["index"] for item in members],
                "episodeCount": len(members),
                "renderedCount": len(rendered_members),
                "durationSeconds": sum(item["durationSeconds"] for item in rendered_members),
                "passedCount": sum(item["status"] == "passed" for item in members),
                "issueCount": sum(item["status"] != "passed" for item in members),
                "targetPassed": sum(
                    item["roles"]["target"]["qcStatus"] == "passed" for item in members
                ),
                "receiverPassed": sum(
                    item["roles"]["receiver"]["qcStatus"] == "passed" for item in members
                ),
                "s3Count": sum(
                    any(role["enteredS3"] for role in item["roles"].values()) for item in members
                ),
            }
        )
    groups.sort(key=lambda item: min(item["episodeIds"]))

    stats = {
        "inputEpisodes": len(episodes),
        "renderedEpisodes": len(rendered_episodes),
        "missingEpisodes": sum(not episode["hasVideo"] for episode in episodes),
        "inputTasks": len(groups),
        "renderedTasks": len({item["taskText"] for item in rendered_episodes}),
        "totalDurationSeconds": sum(item["durationSeconds"] for item in rendered_episodes),
        "totalFrames": sum(item["frameCount"] for item in rendered_episodes),
        "processCompleted": process_counts["completed"],
        "processIncomplete": process_counts["sam_incomplete"],
        "processFailed": process_counts["failed"],
        "episodeStatus": dict(status_counts),
        "roleCount": len(role_values),
        "qcPassedRoles": sum(role["qcStatus"] == "passed" for role in role_values),
        "validRoles": sum(role["annotationStatus"] == "valid" for role in role_values),
        "resolution": dict(resolution_counts),
        "s3AttemptedRoles": sum(role["enteredS3"] for role in role_values),
        "s3PassedRoles": sum(role["resolution"] == "bbox" for role in role_values),
        "qcByRole": {role: dict(counts) for role, counts in qc_by_role.items()},
        "annotationByRole": {role: dict(counts) for role, counts in annotation_by_role.items()},
        "promptVersions": dict(prompt_versions),
        "openSetEpisodeCount": sum(prompt_versions.values()),
    }
    return {
        "format": REPORT_FORMAT,
        "generatedAt": datetime.now(UTC).isoformat(),
        "meta": {
            "runId": run_id,
            "task": task,
            "camera": camera,
            "annotationMode": process.get("annotation_mode", "pick_place"),
            "stageMode": process.get("stage_mode", "unknown"),
            "objectBackend": process.get("backend", {}).get("object_masks", "sam"),
            "gripperBackend": process.get("backend", {}).get("gripper"),
            "qwenModel": process.get("qwen_health", {}).get("model", "unknown"),
            "renderCreatedAt": render_manifest.get("created_at"),
            "maskFormat": next(
                (
                    item.get("mask_format")
                    for item in render_manifest.get("episodes", [])
                    if isinstance(item, dict) and item.get("mask_format")
                ),
                "unknown",
            ),
            "renderFormat": render_manifest.get("format"),
            "alpha": render_manifest.get("alpha"),
            "colors": render_manifest.get("colors_rgb", {}),
            "renderStyle": render_manifest.get("render_style", {}),
        },
        "stats": stats,
        "groups": groups,
        "episodes": episodes,
        "anomalies": [episode for episode in episodes if episode["status"] != "passed"],
        "reviewSheets": [],
    }


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def build_overall_index(
    sheet_paths: list[Path],
    output_path: Path,
    report: dict[str, Any],
) -> Path:
    """Compose the four role/moment sheets into one presentation index."""

    by_stem = {path.stem: path for path in sheet_paths}
    ordered = [by_stem[name] for name in SHEET_ORDER if name in by_stem]
    if not ordered:
        raise ValueError("no target/receiver review sheets were generated")

    canvas_width = 2560
    margin = 64
    gap = 32
    header_height = 250
    title_height = 72
    columns = 2
    card_width = (canvas_width - 2 * margin - gap) // columns
    source_images = [Image.open(path).convert("RGB") for path in ordered]
    max_ratio = max(image.height / image.width for image in source_images)
    image_height = math.ceil(card_width * max_ratio)
    rows = math.ceil(len(source_images) / columns)
    canvas_height = header_height + rows * (title_height + image_height + gap) + margin
    canvas = Image.new("RGB", (canvas_width, canvas_height), (8, 16, 31))
    draw = ImageDraw.Draw(canvas)
    title_font = _font(58)
    subtitle_font = _font(28)
    card_font = _font(30)
    draw.text(
        (margin, 48), "PICK & PLACE · FULL REVIEW INDEX", fill=(242, 246, 251), font=title_font
    )
    stats = report["stats"]
    subtitle = (
        f"{stats['renderedEpisodes']}/{stats['inputEpisodes']} rendered  ·  "
        f"{stats['processCompleted']} complete episodes  ·  "
        f"{stats['validRoles']}/{stats['roleCount']} publishable roles"
    )
    draw.text((margin, 132), subtitle, fill=(154, 170, 194), font=subtitle_font)
    legend_x = margin
    for color, label in (
        ((36, 180, 92), "TARGET"),
        ((35, 116, 224), "RECEIVER"),
        ((255, 255, 0), "GRASP HOLD"),
        ((232, 67, 55), "ISSUE"),
    ):
        draw.ellipse((legend_x, 192, legend_x + 20, 212), fill=color)
        draw.text((legend_x + 30, 184), label, fill=(190, 201, 218), font=_font(20))
        legend_x += 210

    labels = {
        "target_early": "TARGET · EARLY",
        "target_late": "TARGET · LATE",
        "receiver_early": "RECEIVER · EARLY",
        "receiver_late": "RECEIVER · LATE",
    }
    for index, (path, source) in enumerate(zip(ordered, source_images, strict=True)):
        row, column = divmod(index, columns)
        x = margin + column * (card_width + gap)
        y = header_height + row * (title_height + image_height + gap)
        draw.rounded_rectangle(
            (x, y, x + card_width, y + title_height + image_height),
            radius=18,
            fill=(18, 30, 50),
            outline=(45, 61, 84),
            width=2,
        )
        draw.text((x + 24, y + 18), labels[path.stem], fill=(228, 235, 245), font=card_font)
        resized = source.copy()
        resized.thumbnail((card_width, image_height), Image.Resampling.LANCZOS)
        paste_x = x + (card_width - resized.width) // 2
        paste_y = y + title_height + (image_height - resized.height) // 2
        canvas.paste(resized, (paste_x, paste_y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=90, optimize=True)
    return output_path


def generate_review_sheets(render_dir: Path, report: dict[str, Any], columns: int) -> list[Path]:
    manifest_path = render_dir / "manifest.json"
    output_dir = render_dir / "review_sheets"
    sheet_paths = build_sheets(manifest_path, output_dir, columns=columns)
    overall = build_overall_index(sheet_paths, output_dir / "overall_index.jpg", report)
    manifest = _read_json(manifest_path)
    manifest["review_sheets"] = [str(path.relative_to(render_dir)) for path in sheet_paths]
    ArtifactStore.write_json(manifest_path, manifest)
    return [overall, *sheet_paths]


def _sheet_records(render_dir: Path, sheet_paths: list[Path]) -> list[dict[str, Any]]:
    labels = {
        "overall_index": ("全局总览", "overall", "index"),
        "target_early": ("Target · Early", "target", "early"),
        "target_late": ("Target · Late", "target", "late"),
        "receiver_early": ("Receiver · Early", "receiver", "early"),
        "receiver_late": ("Receiver · Late", "receiver", "late"),
    }
    output = []
    for path in sheet_paths:
        with Image.open(path) as image:
            width, height = image.size
        title, role, moment = labels.get(path.stem, (path.stem, "other", "other"))
        output.append(
            {
                "title": title,
                "role": role,
                "moment": moment,
                "path": str(path.relative_to(render_dir)),
                "width": width,
                "height": height,
            }
        )
    return output


def render_html(
    report: dict[str, Any],
    template_path: Path,
    output_path: Path,
) -> Path:
    template = template_path.read_text(encoding="utf-8")
    payload = json.dumps(report, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    marker = "__REPORT_JSON__"
    if template.count(marker) != 1:
        raise ValueError(f"template must contain exactly one {marker}: {template_path}")
    html = template.replace(marker, payload)
    output_path.write_text(html, encoding="utf-8")
    return output_path


def validate_report(report: dict[str, Any], render_dir: Path, output_path: Path) -> None:
    if not output_path.is_file() or output_path.stat().st_size < 20_000:
        raise ValueError(f"report HTML was not written correctly: {output_path}")
    for episode in report["episodes"]:
        if episode["hasVideo"] and not (render_dir / episode["video"]).is_file():
            raise FileNotFoundError(f"missing report video: {episode['video']}")
    for sheet in report["reviewSheets"]:
        if not (render_dir / sheet["path"]).is_file():
            raise FileNotFoundError(f"missing report review sheet: {sheet['path']}")
    if len(report["episodes"]) != report["stats"]["inputEpisodes"]:
        raise ValueError("episode count does not match report statistics")


def main() -> None:
    args = _parse_args()
    if args.columns < 1:
        raise ValueError("--columns must be positive")
    render_dir = args.rendered_videos.resolve()
    output_path = args.output.resolve() if args.output else render_dir / "report.html"
    report = collect_report(render_dir)
    sheets = generate_review_sheets(render_dir, report, args.columns)
    report["reviewSheets"] = _sheet_records(render_dir, sheets)
    render_html(report, args.template.resolve(), output_path)
    validate_report(report, render_dir, output_path)
    print(
        json.dumps(
            {
                "report": str(output_path),
                "review_sheets": [str(path) for path in sheets],
                "episodes": report["stats"]["inputEpisodes"],
                "rendered": report["stats"]["renderedEpisodes"],
                "anomalies": len(report["anomalies"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
