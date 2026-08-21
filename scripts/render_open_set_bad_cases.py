"""Render explicitly pinned open-set bad cases across tasks.

The input manifest is the authority: this command never scans run roots or selects a
"best" artifact.  Each case points to one immutable episode directory and RGB video.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from robotwin_annotation_v2.adapters import rendering as renderer
from robotwin_annotation_v2.adapters.artifact_store import ArtifactStore

INPUT_FORMAT = "robotwin_open_set_bad_case_render_input_v1"
OUTPUT_FORMAT = "robotwin_open_set_bad_case_videos_v1"
ALLOWED_GROUPS = frozenset({"manual_review", "unresolved"})
ALLOWED_STAGES = frozenset({"S1", "S2", "S3"})
_EPISODE_DIRECTORY = re.compile(r"episode_(\d+)")


@dataclass(frozen=True)
class RenderCase:
    group: str
    stage: str
    episode_dir: Path
    video_path: Path
    source_run: str
    task: str
    episode_id: int
    camera: str

    @property
    def masks_path(self) -> Path:
        return self.episode_dir / "masks.npz"

    @property
    def run_manifest_path(self) -> Path:
        return self.episode_dir / "run_manifest.json"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=renderer.DEFAULT_FILL_ALPHA)
    parser.add_argument("--outline-radius", type=int, default=renderer.DEFAULT_OUTLINE_RADIUS)
    parser.add_argument("--halo-radius", type=int, default=renderer.DEFAULT_HALO_RADIUS)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _required_string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{label} must not contain surrounding whitespace")
    return value


def _resolve_path(value: Any, *, label: str, base_dir: Path) -> Path:
    raw = _required_string(value, label=label)
    path = Path(raw).expanduser()
    return (path if path.is_absolute() else base_dir / path).resolve()


def _case_from_json(value: Any, *, index: int, base_dir: Path) -> RenderCase:
    record = _mapping(value, label=f"cases[{index}]")
    group = _required_string(record.get("group"), label=f"cases[{index}].group")
    if group not in ALLOWED_GROUPS:
        raise ValueError(f"cases[{index}].group must be one of {sorted(ALLOWED_GROUPS)}")
    stage = _required_string(record.get("stage"), label=f"cases[{index}].stage").upper()
    if stage not in ALLOWED_STAGES:
        raise ValueError(f"cases[{index}].stage must be one of {sorted(ALLOWED_STAGES)}")

    episode_dir = _resolve_path(
        record.get("episode_dir"),
        label=f"cases[{index}].episode_dir",
        base_dir=base_dir,
    )
    video_path = _resolve_path(
        record.get("video_path"),
        label=f"cases[{index}].video_path",
        base_dir=base_dir,
    )
    match = _EPISODE_DIRECTORY.fullmatch(episode_dir.parent.name)
    if match is None:
        raise ValueError(f"cases[{index}].episode_dir must end in <task>/episode_<id>/<camera>")
    task = episode_dir.parents[1].name
    source_run = episode_dir.parents[2].name
    camera = episode_dir.name
    if not task or not source_run or not camera:
        raise ValueError(f"cases[{index}].episode_dir has an incomplete run layout")
    return RenderCase(
        group=group,
        stage=stage,
        episode_dir=episode_dir,
        video_path=video_path,
        source_run=source_run,
        task=task,
        episode_id=int(match.group(1)),
        camera=camera,
    )


def load_cases(path: Path) -> tuple[RenderCase, ...]:
    resolved = path.expanduser().resolve()
    root = _mapping(json.loads(resolved.read_text(encoding="utf-8")), label="manifest")
    if root.get("format_version") != INPUT_FORMAT:
        raise ValueError(f"manifest.format_version must be {INPUT_FORMAT!r}")
    raw_cases = root.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("manifest.cases must be a non-empty list")
    cases = tuple(
        _case_from_json(value, index=index, base_dir=resolved.parent)
        for index, value in enumerate(raw_cases)
    )
    keys = tuple((case.task, case.episode_id, case.camera) for case in cases)
    if len(keys) != len(set(keys)):
        raise ValueError("manifest.cases contains duplicate task/episode/camera entries")
    return cases


def _publication_status(artifact: renderer.MaskArtifact) -> str:
    statuses = tuple(
        status
        for role, status in zip(
            artifact.roles,
            artifact.annotation_status,
            strict=True,
        )
        if role in {"target", "receiver"} and status != "not_applicable"
    )
    if not statuses:
        return "not_applicable"
    if all(status == "valid" for status in statuses):
        return "completed"
    for status in ("quarantined", "failed"):
        if status in statuses:
            return status
    return "incomplete"


def _validate_sources(cases: Sequence[RenderCase]) -> None:
    missing: list[Path] = []
    for case in cases:
        if not case.episode_dir.is_dir():
            missing.append(case.episode_dir)
        missing.extend(
            path
            for path in (case.masks_path, case.run_manifest_path, case.video_path)
            if not path.is_file()
        )
    unique_missing = tuple(dict.fromkeys(missing))
    if unique_missing:
        preview = "\n".join(f"  - {path}" for path in unique_missing)
        raise FileNotFoundError(f"render sources are missing:\n{preview}")


def _run_manifest(case: RenderCase) -> Mapping[str, Any]:
    payload = _mapping(
        json.loads(case.run_manifest_path.read_text(encoding="utf-8")),
        label=str(case.run_manifest_path),
    )
    episode = _mapping(payload.get("episode"), label="run_manifest.episode")
    actual = (
        payload.get("run_id"),
        episode.get("task"),
        episode.get("episode_index"),
        episode.get("camera"),
    )
    expected = (case.source_run, case.task, case.episode_id, case.camera)
    if actual != expected:
        raise ValueError(f"run manifest identity mismatch: expected {expected}, got {actual}")
    return payload


def render_cases(
    input_manifest: Path,
    output_dir: Path,
    *,
    alpha: float = renderer.DEFAULT_FILL_ALPHA,
    outline_radius: int = renderer.DEFAULT_OUTLINE_RADIUS,
    halo_radius: int = renderer.DEFAULT_HALO_RADIUS,
    crf: int = 18,
    preset: str = "medium",
    overwrite: bool = False,
) -> Path:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if outline_radius < 0 or halo_radius < outline_radius:
        raise ValueError("halo_radius must be at least a non-negative outline_radius")
    if not 0 <= crf <= 51:
        raise ValueError("crf must be between 0 and 51")

    resolved_input = input_manifest.expanduser().resolve()
    cases = load_cases(resolved_input)
    _validate_sources(cases)
    destination = output_dir.expanduser().resolve()
    manifest_path = destination / "manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"output manifest already exists; pass --overwrite: {manifest_path}")
    destination.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for case in cases:
        run_manifest = _run_manifest(case)
        artifact = renderer.load_masks(case.masks_path)
        relative_video = (
            Path(case.group) / f"{case.task}__episode_{case.episode_id:06d}" / "mask_overlay.mp4"
        )
        output_video = destination / relative_video
        video_metadata = renderer.render_video(
            case.video_path,
            artifact,
            output_video,
            alpha=alpha,
            outline_radius=outline_radius,
            halo_radius=halo_radius,
            crf=crf,
            preset=preset,
            overwrite=overwrite,
        )
        records.append(
            {
                "group": case.group,
                "stage": case.stage,
                "task": case.task,
                "episode_index": case.episode_id,
                "episode_id": f"{case.episode_id:06d}",
                "camera": case.camera,
                "publication_status": _publication_status(artifact),
                "source_run": case.source_run,
                "source_episode_dir": str(case.episode_dir),
                "source_run_manifest": str(case.run_manifest_path),
                "source_run_manifest_sha256": renderer.file_sha256(case.run_manifest_path),
                "source_masks": str(case.masks_path),
                "source_masks_sha256": renderer.file_sha256(case.masks_path),
                "source_video": str(case.video_path),
                "source_video_sha256": renderer.file_sha256(case.video_path),
                "annotation_status": dict(
                    zip(artifact.instance_names, artifact.annotation_status, strict=True)
                ),
                "qc_status": dict(zip(artifact.instance_names, artifact.qc_status, strict=True)),
                "role_results": run_manifest.get("roles"),
                "output_video": str(relative_video),
                "output_video_sha256": renderer.file_sha256(output_video),
                "output_bytes": output_video.stat().st_size,
                **video_metadata,
            }
        )

    group_counts = Counter(record["group"] for record in records)
    status_counts = Counter(record["publication_status"] for record in records)
    manifest = {
        "format_version": OUTPUT_FORMAT,
        "created_at": datetime.now(UTC).isoformat(),
        "input_manifest": str(resolved_input),
        "input_manifest_sha256": renderer.file_sha256(resolved_input),
        "selection": "explicit episode_dir/video_path only; no run discovery or ranking",
        "episode_count": len(records),
        "group_counts": dict(sorted(group_counts.items())),
        "publication_status_counts": dict(sorted(status_counts.items())),
        "episodes": records,
    }
    return ArtifactStore.write_json(manifest_path, manifest)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    manifest = render_cases(
        args.manifest,
        args.output_dir,
        alpha=args.alpha,
        outline_radius=args.outline_radius,
        halo_radius=args.halo_radius,
        crf=args.crf,
        preset=args.preset,
        overwrite=args.overwrite,
    )
    print(json.dumps({"manifest": str(manifest)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
