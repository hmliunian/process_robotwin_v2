#!/usr/bin/env python3
"""Render full-length coverage20 videos with the best saved mask run per episode."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np
from robotwin_annotation_v2.adapters import ArtifactStore, RoboTwinDataset
from robotwin_annotation_v2.config import PipelineConfig, load_config
from robotwin_annotation_v2.models import EpisodeRef


PROJECT_ROOT = Path(__file__).resolve().parents[1]


ROLE_COLORS: dict[str, tuple[int, int, int]] = {
    "target": (36, 180, 92),
    "receiver": (35, 116, 224),
    "gripper": (232, 67, 55),
}
DEFAULT_COLOR = (255, 196, 0)
HALO_COLOR = (0, 0, 0)
DEFAULT_FILL_ALPHA = 0.32
DEFAULT_OUTLINE_RADIUS = 3
DEFAULT_HALO_RADIUS = 5


@dataclass(frozen=True)
class MaskCandidate:
    path: Path
    run_id: str
    role_status: dict[str, str]
    valid_role_count: int
    modified_ns: int

    @property
    def score(self) -> tuple[int, int]:
        return self.valid_role_count, self.modified_ns


@dataclass(frozen=True)
class MaskArtifact:
    masks: np.ndarray
    instance_names: tuple[str, ...]
    roles: tuple[str, ...]
    annotation_status: tuple[str, ...]
    frame_count: int
    format_version: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "pilot_move_pillbottle_pad.yaml",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        help="Saved run root; defaults to output.root from the pipeline config",
    )
    parser.add_argument(
        "--run-id",
        help="Render only this exact run instead of selecting across all saved runs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "rendered_videos" / "coverage20_best_current",
    )
    parser.add_argument("--episode-ids", type=int, nargs="*")
    parser.add_argument("--alpha", type=float, default=DEFAULT_FILL_ALPHA)
    parser.add_argument(
        "--outline-radius",
        type=int,
        default=DEFAULT_OUTLINE_RADIUS,
        help="Colored outline expansion in pixels outside the mask",
    )
    parser.add_argument(
        "--halo-radius",
        type=int,
        default=DEFAULT_HALO_RADIUS,
        help="Total black halo expansion in pixels outside the mask",
    )
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _small_strings(archive: Any, key: str) -> tuple[str, ...]:
    return tuple(str(value) for value in archive[key].tolist())


def _candidate(path: Path, runs_root: Path) -> MaskCandidate:
    relative = path.relative_to(runs_root)
    if len(relative.parts) != 5:
        raise ValueError(f"unexpected mask artifact layout: {path}")
    run_id = relative.parts[0]
    with np.load(path, allow_pickle=False) as archive:
        roles = _small_strings(archive, "roles")
        statuses = _small_strings(archive, "annotation_status")
    if len(roles) != len(statuses):
        raise ValueError(f"roles/status length mismatch: {path}")
    role_status = {
        role: status
        for role, status in zip(roles, statuses, strict=True)
        if role in {"target", "receiver"}
    }
    valid_role_count = sum(role_status.get(role) == "valid" for role in ("target", "receiver"))
    return MaskCandidate(
        path=path.resolve(),
        run_id=run_id,
        role_status=role_status,
        valid_role_count=valid_role_count,
        modified_ns=path.stat().st_mtime_ns,
    )


def select_best_masks(
    runs_root: Path,
    *,
    task: str,
    camera: str,
    episode_ids: tuple[int, ...],
    run_id: str | None = None,
) -> dict[int, MaskCandidate]:
    wanted = set(episode_ids)
    candidates: dict[int, list[MaskCandidate]] = {episode_id: [] for episode_id in episode_ids}
    pattern = f"*/{task}/episode_*/{camera}/masks.npz"
    for path in runs_root.glob(pattern):
        if run_id is not None and path.relative_to(runs_root).parts[0] != run_id:
            continue
        episode_name = path.parents[1].name
        try:
            episode_id = int(episode_name.removeprefix("episode_"))
        except ValueError as exc:
            raise ValueError(f"invalid episode artifact directory: {path}") from exc
        if episode_id in wanted:
            candidates[episode_id].append(_candidate(path, runs_root))
    missing = [episode_id for episode_id, values in candidates.items() if not values]
    if missing:
        raise FileNotFoundError(f"no masks.npz found for episodes: {missing}")
    return {
        episode_id: max(values, key=lambda value: value.score)
        for episode_id, values in candidates.items()
    }


def _load_masks(path: Path) -> MaskArtifact:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "format_version",
            "frame_count",
            "masks",
            "instance_names",
            "roles",
            "annotation_status",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"mask archive is missing {missing}: {path}")
        masks = np.asarray(archive["masks"], dtype=bool)
        names = _small_strings(archive, "instance_names")
        roles = _small_strings(archive, "roles")
        statuses = _small_strings(archive, "annotation_status")
        frame_count = int(archive["frame_count"])
        format_version = str(archive["format_version"])
    if masks.ndim != 4:
        raise ValueError(f"masks must be [N,T,H,W], got {masks.shape}: {path}")
    if masks.shape[0] != len(names) or len(names) != len(roles) or len(roles) != len(statuses):
        raise ValueError(f"mask metadata length mismatch: {path}")
    if masks.shape[1] != frame_count:
        raise ValueError(f"mask frame_count mismatch: {path}")
    return MaskArtifact(masks, names, roles, statuses, frame_count, format_version)


def _dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Dilate a mask with a square kernel using a summed-area table."""

    value = np.asarray(mask, dtype=bool)
    if value.ndim != 2:
        raise ValueError("mask must be 2-D")
    if radius < 0:
        raise ValueError("dilation radius must be non-negative")
    if radius == 0:
        return value.copy()

    padded = np.pad(value.astype(np.uint8), radius)
    integral = np.pad(padded, ((1, 0), (1, 0)))
    integral = integral.cumsum(axis=0, dtype=np.int32).cumsum(axis=1, dtype=np.int32)
    kernel_size = radius * 2 + 1
    counts = (
        integral[kernel_size:, kernel_size:]
        - integral[:-kernel_size, kernel_size:]
        - integral[kernel_size:, :-kernel_size]
        + integral[:-kernel_size, :-kernel_size]
    )
    return counts > 0


def _external_outline_layers(
    mask: np.ndarray,
    *,
    outline_radius: int,
    halo_radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return black halo and colored-outline pixels, both strictly outside the mask."""

    if outline_radius < 0:
        raise ValueError("outline_radius must be non-negative")
    if halo_radius < outline_radius:
        raise ValueError("halo_radius must be at least outline_radius")
    outlined = _dilate_mask(mask, outline_radius)
    haloed = _dilate_mask(mask, halo_radius)
    colored_outline = outlined & ~mask
    black_halo = haloed & ~outlined
    return black_halo, colored_outline


def overlay_frame(
    frame: np.ndarray,
    artifact: MaskArtifact,
    frame_id: int,
    alpha: float,
    *,
    outline_radius: int = DEFAULT_OUTLINE_RADIUS,
    halo_radius: int = DEFAULT_HALO_RADIUS,
) -> np.ndarray:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if frame_id >= artifact.frame_count:
        return frame
    output = frame.astype(np.float32)
    for index, (role, status) in enumerate(
        zip(artifact.roles, artifact.annotation_status, strict=True)
    ):
        if status != "valid":
            continue
        mask = artifact.masks[index, frame_id]
        if not mask.any():
            continue
        color = np.asarray(ROLE_COLORS.get(role, DEFAULT_COLOR), dtype=np.float32)
        black_halo, colored_outline = _external_outline_layers(
            mask,
            outline_radius=outline_radius,
            halo_radius=halo_radius,
        )
        output[black_halo] = HALO_COLOR
        output[colored_outline] = color
        output[mask] = output[mask] * (1.0 - alpha) + color * alpha
    return np.clip(output, 0, 255).astype(np.uint8)


def _stream_rate(stream: Any) -> Fraction:
    rate = stream.average_rate or stream.base_rate
    if rate is None or rate <= 0:
        raise ValueError("source video does not expose a positive frame rate")
    return Fraction(rate.numerator, rate.denominator)


def render_video(
    video_path: Path,
    artifact: MaskArtifact,
    output_path: Path,
    *,
    alpha: float,
    outline_radius: int,
    halo_radius: int,
    crf: int,
    preset: str,
    overwrite: bool,
) -> dict[str, Any]:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output already exists; pass --overwrite: {output_path}")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to encode overlay videos")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp.mp4")
    temporary.unlink(missing_ok=True)

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        frame_rate = _stream_rate(stream)
        width, height = int(stream.width), int(stream.height)
        if artifact.masks.shape[2:] != (height, width):
            raise ValueError(
                f"video/mask shape mismatch: video={(height, width)}, "
                f"masks={artifact.masks.shape[2:]}"
            )
        rate_text = f"{frame_rate.numerator}/{frame_rate.denominator}"
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{width}x{height}",
            "-pix_fmt",
            "rgb24",
            "-r",
            rate_text,
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(temporary),
        ]
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        if process.stdin is None or process.stderr is None:
            process.kill()
            raise RuntimeError("failed to open ffmpeg pipes")
        decoded_count = 0
        try:
            for frame_id, video_frame in enumerate(container.decode(stream)):
                rgb = video_frame.to_ndarray(format="rgb24")
                overlaid = overlay_frame(
                    rgb,
                    artifact,
                    frame_id,
                    alpha,
                    outline_radius=outline_radius,
                    halo_radius=halo_radius,
                )
                process.stdin.write(overlaid.tobytes())
                decoded_count += 1
            process.stdin.close()
            stderr = process.stderr.read()
            return_code = process.wait()
            if return_code != 0:
                raise RuntimeError(stderr.decode(errors="replace").strip())
        except BaseException:
            process.kill()
            process.wait()
            temporary.unlink(missing_ok=True)
            raise
    if decoded_count < artifact.frame_count:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"source video has {decoded_count} frames but masks require {artifact.frame_count}"
        )
    temporary.replace(output_path)
    return {
        "frame_count": decoded_count,
        "mask_frame_count": artifact.frame_count,
        "unmasked_trailing_frames": decoded_count - artifact.frame_count,
        "frame_rate": rate_text,
        "width": width,
        "height": height,
        "duration_seconds": float(Fraction(decoded_count, 1) / frame_rate),
    }


def _dataset(config: PipelineConfig) -> RoboTwinDataset:
    return RoboTwinDataset(
        config.dataset.root,
        task=config.dataset.task,
        camera=config.dataset.camera,
        manifest_path=config.dataset.manifest,
    )


def main() -> None:
    args = _parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be between 0 and 1")
    if args.outline_radius < 0:
        raise ValueError("--outline-radius must be non-negative")
    if args.halo_radius < args.outline_radius:
        raise ValueError("--halo-radius must be at least --outline-radius")
    if not 0 <= args.crf <= 51:
        raise ValueError("--crf must be between 0 and 51")
    config = load_config(args.config)
    configured = set(config.dataset.regression_episode_ids)
    episode_ids = tuple(args.episode_ids or config.dataset.regression_episode_ids)
    unknown = sorted(set(episode_ids) - configured)
    if unknown:
        raise ValueError(f"episodes are outside the configured coverage20 set: {unknown}")

    runs_root = (args.runs_root or config.output_root).expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = _dataset(config)
    selected = select_best_masks(
        runs_root,
        task=config.dataset.task,
        camera=config.dataset.camera,
        episode_ids=episode_ids,
        run_id=args.run_id,
    )

    records: list[dict[str, Any]] = []
    for position, episode_id in enumerate(episode_ids, start=1):
        candidate = selected[episode_id]
        artifact = _load_masks(candidate.path)
        ref = EpisodeRef(config.dataset.task, episode_id, config.dataset.camera)
        video_path = dataset.paths(ref).video
        output_path = output_dir / f"episode_{episode_id:06d}_{config.dataset.camera}_overlay.mp4"
        video = render_video(
            video_path,
            artifact,
            output_path,
            alpha=args.alpha,
            outline_radius=args.outline_radius,
            halo_radius=args.halo_radius,
            crf=args.crf,
            preset=args.preset,
            overwrite=args.overwrite,
        )
        nonempty_frames = {
            name: int(mask.reshape(mask.shape[0], -1).any(axis=1).sum())
            for name, mask in zip(artifact.instance_names, artifact.masks, strict=True)
        }
        record = {
            "episode_index": episode_id,
            "run_id": candidate.run_id,
            "source_video": str(video_path),
            "source_masks": str(candidate.path),
            "mask_sha256": _sha256(candidate.path),
            "mask_format": artifact.format_version,
            "annotation_status": dict(
                zip(artifact.instance_names, artifact.annotation_status, strict=True)
            ),
            "nonempty_frames": nonempty_frames,
            "output_video": output_path.name,
            "output_sha256": _sha256(output_path),
            "output_bytes": output_path.stat().st_size,
            **video,
        }
        records.append(record)
        print(
            json.dumps(
                {
                    "progress": f"{position}/{len(episode_ids)}",
                    "episode": episode_id,
                    "run_id": candidate.run_id,
                    "status": candidate.role_status,
                    "output": str(output_path),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    manifest = {
        "format": "robotwin_coverage20_overlay_videos_v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_rule": (
            f"exact run_id={args.run_id}"
            if args.run_id is not None
            else "maximize valid target/receiver roles, then newest masks.npz mtime"
        ),
        "requested_run_id": args.run_id,
        "config": str(config.config_path),
        "dataset_root": str(config.dataset.root),
        "runs_root": str(runs_root),
        "task": config.dataset.task,
        "camera": config.dataset.camera,
        "episode_count": len(records),
        "alpha": args.alpha,
        "colors_rgb": {key: list(value) for key, value in ROLE_COLORS.items()},
        "render_style": {
            "fill_alpha": args.alpha,
            "outline_mode": "external",
            "outline_radius": args.outline_radius,
            "halo_radius": args.halo_radius,
            "halo_color_rgb": list(HALO_COLOR),
        },
        "episodes": records,
    }
    manifest_path = ArtifactStore.write_json(output_dir / "manifest.json", manifest)
    print(json.dumps({"manifest": str(manifest_path), "episode_count": len(records)}, indent=2))


if __name__ == "__main__":
    main()
