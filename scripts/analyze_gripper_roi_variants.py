#!/usr/bin/env python3
"""Compare a legacy gripper-mask baseline with one or more ROI variants."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


FORMAT_VERSION = "robotwin_gripper_roi_variant_analysis_v1"


@dataclass(frozen=True)
class EpisodeData:
    """Loaded masks plus their JSON-safe episode metrics."""

    episode: int
    final: np.ndarray
    hard_roi: np.ndarray | None
    active_window: tuple[int, int]
    metrics: dict[str, Any]


@dataclass(frozen=True)
class RunData:
    """One batch output root, including unavailable episode diagnostics."""

    root: Path
    batch_manifest: dict[str, Any]
    expected_episodes: tuple[int, ...]
    episodes: dict[int, EpisodeData]
    unavailable: dict[int, dict[str, Any]]
    explicit_failures: dict[int, list[dict[str, Any]]]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-root",
        type=Path,
        required=True,
        help="Legacy batch output root containing batch_manifest.json",
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        required=True,
        metavar="LABEL=PATH",
        help="ROI variant batch root; repeat for multiple variants",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser.parse_args()


def parse_variants(values: Sequence[str]) -> dict[str, Path]:
    """Parse repeatable ``LABEL=PATH`` CLI values and reject ambiguous labels."""

    variants: dict[str, Path] = {}
    for value in values:
        label, separator, raw_path = value.partition("=")
        label = label.strip()
        raw_path = raw_path.strip()
        if not separator or not label or not raw_path:
            raise ValueError(f"variant must use non-empty LABEL=PATH syntax: {value!r}")
        if label == "baseline":
            raise ValueError("variant label 'baseline' is reserved")
        if label in variants:
            raise ValueError(f"duplicate variant label: {label}")
        variants[label] = Path(raw_path).expanduser().resolve()
    if not variants:
        raise ValueError("at least one --variant LABEL=PATH is required")
    return variants


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"required manifest does not exist: {path}") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON manifest {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"manifest must contain a JSON object: {path}")
    return value


def _episode_id(record: Mapping[str, Any]) -> int | None:
    value = record.get("episode", record.get("episode_index"))
    if isinstance(value, bool):
        return None
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _batch_episode_ids(batch: Mapping[str, Any], root: Path) -> tuple[int, ...]:
    episode_ids: set[int] = set()
    raw_ids = batch.get("episode_ids", [])
    if isinstance(raw_ids, list):
        for value in raw_ids:
            try:
                episode_ids.add(int(value))
            except (TypeError, ValueError):
                continue
    for key in ("episodes", "failures"):
        records = batch.get(key, [])
        if not isinstance(records, list):
            continue
        for record in records:
            if isinstance(record, dict) and (episode := _episode_id(record)) is not None:
                episode_ids.add(episode)
    for path in root.glob("episode_*/manifest.json"):
        try:
            episode_ids.add(int(path.parent.name.removeprefix("episode_")))
        except ValueError:
            continue
    return tuple(sorted(episode_ids))


def _explicit_failures(batch: Mapping[str, Any]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    failures = batch.get("failures", [])
    if not isinstance(failures, list):
        return grouped
    for record in failures:
        if not isinstance(record, dict) or (episode := _episode_id(record)) is None:
            continue
        grouped.setdefault(episode, []).append(dict(record))
    return grouped


def _bool_array(
    archive: Mapping[str, np.ndarray],
    keys: Iterable[str],
    *,
    ndim: int,
) -> tuple[np.ndarray | None, str | None]:
    for key in keys:
        if key not in archive:
            continue
        value = np.asarray(archive[key], dtype=bool)
        if value.ndim != ndim:
            raise ValueError(f"archive key {key!r} must have {ndim} dimensions, got {value.shape}")
        return value, key
    return None, None


def _active_window(
    archive: Mapping[str, np.ndarray],
    manifest: Mapping[str, Any],
    frame_count: int,
) -> tuple[int, int]:
    raw: Any = archive.get("active_window")
    if raw is None:
        episode = manifest.get("episode", {})
        raw = episode.get("active_window") if isinstance(episode, dict) else None
    if raw is None:
        raw = [0, frame_count - 1]
    values = np.asarray(raw).reshape(-1)
    if values.size != 2:
        raise ValueError(f"active_window must contain two frame ids, got {values.tolist()}")
    start, end = (int(value) for value in values)
    if not 0 <= start <= end < frame_count:
        raise ValueError(
            f"active_window [{start}, {end}] is outside a {frame_count}-frame archive"
        )
    return start, end


def _check_track_shapes(reference: np.ndarray, tracks: Mapping[str, np.ndarray | None]) -> None:
    for name, track in tracks.items():
        if track is not None and track.shape != reference.shape:
            raise ValueError(
                f"{name} shape {track.shape} does not match final track {reference.shape}"
            )


def _windowed(track: np.ndarray, window: tuple[int, int]) -> np.ndarray:
    return track[window[0] : window[1] + 1]


def _adjacent_ious(track: np.ndarray) -> list[float]:
    values: list[float] = []
    for left, right in zip(track[:-1], track[1:], strict=True):
        union = int((left | right).sum())
        if union:
            values.append(int((left & right).sum()) / union)
    return values


def _track_metrics(track: np.ndarray, window: tuple[int, int]) -> dict[str, Any]:
    active = _windowed(track, window)
    pixels = active.reshape(active.shape[0], -1).sum(axis=1)
    nonempty = pixels[pixels > 0]
    adjacent = _adjacent_ious(active)
    return {
        "window_inclusive": list(window),
        "window_frames": int(active.shape[0]),
        "nonempty_frames": int(nonempty.size),
        "coverage": float(nonempty.size / active.shape[0]),
        "pixel_frame_sum": int(pixels.sum()),
        "area_mean_all_frames": float(pixels.mean()),
        "area_mean_nonempty": None if not nonempty.size else float(nonempty.mean()),
        "area_median_nonempty": None if not nonempty.size else float(np.median(nonempty)),
        "area_min_nonempty": None if not nonempty.size else int(nonempty.min()),
        "area_max": int(pixels.max()),
        "adjacent_iou_mean": None if not adjacent else float(np.mean(adjacent)),
        "adjacent_iou_p05": None if not adjacent else float(np.quantile(adjacent, 0.05)),
        "adjacent_iou_pairs": len(adjacent),
    }


def _pixel_audit(track: np.ndarray | None, window: tuple[int, int]) -> dict[str, Any] | None:
    if track is None:
        return None
    active = _windowed(track, window)
    per_frame = active.reshape(active.shape[0], -1).sum(axis=1)
    return {
        "pixel_frame_sum": int(per_frame.sum()),
        "affected_frames": int((per_frame > 0).sum()),
        "max_pixels_per_frame": int(per_frame.max()),
    }


def _selected_candidate(seed: Mapping[str, Any]) -> Mapping[str, Any] | None:
    selected = seed.get("selected_candidate")
    qc = seed.get("qc", {})
    if selected is None and isinstance(qc, dict):
        selected = qc.get("selected_candidate")
    candidates = qc.get("candidates", []) if isinstance(qc, dict) else []
    if not isinstance(candidates, list):
        return None
    for candidate in candidates:
        if isinstance(candidate, dict) and str(candidate.get("candidate_id")) == str(selected):
            return candidate
    return None


def _optional_int(*values: Any) -> int | None:
    for value in values:
        if isinstance(value, bool) or value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _seed_metrics(
    manifest: Mapping[str, Any],
    clean_mask: np.ndarray | None,
) -> dict[str, Any]:
    raw_seed = manifest.get("seed", {})
    seed = raw_seed if isinstance(raw_seed, dict) else {}
    candidate = _selected_candidate(seed) or {}
    clean_from_archive = None if clean_mask is None else int(clean_mask.sum())
    return {
        "selected_candidate": seed.get("selected_candidate"),
        "frame": _optional_int(seed.get("frame")),
        "prompt_mode": seed.get("prompt_mode"),
        "raw_pixels": _optional_int(candidate.get("raw_pixels"), seed.get("raw_pixels")),
        "cropped_pixels": _optional_int(
            candidate.get("cropped_pixels"),
            seed.get("cropped_pixels"),
        ),
        "clean_pixels": _optional_int(
            clean_from_archive,
            candidate.get("clean_pixels"),
            seed.get("clean_pixels"),
        ),
    }


def _roi_manifest_geometry(manifest: Mapping[str, Any]) -> dict[str, Any] | None:
    roi_policy = manifest.get("roi_policy")
    if isinstance(roi_policy, dict):
        return dict(roi_policy)
    for parent_key in ("roi_geometry", "roi", "geometry"):
        value = manifest.get(parent_key)
        if isinstance(value, dict):
            return dict(value)
    contract = manifest.get("contract", {})
    if isinstance(contract, dict):
        for key in ("roi_geometry", "gripper_roi"):
            value = contract.get(key)
            if isinstance(value, dict):
                return dict(value)
    return None


def _load_episode(root: Path, episode: int) -> EpisodeData:
    episode_dir = root / f"episode_{episode:06d}"
    manifest = _read_json(episode_dir / "manifest.json")
    archive_path = episode_dir / "gripper_masks.npz"
    try:
        loaded = np.load(archive_path, allow_pickle=False)
    except FileNotFoundError:
        raise FileNotFoundError(f"required mask archive does not exist: {archive_path}") from None
    try:
        files = set(loaded.files)
        wanted = {
            "active_window",
            "gripper_track",
            "final_gripper_track",
            "final_track",
            "native_track",
            "prompt_roi_track",
            "hard_roi_track",
            "target_removed",
            "receiver_removed",
            "seed_mask",
            "clean_seed_mask",
        }
        if "prompt_roi_track" not in files or "hard_roi_track" not in files:
            wanted.add("roi_track")
        archive = {key: loaded[key] for key in wanted & files}
    finally:
        loaded.close()

    final, final_source = _bool_array(
        archive,
        ("gripper_track", "final_gripper_track", "final_track"),
        ndim=3,
    )
    if final is None or final_source is None:
        raise ValueError(f"archive has no final gripper track: {archive_path}")
    native, native_source = _bool_array(archive, ("native_track",), ndim=3)
    explicit_prompt, prompt_source = _bool_array(archive, ("prompt_roi_track",), ndim=3)
    explicit_hard, hard_source = _bool_array(archive, ("hard_roi_track",), ndim=3)
    legacy_roi, legacy_source = _bool_array(archive, ("roi_track",), ndim=3)
    prompt_roi = explicit_prompt if explicit_prompt is not None else legacy_roi
    hard_roi = explicit_hard if explicit_hard is not None else legacy_roi
    prompt_source = prompt_source or legacy_source
    hard_source = hard_source or legacy_source
    target_removed, target_source = _bool_array(archive, ("target_removed",), ndim=3)
    receiver_removed, receiver_source = _bool_array(archive, ("receiver_removed",), ndim=3)
    clean_mask, clean_source = _bool_array(
        archive,
        ("seed_mask", "clean_seed_mask"),
        ndim=2,
    )
    _check_track_shapes(
        final,
        {
            "native_track": native,
            "prompt_roi_track": prompt_roi,
            "hard_roi_track": hard_roi,
            "target_removed": target_removed,
            "receiver_removed": receiver_removed,
        },
    )
    window = _active_window(archive, manifest, final.shape[0])
    native_clipped = None
    if native is not None and hard_roi is not None:
        native_clipped = native & ~hard_roi
    clipped_metrics = _pixel_audit(native_clipped, window)
    if clipped_metrics is not None:
        native_pixels = int(_windowed(native, window).sum()) if native is not None else 0
        clipped_metrics["fraction_of_native_pixels"] = (
            None if not native_pixels else clipped_metrics["pixel_frame_sum"] / native_pixels
        )
    compatibility = {
        "legacy_prompt_roi_fallback": explicit_prompt is None and legacy_roi is not None,
        "legacy_hard_roi_fallback": explicit_hard is None and legacy_roi is not None,
        "missing_prompt_roi": prompt_roi is None,
        "missing_hard_roi": hard_roi is None,
    }
    metrics = {
        "status": "completed",
        "episode_manifest_status": manifest.get("status"),
        "seed": _seed_metrics(manifest, clean_mask),
        "final": _track_metrics(final, window),
        "native_clipped_by_hard_roi": clipped_metrics,
        "target_removed": _pixel_audit(target_removed, window),
        "receiver_removed": _pixel_audit(receiver_removed, window),
        "roi_geometry": _roi_manifest_geometry(manifest),
        "archive_keys": {
            "final": final_source,
            "native": native_source,
            "prompt_roi": prompt_source,
            "hard_roi": hard_source,
            "seed_clean": clean_source,
            "target_removed": target_source,
            "receiver_removed": receiver_source,
        },
        "compatibility": compatibility,
        "paths": {
            "manifest": str((episode_dir / "manifest.json").resolve()),
            "mask_archive": str(archive_path.resolve()),
        },
    }
    return EpisodeData(
        episode=episode,
        final=final,
        hard_roi=hard_roi,
        active_window=window,
        metrics=metrics,
    )


def load_run(root: Path) -> RunData:
    """Load a batch root while retaining per-episode failures in the report."""

    resolved = root.expanduser().resolve()
    batch_path = resolved / "batch_manifest.json"
    batch = _read_json(batch_path)
    expected = _batch_episode_ids(batch, resolved)
    failures = _explicit_failures(batch)
    episodes: dict[int, EpisodeData] = {}
    unavailable: dict[int, dict[str, Any]] = {}
    for episode in expected:
        try:
            episodes[episode] = _load_episode(resolved, episode)
        except (FileNotFoundError, OSError, ValueError) as error:
            unavailable[episode] = {
                "status": "failed" if episode in failures else "unavailable",
                "error": f"{type(error).__name__}: {error}",
                "batch_failures": failures.get(episode, []),
            }
    return RunData(
        root=resolved,
        batch_manifest=batch,
        expected_episodes=expected,
        episodes=episodes,
        unavailable=unavailable,
        explicit_failures=failures,
    )


def _mean_or_none(values: Sequence[float]) -> float | None:
    return None if not values else float(np.mean(values))


def _seed_summary(episodes: Iterable[EpisodeData], key: str) -> dict[str, Any]:
    values = [
        int(value)
        for episode in episodes
        if (value := episode.metrics["seed"].get(key)) is not None
    ]
    return {
        "episodes_with_value": len(values),
        "pixel_sum": int(sum(values)),
        "mean_pixels": _mean_or_none([float(value) for value in values]),
    }


def _run_summary(run: RunData) -> dict[str, Any]:
    episodes = list(run.episodes.values())
    failed_ids = (set(run.explicit_failures) | set(run.unavailable)) - set(run.episodes)
    total_frames = sum(episode.metrics["final"]["window_frames"] for episode in episodes)
    nonempty_frames = sum(
        episode.metrics["final"]["nonempty_frames"] for episode in episodes
    )
    final_pixels = sum(episode.metrics["final"]["pixel_frame_sum"] for episode in episodes)
    adjacent: list[float] = []
    for episode in episodes:
        adjacent.extend(_adjacent_ious(_windowed(episode.final, episode.active_window)))

    def audit_sum(key: str) -> dict[str, Any]:
        audits = [
            audit
            for episode in episodes
            if (audit := episode.metrics.get(key)) is not None
        ]
        return {
            "episodes_with_value": len(audits),
            "pixel_frame_sum": int(sum(audit["pixel_frame_sum"] for audit in audits)),
            "affected_frames": int(sum(audit["affected_frames"] for audit in audits)),
        }

    compatibility = {
        key: sum(bool(episode.metrics["compatibility"][key]) for episode in episodes)
        for key in (
            "legacy_prompt_roi_fallback",
            "legacy_hard_roi_fallback",
            "missing_prompt_roi",
            "missing_hard_roi",
        )
    }
    batch_episodes = run.batch_manifest.get("episodes", [])
    batch_failures = run.batch_manifest.get("failures", [])
    return {
        "root": str(run.root),
        "batch_manifest": str((run.root / "batch_manifest.json").resolve()),
        "batch_status": run.batch_manifest.get("status"),
        "completion": {
            "requested_episodes": len(run.expected_episodes),
            "batch_completed_records": (
                len(batch_episodes) if isinstance(batch_episodes, list) else 0
            ),
            "batch_failure_records": len(batch_failures) if isinstance(batch_failures, list) else 0,
            "analyzed_episodes": len(episodes),
            "failed_or_unavailable_episodes": len(failed_ids),
            "failed_or_unavailable_episode_ids": sorted(failed_ids),
        },
        "seed": {
            "raw": _seed_summary(episodes, "raw_pixels"),
            "cropped": _seed_summary(episodes, "cropped_pixels"),
            "clean": _seed_summary(episodes, "clean_pixels"),
        },
        "final": {
            "window_frames": int(total_frames),
            "nonempty_frames": int(nonempty_frames),
            "coverage": None if not total_frames else nonempty_frames / total_frames,
            "pixel_frame_sum": int(final_pixels),
            "area_mean_all_frames": None if not total_frames else final_pixels / total_frames,
            "adjacent_iou_mean": _mean_or_none(adjacent),
            "adjacent_iou_p05": (
                None if not adjacent else float(np.quantile(adjacent, 0.05))
            ),
            "adjacent_iou_pairs": len(adjacent),
        },
        "native_clipped_by_hard_roi": audit_sum("native_clipped_by_hard_roi"),
        "target_removed": audit_sum("target_removed"),
        "receiver_removed": audit_sum("receiver_removed"),
        "compatibility": compatibility,
        "roi_policy": run.batch_manifest.get("roi_policy"),
    }


def _comparison_unavailable(reason: str) -> dict[str, Any]:
    return {"available": False, "reason": reason}


def compare_episode(baseline: EpisodeData, variant: EpisodeData) -> dict[str, Any]:
    """Compare two final tracks and attribute gains to the variant's new hard-ROI band."""

    if variant.final.shape != baseline.final.shape:
        return _comparison_unavailable(
            "final track shape mismatch: "
            f"baseline={baseline.final.shape}, variant={variant.final.shape}"
        )
    start = min(baseline.active_window[0], variant.active_window[0])
    end = max(baseline.active_window[1], variant.active_window[1])
    baseline_final = baseline.final[start : end + 1]
    variant_final = variant.final[start : end + 1]
    gained = variant_final & ~baseline_final
    lost = baseline_final & ~variant_final
    union = baseline_final | variant_final
    intersection = baseline_final & variant_final
    gained_pixels = int(gained.sum())
    lost_pixels = int(lost.sum())
    gain_per_frame = gained.reshape(gained.shape[0], -1).sum(axis=1)
    loss_per_frame = lost.reshape(lost.shape[0], -1).sum(axis=1)
    result: dict[str, Any] = {
        "available": True,
        "comparison_window_inclusive": [start, end],
        "gained_pixel_frames": gained_pixels,
        "lost_pixel_frames": lost_pixels,
        "net_pixel_frames": gained_pixels - lost_pixels,
        "frames_with_gains": int((gain_per_frame > 0).sum()),
        "frames_with_losses": int((loss_per_frame > 0).sum()),
        "final_iou": None if not union.any() else int(intersection.sum()) / int(union.sum()),
        "new_hard_roi_band": None,
    }
    if baseline.hard_roi is None or variant.hard_roi is None:
        result["new_hard_roi_band_unavailable_reason"] = (
            "baseline or variant archive has neither hard_roi_track nor legacy roi_track"
        )
        return result
    if baseline.hard_roi.shape != variant.hard_roi.shape:
        result["new_hard_roi_band_unavailable_reason"] = (
            "hard ROI shape mismatch: "
            f"baseline={baseline.hard_roi.shape}, variant={variant.hard_roi.shape}"
        )
        return result
    new_band = variant.hard_roi[start : end + 1] & ~baseline.hard_roi[start : end + 1]
    gained_in_band = int((gained & new_band).sum())
    result["new_hard_roi_band"] = {
        "pixel_frame_sum": int(new_band.sum()),
        "gained_pixel_frames_in_band": gained_in_band,
        "fraction_of_gained_pixels": (
            None if not gained_pixels else gained_in_band / gained_pixels
        ),
    }
    return result


def _comparison_summary(comparisons: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    records = list(comparisons)
    available = [comparison for comparison in records if comparison.get("available")]
    gained = sum(int(comparison["gained_pixel_frames"]) for comparison in available)
    lost = sum(int(comparison["lost_pixel_frames"]) for comparison in available)
    band_records = [
        band
        for comparison in available
        if isinstance((band := comparison.get("new_hard_roi_band")), dict)
    ]
    gained_in_band = sum(int(band["gained_pixel_frames_in_band"]) for band in band_records)
    return {
        "episodes_compared": len(available),
        "episodes_unavailable": sum(1 for value in records if not value.get("available")),
        "gained_pixel_frames": int(gained),
        "lost_pixel_frames": int(lost),
        "net_pixel_frames": int(gained - lost),
        "new_hard_roi_band": {
            "episodes_with_value": len(band_records),
            "pixel_frame_sum": int(sum(int(value["pixel_frame_sum"]) for value in band_records)),
            "gained_pixel_frames_in_band": int(gained_in_band),
            "fraction_of_gained_pixels": None if not gained else gained_in_band / gained,
        },
    }


def _episode_payload(run: RunData, episode: int) -> dict[str, Any]:
    if episode in run.episodes:
        return dict(run.episodes[episode].metrics)
    if episode in run.unavailable:
        return dict(run.unavailable[episode])
    failures = run.explicit_failures.get(episode, [])
    return {
        "status": "failed" if failures else "not_requested",
        "batch_failures": failures,
    }


def build_report(baseline_root: Path, variants: Mapping[str, Path]) -> dict[str, Any]:
    """Build the full JSON-safe comparison report."""

    baseline = load_run(baseline_root)
    baseline_episodes = {
        str(episode): _episode_payload(baseline, episode)
        for episode in baseline.expected_episodes
    }
    variant_payloads: dict[str, Any] = {}
    # Load and summarize one variant at a time. A full batch contains dense
    # tracks, so retaining every variant would multiply peak memory by the
    # number of experiment arms without helping any cross-variant metric.
    for label, path in variants.items():
        run = load_run(path)
        episode_payloads: dict[str, Any] = {}
        comparisons: list[dict[str, Any]] = []
        # A smoke variant may intentionally run only a subset of the baseline.
        # Baseline-only episodes are not variant failures and must not dilute
        # the comparison availability count.
        for episode in run.expected_episodes:
            payload = _episode_payload(run, episode)
            baseline_data = baseline.episodes.get(episode)
            variant_data = run.episodes.get(episode)
            if baseline_data is None:
                comparison = _comparison_unavailable("baseline episode is unavailable")
            elif variant_data is None:
                comparison = _comparison_unavailable("variant episode is unavailable")
            else:
                comparison = compare_episode(baseline_data, variant_data)
            payload["comparison_to_baseline"] = comparison
            comparisons.append(comparison)
            episode_payloads[str(episode)] = payload
        variant_payloads[label] = {
            "summary": _run_summary(run),
            "comparison_to_baseline": _comparison_summary(comparisons),
            "episodes": episode_payloads,
        }
    return {
        "format_version": FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pixel_count_unit": "foreground pixel x frame within active_window",
        "compatibility_policy": {
            "prompt_roi": "prompt_roi_track, falling back to legacy roi_track",
            "hard_roi": "hard_roi_track, falling back to legacy roi_track",
            "seed_counts": (
                "selected seed QC candidate fields, with clean seed_mask archive fallback"
            ),
        },
        "baseline": {
            "summary": _run_summary(baseline),
            "episodes": baseline_episodes,
        },
        "variants": variant_payloads,
    }


def _number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _escape_cell(value: Any) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ")


def _run_markdown_row(label: str, summary: Mapping[str, Any]) -> str:
    completion = summary["completion"]
    seed = summary["seed"]
    final = summary["final"]
    clipped = summary["native_clipped_by_hard_roi"]
    target = summary["target_removed"]
    receiver = summary["receiver_removed"]
    return "| " + " | ".join(
        _escape_cell(value)
        for value in (
            label,
            f"{completion['analyzed_episodes']}/{completion['requested_episodes']}",
            completion["failed_or_unavailable_episodes"],
            _number(seed["raw"]["mean_pixels"], 1),
            _number(seed["cropped"]["mean_pixels"], 1),
            _number(seed["clean"]["mean_pixels"], 1),
            f"{final['nonempty_frames']}/{final['window_frames']}",
            _number(final["area_mean_all_frames"], 1),
            _number(final["adjacent_iou_mean"]),
            clipped["pixel_frame_sum"],
            target["pixel_frame_sum"],
            receiver["pixel_frame_sum"],
        )
    ) + " |"


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact human review report from ``build_report`` output."""

    lines = [
        "# Gripper ROI variant analysis",
        "",
        "Pixel counts are foreground pixel × frame within each episode's active window.",
        "",
        "## Run summary",
        "",
        (
            "| Run | Complete | Failed/unavailable | Seed raw mean | Seed cropped mean | "
            "Seed clean mean | Final nonempty | Final area mean | Adjacent IoU | "
            "Native clipped | Target removed | Receiver removed |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        _run_markdown_row("baseline", report["baseline"]["summary"]),
    ]
    variants = report.get("variants", {})
    for label, variant in variants.items():
        lines.append(_run_markdown_row(str(label), variant["summary"]))

    lines.extend(
        [
            "",
            "## Global delta versus baseline",
            "",
            (
                "| Variant | Episodes | Gained | Lost | Net | Gained in new hard-ROI band | "
                "Gain-in-band fraction |"
            ),
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for label, variant in variants.items():
        comparison = variant["comparison_to_baseline"]
        band = comparison["new_hard_roi_band"]
        lines.append(
            "| "
            + " | ".join(
                _escape_cell(value)
                for value in (
                    label,
                    comparison["episodes_compared"],
                    comparison["gained_pixel_frames"],
                    comparison["lost_pixel_frames"],
                    comparison["net_pixel_frames"],
                    band["gained_pixel_frames_in_band"],
                    _number(band["fraction_of_gained_pixels"]),
                )
            )
            + " |"
        )

    for label, variant in variants.items():
        lines.extend(
            [
                "",
                f"## Per-episode: {_escape_cell(label)}",
                "",
                (
                    "| Episode | Status | Seed raw/cropped/clean | Final nonempty | Area mean | "
                    "Adj. IoU | Native clipped | Target/receiver removed | Gained | Lost | "
                    "Gained in new band |"
                ),
                "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for episode, payload in variant["episodes"].items():
            comparison = payload["comparison_to_baseline"]
            if payload.get("status") != "completed":
                lines.append(
                    f"| {episode} | {_escape_cell(payload.get('status'))} | n/a | n/a | n/a | "
                    "n/a | n/a | n/a | n/a | n/a | n/a |"
                )
                continue
            seed = payload["seed"]
            final = payload["final"]
            clipped = payload.get("native_clipped_by_hard_roi")
            target = payload.get("target_removed")
            receiver = payload.get("receiver_removed")
            band = comparison.get("new_hard_roi_band") if comparison.get("available") else None
            row = (
                episode,
                payload["status"],
                (
                    f"{_number(seed['raw_pixels'])}/{_number(seed['cropped_pixels'])}/"
                    f"{_number(seed['clean_pixels'])}"
                ),
                f"{final['nonempty_frames']}/{final['window_frames']}",
                _number(final["area_mean_all_frames"], 1),
                _number(final["adjacent_iou_mean"]),
                "n/a" if clipped is None else clipped["pixel_frame_sum"],
                (
                    f"{'n/a' if target is None else target['pixel_frame_sum']}/"
                    f"{'n/a' if receiver is None else receiver['pixel_frame_sum']}"
                ),
                comparison.get("gained_pixel_frames", "n/a"),
                comparison.get("lost_pixel_frames", "n/a"),
                "n/a" if band is None else band["gained_pixel_frames_in_band"],
            )
            lines.append("| " + " | ".join(_escape_cell(value) for value in row) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parse_args()
    variants = parse_variants(args.variant)
    report = build_report(args.baseline_root, variants)
    output_json = args.output_json.expanduser().resolve()
    output_markdown = args.output_markdown.expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    output_markdown.write_text(render_markdown(report), encoding="utf-8")
    print(output_json)
    print(output_markdown)


if __name__ == "__main__":
    main()
