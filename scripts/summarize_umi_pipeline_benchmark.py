"""Compare full UMI trial artifacts offline; IoU measures agreement, not ground truth."""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from robotwin_annotation_v2.adapters.canonical_masks import (
    CanonicalMaskBundle,
    read_canonical_masks,
)
from robotwin_annotation_v2.domain import AnnotationMode, annotation_spec

MODES = ("baseline", "resident", "async")
OBJECT_ROLES = ("target", "receiver")
type EpisodeKey = tuple[str, int, str]
type BoolMasks = NDArray[np.bool_]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value: dict[str, Any] = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


@dataclass
class Episode:
    directory: Path
    manifest: dict[str, Any]
    row: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> EpisodeKey:
        episode = self.manifest.get("episode", {})
        index = episode.get("episode_index", self.row.get("episode", self.row.get("episode_index")))
        if index is None:
            index = int(self.directory.parent.name.removeprefix("episode_"))
        return (
            str(episode.get("task", self.row.get("task", self.directory.parent.parent.name))),
            int(index),
            str(episode.get("camera", self.row.get("camera", self.directory.name))),
        )


def discover(directory: Path) -> dict[EpisodeKey, Episode]:
    if not directory.is_dir():
        raise FileNotFoundError(f"missing benchmark mode directory: {directory}")
    episodes: dict[EpisodeKey, Episode] = {}
    for row in read_json(directory / "summary.json").get("episodes", []):
        path = Path(row["artifact_dir"])
        if not path.is_absolute():
            path = directory / path
        item = Episode(path, read_json(path / "run_manifest.json"), row)
        if item.key in episodes:
            raise ValueError(f"duplicate episode identity: {directory}: {item.key}")
        episodes[item.key] = item
    artifact_dirs = {path.parent for path in directory.rglob("run_manifest.json")}
    artifact_dirs.update(path.parent for path in directory.rglob("masks.npz"))
    for path in sorted(artifact_dirs):
        item = Episode(path, read_json(path / "run_manifest.json"))
        if item.key in episodes:
            if episodes[item.key].directory.resolve() != item.directory.resolve():
                raise ValueError(f"duplicate artifacts: {directory}: {item.key}")
        else:
            episodes[item.key] = item
    return episodes


def canonical(episode: Episode | None) -> tuple[CanonicalMaskBundle | None, str | None]:
    if episode is None:
        return None, "episode_not_found"
    try:
        bundle = read_canonical_masks(episode.directory / "masks.npz")
        expected = episode.manifest.get("frame_count", episode.row.get("frame_count"))
        if expected is not None and bundle.frame_count != int(expected):
            raise ValueError("canonical masks do not cover the declared full episode")
        return bundle, None
    except (OSError, ValueError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def required_roles(episode: Episode | None, bundle: CanonicalMaskBundle | None) -> set[str]:
    roles = set() if episode is None else set(episode.manifest.get("required_object_roles", []))
    if episode is not None and not roles:
        roles.update(role["role"] for role in episode.manifest.get("roles", []))
    if bundle is not None:
        roles.update(
            role
            for i, role in enumerate(OBJECT_ROLES)
            if bundle.annotation_status[i] not in {"not_applicable", "not_annotated"}
        )
    if episode is not None and not roles:
        annotation = episode.row.get("input_identity", {}).get("annotation", {})
        mode = annotation.get("mode", episode.manifest.get("annotation_mode"))
        if mode is None:
            raise ValueError(
                f"cannot determine required roles for missing artifacts: {episode.key}"
            )
        roles.update(annotation_spec(AnnotationMode(mode)).required_role_names)
    return roles.intersection(OBJECT_ROLES)


def role_state(
    episode: Episode | None, bundle: CanonicalMaskBundle | None, role: str
) -> dict[str, Any]:
    if episode is None:
        return {"episode_missing": True}
    manifest_role: dict[str, Any] = next(
        (item for item in episode.manifest.get("roles", []) if item["role"] == role), {}
    )
    qc = read_json(episode.directory / "mask_qc.json").get("roles", {}).get(role, {})
    temporal = manifest_role.get("temporal_qc_path", f"{role}_0/temporal_qc.json")
    state = {
        "artifact_dir": str(episode.directory),
        "run_manifest": episode.manifest,
        "manifest_role": manifest_role,
        "semantic": read_json(episode.directory / "semantic_plan.json").get("roles", {}).get(role),
        "qc": {
            key: qc.get(key)
            for key in (
                "status",
                "selected_candidate",
                "selected_seed_frame_id",
                "selected_query",
                "selected_query_field",
                "reason",
            )
        },
        "qc_attempts": [
            {key: attempt.get(key) for key in ("seed_frame_id", "status", "method", "reason")}
            for attempt in qc.get("attempts", [])
        ],
        "temporal_qc": read_json(episode.directory / temporal) if temporal else {},
        "process_status": episode.row.get("process_status"),
        "objects_ready": episode.row.get("objects_ready"),
        "error": episode.row.get("error"),
    }
    if bundle is not None:
        index = OBJECT_ROLES.index(role)
        state.update(
            annotation_status=bundle.annotation_status[index],
            qc_status=bundle.qc_status[index],
            required_role_passed=(
                bundle.annotation_status[index] == "valid"
                and bundle.qc_status[index] == "passed"
                and not manifest_role.get("failure")
            ),
        )
    return state


def native(episode: Episode | None, role: str) -> tuple[BoolMasks | None, str | None]:
    if episode is None:
        return None, "episode_not_found"
    entry: dict[str, Any] = next(
        (item for item in episode.manifest.get("roles", []) if item["role"] == role), {}
    )
    relative = entry.get("native_track_path") or f"{role}_0/native_track.npz"
    try:
        with np.load(episode.directory / relative, allow_pickle=False) as archive:
            masks: BoolMasks = archive["masks"]
        if masks.dtype != bool or masks.ndim != 3 or masks.shape[0] == 0:
            raise ValueError("native masks must have bool shape [T,H,W]")
        expected = episode.manifest.get("frame_count", episode.row.get("frame_count"))
        if expected is not None and masks.shape[0] != int(expected):
            raise ValueError("native masks do not cover the declared full episode")
        return masks, None
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def agreement(
    reference: BoolMasks | None,
    actual: BoolMasks | None,
    *,
    reference_error: str | None = None,
    actual_error: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Keep missing artifacts separate from valid, mutually empty mask frames."""
    unavailable = {
        "status": "unavailable",
        "exact": False,
        "baseline_error": reference_error,
        "actual_error": actual_error,
    }
    if reference is None or actual is None:
        return unavailable, []
    if (
        reference.shape != actual.shape
        or reference.dtype != bool
        or actual.dtype != bool
        or reference.ndim != 3
        or reference.shape[0] == 0
    ):
        return {
            **unavailable,
            "status": "shape_or_dtype_mismatch",
            "baseline_shape": list(reference.shape),
            "actual_shape": list(actual.shape),
        }, []
    frames = []
    for frame, (left, right) in enumerate(zip(reference, actual, strict=True)):
        intersection = int(np.count_nonzero(left & right))
        union = int(np.count_nonzero(left | right))
        frames.append(
            {
                "frame": frame,
                "iou": intersection / union if union else 1.0,
                "intersection": intersection,
                "union": union,
                "baseline_pixels": int(np.count_nonzero(left)),
                "actual_pixels": int(np.count_nonzero(right)),
                "changed_pixels": int(np.count_nonzero(left != right)),
            }
        )
    iou = np.asarray([row["iou"] for row in frames])
    nonempty = [row["iou"] for row in frames if row["union"] > 0]
    changed = sum(row["changed_pixels"] for row in frames)
    return {
        "status": "compared",
        "exact": changed == 0,
        "frame_count": len(frames),
        "mean_iou": float(iou.mean()),
        "min_iou": float(iou.min()),
        "p05_iou": float(np.percentile(iou, 5)),
        "nonempty_mean_iou": float(np.mean(nonempty)) if nonempty else None,
        "nonempty_min_iou": min(nonempty) if nonempty else None,
        "nonempty_p05_iou": float(np.percentile(nonempty, 5)) if nonempty else None,
        "dual_empty_frames": sum(row["union"] == 0 for row in frames),
        "baseline_only_frames": sum(
            row["baseline_pixels"] > 0 and row["actual_pixels"] == 0 for row in frames
        ),
        "actual_only_frames": sum(
            row["actual_pixels"] > 0 and row["baseline_pixels"] == 0 for row in frames
        ),
        "changed_pixels": changed,
    }, frames


def decision_changes(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, tuple[Any, ...]] = {
        key: (left.get(key), right.get(key))
        for key in ("annotation_status", "qc_status", "qc", "qc_attempts", "temporal_qc")
    }
    for key in ("seed_frame_id", "primary_query", "status", "failure", "output_window"):
        values[key] = (
            left.get("manifest_role", {}).get(key),
            right.get("manifest_role", {}).get(key),
        )
    semantic_keys = (
        "seed_frame_id",
        "category_query",
        "color_category_query",
        "shape_category_query",
        "general_fallback_query",
        "recommended_order",
        "exclude",
        "status",
    )
    values["semantic_request"] = tuple(
        {key: (state.get("semantic") or {}).get(key) for key in semantic_keys}
        for state in (left, right)
    )
    return {key: {"baseline": a, "actual": b} for key, (a, b) in values.items() if a != b}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_quality(root: Path, output: Path) -> dict[str, Any]:
    indexed = {mode: discover(root / mode) for mode in MODES}
    keys = sorted(set().union(*(set(episodes) for episodes in indexed.values())))
    comparisons: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    for key in keys:
        baseline = indexed["baseline"].get(key)
        baseline_masks, baseline_error = canonical(baseline)
        for mode in MODES[1:]:
            actual = indexed[mode].get(key)
            actual_masks, actual_error = canonical(actual)
            roles = required_roles(baseline, baseline_masks) | required_roles(actual, actual_masks)
            for role in sorted(roles):
                left = role_state(baseline, baseline_masks, role)
                right = role_state(actual, actual_masks, role)
                row = {
                    "mode": mode,
                    "task": key[0],
                    "episode": key[1],
                    "camera": key[2],
                    "role": role,
                    "baseline": left,
                    "actual": right,
                    "decision_changes": decision_changes(left, right),
                }
                index = OBJECT_ROLES.index(role)
                native_left, native_error = native(baseline, role)
                native_right, actual_native_error = native(actual, role)
                layers = {
                    "canonical": (
                        None if baseline_masks is None else baseline_masks.masks[index],
                        None if actual_masks is None else actual_masks.masks[index],
                        baseline_error,
                        actual_error,
                    ),
                    "native": (native_left, native_right, native_error, actual_native_error),
                }
                for layer, (reference, masks, error, mask_error) in layers.items():
                    result, frames = agreement(
                        reference, masks, reference_error=error, actual_error=mask_error
                    )
                    row[layer] = result
                    frame_rows.extend(
                        {
                            "mode": mode,
                            "task": key[0],
                            "episode": key[1],
                            "camera": key[2],
                            "role": role,
                            "layer": layer,
                            **frame,
                            "baseline_annotation_status": left.get("annotation_status"),
                            "actual_annotation_status": right.get("annotation_status"),
                            "baseline_qc_status": left.get("qc_status"),
                            "actual_qc_status": right.get("qc_status"),
                        }
                        for frame in frames
                    )
                comparisons.append(row)
    aggregates: dict[str, Any] = {}
    for mode in MODES[1:]:
        selected = [row for row in comparisons if row["mode"] == mode]
        aggregates[mode] = {}
        for layer in ("canonical", "native"):
            values = [row[layer] for row in selected]
            frames = [r for r in frame_rows if r["mode"] == mode and r["layer"] == layer]
            nonempty = [r["iou"] for r in frames if r["union"] > 0]
            aggregates[mode][layer] = {
                "role_comparisons": len(values),
                "statuses": dict(Counter(value["status"] for value in values)),
                "all_exact": bool(values) and all(value["exact"] for value in values),
                "all_required_roles_passed": bool(selected)
                and all(
                    row["baseline"].get("required_role_passed", False)
                    and row["actual"].get("required_role_passed", False)
                    for row in selected
                ),
                "compared_role_frames": len(frames),
                "frame_mean_iou": float(np.mean([r["iou"] for r in frames])) if frames else None,
                "nonempty_frame_mean_iou": float(np.mean(nonempty)) if nonempty else None,
                "dual_empty_frames": sum(r["union"] == 0 for r in frames),
                "changed_pixels": sum(r["changed_pixels"] for r in frames),
            }
            aggregates[mode][layer]["all_successful_exact"] = (
                aggregates[mode][layer]["all_exact"]
                and aggregates[mode][layer]["all_required_roles_passed"]
            )
    result = {
        "scope": "Full-episode output agreement against baseline, not ground-truth accuracy",
        "dual_empty_policy": "IoU=1; report nonempty IoU separately; missing files are unavailable",
        "episode_counts": {mode: len(items) for mode, items in indexed.items()},
        "summary": aggregates,
        "comparisons": comparisons,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "quality.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    write_csv(output / "per_frame_iou.csv", frame_rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize_quality(args.root, args.output or args.root / "analysis")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
