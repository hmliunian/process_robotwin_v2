from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from robotwin_annotation_v2.adapters.canonical_masks import build_canonical_mask_bundle
from scripts.summarize_umi_pipeline_benchmark import agreement, summarize_quality


def write_episode(
    root: Path, mode: str, native_masks: NDArray[np.bool_], *, canonical_exists: bool = True
) -> None:
    directory = root / mode / "runs/task/episode_000007/cam_high"
    directory.mkdir(parents=True)
    manifest = {
        "episode": {"task": "task", "episode_index": 7, "camera": "cam_high"},
        "frame_count": len(native_masks),
        "required_object_roles": ["target"],
        "roles": [
            {
                "role": "target",
                "status": "quarantined",
                "failure": "temporal_qc_quarantine",
                "native_track_path": "target_0/native_track.npz",
            }
        ],
    }
    (directory / "run_manifest.json").write_text(json.dumps(manifest))
    (directory / "target_0").mkdir()
    np.savez_compressed(directory / "target_0/native_track.npz", masks=native_masks)
    if canonical_exists:
        masks = np.zeros((4, *native_masks.shape), dtype=bool)
        bundle = build_canonical_mask_bundle(
            directory / "masks.npz",
            frame_count=len(native_masks),
            masks=masks,
            annotation_status=["quarantined", "not_applicable", "not_annotated", "not_annotated"],
            qc_status=["passed", "not_applicable", "not_run", "not_run"],
            frame_encoding=np.zeros((4, len(native_masks)), dtype=np.uint8),
        )
        np.savez_compressed(bundle.path, **bundle.to_payload())


def test_quality_keeps_missing_required_masks_and_quarantined_tracks_visible(
    tmp_path: Path,
) -> None:
    native = np.ones((3, 2, 2), dtype=bool)
    write_episode(tmp_path, "baseline", native)
    write_episode(tmp_path, "resident", native)
    write_episode(tmp_path, "async", native, canonical_exists=False)
    output = tmp_path / "analysis"
    result = summarize_quality(tmp_path, output)
    resident, asynchronous = result["comparisons"]
    assert resident["role"] == asynchronous["role"] == "target"
    assert resident["canonical"]["exact"] is True
    assert resident["canonical"]["dual_empty_frames"] == 3
    assert resident["canonical"]["nonempty_mean_iou"] is None
    assert resident["native"]["nonempty_mean_iou"] == 1
    assert resident["native"]["dual_empty_frames"] == 0
    assert result["summary"]["resident"]["canonical"]["all_successful_exact"] is False
    assert asynchronous["canonical"]["status"] == "unavailable"
    assert asynchronous["canonical"]["exact"] is False
    assert asynchronous["native"]["exact"] is True
    assert result["summary"]["async"]["canonical"]["all_exact"] is False
    with (output / "per_frame_iou.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 9  # No fabricated canonical frames, receiver or gripper comparisons.
    assert {row["role"] for row in rows} == {"target"}
    assert json.loads((output / "quality.json").read_text())["episode_counts"] == {
        "baseline": 1,
        "resident": 1,
        "async": 1,
    }


def test_agreement_does_not_hide_one_sided_empty_frames_in_nonempty_iou() -> None:
    baseline = np.zeros((3, 2, 2), dtype=bool)
    actual = baseline.copy()
    baseline[0, 0, 0] = True
    actual[1, 0, 0] = True
    result, frames = agreement(baseline, actual)
    assert [frame["iou"] for frame in frames] == [0, 0, 1]
    assert result["mean_iou"] == pytest.approx(1 / 3)
    assert result["nonempty_mean_iou"] == 0
    assert result["dual_empty_frames"] == 1
    assert result["baseline_only_frames"] == result["actual_only_frames"] == 1
    assert result["changed_pixels"] == 2 and result["exact"] is False
    assert agreement(baseline, actual[:2])[0]["status"] == "shape_or_dtype_mismatch"


def test_equal_truncated_outputs_are_not_reported_as_full_episode_agreement(tmp_path: Path) -> None:
    native = np.ones((3, 2, 2), dtype=bool)
    for mode in ("baseline", "resident", "async"):
        write_episode(tmp_path, mode, native)
        path = tmp_path / mode / "runs/task/episode_000007/cam_high/run_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["frame_count"] = 10
        path.write_text(json.dumps(manifest))
    result = summarize_quality(tmp_path, tmp_path / "analysis")
    for row in result["comparisons"]:
        assert row["canonical"]["exact"] is False
        assert row["native"]["exact"] is False
        assert "full episode" in row["native"]["baseline_error"]


def test_unstarted_jobs_without_any_artifacts_remain_required_failures(tmp_path: Path) -> None:
    for mode in ("baseline", "resident", "async"):
        directory = tmp_path / mode
        directory.mkdir()
        (directory / "summary.json").write_text(
            json.dumps(
                {
                    "episodes": [
                        {
                            "task": "task",
                            "episode": 7,
                            "camera": "cam_high",
                            "artifact_dir": "runs/task/episode_000007/cam_high",
                            "objects_ready": False,
                            "process_status": "not_run_after_benchmark_failure",
                            "input_identity": {"annotation": {"mode": "tool_use"}},
                        }
                    ]
                }
            )
        )
    result = summarize_quality(tmp_path, tmp_path / "analysis")
    assert len(result["comparisons"]) == 4
    assert {row["role"] for row in result["comparisons"]} == {"target", "receiver"}
    assert all(row["canonical"]["exact"] is False for row in result["comparisons"])
    assert result["summary"]["resident"]["canonical"]["statuses"] == {"unavailable": 2}
