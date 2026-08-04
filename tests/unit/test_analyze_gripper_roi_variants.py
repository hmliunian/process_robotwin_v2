from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.analyze_gripper_roi_variants import (
    build_report,
    parse_variants,
    render_markdown,
)


EPISODE = 7152


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_batch(root: Path, *, failed_episode: int | None = 7156) -> None:
    failures = []
    episode_ids = [EPISODE]
    if failed_episode is not None:
        episode_ids.append(failed_episode)
        failures.append(
            {
                "episode": failed_episode,
                "status": "failed",
                "error": "synthetic inference failure",
            }
        )
    _write_json(
        root / "batch_manifest.json",
        {
            "status": "completed_with_failures" if failures else "completed",
            "episode_ids": episode_ids,
            "episodes": [{"episode": EPISODE, "status": "review_required"}],
            "failures": failures,
        },
    )


def _episode_manifest(*, raw: int, cropped: int, clean: int) -> dict[str, object]:
    return {
        "status": "review_required",
        "episode": {"episode_index": EPISODE, "active_window": [0, 2]},
        "seed": {
            "selected_candidate": "B",
            "frame": 1,
            "prompt_mode": "text_box",
            "clean_pixels": clean,
            "qc": {
                "selected_candidate": "B",
                "candidates": [
                    {
                        "candidate_id": "A",
                        "raw_pixels": 999,
                        "cropped_pixels": 999,
                        "clean_pixels": 999,
                    },
                    {
                        "candidate_id": "B",
                        "raw_pixels": raw,
                        "cropped_pixels": cropped,
                        "clean_pixels": clean,
                    },
                ],
            },
        },
    }


def _base_tracks() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    final = np.zeros((3, 2, 3), dtype=bool)
    final[0, 0, 0] = True
    final[1, 0, 0] = True
    native = np.ones_like(final)
    legacy_roi = np.zeros_like(final)
    legacy_roi[:, 0, 0] = True
    return final, native, legacy_roi


def _write_baseline(root: Path) -> None:
    _write_batch(root)
    episode_dir = root / f"episode_{EPISODE:06d}"
    _write_json(episode_dir / "manifest.json", _episode_manifest(raw=10, cropped=7, clean=2))
    final, native, legacy_roi = _base_tracks()
    target_removed = np.zeros_like(final)
    target_removed[0, 1, 0] = True
    receiver_removed = np.zeros_like(final)
    receiver_removed[1, 1, 1] = True
    np.savez_compressed(
        episode_dir / "gripper_masks.npz",
        active_window=np.asarray([0, 2]),
        seed_mask=np.asarray([[True, True, False]]),
        native_track=native,
        roi_track=legacy_roi,
        target_removed=target_removed,
        receiver_removed=receiver_removed,
        gripper_track=final,
    )


def _write_variant(root: Path) -> None:
    _write_batch(root)
    episode_dir = root / f"episode_{EPISODE:06d}"
    _write_json(episode_dir / "manifest.json", _episode_manifest(raw=12, cropped=9, clean=4))
    baseline_final, _, baseline_hard = _base_tracks()
    final = np.zeros_like(baseline_final)
    final[0, 0, 0:2] = True
    final[1, 0, 1] = True
    final[2, 0, 1] = True
    hard_roi = baseline_hard.copy()
    hard_roi[:, 0, 1] = True
    prompt_roi = hard_roi.copy()
    prompt_roi[:, 1, 0] = True
    native = hard_roi.copy()
    native[:, 0, 2] = True
    target_removed = np.zeros_like(final)
    target_removed[0, 1, 0:2] = True
    receiver_removed = np.zeros_like(final)
    receiver_removed[1:, 1, 2] = True
    np.savez_compressed(
        episode_dir / "gripper_masks.npz",
        active_window=np.asarray([0, 2]),
        seed_mask=np.asarray([[True, True, True, True]]),
        native_track=native,
        prompt_roi_track=prompt_roi,
        hard_roi_track=hard_roi,
        roi_track=hard_roi,
        target_removed=target_removed,
        receiver_removed=receiver_removed,
        gripper_track=final,
    )


def test_report_compares_variant_and_uses_legacy_roi_fallback(tmp_path: Path) -> None:
    baseline_root = tmp_path / "baseline"
    variant_root = tmp_path / "back080"
    _write_baseline(baseline_root)
    _write_variant(variant_root)

    report = build_report(baseline_root, {"back080": variant_root})

    baseline = report["baseline"]
    baseline_episode = baseline["episodes"][str(EPISODE)]
    assert baseline["summary"]["completion"] == {
        "requested_episodes": 2,
        "batch_completed_records": 1,
        "batch_failure_records": 1,
        "analyzed_episodes": 1,
        "failed_or_unavailable_episodes": 1,
        "failed_or_unavailable_episode_ids": [7156],
    }
    assert baseline_episode["compatibility"]["legacy_prompt_roi_fallback"] is True
    assert baseline_episode["compatibility"]["legacy_hard_roi_fallback"] is True
    assert baseline_episode["archive_keys"]["prompt_roi"] == "roi_track"
    assert baseline_episode["archive_keys"]["hard_roi"] == "roi_track"
    assert baseline_episode["seed"]["raw_pixels"] == 10
    assert baseline_episode["seed"]["cropped_pixels"] == 7
    assert baseline_episode["seed"]["clean_pixels"] == 2

    variant = report["variants"]["back080"]
    episode = variant["episodes"][str(EPISODE)]
    comparison = episode["comparison_to_baseline"]
    assert episode["final"]["nonempty_frames"] == 3
    assert episode["final"]["pixel_frame_sum"] == 4
    assert episode["final"]["area_mean_all_frames"] == pytest.approx(4 / 3)
    assert episode["final"]["adjacent_iou_mean"] == pytest.approx(0.75)
    assert episode["native_clipped_by_hard_roi"]["pixel_frame_sum"] == 3
    assert episode["target_removed"]["pixel_frame_sum"] == 2
    assert episode["receiver_removed"]["pixel_frame_sum"] == 2
    assert comparison["gained_pixel_frames"] == 3
    assert comparison["lost_pixel_frames"] == 1
    assert comparison["new_hard_roi_band"] == {
        "pixel_frame_sum": 3,
        "gained_pixel_frames_in_band": 3,
        "fraction_of_gained_pixels": 1.0,
    }
    assert variant["comparison_to_baseline"]["episodes_compared"] == 1
    assert variant["comparison_to_baseline"]["episodes_unavailable"] == 1
    assert variant["comparison_to_baseline"]["net_pixel_frames"] == 2

    markdown = render_markdown(report)
    assert "# Gripper ROI variant analysis" in markdown
    assert "| back080 | 1 | 3 | 1 | 2 | 3 | 1.000 |" in markdown
    assert f"| {EPISODE} | completed | 12/9/4 |" in markdown
    assert "| 7156 | failed | n/a |" in markdown


def test_shape_mismatch_is_reported_without_aborting_other_metrics(tmp_path: Path) -> None:
    baseline_root = tmp_path / "baseline"
    variant_root = tmp_path / "variant"
    _write_baseline(baseline_root)
    _write_variant(variant_root)
    _write_batch(variant_root, failed_episode=None)
    archive_path = variant_root / f"episode_{EPISODE:06d}" / "gripper_masks.npz"
    with np.load(archive_path, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    for key, value in tuple(values.items()):
        if value.ndim == 3:
            values[key] = np.pad(value, ((0, 0), (0, 0), (0, 1)))
    np.savez_compressed(archive_path, **values)

    report = build_report(baseline_root, {"wider": variant_root})

    variant = report["variants"]["wider"]
    comparison = variant["episodes"][str(EPISODE)]["comparison_to_baseline"]
    assert comparison["available"] is False
    assert "shape mismatch" in comparison["reason"]
    assert variant["comparison_to_baseline"]["episodes_unavailable"] == 1
    assert variant["summary"]["final"]["pixel_frame_sum"] == 4


def test_parse_variants_requires_unique_non_reserved_labels(tmp_path: Path) -> None:
    parsed = parse_variants([f"wide={tmp_path}"])
    assert parsed == {"wide": tmp_path.resolve()}

    with pytest.raises(ValueError, match="duplicate"):
        parse_variants([f"wide={tmp_path}", f"wide={tmp_path}"])
    with pytest.raises(ValueError, match="reserved"):
        parse_variants([f"baseline={tmp_path}"])
    with pytest.raises(ValueError, match="at least one"):
        parse_variants([])
