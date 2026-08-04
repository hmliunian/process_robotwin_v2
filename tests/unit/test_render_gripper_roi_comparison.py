from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from scripts.render_gripper_roi_comparison import (
    CANVAS_WIDTH,
    CONTACT_HEIGHT,
    GLOBAL_HEADER_HEIGHT,
    RUN_GAP,
    RUN_HEADER_HEIGHT,
    SEED_STRIP_HEIGHT,
    RunSpec,
    build_run_specs,
    load_episode_visual,
    parse_variants,
    render_comparisons,
)


EPISODE = 7152


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _manifest(*, explicit_roi: bool) -> dict[str, object]:
    manifest: dict[str, object] = {
        "status": "review_required",
        "seed": {
            "selected_candidate": "D",
            "frame": 93,
            "clean_pixels": 1218,
            "candidate_artifacts": {
                "panels": {"D": "seed_candidates/candidate_D.png"}
            },
        },
    }
    if explicit_roi:
        manifest["roi_policy"] = {
            "prompt": {
                "geometry": {"axial_back_m": 0.080, "axial_front_m": 0.060}
            },
            "hard": {
                "geometry": {"axial_back_m": 0.080, "axial_front_m": 0.045}
            },
        }
    return manifest


def _write_episode(root: Path, *, explicit_roi: bool, color: tuple[int, int, int]) -> None:
    episode_dir = root / f"episode_{EPISODE:06d}"
    _write_json(episode_dir / "manifest.json", _manifest(explicit_roi=explicit_roi))
    _write_json(root / "batch_manifest.json", {"status": "completed"})
    Image.new("RGB", (1280, 720), color).save(
        episode_dir / "episode_contact_sheet.jpg",
        quality=100,
        subsampling=0,
    )
    seed_path = episode_dir / "seed_candidates" / "candidate_D.png"
    seed_path.parent.mkdir(parents=True)
    Image.new("RGB", (320, 240), (20, 200, 100)).save(seed_path)
    (episode_dir / "episode_gripper_review.mp4").write_bytes(b"synthetic")


def test_load_visual_labels_legacy_and_split_roi_parameters(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    variant = tmp_path / "split"
    _write_episode(baseline, explicit_roi=False, color=(200, 30, 30))
    _write_episode(variant, explicit_roi=True, color=(30, 30, 200))

    baseline_visual = load_episode_visual(
        RunSpec(label="A", root=baseline, is_baseline=True),
        EPISODE,
    )
    split_visual = load_episode_visual(RunSpec(label="S", root=variant), EPISODE)

    assert baseline_visual.status == "complete"
    assert "prompt back/front=0.025/0.060 m" in baseline_visual.roi_label
    assert "hard back/front=0.025/0.060 m" in baseline_visual.roi_label
    assert "selected seed=D" in baseline_visual.seed_label
    assert "clean=1,218 px" in baseline_visual.seed_label
    assert split_visual.status == "complete"
    assert "prompt back/front=0.080/0.060 m" in split_visual.roi_label
    assert "hard back/front=0.080/0.045 m" in split_visual.roi_label


def test_render_comparison_preserves_1280_contact_and_marks_missing_variant(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline"
    complete = tmp_path / "complete"
    missing = tmp_path / "missing"
    output = tmp_path / "rendered"
    _write_episode(baseline, explicit_roi=False, color=(200, 30, 30))
    _write_episode(complete, explicit_roi=True, color=(30, 30, 200))
    flagged = _manifest(explicit_roi=True)
    seed = flagged["seed"]
    assert isinstance(seed, dict)
    seed.update(
        {
            "prompt_mode": "box_only",
            "selection_source": "forced_fallback",
            "qc": {"forced_fallback": True},
        }
    )
    _write_json(complete / f"episode_{EPISODE:06d}" / "manifest.json", flagged)

    overviews, index = render_comparisons(
        baseline,
        {"C": complete, "S": missing},
        [EPISODE],
        output,
    )

    expected_height = (
        GLOBAL_HEADER_HEIGHT
        + 3 * (RUN_HEADER_HEIGHT + CONTACT_HEIGHT + SEED_STRIP_HEIGHT)
        + 2 * RUN_GAP
    )
    with Image.open(overviews[0]) as image:
        assert image.size == (CANVAS_WIDTH, expected_height)
        contact_y = GLOBAL_HEADER_HEIGHT + RUN_HEADER_HEIGHT + CONTACT_HEIGHT // 2
        red, green, blue = image.convert("RGB").getpixel((CANVAS_WIDTH // 2, contact_y))
        assert red > 180
        assert green < 50
        assert blue < 50

    markdown = index.read_text(encoding="utf-8")
    assert "episode_007152_roi_comparison.jpg" in markdown
    assert "episode_gripper_review.mp4" in markdown
    assert "episode_contact_sheet.jpg" in markdown
    assert "manifest.json" in markdown
    assert "seed_candidates/candidate_D.png" in markdown
    assert "**missing**" in markdown
    assert "QC FLAG: forced_fallback/box_only" in markdown


def test_parse_variants_groups_duplicate_labels_as_ordered_shards(tmp_path: Path) -> None:
    shard_a = tmp_path / "shard_a"
    shard_b = tmp_path / "shard_b"
    assert parse_variants([f"F12={shard_a}", f"F12={shard_b}"]) == {
        "F12": (shard_a.resolve(), shard_b.resolve())
    }
    with pytest.raises(ValueError, match="reserved"):
        parse_variants([f"A={tmp_path}"])


def test_load_visual_finds_episode_in_second_variant_shard(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    shard_a = tmp_path / "shard_a"
    shard_b = tmp_path / "shard_b"
    _write_episode(shard_b, explicit_roi=True, color=(30, 30, 200))

    runs = build_run_specs(baseline, {"F12": (shard_a, shard_b)})
    visual = load_episode_visual(runs[1], EPISODE)

    assert visual.status == "complete"
    assert visual.episode_dir.parent == shard_b.resolve()
