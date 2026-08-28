from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from scripts.generate_render_report import (
    DEFAULT_TEMPLATE,
    build_overall_index,
    collect_report,
    render_html,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_collect_report_keeps_identity_and_temporal_qc_separate(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-a"
    render_dir = run_dir / "rendered_videos"
    dataset_root = tmp_path / "dataset"
    render_dir.mkdir(parents=True)
    (render_dir / "episode_000000.mp4").touch()
    metadata_path = dataset_root / "meta" / "episodes.jsonl"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "episode_index": 0,
                        "tasks": ["Move the cup to the tray."],
                        "task_text_zh": "把杯子移到托盘。",
                        "length": 12,
                    }
                ),
                json.dumps(
                    {
                        "episode_index": 1,
                        "tasks": ["Move the bowl to the table."],
                        "length": 10,
                    }
                ),
            )
        ),
        encoding="utf-8",
    )
    _write_json(
        render_dir / "manifest.json",
        {
            "format": "render-v1",
            "requested_run_id": "run-a",
            "dataset_root": str(dataset_root),
            "task": "task-a",
            "camera": "cam_high",
            "episodes": [
                {
                    "episode_index": 0,
                    "task_text": "Move the cup to the tray.",
                    "source_masks": str(
                        run_dir / "task-a" / "episode_000000" / "cam_high" / "masks.npz"
                    ),
                    "output_video": "episode_000000.mp4",
                    "annotation_status": {
                        "target_0": "valid",
                        "receiver_0": "quarantined",
                    },
                    "qc_status": {"target_0": "passed", "receiver_0": "passed"},
                    "nonempty_frames": {"target_0": 8, "receiver_0": 0},
                    "duration_seconds": 1.2,
                    "frame_count": 12,
                    "width": 640,
                    "height": 352,
                    "frame_rate": "10/1",
                }
            ],
        },
    )
    _write_json(
        run_dir / "process_summary.json",
        {
            "run_id": "run-a",
            "dataset_root": str(dataset_root),
            "task": "task-a",
            "camera": "cam_high",
            "requested_episode_ids": [0, 1],
            "records": [
                {"episode": 0, "status": "sam_incomplete"},
                {"episode": 1, "status": "failed", "error": "stage failed"},
            ],
            "backend": {"object_masks": "sam", "gripper": None},
            "qwen_health": {"model": "qwen-test"},
        },
    )
    episode_dir = run_dir / "task-a" / "episode_000000" / "cam_high"
    _write_json(
        episode_dir / "run_manifest.json",
        {
            "roles": [
                {"role": "target", "status": "ok", "qc_status": "passed"},
                {
                    "role": "receiver",
                    "status": "quarantined",
                    "qc_status": "passed",
                    "failure": "temporal_qc_quarantine",
                },
            ]
        },
    )
    _write_json(
        episode_dir / "mask_qc.json",
        {
            "roles": {
                "target": {
                    "status": "passed",
                    "selected_query_field": "category_query",
                    "attempts": [
                        {
                            "method": "text_query",
                            "seed_frame_id": 0,
                            "status": "passed",
                            "selected_query_field": "category_query",
                        }
                    ],
                },
                "receiver": {
                    "status": "passed",
                    "selected_query_field": "bbox_fallback",
                    "attempts": [
                        {"method": "text_query", "seed_frame_id": 0, "status": "rejected"},
                        {
                            "method": "bbox_fallback",
                            "seed_frame_id": 0,
                            "status": "passed",
                        },
                    ],
                },
            }
        },
    )
    _write_json(
        episode_dir / "semantic_plan.json",
        {"prompt_version": "object_roles_semantic_v2"},
    )

    report = collect_report(render_dir)

    assert report["stats"]["inputEpisodes"] == 2
    assert report["stats"]["renderedEpisodes"] == 1
    assert report["stats"]["roleCount"] == 2
    assert report["stats"]["qcPassedRoles"] == 2
    assert report["stats"]["validRoles"] == 1
    assert report["stats"]["s3AttemptedRoles"] == 1
    assert report["stats"]["s3PassedRoles"] == 1
    assert report["stats"]["inputTasks"] == 2
    assert [item["status"] for item in report["anomalies"]] == ["quarantined", "missing"]


def test_build_overall_index_and_render_html(tmp_path: Path) -> None:
    sheets = []
    for index, name in enumerate(
        ("target_early", "target_late", "receiver_early", "receiver_late")
    ):
        path = tmp_path / f"{name}.jpg"
        Image.new("RGB", (320, 240), (20 + index * 20, 40, 70)).save(path)
        sheets.append(path)
    report = {
        "stats": {
            "renderedEpisodes": 1,
            "inputEpisodes": 2,
            "processCompleted": 1,
            "validRoles": 2,
            "roleCount": 2,
        }
    }
    output = build_overall_index(sheets, tmp_path / "overall_index.jpg", report)

    assert output.is_file()
    with Image.open(output) as image:
        assert image.width == 2560
        assert image.height > 1000

    template = tmp_path / "template.html"
    template.write_text("<script>__REPORT_JSON__</script>", encoding="utf-8")
    html_path = render_html({"text": "safe </script> payload"}, template, tmp_path / "report.html")
    html = html_path.read_text(encoding="utf-8")
    assert "<\\/script>" in html
    assert html.count("__REPORT_JSON__") == 0


def test_default_template_contains_all_report_sections() -> None:
    template = DEFAULT_TEMPLATE.read_text(encoding="utf-8")

    assert template.count("__REPORT_JSON__") == 1
    for section_id in (
        "overview",
        "scope",
        "pipeline",
        "open-set",
        "review",
        "dashboard",
        "tasks",
        "anomalies",
        "videos",
        "lineage",
        "appendix",
    ):
        assert f'id="{section_id}"' in template
