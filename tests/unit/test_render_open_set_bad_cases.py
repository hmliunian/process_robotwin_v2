from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import scripts.render_open_set_bad_cases as render_bad_cases
from scripts.render_coverage20_videos import MaskArtifact


def _case(tmp_path: Path, run: str, task: str, episode_id: int) -> tuple[Path, Path]:
    episode_dir = tmp_path / "runs" / run / task / f"episode_{episode_id:06d}" / "cam_high"
    episode_dir.mkdir(parents=True)
    (episode_dir / "masks.npz").write_bytes(b"masks")
    (episode_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_id": run,
                "episode": {
                    "task": task,
                    "episode_index": episode_id,
                    "camera": "cam_high",
                },
                "roles": [],
            }
        ),
        encoding="utf-8",
    )
    video_path = tmp_path / "videos" / task / f"episode_{episode_id:06d}.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(b"video")
    return episode_dir, video_path


def _input_manifest(tmp_path: Path, cases: list[dict[str, Any]]) -> Path:
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps(
            {
                "format_version": render_bad_cases.INPUT_FORMAT,
                "cases": cases,
            }
        ),
        encoding="utf-8",
    )
    return path


def _artifact(statuses: tuple[str, str]) -> MaskArtifact:
    return MaskArtifact(
        masks=np.zeros((2, 1, 2, 3), dtype=bool),
        instance_names=("target_0", "receiver_0"),
        roles=("target", "receiver"),
        annotation_status=statuses,
        frame_count=1,
        format_version="robotwin_visible_masks_v3",
        qc_status=("passed", "passed"),
        frame_encoding=np.zeros((2, 1), dtype=np.uint8),
    )


def test_render_cases_uses_only_explicit_cross_task_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_dir, first_video = _case(tmp_path, "run-a", "place_fan", 16500)
    second_dir, second_video = _case(tmp_path, "run-b", "place_phone_stand", 19545)
    input_manifest = _input_manifest(
        tmp_path,
        [
            {
                "group": "manual_review",
                "stage": "S3",
                "episode_dir": str(first_dir),
                "video_path": str(first_video),
            },
            {
                "group": "unresolved",
                "stage": "S3",
                "episode_dir": str(second_dir),
                "video_path": str(second_video),
            },
        ],
    )
    loaded: list[Path] = []
    rendered: list[tuple[Path, Path]] = []

    def fake_load(path: Path) -> MaskArtifact:
        loaded.append(path)
        return _artifact(("valid", "valid") if "run-a" in path.parts else ("valid", "quarantined"))

    def fake_render(
        video_path: Path,
        _artifact_value: MaskArtifact,
        output_path: Path,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        rendered.append((video_path, output_path))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(f"rendered:{video_path}".encode())
        return {"frame_count": 11, "frame_rate": "50/1"}

    monkeypatch.setattr(render_bad_cases.renderer, "_load_masks", fake_load)
    monkeypatch.setattr(render_bad_cases.renderer, "render_video", fake_render)

    output_manifest = render_bad_cases.render_cases(input_manifest, tmp_path / "output")
    payload = json.loads(output_manifest.read_text(encoding="utf-8"))

    assert loaded == [first_dir / "masks.npz", second_dir / "masks.npz"]
    assert [item[0] for item in rendered] == [first_video, second_video]
    assert [record["source_run"] for record in payload["episodes"]] == ["run-a", "run-b"]
    assert [record["episode_id"] for record in payload["episodes"]] == ["016500", "019545"]
    assert payload["group_counts"] == {"manual_review": 1, "unresolved": 1}
    assert payload["publication_status_counts"] == {"completed": 1, "quarantined": 1}
    assert payload["episodes"][0]["output_video"].startswith("manual_review/")
    assert payload["episodes"][1]["output_video"].startswith("unresolved/")
    for record in payload["episodes"]:
        assert len(record["source_masks_sha256"]) == 64
        assert len(record["source_video_sha256"]) == 64
        assert len(record["source_run_manifest_sha256"]) == 64
        assert len(record["output_video_sha256"]) == 64
        assert record["annotation_status"]
        assert record["qc_status"]


def test_manifest_rejects_s4_without_searching_sources(tmp_path: Path) -> None:
    input_manifest = _input_manifest(
        tmp_path,
        [
            {
                "group": "unresolved",
                "stage": "S4",
                "episode_dir": "missing/run/task/episode_000001/cam_high",
                "video_path": "missing.mp4",
            }
        ],
    )

    with pytest.raises(ValueError, match="stage must be one of"):
        render_bad_cases.load_cases(input_manifest)


def test_manifest_rejects_duplicate_episode_sources(tmp_path: Path) -> None:
    episode_dir, video_path = _case(tmp_path, "run-a", "place_fan", 16500)
    record = {
        "group": "manual_review",
        "stage": "S3",
        "episode_dir": str(episode_dir),
        "video_path": str(video_path),
    }
    input_manifest = _input_manifest(tmp_path, [record, record])

    with pytest.raises(ValueError, match="duplicate task/episode/camera"):
        render_bad_cases.load_cases(input_manifest)


def test_run_manifest_identity_must_match_pinned_path(tmp_path: Path) -> None:
    episode_dir, video_path = _case(tmp_path, "run-a", "place_fan", 16500)
    (episode_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_id": "different-run",
                "episode": {
                    "task": "place_fan",
                    "episode_index": 16500,
                    "camera": "cam_high",
                },
            }
        ),
        encoding="utf-8",
    )
    manifest = _input_manifest(
        tmp_path,
        [
            {
                "group": "manual_review",
                "stage": "S3",
                "episode_dir": str(episode_dir),
                "video_path": str(video_path),
            }
        ],
    )

    with pytest.raises(ValueError, match="identity mismatch"):
        render_bad_cases._run_manifest(render_bad_cases.load_cases(manifest)[0])
