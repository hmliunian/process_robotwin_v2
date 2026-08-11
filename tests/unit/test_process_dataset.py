from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

import scripts.process_dataset as process_module
import scripts.render_urdf_gripper_masks as urdf_module
from scripts.process_dataset import (
    DiscoveredEpisode,
    build_dynamic_manifest,
    discover_episodes,
    select_urdf_source_episodes,
)


def _touch_episode(
    root: Path,
    episode_id: int,
    *,
    camera: str = "cam_high",
    sidecar: bool = True,
    video: bool = True,
    depth: bool = True,
    frame_count: int = 6,
) -> None:
    chunk = f"chunk-{episode_id // 1000:03d}"
    parquet = root / "data" / chunk / f"episode_{episode_id:06d}.parquet"
    parquet.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"frame_index": np.arange(frame_count, dtype=np.int64)}).to_parquet(
        parquet,
        index=False,
    )
    if video:
        video_path = (
            root
            / "videos"
            / chunk
            / f"observation.images.{camera}"
            / f"episode_{episode_id:06d}.mp4"
        )
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.touch()
    if sidecar:
        sidecar_path = root / "sidecars" / f"episode_{episode_id:06d}.hdf5"
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_path.touch()
    if depth:
        depth_path = (
            root
            / "sidecars"
            / "videos"
            / chunk
            / f"observation.depths.{camera}"
            / f"episode_{episode_id:06d}.mkv"
        )
        depth_path.parent.mkdir(parents=True, exist_ok=True)
        depth_path.touch()


def _write_source_summary(
    source_run: Path,
    dataset_root: Path,
    records: list[dict[str, Any]],
    *,
    task: str = "task",
    camera: str = "cam_high",
) -> None:
    source_run.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": "robotwin_process_dataset_summary_v1",
        "run_id": source_run.name,
        "dataset_root": str(dataset_root.resolve()),
        "task": task,
        "camera": camera,
        "dynamic_manifest": {
            "format_version": "robotwin_dataset_manifest_dynamic_v1",
            "task": task,
            "camera": camera,
            "frame_shape_hw": [2, 3],
            "raw_video_frame_surplus": 0,
            "usable_frame_count_source": "parquet",
            "dataset_root": str(dataset_root.resolve()),
            "smoke_episode_ids": [int(records[0]["episode"])],
            "regression_episode_ids": [
                int(record["episode"]) for record in records
            ],
            "required_relative_files": [],
        },
        "records": records,
    }
    (source_run / "process_summary.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def _write_source_episode(
    source_run: Path,
    episode_id: int,
    *,
    task: str = "task",
    camera: str = "cam_high",
    annotation_status: tuple[str, ...] = (
        "valid",
        "valid",
        "not_annotated",
        "valid",
    ),
    qc_status: tuple[str, ...] = (
        "passed",
        "passed",
        "not_run",
        "rejected",
    ),
    include_format_version: bool = True,
    mask_shape: tuple[int, ...] = (4, 6, 2, 3),
    stored_frame_count: int = 6,
    target_window: tuple[int, int] | None = (0, 2),
) -> None:
    episode_dir = source_run / task / f"episode_{episode_id:06d}" / camera
    episode_dir.mkdir(parents=True, exist_ok=True)
    mask_payload = {
        "frame_count": np.asarray(stored_frame_count, dtype=np.int64),
        "masks": np.zeros(mask_shape, dtype=bool),
        "instance_names": np.asarray(
            ("target_0", "receiver_0", "gripper_left", "gripper_right")
        ),
        "roles": np.asarray(("target", "receiver", "gripper", "gripper")),
        "annotation_status": np.asarray(annotation_status),
        "qc_status": np.asarray(qc_status),
    }
    if include_format_version:
        mask_payload["format_version"] = np.asarray("robotwin_visible_masks_v2")
    np.savez_compressed(episode_dir / "masks.npz", **mask_payload)
    summary = json.loads(
        (source_run / "process_summary.json").read_text(encoding="utf-8")
    )
    dataset_root = Path(summary["dataset_root"])
    chunk = f"chunk-{episode_id // 1000:03d}"
    loop = {
        "format_version": "robotwin_loop_context_v1",
        "episode": {
            "task": task,
            "episode_index": episode_id,
            "episode_id": f"{episode_id:06d}",
            "camera": camera,
        },
        "task_text": "test",
        "frame_count": stored_frame_count,
        "events": {
            "active_arm": "right",
            "t_move_start": 0,
            "t_close_start": 1,
            "t_close_done": 2,
            "t_open_start": 4,
            "t_open_done": 5,
        },
        "windows": {
            "loop": [0, 5],
            "target_0": [0, 2],
            "receiver_0": [2, 5],
        },
        "semantic_frames": [],
        "sources": {
            "state": str(
                dataset_root
                / "data"
                / chunk
                / f"episode_{episode_id:06d}.parquet"
            ),
            "video": str(
                dataset_root
                / "videos"
                / chunk
                / f"observation.images.{camera}"
                / f"episode_{episode_id:06d}.mp4"
            ),
        },
    }
    (episode_dir / "loop.json").write_text(json.dumps(loop), encoding="utf-8")
    for instance_name in ("target_0", "receiver_0"):
        artifact = episode_dir / instance_name / "native_track.npz"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(f"{instance_name}-{episode_id}".encode())
    target_role: dict[str, Any] = {
        "role": "target",
        "status": "ok",
        "qc_status": "passed",
    }
    if target_window is not None:
        target_role["output_window"] = list(target_window)
    manifest = {
        "format_version": "robotwin_mask_run_v2",
        "run_id": source_run.name,
        "episode": {
            "task": task,
            "episode_index": episode_id,
            "episode_id": f"{episode_id:06d}",
            "camera": camera,
        },
        "frame_count": stored_frame_count,
        "roles": [
            {
                **target_role,
                "native_track_path": "target_0/native_track.npz",
            },
            {
                "role": "receiver",
                "status": "ok",
                "qc_status": "passed",
                "output_window": [2, 5],
                "native_track_path": "receiver_0/native_track.npz",
            },
        ],
        "algorithm": {"source": "sam-test"},
    }
    (episode_dir / "run_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    provenance = {
        "format_version": "robotwin_frame_provenance_v2",
        "channels": {
            "target_0": {
                "status": "ok",
                "qc_status": "passed",
                "output_window": [0, 2],
            },
            "receiver_0": {
                "status": "ok",
                "qc_status": "passed",
                "output_window": [2, 5],
            },
        },
    }
    (episode_dir / "frame_provenance.json").write_text(
        json.dumps(provenance), encoding="utf-8"
    )


def _source_lineage(
    source_run: Path,
    episode_id: int,
    *,
    task: str = "task",
    camera: str = "cam_high",
) -> dict[str, Any]:
    summary = json.loads(
        (source_run / "process_summary.json").read_text(encoding="utf-8")
    )
    validated = process_module.validate_derivation_source_episode(
        source_run / task / f"episode_{episode_id:06d}" / camera,
        task=task,
        camera=camera,
        episode_index=episode_id,
        expected_dataset_root=Path(summary["dataset_root"]),
    )
    return dict(validated.lineage)


def _cli_config(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        dataset=SimpleNamespace(
            root=tmp_path / "configured-dataset",
            task="configured-task",
            camera="cam_high",
        )
    )


def test_discover_episodes_cross_checks_video_and_sidecar(tmp_path: Path) -> None:
    _touch_episode(tmp_path, 7)
    _touch_episode(tmp_path, 1001, sidecar=False)

    result = discover_episodes(tmp_path, camera="cam_high")

    assert result.episode_ids == (7,)
    assert result.skipped == (
        {
            "episode": 1001,
            "status": "discovery_skipped",
            "missing": ["sidecar"],
            "parquet": str(
                tmp_path / "data/chunk-001/episode_001001.parquet"
            ),
        },
    )


def test_discover_episodes_can_require_depth_video(tmp_path: Path) -> None:
    _touch_episode(tmp_path, 7, depth=False)

    legacy = discover_episodes(tmp_path, camera="cam_high")
    urdf = discover_episodes(tmp_path, camera="cam_high", require_depth=True)

    assert legacy.episode_ids == (7,)
    assert urdf.episode_ids == ()
    assert urdf.skipped[0]["missing"] == ["depth_video"]


def test_discover_episodes_rejects_invalid_chunk_name(tmp_path: Path) -> None:
    bad = tmp_path / "data/not-a-chunk"
    bad.mkdir(parents=True)
    (bad / "episode_000001.parquet").touch()

    with pytest.raises(ValueError, match="invalid chunk directory name"):
        discover_episodes(tmp_path, camera="cam_high")


def test_dynamic_manifest_contains_measured_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = DiscoveredEpisode(
        episode_id=7,
        parquet=tmp_path / "episode_000007.parquet",
        video=tmp_path / "episode_000007.mp4",
        sidecar=tmp_path / "episode_000007.hdf5",
    )
    second = DiscoveredEpisode(
        episode_id=8,
        parquet=tmp_path / "episode_000008.parquet",
        video=tmp_path / "episode_000008.mp4",
        sidecar=tmp_path / "episode_000008.hdf5",
    )
    monkeypatch.setattr(
        process_module,
        "_measure_episode",
        lambda _episode: (24, (240, 320), 1),
    )

    manifest = build_dynamic_manifest(
        tmp_path,
        task="task",
        camera="cam_high",
        episodes=(first, second),
    )

    assert manifest["format_version"] == "robotwin_dataset_manifest_dynamic_v1"
    assert manifest["frame_shape_hw"] == [240, 320]
    assert manifest["raw_video_frame_surplus"] == 1
    assert manifest["regression_episode_ids"] == [7, 8]
    assert manifest["dataset_root"] == str(tmp_path.resolve())


def test_parse_args_defaults_to_sam_and_preserves_just_sentinel_paths() -> None:
    args = process_module._parse_args(
        ["--source-run-dir", "-", "--urdf-path", "-"]
    )

    assert args.gripper_backend == "sam"
    assert args.urdf_depth_tolerance_mm is None
    assert args.urdf_minimum_eligible_nonempty_fraction is None
    assert args.source_run_dir == "-"
    assert args.urdf_path == "-"
    assert process_module._optional_cli_path(args.source_run_dir) is None
    assert process_module._optional_cli_path(args.urdf_path) is None


def test_main_legacy_cli_dispatches_sam_without_urdf_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _cli_config(tmp_path)
    calls: dict[str, Any] = {}

    def fake_process(received_config: Any, **kwargs: Any) -> dict[str, Any]:
        calls["config"] = received_config
        calls.update(kwargs)
        return {"passed": True, "gripper_backend": "sam"}

    monkeypatch.setattr(process_module, "load_config", lambda _path: config)
    monkeypatch.setattr(process_module, "process_dataset", fake_process)
    monkeypatch.setattr(
        process_module,
        "process_urdf_source_run",
        lambda **_kwargs: pytest.fail("legacy SAM CLI must not enter URDF mode"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "process_dataset.py",
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--output-dir",
            str(tmp_path / "output"),
            "--source-run-dir",
            "-",
            "--urdf-path",
            "-",
        ],
    )

    process_module.main()

    assert calls["config"] is config
    assert calls["dataset_root"] == tmp_path / "dataset"
    assert calls["output_root"] == tmp_path / "output"
    assert calls["run_id"] is None
    assert calls["episode_ids"] is None
    assert calls["force"] is False
    assert calls["skip_render"] is False


@pytest.mark.parametrize(
    ("extra_args", "message"),
    (
        (("--gripper-backend", "urdf", "--urdf-path", "aloha.urdf"), "source-run-dir"),
        (("--gripper-backend", "urdf", "--source-run-dir", "source"), "urdf-path"),
        (
            (
                "--gripper-backend",
                "urdf",
                "--source-run-dir",
                "source",
                "--urdf-path",
                "aloha.urdf",
                "--resume",
            ),
            "explicit --run-id",
        ),
        (("--source-run-dir", "source"), "URDF-only options"),
        (("--urdf-depth-tolerance-mm", "9"), "URDF-only options"),
        (
            ("--urdf-minimum-eligible-nonempty-fraction", "0.8"),
            "URDF-only options",
        ),
        (
            (
                "--gripper-backend",
                "urdf",
                "--source-run-dir",
                "source",
                "--urdf-path",
                "aloha.urdf",
                "--force",
            ),
            "--force is not supported",
        ),
        (
            (
                "--gripper-backend",
                "urdf",
                "--source-run-dir",
                "source",
                "--urdf-path",
                "aloha.urdf",
                "--run-id",
                "resume-run",
                "--dry-run",
                "--resume",
            ),
            "cannot be used together",
        ),
    ),
)
def test_main_rejects_invalid_urdf_parameter_combinations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra_args: tuple[str, ...],
    message: str,
) -> None:
    monkeypatch.setattr(process_module, "load_config", lambda _path: _cli_config(tmp_path))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "process_dataset.py",
            "--dataset-root",
            str(tmp_path / "dataset"),
            *extra_args,
        ],
    )

    with pytest.raises(ValueError, match=message):
        process_module.main()


def test_select_urdf_source_episodes_filters_failed_object_channels(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    source = tmp_path / "source-run"
    _write_source_summary(
        source,
        dataset,
        [
            {"episode": 1, "status": "completed"},
            {"episode": 2, "status": "sam_incomplete"},
            {"episode": 3, "status": "completed"},
            {"episode": 4, "status": "skipped_complete"},
        ],
    )
    # The old gripper channels are deliberately rejected; URDF replaces them.
    _write_source_episode(source, 1)
    # A derived run inherits only complete, frozen source episodes.
    _write_source_episode(source, 2)
    _write_source_episode(
        source,
        3,
        annotation_status=("failed", "valid", "not_annotated", "not_annotated"),
        qc_status=("rejected", "passed", "not_run", "not_run"),
    )
    _write_source_episode(source, 4)

    selection = select_urdf_source_episodes(
        source,
        dataset_root=dataset,
        task="task",
        camera="cam_high",
        discovered_episode_ids=(1, 2, 3, 4),
    )

    assert selection.episode_ids == (1, 4)
    assert {record["episode"] for record in selection.excluded} == {2, 3}
    assert all(
        record["reason"] == "source_contract_error:UrdfGripperPublishError"
        for record in selection.excluded
    )
    assert "not complete" in next(
        record["error"] for record in selection.excluded if record["episode"] == 2
    )
    assert "valid and QC-passed" in next(
        record["error"] for record in selection.excluded if record["episode"] == 3
    )
    with pytest.raises(
        ValueError,
        match=r"3 \(source_contract_error:UrdfGripperPublishError\)",
    ):
        select_urdf_source_episodes(
            source,
            dataset_root=dataset,
            task="task",
            camera="cam_high",
            discovered_episode_ids=(1, 2, 3, 4),
            requested_episode_ids=(3,),
        )


@pytest.mark.parametrize(
    "source_kwargs",
    (
        {"include_format_version": False},
        {"mask_shape": (3, 6, 2, 3)},
        {"stored_frame_count": 2},
        {"target_window": None},
    ),
)
def test_select_urdf_source_episodes_rejects_renderer_contract_mismatches(
    tmp_path: Path,
    source_kwargs: dict[str, Any],
) -> None:
    dataset = tmp_path / "dataset"
    source = tmp_path / "source-run"
    _write_source_summary(
        source,
        dataset,
        [{"episode": 1, "status": "completed"}],
    )
    _write_source_episode(source, 1, **source_kwargs)

    with pytest.raises(
        ValueError,
        match="source_contract_error:UrdfGripperPublishError",
    ):
        select_urdf_source_episodes(
            source,
            dataset_root=dataset,
            task="task",
            camera="cam_high",
            discovered_episode_ids=(1,),
            requested_episode_ids=(1,),
            expected_frame_counts={1: 6},
        )


def test_select_urdf_source_episodes_checks_parquet_frame_count(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    source = tmp_path / "source-run"
    _write_source_summary(
        source,
        dataset,
        [{"episode": 1, "status": "completed"}],
    )
    _write_source_episode(source, 1)

    with pytest.raises(
        ValueError,
        match="source_contract_error:UrdfGripperPublishError",
    ):
        select_urdf_source_episodes(
            source,
            dataset_root=dataset,
            task="task",
            camera="cam_high",
            discovered_episode_ids=(1,),
            requested_episode_ids=(1,),
            expected_frame_counts={1: 7},
        )


def test_main_urdf_cli_forwards_parameters_without_legacy_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    source = tmp_path / "source-run"
    _write_source_summary(
        source,
        dataset,
        [{"episode": 7, "status": "completed"}],
    )
    config = _cli_config(tmp_path)
    calls: dict[str, Any] = {}

    def fake_urdf(**kwargs: Any) -> dict[str, Any]:
        calls.update(kwargs)
        return {"passed": True, "gripper_backend": "urdf"}

    monkeypatch.setattr(process_module, "load_config", lambda _path: config)
    monkeypatch.setattr(process_module, "process_urdf_source_run", fake_urdf)
    monkeypatch.setattr(
        process_module,
        "process_dataset",
        lambda *_args, **_kwargs: pytest.fail("URDF CLI must not run legacy SAM pipeline"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "process_dataset.py",
            "--gripper-backend",
            "urdf",
            "--source-run-dir",
            str(source),
            "--urdf-path",
            str(tmp_path / "aloha.urdf"),
            "--dataset-root",
            str(dataset),
            "--output-dir",
            str(tmp_path / "output"),
            "--task",
            "task",
            "--camera",
            "cam_high",
            "--run-id",
            "urdf-test",
            "--episode-ids",
            "7",
            "--dry-run",
            "--urdf-depth-tolerance-mm",
            "5.5",
            "--urdf-minimum-eligible-nonempty-fraction",
            "0.8",
            "--skip-render",
            "--allow-partial-source",
        ],
    )

    process_module.main()

    assert calls["dataset_root"] == dataset
    assert calls["source_run_dir"] == source
    assert calls["task"] == "task"
    assert calls["camera"] == "cam_high"
    assert calls["output_root"] == tmp_path / "output"
    assert calls["urdf_path"] == tmp_path / "aloha.urdf"
    assert calls["run_id"] == "urdf-test"
    assert calls["episode_ids"] == (7,)
    assert calls["resume"] is False
    assert calls["dry_run"] is True
    assert calls["depth_tolerance_mm"] == 5.5
    assert calls["minimum_eligible_nonempty_fraction"] == 0.8
    assert calls["skip_render"] is True
    assert calls["allow_partial_source"] is True


@pytest.mark.parametrize("run_id", ("../escape", "/absolute/run", "bad\\name"))
def test_process_urdf_source_run_rejects_unsafe_run_id_before_output_paths(
    tmp_path: Path,
    run_id: str,
) -> None:
    output_root = tmp_path / "output"

    with pytest.raises(ValueError, match="simple non-empty directory name"):
        process_module.process_urdf_source_run(
            dataset_root=tmp_path / "missing-dataset",
            source_run_dir=tmp_path / "missing-source",
            task="task",
            camera="cam_high",
            output_root=output_root,
            urdf_path=tmp_path / "missing.urdf",
            run_id=run_id,
            dry_run=True,
        )

    assert not output_root.exists()


def test_urdf_run_ownership_rejects_existing_non_resume_and_foreign_resume(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "output" / "same-run"
    run_dir.mkdir(parents=True)
    sentinel = run_dir / "process_summary.json"
    original = {
        "format_version": "robotwin_process_dataset_summary_v1",
        "gripper_backend": "sam",
        "backend": {"type": "sam"},
        "run_id": "same-run",
    }
    sentinel.write_text(json.dumps(original), encoding="utf-8")

    with pytest.raises(FileExistsError, match="canonical output run already exists"):
        process_module._validate_urdf_run_ownership(
            run_dir,
            run_id="same-run",
            resume=False,
        )

    backend_manifest = run_dir / "_backend" / "urdf" / "manifest.json"
    backend_manifest.parent.mkdir(parents=True)
    backend_manifest.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="not owned by the URDF backend"):
        process_module._validate_urdf_run_ownership(
            run_dir,
            run_id="same-run",
            resume=True,
        )

    assert json.loads(sentinel.read_text(encoding="utf-8")) == original


def test_run_ownership_preserves_legacy_sam_and_interrupted_urdf_resume(
    tmp_path: Path,
) -> None:
    sam_dir = tmp_path / "legacy-sam"
    sam_dir.mkdir()
    (sam_dir / "process_summary.json").write_text(
        json.dumps({"run_id": "legacy-sam"}),
        encoding="utf-8",
    )
    process_module._validate_sam_run_ownership(sam_dir, run_id="legacy-sam")

    urdf_dir = tmp_path / "interrupted-urdf"
    manifest = urdf_dir / "_backend" / "urdf" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}", encoding="utf-8")
    process_module._validate_urdf_run_ownership(
        urdf_dir,
        run_id="interrupted-urdf",
        resume=True,
    )

    with pytest.raises(ValueError, match="owned by the URDF backend"):
        process_module._validate_sam_run_ownership(
            urdf_dir,
            run_id="interrupted-urdf",
        )


def test_process_urdf_source_run_never_calls_sam_or_legacy_renderer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    source = tmp_path / "source-run"
    _touch_episode(dataset, 7)
    _write_source_summary(
        source,
        dataset,
        [{"episode": 7, "status": "completed"}],
    )
    _write_source_episode(source, 7)
    observed: list[Any] = []

    def fake_experiment(config: Any) -> dict[str, Any]:
        observed.append(config)
        return {"dry_run": True, "episode_count": len(config.episode_ids)}

    def unexpected(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("URDF source-run mode called a legacy SAM/gripper path")

    monkeypatch.setattr(process_module, "_load_sam_runtime", unexpected)
    monkeypatch.setattr(process_module, "_render_processed", unexpected)

    summary = process_module.process_urdf_source_run(
        dataset_root=dataset,
        source_run_dir=source,
        task="task",
        camera="cam_high",
        output_root=tmp_path / "output",
        urdf_path=tmp_path / "aloha.urdf",
        run_id="urdf-test",
        episode_ids=(7,),
        dry_run=True,
        experiment_runner=fake_experiment,
    )

    assert summary["passed"] is True
    assert summary["requested_episode_ids"] == [7]
    assert summary["backend"]["selected_episode_ids"] == [7]
    assert len(observed) == 1
    assert observed[0].episode_ids == (7,)
    assert observed[0].run_id == "urdf"
    assert observed[0].skip_overlay is True
    assert observed[0].run_dir == (
        tmp_path / "output/urdf-test/_backend/urdf"
    ).resolve()


def test_process_urdf_source_run_requires_explicit_partial_source_opt_in(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    source = tmp_path / "source-run"
    _touch_episode(dataset, 7)
    _touch_episode(dataset, 8)
    _write_source_summary(
        source,
        dataset,
        [
            {"episode": 7, "status": "completed"},
            {"episode": 8, "status": "sam_incomplete"},
        ],
    )
    _write_source_episode(source, 7)
    observed: list[Any] = []

    def fake_experiment(config: Any) -> dict[str, Any]:
        observed.append(config)
        return {"dry_run": True, "episode_count": len(config.episode_ids)}

    common = {
        "dataset_root": dataset,
        "source_run_dir": source,
        "task": "task",
        "camera": "cam_high",
        "output_root": tmp_path / "output",
        "urdf_path": tmp_path / "aloha.urdf",
        "run_id": "urdf-test",
        "dry_run": True,
        "experiment_runner": fake_experiment,
    }
    with pytest.raises(ValueError, match="--allow-partial-source"):
        process_module.process_urdf_source_run(**common)

    summary = process_module.process_urdf_source_run(
        **common,
        allow_partial_source=True,
    )

    assert len(observed) == 1
    assert observed[0].episode_ids == (7,)
    assert summary["passed"] is True
    assert summary["backend"]["source_selection_complete"] is False
    assert summary["backend"]["allow_partial_source"] is True
    assert summary["backend"]["source_excluded"] == [
        {
            "episode": 8,
            "status": "source_excluded",
            "reason": "source_contract_error:FileNotFoundError",
            "error": str(
                "source episode directory is missing: "
                f"{(source / 'task/episode_000008/cam_high').resolve()}"
            ),
        }
    ]
    assert summary["records"] == [
        summary["backend"]["source_excluded"][0],
        {
            "episode": 7,
            "status": "planned",
            "gripper_backend": "urdf",
            "source_lineage_sha256": _source_lineage(source, 7)[
                "lineage_sha256"
            ],
        },
    ]


def test_process_urdf_source_run_records_incomplete_dataset_inputs(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    source = tmp_path / "source-run"
    _touch_episode(dataset, 7)
    _touch_episode(dataset, 8, depth=False)
    _write_source_summary(
        source,
        dataset,
        [
            {"episode": 7, "status": "completed"},
            {"episode": 8, "status": "completed"},
        ],
    )
    _write_source_episode(source, 7)
    _write_source_episode(source, 8)
    observed: list[Any] = []

    def fake_experiment(config: Any) -> dict[str, Any]:
        observed.append(config)
        return {"dry_run": True, "episode_count": len(config.episode_ids)}

    common = {
        "dataset_root": dataset,
        "source_run_dir": source,
        "task": "task",
        "camera": "cam_high",
        "output_root": tmp_path / "output",
        "urdf_path": tmp_path / "aloha.urdf",
        "run_id": "urdf-test",
        "dry_run": True,
        "experiment_runner": fake_experiment,
    }
    with pytest.raises(ValueError, match="--allow-partial-source"):
        process_module.process_urdf_source_run(**common)

    summary = process_module.process_urdf_source_run(
        **common,
        allow_partial_source=True,
    )

    assert observed[0].episode_ids == (7,)
    assert summary["requested_episode_ids"] == [7, 8]
    assert summary["discovered_episode_ids"] == [7, 8]
    assert summary["backend"]["source_selection_complete"] is False
    assert summary["backend"]["source_excluded"] == []
    assert summary["backend"]["dataset_excluded"] == [
        {
            "episode": 8,
            "status": "dataset_excluded",
            "reason": "dataset_inputs_missing",
            "missing": ["depth_video"],
            "parquet": str(
                dataset / "data/chunk-000/episode_000008.parquet"
            ),
        }
    ]
    assert summary["records"] == [
        summary["backend"]["dataset_excluded"][0],
        {
            "episode": 7,
            "status": "planned",
            "gripper_backend": "urdf",
            "source_lineage_sha256": _source_lineage(source, 7)[
                "lineage_sha256"
            ],
        },
    ]

    observed.clear()
    explicit = process_module.process_urdf_source_run(
        **common,
        episode_ids=(7,),
    )

    assert observed[0].episode_ids == (7,)
    assert explicit["requested_episode_ids"] == [7]
    assert explicit["backend"]["dataset_excluded"] == []
    assert explicit["backend"]["source_excluded"] == []
    assert explicit["backend"]["source_selection_complete"] is True


def test_process_urdf_source_run_records_shared_render_failure(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    source = tmp_path / "source-run"
    _touch_episode(dataset, 7)
    _write_source_summary(
        source,
        dataset,
        [{"episode": 7, "status": "completed"}],
    )
    _write_source_episode(source, 7)

    def fake_experiment(config: Any) -> dict[str, Any]:
        config.run_dir.mkdir(parents=True)
        return {
            "status": "complete",
            "episodes": [
                {
                    "episode_index": 7,
                    "status": "complete",
                    "output_dir": "episode_000007",
                    "active_arm": "right",
                    "source_lineage": _source_lineage(source, 7),
                }
            ],
        }

    def fake_publish(**_kwargs: Any) -> dict[str, Any]:
        return {"status": "completed"}

    def failed_render(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("sheet generation failed")

    summary = process_module.process_urdf_source_run(
        pipeline_config=process_module.load_config(
            Path("configs/pilot_move_pillbottle_pad.yaml")
        ),
        dataset_root=dataset,
        source_run_dir=source,
        task="task",
        camera="cam_high",
        output_root=tmp_path / "output",
        urdf_path=tmp_path / "aloha.urdf",
        run_id="urdf-test",
        episode_ids=(7,),
        experiment_runner=fake_experiment,
        episode_publisher=fake_publish,
        episode_validator=lambda **_kwargs: {"status": "completed"},
        render_builder=failed_render,
    )

    assert summary["passed"] is False
    assert summary["render"] is None
    assert summary["records"][-1] == {
        "status": "render_failed",
        "gripper_backend": "urdf",
        "error": "RuntimeError: sheet generation failed",
    }
    assert Path(summary["artifact"]).is_file()


def test_process_urdf_source_run_blocks_render_when_source_changes_after_publish(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    source = tmp_path / "source-run"
    _touch_episode(dataset, 7)
    _write_source_summary(
        source,
        dataset,
        [{"episode": 7, "status": "completed"}],
    )
    _write_source_episode(source, 7)

    def fake_experiment(config: Any) -> dict[str, Any]:
        config.run_dir.mkdir(parents=True)
        return {
            "status": "complete",
            "episodes": [
                {
                    "episode_index": 7,
                    "status": "complete",
                    "output_dir": "episode_000007",
                    "active_arm": "right",
                    "source_lineage": _source_lineage(source, 7),
                }
            ],
        }

    def fake_publish(**_kwargs: Any) -> dict[str, Any]:
        artifact = source / "task/episode_000007/cam_high/target_0/native_track.npz"
        artifact.write_bytes(artifact.read_bytes() + b"-changed")
        return {"status": "completed"}

    summary = process_module.process_urdf_source_run(
        pipeline_config=process_module.load_config(
            Path("configs/pilot_move_pillbottle_pad.yaml")
        ),
        dataset_root=dataset,
        source_run_dir=source,
        task="task",
        camera="cam_high",
        output_root=tmp_path / "output",
        urdf_path=tmp_path / "aloha.urdf",
        run_id="urdf-source-change",
        episode_ids=(7,),
        experiment_runner=fake_experiment,
        episode_publisher=fake_publish,
        episode_validator=lambda **_kwargs: pytest.fail(
            "canonical validation must not run after source lineage changes"
        ),
        render_builder=lambda *_args, **_kwargs: pytest.fail(
            "shared renderer must not run after source lineage changes"
        ),
    )

    assert summary["passed"] is False
    assert summary["render"] is None
    assert summary["records"][-1]["status"] == "source_lineage_changed"
    assert not any(
        record.get("status") == "render_failed" for record in summary["records"]
    )


def test_process_urdf_source_run_blocks_render_on_canonical_revalidation_failure(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    source = tmp_path / "source-run"
    _touch_episode(dataset, 7)
    _write_source_summary(
        source,
        dataset,
        [{"episode": 7, "status": "completed"}],
    )
    _write_source_episode(source, 7)

    def fake_experiment(config: Any) -> dict[str, Any]:
        config.run_dir.mkdir(parents=True)
        return {
            "status": "complete",
            "episodes": [
                {
                    "episode_index": 7,
                    "status": "complete",
                    "output_dir": "episode_000007",
                    "active_arm": "right",
                    "source_lineage": _source_lineage(source, 7),
                }
            ],
        }

    def reject_canonical(**_kwargs: Any) -> dict[str, Any]:
        raise process_module.UrdfGripperPublishError(
            "published masks differ from the canonical contract"
        )

    summary = process_module.process_urdf_source_run(
        pipeline_config=process_module.load_config(
            Path("configs/pilot_move_pillbottle_pad.yaml")
        ),
        dataset_root=dataset,
        source_run_dir=source,
        task="task",
        camera="cam_high",
        output_root=tmp_path / "output",
        urdf_path=tmp_path / "aloha.urdf",
        run_id="urdf-canonical-change",
        episode_ids=(7,),
        experiment_runner=fake_experiment,
        episode_publisher=lambda **_kwargs: {"status": "completed"},
        episode_validator=reject_canonical,
        render_builder=lambda *_args, **_kwargs: pytest.fail(
            "shared renderer must not run after canonical validation fails"
        ),
    )

    assert summary["passed"] is False
    assert summary["render"] is None
    assert summary["records"][-1]["status"] == "canonical_validation_failed"
    assert not any(
        record.get("status") == "render_failed" for record in summary["records"]
    )


def test_process_urdf_source_run_renders_successes_after_partial_backend_failure(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    source = tmp_path / "source-run"
    for episode_id in (7, 8):
        _touch_episode(dataset, episode_id)
    _write_source_summary(
        source,
        dataset,
        [
            {"episode": 7, "status": "completed"},
            {"episode": 8, "status": "completed"},
        ],
    )
    _write_source_episode(source, 7)
    _write_source_episode(source, 8)
    published: list[int] = []
    rendered: list[tuple[int, ...]] = []

    def fake_experiment(config: Any) -> dict[str, Any]:
        config.run_dir.mkdir(parents=True)
        manifest = {
            "format_version": "robotwin_urdf_gripper_run_v2",
            "run_id": "urdf",
            "status": "failed",
            "episodes": [
                {
                    "episode_index": 7,
                    "status": "complete",
                    "output_dir": "episode_000007",
                    "active_arm": "right",
                    "source_lineage": _source_lineage(source, 7),
                },
                {
                    "episode_index": 8,
                    "status": "failed",
                    "error": (
                        "eligible nonempty fraction is below the run threshold: "
                        "0.898649 < 0.900000"
                    ),
                },
            ],
        }
        (config.run_dir / "manifest.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )
        raise urdf_module.UrdfBatchIncompleteError(
            "URDF gripper run failed for episodes [8]",
            result=manifest,
        )

    def fake_publish(**kwargs: Any) -> dict[str, Any]:
        published.append(int(kwargs["backend_episode_record"]["episode_index"]))
        return {"status": "completed"}

    def fake_render(
        _config: Any,
        *,
        run_id: str,
        episode_ids: tuple[int, ...],
        output_dir: Path,
    ) -> dict[str, Any]:
        assert run_id == "urdf-partial"
        assert output_dir == (tmp_path / "output/urdf-partial").resolve()
        rendered.append(episode_ids)
        return {
            "manifest": str(output_dir / "rendered_videos/manifest.json"),
            "episode_count": len(episode_ids),
            "review_sheets": [],
        }

    summary = process_module.process_urdf_source_run(
        pipeline_config=process_module.load_config(
            Path("configs/pilot_move_pillbottle_pad.yaml")
        ),
        dataset_root=dataset,
        source_run_dir=source,
        task="task",
        camera="cam_high",
        output_root=tmp_path / "output",
        urdf_path=tmp_path / "aloha.urdf",
        run_id="urdf-partial",
        episode_ids=(7, 8),
        experiment_runner=fake_experiment,
        episode_publisher=fake_publish,
        episode_validator=lambda **_kwargs: {"status": "completed"},
        render_builder=fake_render,
    )

    assert published == [7]
    assert rendered == [(7,)]
    assert summary["passed"] is False
    expected_error = (
        "UrdfBatchIncompleteError: URDF gripper run failed for episodes [8]"
    )
    assert summary["fatal_error"] == expected_error
    assert summary["render"]["episode_count"] == 1
    assert summary["backend"]["error"] == expected_error
    assert [record["status"] for record in summary["records"]] == [
        "completed",
        "gripper_incomplete",
    ]
    persisted = json.loads(Path(summary["artifact"]).read_text(encoding="utf-8"))
    assert set(persisted) == {
        "format_version",
        "gripper_backend",
        "run_id",
        "dataset_root",
        "task",
        "camera",
        "discovered_episode_ids",
        "requested_episode_ids",
        "dynamic_manifest",
        "qwen_health",
        "records",
        "render",
        "fatal_error",
        "backend",
        "passed",
    }


@pytest.mark.parametrize(
    "runner_error",
    [
        urdf_module.UrdfMaskRunError(
            "resume configuration/assets differ from the immutable run contract"
        ),
        ValueError("invalid URDF configuration"),
        TypeError("synthetic programming error"),
    ],
    ids=("resume-contract", "configuration", "programming"),
)
def test_process_urdf_source_run_propagates_non_batch_runner_errors(
    tmp_path: Path,
    runner_error: Exception,
) -> None:
    dataset = tmp_path / "dataset"
    source = tmp_path / "source-run"
    _touch_episode(dataset, 7)
    _write_source_summary(
        source,
        dataset,
        [{"episode": 7, "status": "completed"}],
    )
    _write_source_episode(source, 7)
    run_dir = tmp_path / "output/urdf-resume/_backend/urdf"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "episodes": [
                    {
                        "episode_index": 7,
                        "status": "complete",
                        "output_dir": "stale_episode_000007",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    publisher_called = False
    renderer_called = False

    def failed_runner(_config: Any) -> dict[str, Any]:
        raise runner_error

    def unexpected_publish(**_kwargs: Any) -> dict[str, Any]:
        nonlocal publisher_called
        publisher_called = True
        return {"status": "completed"}

    def unexpected_render(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        nonlocal renderer_called
        renderer_called = True
        return {"episode_count": 1}

    with pytest.raises(type(runner_error), match=str(runner_error)):
        process_module.process_urdf_source_run(
            pipeline_config=process_module.load_config(
                Path("configs/pilot_move_pillbottle_pad.yaml")
            ),
            dataset_root=dataset,
            source_run_dir=source,
            task="task",
            camera="cam_high",
            output_root=tmp_path / "output",
            urdf_path=tmp_path / "aloha.urdf",
            run_id="urdf-resume",
            episode_ids=(7,),
            resume=True,
            experiment_runner=failed_runner,
            episode_publisher=unexpected_publish,
            render_builder=unexpected_render,
        )

    assert publisher_called is False
    assert renderer_called is False
    assert not (tmp_path / "output/urdf-resume/process_summary.json").exists()
