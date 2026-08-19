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


class _EpisodeRecordingUI(process_module.ProcessUI):
    def __init__(self) -> None:
        super().__init__(emit_json_summary=False, verbose=False)
        self.finished_episodes: list[tuple[int, str]] = []

    def episode_finished(
        self,
        episode_id: int,
        *,
        status: str,
        detail: str | None = None,
    ) -> None:
        del detail
        self.finished_episodes.append((episode_id, status))


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
        ),
        sam3=SimpleNamespace(gpus=(2,)),
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


def test_process_dataset_reports_sam_stages_without_embedded_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset = tmp_path / "dataset"
    _touch_episode(dataset, 7)
    monkeypatch.setattr(
        process_module,
        "_measure_episode",
        lambda _episode: (6, (2, 3), 0),
    )
    backend_shutdown: list[bool] = []
    runtime = SimpleNamespace(
        qwen_client_factory=lambda **_kwargs: SimpleNamespace(
            health=lambda: {"status": "ok"}
        ),
        backend_factory=lambda **_kwargs: SimpleNamespace(
            shutdown=lambda: backend_shutdown.append(True)
        ),
        execution_errors=(RuntimeError,),
        emit_gripper_result=lambda *_args: print('{"stage":"gripper"}') or True,
        emit_sam_result=lambda *_args: print('{"stage":"sam"}') or True,
        execute_gripper_episode=lambda *_args: object(),
        execute_sam_episode=lambda *_args: object(),
        fatal_cuda_error=lambda _exc: False,
        gripper_episode_complete=lambda *_args: False,
        run_qwen=lambda *_args: print('{"stage":"qwen"}'),
    )
    monkeypatch.setattr(process_module, "_load_sam_runtime", lambda: runtime)

    class RecordingUI(process_module.ProcessUI):
        def __init__(self) -> None:
            super().__init__(emit_json_summary=False, verbose=False)
            self.stages: list[tuple[str, int, str, str | None]] = []
            self.episodes: list[tuple[int, str]] = []

        def stage_started(self, episode_id: int, label: str) -> None:
            self.stages.append(("started", episode_id, label, None))

        def stage_finished(
            self,
            episode_id: int,
            label: str,
            *,
            status: str = "completed",
            detail: str | None = None,
        ) -> None:
            self.stages.append(("finished", episode_id, label, status))

        def episode_finished(
            self,
            episode_id: int,
            *,
            status: str,
            detail: str | None = None,
        ) -> None:
            self.episodes.append((episode_id, status))

    reporter = RecordingUI()
    summary = process_module.process_dataset(
        process_module.load_config(Path("configs/pilot_move_pillbottle_pad.yaml")),
        dataset_root=dataset,
        task="task",
        camera="cam_high",
        output_root=tmp_path / "output",
        run_id="ui-sam-test",
        episode_ids=(7,),
        skip_render=True,
        reporter=reporter,
    )

    assert capsys.readouterr().out == ""
    assert summary["passed"] is True
    assert summary["records"] == [{"episode": 7, "status": "completed"}]
    assert reporter.stages == [
        ("started", 7, "qwen", None),
        ("finished", 7, "qwen", "completed"),
        ("started", 7, "object_sam", None),
        ("finished", 7, "object_sam", "completed"),
        ("started", 7, "gripper_sam", None),
        ("finished", 7, "gripper_sam", "completed"),
    ]
    assert reporter.episodes == [(7, "completed")]
    assert backend_shutdown == [True]


def test_process_dataset_target_receiver_only_skips_gripper_and_uses_sam_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    _touch_episode(dataset, 7)
    monkeypatch.setattr(
        process_module,
        "_measure_episode",
        lambda _episode: (6, (2, 3), 0),
    )
    calls: list[str] = []

    def unexpected_gripper(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("target/receiver-only processing must not inspect or run gripper")

    runtime = SimpleNamespace(
        qwen_client_factory=lambda **_kwargs: SimpleNamespace(
            health=lambda: {"status": "ok"}
        ),
        backend_factory=lambda **_kwargs: SimpleNamespace(
            shutdown=lambda: calls.append("shutdown")
        ),
        execution_errors=(RuntimeError,),
        emit_gripper_result=unexpected_gripper,
        emit_sam_result=lambda *_args: calls.append("emit_sam") or True,
        execute_gripper_episode=unexpected_gripper,
        execute_sam_episode=lambda *_args: calls.append("execute_sam") or object(),
        fatal_cuda_error=lambda _exc: False,
        gripper_episode_complete=unexpected_gripper,
        sam_episode_complete=lambda *_args: calls.append("sam_resume_scan") or False,
        run_qwen=lambda *_args: calls.append("qwen"),
    )
    monkeypatch.setattr(process_module, "_load_sam_runtime", lambda: runtime)

    summary = process_module.process_dataset(
        process_module.load_config(Path("configs/pilot_move_pillbottle_pad.yaml")),
        dataset_root=dataset,
        task="task",
        camera="cam_high",
        output_root=tmp_path / "output",
        run_id="target-receiver-source",
        episode_ids=(7,),
        skip_render=True,
        target_receiver_only=True,
    )

    assert calls == [
        "sam_resume_scan",
        "qwen",
        "execute_sam",
        "emit_sam",
        "shutdown",
    ]
    assert summary["passed"] is True
    assert summary["records"] == [{"episode": 7, "status": "completed"}]
    assert summary["annotation_mode"] == "pick_place"
    assert summary["required_object_roles"] == ["target", "receiver"]
    assert summary["gripper_backend"] is None
    assert summary["backend"] == {"object_masks": "sam", "gripper": None}
    assert summary["stage_mode"] == "object_source_only"


def test_target_only_rejects_sam_gripper_before_loading_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_runtime_load() -> Any:
        pytest.fail("unsupported target-only SAM gripper must fail before model loading")

    monkeypatch.setattr(process_module, "_load_sam_runtime", unexpected_runtime_load)
    config = process_module.load_config(
        Path("configs/pilot_adjust_bottle_target_only.yaml")
    )

    with pytest.raises(ValueError, match="target_only.*URDF"):
        process_module.process_dataset(
            config,
            dataset_root=tmp_path / "unused",
            task="adjust_bottle",
            camera="cam_high",
            output_root=tmp_path / "output",
            episode_ids=(0,),
            skip_render=True,
        )


def test_target_only_review_manifest_lists_only_applicable_roles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_config = process_module.load_config(
        Path("configs/pilot_adjust_bottle_target_only.yaml")
    )
    config = SimpleNamespace(
        config_path=target_config.config_path,
        output_root=tmp_path / "runs",
        annotation=target_config.annotation,
        dataset=SimpleNamespace(
            root=tmp_path / "dataset",
            task="adjust_bottle",
            camera="cam_high",
            manifest=None,
            manifest_data={},
        ),
    )
    source_video = tmp_path / "episode_000000.mp4"
    source_video.touch()

    class FakeDataset:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def paths(self, _ref: Any) -> Any:
            return SimpleNamespace(video=source_video)

        def task_text(self, _episode_id: int) -> str:
            return "adjust bottle"

    masks_path = tmp_path / "masks.npz"
    masks_path.touch()
    artifact = SimpleNamespace(
        format_version="robotwin_visible_masks_v2",
        instance_names=(
            "target_0",
            "receiver_0",
            "gripper_left",
            "gripper_right",
        ),
        annotation_status=("valid", "not_applicable", "valid", "not_annotated"),
        qc_status=("passed", "not_applicable", "passed", "not_run"),
    )

    def render_video(
        _video_path: Path,
        _artifact: Any,
        output_path: Path,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        output_path.write_bytes(b"video")
        return {"frame_count": 1}

    fake_render = SimpleNamespace(
        DEFAULT_FILL_ALPHA=0.36,
        DEFAULT_OUTLINE_RADIUS=1,
        DEFAULT_HALO_RADIUS=2,
        ROLE_COLORS={"target": (1, 2, 3), "gripper": (4, 5, 6)},
        select_best_masks=lambda *_args, **_kwargs: {
            0: SimpleNamespace(path=masks_path, run_id="target-only")
        },
        _load_masks=lambda _path: artifact,
        _output_video_name=lambda **_kwargs: "episode_000000.mp4",
        render_video=render_video,
        _sha256=lambda _path: "sha256",
        build_sheets=lambda *_args, **_kwargs: [],
    )
    monkeypatch.setitem(sys.modules, "render_coverage20_videos", fake_render)
    monkeypatch.setattr(process_module, "RoboTwinDataset", FakeDataset)

    report = process_module._render_processed(
        config,
        run_id="target-only",
        episode_ids=(0,),
        output_dir=tmp_path / "output",
    )

    manifest = json.loads(Path(report["manifest"]).read_text(encoding="utf-8"))
    assert manifest["rendered_roles"] == ["target", "gripper"]


def test_default_bundled_urdf_path_is_repository_asset() -> None:
    expected = (
        Path(__file__).resolve().parents[2]
        / "configs/assets/aloha-agilex/arx5_description_isaac_gripper.urdf"
    ).resolve()

    assert process_module.DEFAULT_BUNDLED_URDF_PATH == expected
    assert process_module.DEFAULT_BUNDLED_URDF_PATH.is_file()


def test_parse_args_defaults_to_urdf_and_preserves_just_sentinel_paths() -> None:
    args = process_module._parse_args(
        ["--source-run-dir", "-", "--urdf-path", "-"]
    )

    assert args.gripper_backend == "urdf"
    assert args.urdf_depth_tolerance_mm is None
    assert args.urdf_minimum_eligible_nonempty_fraction is None
    assert args.source_run_dir == "-"
    assert args.urdf_path == "-"
    assert process_module._optional_cli_path(args.source_run_dir) is None
    assert process_module._optional_cli_path(args.urdf_path) is None
    assert args.ui == "auto"
    assert args.verbose is False
    assert args.urdf_egl_device_id is None
    assert args.urdf_pipeline_buffer_size == 2
    assert args.no_urdf_pipeline is False


def test_parse_args_accepts_urdf_pipeline_controls() -> None:
    args = process_module._parse_args(
        [
            "--gripper-backend",
            "urdf",
            "--urdf-egl-device-id",
            "3",
            "--urdf-pipeline-buffer-size",
            "4",
            "--no-urdf-pipeline",
        ]
    )

    assert args.urdf_egl_device_id == 3
    assert args.urdf_pipeline_buffer_size == 4
    assert args.no_urdf_pipeline is True


def test_parse_args_accepts_output_format_alias_and_verbose() -> None:
    args = process_module._parse_args(["--output-format", "json", "--verbose"])

    assert args.ui == "json"
    assert args.verbose is True


@pytest.mark.parametrize(
    ("arguments", "expected"),
    (
        (("--data-path", "dataset", "--target-only"), "target_only"),
        (("--data_path", "dataset", "--pick_place"), "pick_place"),
    ),
)
def test_parse_args_accepts_path_only_modes(
    arguments: tuple[str, ...],
    expected: str,
) -> None:
    args = process_module._parse_args(arguments)

    assert args.data_path == Path("dataset")
    assert args.path_mode == expected


@pytest.mark.parametrize(
    ("mode", "semantic_prompt", "qc_prompt"),
    (
        (
            process_module.AnnotationMode.PICK_PLACE,
            "target_receiver_semantic_open_set.txt",
            "mask_candidate_qc_open_set.txt",
        ),
        (
            process_module.AnnotationMode.TARGET_ONLY,
            "target_only_semantic_open_set.txt",
            "target_only_mask_candidate_qc_open_set.txt",
        ),
    ),
)
def test_path_only_default_profiles_enable_s1_through_s3(
    mode: Any,
    semantic_prompt: str,
    qc_prompt: str,
) -> None:
    config = process_module.load_config(process_module.PATH_MODE_CONFIGS[mode])

    assert config.annotation.mode is mode
    assert config.qwen.prompt_template.name == semantic_prompt
    assert config.mask.qc_prompt_template is not None
    assert config.mask.qc_prompt_template.name == qc_prompt
    assert config.mask.qc_max_candidates == 8
    assert config.mask.qc_query_fallback_enabled
    assert config.mask.qc_seed_fallback_enabled
    assert config.mask.qc_bbox_fallback_enabled
    assert config.mask.qc_bbox_prompt_template is not None
    assert config.mask.qc_bbox_prompt_template.name == (
        "open_set_bbox_localization.txt"
    )


def test_path_only_single_task_dispatches_from_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "adjust_bottle"
    target = SimpleNamespace(
        root=dataset,
        task="adjust_bottle",
        camera="cam_high",
        episode_ids=(0,),
    )
    resolved = SimpleNamespace(root=dataset, targets=(target,), is_collection=False)
    calls: dict[str, Any] = {}
    config = _cli_config(tmp_path)

    monkeypatch.setattr(
        process_module,
        "resolve_dataset_input",
        lambda *_args, **_kwargs: resolved,
    )

    def fake_load_config(path: Path) -> Any:
        calls["config_path"] = path
        return config

    monkeypatch.setattr(process_module, "load_config", fake_load_config)

    def fake_live(**kwargs: Any) -> dict[str, Any]:
        calls.update(kwargs)
        return {"passed": True}

    monkeypatch.setattr(process_module, "process_live_urdf_pipeline", fake_live)
    args = process_module._parse_args(
        [
            "--data-path",
            str(dataset),
            "--target-only",
            "--episode-ids",
            "0",
            "--skip-render",
        ]
    )

    summary = process_module._run_from_args(
        args,
        process_module.ProcessUI(emit_json_summary=False, verbose=False),
    )

    assert summary["passed"] is True
    assert calls["config_path"] == process_module.PATH_MODE_CONFIGS[
        process_module.AnnotationMode.TARGET_ONLY
    ]
    assert calls["dataset_root"] == dataset
    assert calls["task"] == "adjust_bottle"
    assert calls["episode_ids"] == (0,)


def test_path_only_collection_runs_each_task_and_writes_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets = tuple(
        SimpleNamespace(
            root=tmp_path / task,
            task=task,
            camera="cam_high",
            episode_ids=(episode_id,),
        )
        for task, episode_id in (("alpha", 1), ("beta", 2))
    )
    resolved = SimpleNamespace(root=tmp_path, targets=targets, is_collection=True)
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        process_module,
        "resolve_dataset_input",
        lambda *_args, **_kwargs: resolved,
    )
    monkeypatch.setattr(process_module, "load_config", lambda _path: _cli_config(tmp_path))

    def fake_live(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"passed": True, "artifact": f"{kwargs['task']}.json"}

    monkeypatch.setattr(process_module, "process_live_urdf_pipeline", fake_live)
    args = process_module._parse_args(
        [
            "--data-path",
            str(tmp_path),
            "--target-only",
            "--run-id",
            "collection-test",
            "--output-dir",
            str(tmp_path / "output"),
            "--skip-render",
        ]
    )

    summary = process_module._run_from_args(
        args,
        process_module.ProcessUI(emit_json_summary=False, verbose=False),
    )

    assert [call["task"] for call in calls] == ["alpha", "beta"]
    assert [call["run_id"] for call in calls] == [
        "collection-test-alpha",
        "collection-test-beta",
    ]
    assert summary["passed"] is True
    assert Path(summary["artifact"]).is_file()


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
        process_module,
        "process_live_urdf_pipeline",
        lambda **_kwargs: pytest.fail("legacy SAM CLI must not enter live URDF mode"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "process_dataset.py",
            "--gripper-backend",
            "sam",
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
    assert isinstance(calls["reporter"], process_module.ProcessUI)


def test_main_live_urdf_cli_uses_bundled_asset_and_not_derived_entrypoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _cli_config(tmp_path)
    calls: dict[str, Any] = {}

    def fake_live_urdf(**kwargs: Any) -> dict[str, Any]:
        calls.update(kwargs)
        return {"passed": True, "gripper_backend": "urdf"}

    monkeypatch.setattr(process_module, "load_config", lambda _path: config)
    monkeypatch.setattr(
        process_module,
        "process_live_urdf_pipeline",
        fake_live_urdf,
    )
    monkeypatch.setattr(
        process_module,
        "process_urdf_source_run",
        lambda **_kwargs: pytest.fail(
            "URDF without --source-run-dir must use the live pipeline"
        ),
    )
    monkeypatch.setattr(
        process_module,
        "process_dataset",
        lambda *_args, **_kwargs: pytest.fail(
            "CLI must dispatch live URDF through its coordinator"
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "process_dataset.py",
            "--gripper-backend",
            "urdf",
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--output-dir",
            str(tmp_path / "output"),
            "--task",
            "task",
            "--camera",
            "cam_high",
            "--run-id",
            "live-urdf-test",
            "--episode-ids",
            "7",
            "--skip-render",
        ],
    )

    process_module.main()

    assert calls["pipeline_config"] is config
    assert calls["dataset_root"] == tmp_path / "dataset"
    assert calls["task"] == "task"
    assert calls["camera"] == "cam_high"
    assert calls["output_root"] == tmp_path / "output"
    assert calls["urdf_path"] == process_module.DEFAULT_BUNDLED_URDF_PATH
    assert calls["run_id"] == "live-urdf-test"
    assert calls["episode_ids"] == (7,)
    assert calls["skip_render"] is True
    assert calls["urdf_pipeline"] is True
    assert calls["urdf_pipeline_buffer_size"] == 2
    assert calls["urdf_egl_device_id"] is None


def test_live_urdf_pipeline_runs_target_receiver_source_before_derived_urdf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _cli_config(tmp_path)
    dataset = tmp_path / "dataset"
    _touch_episode(dataset, 7)
    output_root = tmp_path / "output"
    events: list[str] = []
    source_calls: dict[str, Any] = {}
    urdf_calls: dict[str, Any] = {}

    def fake_source(received_config: Any, **kwargs: Any) -> dict[str, Any]:
        events.append("target_receiver")
        source_calls["config"] = received_config
        source_calls.update(kwargs)
        source_run_dir = Path(kwargs["output_root"]) / str(kwargs["run_id"])
        source_run_dir.mkdir(parents=True)
        summary = {
            "format_version": "robotwin_process_dataset_summary_v1",
            "run_id": kwargs["run_id"],
            "dataset_root": str(Path(kwargs["dataset_root"]).resolve()),
            "task": kwargs["task"],
            "camera": kwargs["camera"],
            "records": [{"episode": 7, "status": "completed"}],
            "passed": True,
        }
        (source_run_dir / "process_summary.json").write_text(
            json.dumps(summary),
            encoding="utf-8",
        )
        return summary

    def fake_urdf(**kwargs: Any) -> dict[str, Any]:
        events.append("urdf")
        urdf_calls.update(kwargs)
        return {
            "passed": True,
            "gripper_backend": "urdf",
            "run_id": kwargs["run_id"],
        }

    release_report = {
        "gc_collected": 3,
        "cuda_available": True,
        "gpus": [
            {
                "gpu": 2,
                "allocated_before_bytes": 10,
                "reserved_before_bytes": 20,
                "allocated_after_bytes": 0,
                "reserved_after_bytes": 0,
            }
        ],
    }

    def fake_release(gpus: Any) -> dict[str, Any]:
        events.append("release")
        assert tuple(gpus) == config.sam3.gpus
        return release_report

    monkeypatch.setattr(process_module, "process_dataset", fake_source)
    monkeypatch.setattr(process_module, "_release_sam_cuda_cache", fake_release)
    monkeypatch.setattr(process_module, "process_urdf_source_run", fake_urdf)

    summary = process_module.process_live_urdf_pipeline(
        pipeline_config=config,
        dataset_root=dataset,
        task="task",
        camera="cam_high",
        output_root=output_root,
        urdf_path=process_module.DEFAULT_BUNDLED_URDF_PATH,
        run_id="live-urdf-test",
        episode_ids=(7,),
        skip_render=True,
        backend_factory=lambda **_kwargs: object(),
    )

    expected_source_root = (output_root / "_sources").resolve()
    expected_source_run = expected_source_root / "live-urdf-test-object-source"
    assert events == ["target_receiver", "release", "urdf"]
    assert source_calls["config"] is config
    assert source_calls["output_root"] == expected_source_root
    assert source_calls["run_id"] == "live-urdf-test-object-source"
    assert source_calls["object_source_only"] is True
    assert source_calls["skip_render"] is True
    assert urdf_calls["pipeline_config"] is config
    assert urdf_calls["source_run_dir"] == expected_source_run
    assert urdf_calls["output_root"] == output_root.resolve()
    assert urdf_calls["run_id"] == "live-urdf-test"
    assert urdf_calls["urdf_path"] == process_module.DEFAULT_BUNDLED_URDF_PATH
    assert urdf_calls["episode_ids"] == (7,)
    assert urdf_calls["skip_render"] is True
    assert urdf_calls["source_release"] == release_report
    assert summary["gripper_backend"] == "urdf"


def test_live_urdf_pipeline_streams_by_default_and_reuses_backend_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _cli_config(tmp_path)
    dataset = tmp_path / "dataset"
    _touch_episode(dataset, 7)
    streamed: dict[str, Any] = {}
    canonical: dict[str, Any] = {}
    prepared = {"status": "complete", "episodes": []}

    def fake_stream(*_args: Any, **kwargs: Any) -> tuple[Any, Any, Any]:
        streamed.update(kwargs)
        return {"passed": True}, prepared, None

    def fake_canonical(**kwargs: Any) -> dict[str, Any]:
        canonical.update(kwargs)
        return {"passed": True, "gripper_backend": "urdf"}

    monkeypatch.setattr(process_module, "_select_urdf_egl_device", lambda *_args: 3)
    monkeypatch.setattr(
        process_module, "_run_streaming_source_urdf_workers", fake_stream
    )
    monkeypatch.setattr(
        process_module,
        "_release_sam_cuda_cache",
        lambda _gpus: {"gc_collected": 0, "cuda_available": True, "gpus": []},
    )
    monkeypatch.setattr(process_module, "process_urdf_source_run", fake_canonical)

    summary = process_module.process_live_urdf_pipeline(
        pipeline_config=config,
        dataset_root=dataset,
        task="task",
        camera="cam_high",
        output_root=tmp_path / "output",
        urdf_path=process_module.DEFAULT_BUNDLED_URDF_PATH,
        run_id="streaming-test",
        episode_ids=(7,),
        skip_render=True,
    )

    assert streamed["episode_ids"] == (7,)
    assert streamed["buffer_size"] == 2
    assert streamed["urdf_run_config"].egl_device_id == 3
    assert canonical["prepared_backend_result"] is prepared
    assert canonical["prepared_backend_error"] is None
    assert canonical["report_lifecycle"] is False
    assert canonical["egl_device_id"] == 3
    assert summary["passed"] is True


def test_main_json_mode_prints_one_machine_readable_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _cli_config(tmp_path)
    expected = {
        "passed": True,
        "gripper_backend": "sam",
        "records": [{"episode": 7, "status": "completed"}],
    }

    monkeypatch.setattr(process_module, "load_config", lambda _path: config)
    monkeypatch.setattr(
        process_module,
        "process_dataset",
        lambda *_args, **_kwargs: expected,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "process_dataset.py",
            "--gripper-backend",
            "sam",
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--ui",
            "json",
        ],
    )

    process_module.main()

    captured = capsys.readouterr()
    assert json.loads(captured.out) == expected
    assert captured.err == ""


def test_main_json_mode_prints_failed_summary_before_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _cli_config(tmp_path)
    expected = {
        "passed": False,
        "gripper_backend": "sam",
        "records": [{"episode": 7, "status": "failed"}],
    }

    monkeypatch.setattr(process_module, "load_config", lambda _path: config)
    monkeypatch.setattr(
        process_module,
        "process_dataset",
        lambda *_args, **_kwargs: expected,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "process_dataset.py",
            "--gripper-backend",
            "sam",
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--ui",
            "json",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        process_module.main()

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert json.loads(captured.out) == expected
    assert captured.err == ""


@pytest.mark.parametrize(
    ("extra_args", "message"),
    (
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
        (
            ("--gripper-backend", "sam", "--source-run-dir", "source"),
            "URDF-only options",
        ),
        (
            ("--gripper-backend", "sam", "--urdf-depth-tolerance-mm", "9"),
            "URDF-only options",
        ),
        (
            (
                "--gripper-backend",
                "sam",
                "--urdf-minimum-eligible-nonempty-fraction",
                "0.8",
            ),
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
        (
            ("--gripper-backend", "urdf", "--dry-run"),
            "--dry-run.*--source-run-dir",
        ),
        (
            (
                "--gripper-backend",
                "urdf",
                "--run-id",
                "live-resume",
                "--resume",
            ),
            "--resume.*--source-run-dir",
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
        "process_live_urdf_pipeline",
        lambda **_kwargs: pytest.fail(
            "explicit --source-run-dir must preserve the derived URDF path"
        ),
    )
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

    reporter = _EpisodeRecordingUI()
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
        reporter=reporter,
    )

    assert summary["passed"] is False
    assert summary["render"] is None
    assert summary["records"][-1] == {
        "status": "render_failed",
        "gripper_backend": "urdf",
        "error": "RuntimeError: sheet generation failed",
    }
    assert Path(summary["artifact"]).is_file()
    assert reporter.finished_episodes == [(7, "completed")]


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

    reporter = _EpisodeRecordingUI()
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
        reporter=reporter,
    )

    assert summary["passed"] is False
    assert summary["render"] is None
    assert summary["records"][-1]["status"] == "canonical_validation_failed"
    assert not any(
        record.get("status") == "render_failed" for record in summary["records"]
    )
    assert reporter.finished_episodes == [(7, "canonical_validation_failed")]


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
        "annotation_mode",
        "required_object_roles",
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


class _ProtocolReceiveConnection:
    def __init__(self) -> None:
        self.messages: list[tuple[Any, ...]] = []

    def poll(self, _timeout: float = 0.0) -> bool:
        return bool(self.messages)

    def recv(self) -> tuple[Any, ...]:
        return self.messages.pop(0)

    def close(self) -> None:
        pass


class _ProtocolSendConnection:
    def __init__(self, receiver: _ProtocolReceiveConnection) -> None:
        self.receiver = receiver

    def send(self, message: tuple[Any, ...]) -> None:
        self.receiver.messages.append(message)

    def close(self) -> None:
        pass


class _ProtocolQueue:
    def __init__(self, maxsize: int) -> None:
        self.maxsize = maxsize
        self.items: list[Any] = []

    def put_nowait(self, value: Any) -> None:
        if len(self.items) >= self.maxsize:
            raise process_module.Full
        self.items.append(value)

    def close(self) -> None:
        pass

    def cancel_join_thread(self) -> None:
        pass


class _ProtocolProcess:
    def __init__(
        self,
        context: _ProtocolContext,
        *,
        kwargs: dict[str, Any],
        name: str,
    ) -> None:
        self.context = context
        self.kwargs = kwargs
        self.name = name
        self.alive = False
        self.exitcode = 0
        self.terminated = False

    def start(self) -> None:
        sender = self.kwargs["connection"]
        if self.name == "robotwin-streaming-source":
            if self.context.source_error:
                sender.send(("error", "SourceError", "boom", "source traceback"))
            elif self.context.source_hard_exit:
                return
            else:
                sender.send(("source_episode", 7, "completed"))
                sender.send(("result", {"passed": True, "run_id": "source"}))
        else:
            self.alive = True

    def is_alive(self) -> bool:
        if self.name == "robotwin-streaming-urdf" and self.alive:
            queue = self.kwargs["ready_queue"]
            if None in queue.items:
                sender = self.kwargs["connection"]
                result = None if self.context.empty_backend else {"status": "complete"}
                error = "empty backend" if result is None else None
                sender.send(("result", result, error))
                self.alive = False
        return self.alive

    def join(self, timeout: float | None = None) -> None:
        del timeout

    def terminate(self) -> None:
        self.terminated = True
        self.alive = False


class _ProtocolContext:
    def __init__(
        self,
        *,
        source_error: bool = False,
        source_hard_exit: bool = False,
        empty_backend: bool = False,
    ) -> None:
        self.source_error = source_error
        self.source_hard_exit = source_hard_exit
        self.empty_backend = empty_backend
        self.processes: list[_ProtocolProcess] = []

    def Pipe(self, duplex: bool = False) -> tuple[Any, Any]:
        assert duplex is False
        receiver = _ProtocolReceiveConnection()
        return receiver, _ProtocolSendConnection(receiver)

    def Queue(self, maxsize: int) -> _ProtocolQueue:
        return _ProtocolQueue(maxsize)

    def Process(self, *, kwargs: dict[str, Any], name: str, **_rest: Any) -> Any:
        process = _ProtocolProcess(self, kwargs=kwargs, name=name)
        self.processes.append(process)
        return process


def _run_protocol_coordinator(
    monkeypatch: pytest.MonkeyPatch,
    context: _ProtocolContext,
) -> tuple[dict[str, Any], dict[str, Any], str | None]:
    monkeypatch.setattr(process_module.mp, "get_context", lambda _method: context)
    return process_module._run_streaming_source_urdf_workers(
        SimpleNamespace(),
        dataset_root=Path("/dataset"),
        task="task",
        camera="cam_high",
        source_output_root=Path("/sources"),
        source_run_id="source",
        episode_ids=(7,),
        urdf_run_config=SimpleNamespace(episode_ids=(7,)),
        buffer_size=2,
        reporter=None,
    )


def test_streaming_coordinator_returns_both_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, backend, error = _run_protocol_coordinator(
        monkeypatch, _ProtocolContext()
    )

    assert source["passed"] is True
    assert backend == {"status": "complete"}
    assert error is None


def test_streaming_coordinator_terminates_peer_on_child_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _ProtocolContext(source_error=True)

    with pytest.raises(RuntimeError, match="SourceError: boom"):
        _run_protocol_coordinator(monkeypatch, context)

    urdf_process = next(
        process for process in context.processes if process.name == "robotwin-streaming-urdf"
    )
    assert urdf_process.terminated is True


def test_streaming_coordinator_empty_backend_result_is_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(RuntimeError, match="empty backend"):
        _run_protocol_coordinator(
            monkeypatch,
            _ProtocolContext(empty_backend=True),
        )


def test_streaming_coordinator_hard_source_exit_is_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _ProtocolContext(source_hard_exit=True)

    with pytest.raises(RuntimeError, match="source process exited without a result"):
        _run_protocol_coordinator(monkeypatch, context)

    urdf_process = next(
        process for process in context.processes if process.name == "robotwin-streaming-urdf"
    )
    assert urdf_process.terminated is True
