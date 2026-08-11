from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import scripts.render_urdf_gripper_masks as render_module
from robotwin_annotation_v2.urdf_gripper_data import ActiveGripperLoop
from scripts.render_urdf_gripper_masks import (
    RunConfig,
    UrdfMaskProduct,
    collect_asset_identity,
    compose_four_channel_payload,
    decode_depth_video,
    load_four_channel_masks,
    overlay_frame,
    parse_args,
    render_episode_product,
    run_experiment,
    save_episode_artifacts,
)


def _write_masks(path: Path, *, frame_count: int = 3) -> np.ndarray:
    masks = np.zeros((4, frame_count, 2, 3), dtype=bool)
    masks[0, :, 0, 0] = True
    masks[1, :, 0, 1] = True
    masks[2, :, 1, 0] = True
    masks[3, :, 1, 1] = True
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        format_version=np.asarray("robotwin_visible_masks_v2"),
        frame_count=np.asarray(frame_count, dtype=np.int64),
        masks=masks,
        instance_names=np.asarray(
            ("target_0", "receiver_0", "gripper_left", "gripper_right")
        ),
        roles=np.asarray(("target", "receiver", "gripper", "gripper")),
        annotation_status=np.asarray(("valid", "valid", "valid", "valid")),
        qc_status=np.asarray(("passed", "passed", "passed", "passed")),
    )
    return masks


def _episode(tmp_path: Path, *, active_arm: str = "right") -> Any:
    return SimpleNamespace(
        frame_count=3,
        active_arm=active_arm,
        active_window=(1, 2),
        joint_absolute=np.zeros((3, 14), dtype=np.float64),
        paths=SimpleNamespace(
            episode_index=7152,
            parquet=tmp_path / "episode.parquet",
            rgb_video=tmp_path / "rgb.mp4",
            depth_video=tmp_path / "depth.mkv",
            sidecar=tmp_path / "sidecar.hdf5",
        ),
    )


def _derived_loop(*, active_arm: str = "right") -> ActiveGripperLoop:
    return ActiveGripperLoop(
        active_arm=active_arm,
        t_move_start=0,
        t_close_start=1,
        t_close_done=2,
        t_open_start=3,
        t_open_done=5,
    )


def _write_source_loop(
    masks_path: Path,
    *,
    dataset_root: Path | None = None,
    task: str = "move_pillbottle_pad",
    episode_index: int = 7152,
    camera: str = "cam_high",
    active_arm: str = "right",
) -> Path:
    events = _derived_loop(active_arm=active_arm)
    path = masks_path.with_name("loop.json")
    payload = {
        "format_version": "robotwin_loop_context_v1",
        "episode": {
            "task": task,
            "episode_index": episode_index,
            "episode_id": f"{episode_index:06d}",
            "camera": camera,
        },
        "task_text": "test",
        "frame_count": 6,
        "events": events.to_json(),
        "windows": {
            "loop": list(events.inclusive_window),
            "target_0": [events.t_move_start, events.t_close_done],
            "receiver_0": [events.t_close_done, events.t_open_done],
        },
        "semantic_frames": [],
        "sources": (
            {}
            if dataset_root is None
            else {
                "state": str(
                    dataset_root
                    / "data/chunk-007/episode_007152.parquet"
                ),
                "video": str(
                    dataset_root
                    / "videos/chunk-007/observation.images.cam_high/"
                    "episode_007152.mp4"
                ),
            }
        ),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_source_contract(masks_path: Path, *, dataset_root: Path) -> Path:
    """Write one complete canonical source episode used by derived-run tests."""

    source_run = masks_path.parents[3]
    episode_dir = masks_path.parent
    loop_path = _write_source_loop(masks_path, dataset_root=dataset_root)
    for instance_name in ("target_0", "receiver_0"):
        artifact = episode_dir / instance_name / "native_track.npz"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(f"{instance_name}-track".encode())
    manifest = {
        "format_version": "robotwin_mask_run_v2",
        "run_id": source_run.name,
        "episode": {
            "task": "move_pillbottle_pad",
            "episode_index": 7152,
            "episode_id": "007152",
            "camera": "cam_high",
        },
        "frame_count": 6,
        "roles": [
            {
                "role": "target",
                "status": "ok",
                "qc_status": "passed",
                "output_window": [0, 2],
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
        json.dumps(manifest), encoding="utf-8"
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
    summary = {
        "format_version": "robotwin_process_dataset_summary_v1",
        "run_id": source_run.name,
        "dataset_root": str(dataset_root.resolve()),
        "task": "move_pillbottle_pad",
        "camera": "cam_high",
        "dynamic_manifest": {
            "task": "move_pillbottle_pad",
            "camera": "cam_high",
            "dataset_root": str(dataset_root.resolve()),
            "regression_episode_ids": [7152],
            "frame_shape_hw": [2, 3],
        },
        "records": [{"episode": 7152, "status": "completed"}],
    }
    (source_run / "process_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    return loop_path


def _derived_episode(tmp_path: Path, *, active_arm: str = "right") -> Any:
    loop = _derived_loop(active_arm=active_arm)
    return SimpleNamespace(
        frame_count=6,
        active_arm=loop.active_arm,
        active_window=loop.inclusive_window,
        loop=loop,
        joint_absolute=np.zeros((6, 14), dtype=np.float64),
        paths=SimpleNamespace(
            episode_index=7152,
            parquet=tmp_path / "episode.parquet",
            rgb_video=tmp_path / "rgb.mp4",
            depth_video=tmp_path / "depth.mkv",
            sidecar=tmp_path / "sidecar.hdf5",
        ),
    )


def _write_urdf(path: Path, *, mesh_uri: str | None = None) -> None:
    mesh = (
        ""
        if mesh_uri is None
        else f'<visual><geometry><mesh filename="{mesh_uri}"/></geometry></visual>'
    )
    path.write_text(f'<robot name="test"><link name="base">{mesh}</link></robot>\n')


def _write_episode_inputs(episode: Any) -> None:
    for name in ("parquet", "sidecar", "rgb_video", "depth_video"):
        path = Path(getattr(episode.paths, name))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"test-{name}".encode())


def _product() -> UrdfMaskProduct:
    visible = np.zeros((3, 2, 3), dtype=bool)
    visible[1:, 1, 2] = True
    return UrdfMaskProduct(
        gripper_track=visible,
        rendered_amodal_track=visible.copy(),
        depth_evaluable_track=visible.copy(),
        depth_consistent_track=visible.copy(),
        frame_diagnostics=(
            {
                "frame_id": 1,
                "accepted": True,
                "visible_pixels": 1,
                "amodal_pixels": 1,
                "depth_evaluable_pixels": 1,
                "selected_q_by_joint": {"fr_joint7": 0.02, "fr_joint8": 0.021},
                "component_acceptance": {
                    "fr_link6": True,
                    "fr_link7": True,
                    "fr_link8": True,
                },
            },
            {
                "frame_id": 2,
                "accepted": True,
                "visible_pixels": 1,
                "amodal_pixels": 1,
                "depth_evaluable_pixels": 1,
                "selected_q_by_joint": {"fr_joint7": 0.022, "fr_joint8": 0.024},
                "component_acceptance": {
                    "fr_link6": True,
                    "fr_link7": True,
                    "fr_link8": True,
                },
            },
        ),
    )


def _derived_product() -> UrdfMaskProduct:
    visible = np.zeros((6, 2, 3), dtype=bool)
    visible[:, 1, 2] = True
    records = tuple(
        {
            "frame_id": frame_id,
            "accepted": True,
            "visible_pixels": 1,
            "amodal_pixels": 1,
            "depth_evaluable_pixels": 1,
            "selected_q_by_joint": {
                "fr_joint7": 0.02,
                "fr_joint8": 0.021,
            },
            "component_acceptance": {
                "fr_link6": True,
                "fr_link7": True,
                "fr_link8": True,
            },
        }
        for frame_id in range(6)
    )
    return UrdfMaskProduct(
        gripper_track=visible,
        rendered_amodal_track=visible.copy(),
        depth_evaluable_track=visible.copy(),
        depth_consistent_track=visible.copy(),
        frame_diagnostics=records,
    )


def test_parse_args_supports_single_and_batch_episode_modes(tmp_path: Path) -> None:
    common = [
        "--source-run-dir",
        str(tmp_path / "source"),
        "--urdf-path",
        str(tmp_path / "aloha.urdf"),
    ]

    single = parse_args([*common, "--episode-id", "7152", "--dry-run"])
    batch = parse_args([*common, "--episode-ids", "7152", "7157", "--dry-run"])

    assert single.episode_ids == (7152,)
    assert batch.episode_ids == (7152, 7157)
    assert single.dry_run and batch.dry_run


def test_parse_args_requires_explicit_run_id_for_resume(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="explicit --run-id"):
        parse_args(
            [
                "--source-run-dir",
                str(tmp_path / "source"),
                "--urdf-path",
                str(tmp_path / "aloha.urdf"),
                "--episode-id",
                "7152",
                "--resume",
            ]
        )


def test_dry_run_and_resume_are_mutually_exclusive(tmp_path: Path) -> None:
    common = [
        "--source-run-dir",
        str(tmp_path / "source"),
        "--urdf-path",
        str(tmp_path / "aloha.urdf"),
        "--episode-id",
        "7152",
        "--run-id",
        "resume-run",
        "--dry-run",
        "--resume",
    ]

    with pytest.raises(ValueError, match="cannot be used together"):
        parse_args(common)

    config = RunConfig(
        dataset_root=tmp_path / "dataset",
        source_run_dir=tmp_path / "source",
        output_root=tmp_path / "output",
        run_id="resume-run",
        urdf_path=tmp_path / "aloha.urdf",
        mesh_root=None,
        episode_ids=(7152,),
        dry_run=True,
        resume=True,
    )
    with pytest.raises(ValueError, match="cannot be enabled together"):
        render_module.validate_run_config(config)


def test_compose_replaces_both_old_gripper_channels_without_mutating_source(
    tmp_path: Path,
) -> None:
    path = tmp_path / "masks.npz"
    original = _write_masks(path)
    source = load_four_channel_masks(path, frame_count=3)
    replacement = np.zeros((3, 2, 3), dtype=bool)
    replacement[1:, 1, 2] = True

    payload = compose_four_channel_payload(
        source,
        active_arm="right",
        gripper_track=replacement,
    )

    np.testing.assert_array_equal(payload["masks"][0:2], original[0:2])
    assert not payload["masks"][2].any()
    np.testing.assert_array_equal(payload["masks"][3], replacement)
    np.testing.assert_array_equal(source.masks, original)
    assert payload["annotation_status"].tolist() == [
        "valid",
        "valid",
        "not_annotated",
        "valid",
    ]
    with np.load(path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["masks"], original)


def test_save_episode_artifacts_writes_standalone_and_combined_contracts(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source/masks.npz"
    _write_masks(source_path)
    source = load_four_channel_masks(source_path, frame_count=3)
    product = _product()
    visible = product.gripper_track
    output = tmp_path / "new_run/episode_007152"

    combined, diagnostics = save_episode_artifacts(
        output,
        _episode(tmp_path),
        source,
        product,
        tolerance_mm=5.0,
    )

    assert (output / "gripper_masks.npz").is_file()
    assert (output / "masks.npz").is_file()
    assert (output / "diagnostics.json").is_file()
    assert combined.annotation_status == ("valid", "valid", "not_annotated", "valid")
    np.testing.assert_array_equal(combined.masks[3], visible)
    assert diagnostics["depth_tolerance_mm"] == 5.0
    with np.load(output / "gripper_masks.npz", allow_pickle=False) as archive:
        assert archive["active_arm"].item() == "right"
        assert archive["active_window"].tolist() == [1, 2]
        np.testing.assert_array_equal(archive["gripper_track"], visible)


def test_decode_depth_video_uses_gray16le_and_exact_parquet_length(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = np.arange(2 * 2 * 3, dtype="<u2").reshape(2, 2, 3)
    observed: list[str] = []

    def fake_run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        observed.extend(command)
        return subprocess.CompletedProcess(command, 0, stdout=values.tobytes(), stderr=b"")

    monkeypatch.setattr(render_module, "_ffmpeg", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(render_module.subprocess, "run", fake_run)

    decoded = decode_depth_video(tmp_path / "depth.mkv", frame_count=2, frame_shape=(2, 3))

    np.testing.assert_array_equal(decoded, values)
    assert "gray16le" in observed
    assert observed[observed.index("-frames:v") + 1] == "2"


def test_overlay_frame_respects_invalid_channels_and_colors() -> None:
    rgb = np.full((2, 3, 3), 100, dtype=np.uint8)
    masks = np.zeros((4, 1, 2, 3), dtype=bool)
    masks[0, 0, 0, 0] = True
    masks[3, 0, 1, 2] = True

    result = overlay_frame(
        rgb,
        masks,
        ("valid", "valid", "not_annotated", "valid"),
        frame_id=0,
        alpha=1.0,
    )

    np.testing.assert_array_equal(result[0, 0], (36, 180, 92))
    np.testing.assert_array_equal(result[1, 2], (232, 67, 55))
    np.testing.assert_array_equal(result[0, 2], (100, 100, 100))


def test_dry_run_preflights_without_creating_output_or_renderer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    source_run = tmp_path / "source_run"
    output_root = tmp_path / "artifacts"
    dataset.mkdir()
    source_run.mkdir()
    urdf = tmp_path / "aloha.urdf"
    _write_urdf(urdf)
    source_masks = (
        source_run
        / "move_pillbottle_pad/episode_007152/cam_high/masks.npz"
    )
    _write_masks(source_masks, frame_count=6)
    source_loop = _write_source_contract(source_masks, dataset_root=dataset)
    fake_episode = _derived_episode(dataset)
    _write_episode_inputs(fake_episode)
    monkeypatch.setattr(
        render_module,
        "load_urdf_gripper_episode",
        lambda *_args, **_kwargs: fake_episode,
    )
    config = RunConfig(
        dataset_root=dataset,
        source_run_dir=source_run,
        output_root=output_root,
        run_id="dry-test",
        urdf_path=urdf,
        mesh_root=None,
        episode_ids=(7152,),
        dry_run=True,
    )

    result = run_experiment(config)

    assert result["dry_run"] is True
    assert result["episode_count"] == 1
    episode_contract = result["run_contract"]["episode_plans"][0]
    assert set(episode_contract["inputs"]) == {
        "source_masks",
        "source_loop",
        "parquet",
        "sidecar",
        "rgb_video",
        "depth_video",
    }
    for identity in episode_contract["inputs"].values():
        assert set(identity) == {"path", "sha256", "bytes"}
        assert identity["bytes"] > 0
    assert episode_contract["inputs"]["source_loop"]["sha256"] == hashlib.sha256(
        source_loop.read_bytes()
    ).hexdigest()
    assert episode_contract["events"] == _derived_loop().to_json()
    assert episode_contract["source_lineage"]["format_version"] == (
        "robotwin_derivation_source_lineage_v1"
    )
    assert len(episode_contract["source_lineage"]["lineage_sha256"]) == 64
    implementation_paths = {
        item["path"]
        for item in result["run_contract"]["implementation"]["files"]
    }
    assert "src/robotwin_annotation_v2/urdf_gripper_publisher.py" in (
        implementation_paths
    )
    assert result["run_contract"]["minimum_eligible_nonempty_fraction"] == 0.90
    assert not output_root.exists()


def test_run_refuses_existing_output_before_rendering(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    source_run = tmp_path / "source"
    output_root = tmp_path / "output"
    dataset.mkdir()
    source_run.mkdir()
    urdf = tmp_path / "aloha.urdf"
    urdf.touch()
    (output_root / "same-run").mkdir(parents=True)
    config = RunConfig(
        dataset_root=dataset,
        source_run_dir=source_run,
        output_root=output_root,
        run_id="same-run",
        urdf_path=urdf,
        mesh_root=None,
        episode_ids=(7152,),
        dry_run=True,
    )

    with pytest.raises(FileExistsError, match="output run already exists"):
        run_experiment(config)


def test_source_masks_remain_byte_identical_after_composition(tmp_path: Path) -> None:
    source_path = tmp_path / "source.npz"
    _write_masks(source_path)
    before = hashlib.sha256(source_path.read_bytes()).hexdigest()
    source = load_four_channel_masks(source_path, frame_count=3)
    replacement = np.zeros((3, 2, 3), dtype=bool)

    compose_four_channel_payload(
        source,
        active_arm="left",
        gripper_track=replacement,
    )

    after = hashlib.sha256(source_path.read_bytes()).hexdigest()
    assert after == before


def _integration_config(tmp_path: Path, *, run_id: str = "resume-run") -> RunConfig:
    dataset = tmp_path / "dataset"
    source_run = tmp_path / "source"
    output_root = tmp_path / "output"
    dataset.mkdir(exist_ok=True)
    source_run.mkdir(exist_ok=True)
    urdf = tmp_path / "aloha.urdf"
    _write_urdf(urdf)
    source_masks = (
        source_run / "move_pillbottle_pad/episode_007152/cam_high/masks.npz"
    )
    _write_masks(source_masks, frame_count=6)
    _write_source_contract(source_masks, dataset_root=dataset)
    return RunConfig(
        dataset_root=dataset,
        source_run_dir=source_run,
        output_root=output_root,
        run_id=run_id,
        urdf_path=urdf,
        mesh_root=None,
        episode_ids=(7152,),
        skip_overlay=True,
    )


def _mock_integration_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    fail_render: bool = False,
) -> None:
    episode = _derived_episode(tmp_path)
    _write_episode_inputs(episode)
    monkeypatch.setattr(
        render_module,
        "load_urdf_gripper_episode",
        lambda *_args, **_kwargs: episode,
    )
    monkeypatch.setattr(
        render_module,
        "load_camera_calibration",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        render_module,
        "decode_depth_video",
        lambda *_args, **_kwargs: np.zeros((6, 2, 3), dtype=np.uint16),
    )

    def render(*_args: Any, **_kwargs: Any) -> UrdfMaskProduct:
        if fail_render:
            raise RuntimeError("synthetic render failure")
        return _derived_product()

    monkeypatch.setattr(render_module, "render_episode_product", render)


def test_collect_asset_identity_hashes_urdf_and_every_visual_mesh(tmp_path: Path) -> None:
    mesh = tmp_path / "meshes/gripper.dae"
    mesh.parent.mkdir()
    mesh.write_bytes(b"collada-test")
    urdf = tmp_path / "aloha.urdf"
    _write_urdf(urdf, mesh_uri="meshes/gripper.dae")

    assets = collect_asset_identity(urdf, None)

    assert assets["urdf"]["sha256"] == hashlib.sha256(urdf.read_bytes()).hexdigest()
    assert len(assets["visual_meshes"]) == 1
    assert assets["visual_meshes"][0]["sha256"] == hashlib.sha256(
        mesh.read_bytes()
    ).hexdigest()


def test_render_driver_uses_per_joint_temporal_priors_and_component_acceptance() -> None:
    calls: list[dict[str, Any]] = []
    visible = np.zeros((2, 3), dtype=bool)
    visible[1, 2] = True
    amodal = visible.copy()
    amodal[1, 1] = True
    render_depth = np.zeros((2, 3), dtype=np.float64)
    render_depth[amodal] = 100.0

    class FakeRenderer:
        def fit_finger_q(self, *_args: Any, **kwargs: Any) -> Any:
            calls.append(kwargs)
            frame = len(calls)
            acceptance = {
                "fr_link6": True,
                "fr_link7": True,
                "fr_link8": frame == 2,
            }
            return SimpleNamespace(
                accepted=all(acceptance.values()),
                selected_q_m=None,
                selected_q_by_joint={"fr_joint7": 0.01 * frame, "fr_joint8": 0.03},
                component_acceptance=acceptance,
                selected_render=SimpleNamespace(
                    active_gripper_mask=amodal.copy(),
                    active_gripper_depth_mm=render_depth.copy(),
                ),
                visible_mask=visible.copy(),
                diagnostics={"frame": frame},
            )

    episode = _episode(Path("/tmp"))
    calibration = SimpleNamespace(
        intrinsic_cv=np.repeat(np.eye(3)[None, ...], 3, axis=0),
        cam2world_gl=np.repeat(np.eye(4)[None, ...], 3, axis=0),
    )

    product = render_episode_product(
        FakeRenderer(),
        {},
        episode,
        calibration,
        np.full((3, 2, 3), 100, dtype=np.uint16),
        frame_shape=(2, 3),
        tolerance_mm=8.0,
    )

    assert calls[0]["temporal_prior_q_by_joint"] is None
    assert calls[1]["temporal_prior_q_by_joint"] == {"fr_joint7": 0.01}
    assert product.frame_diagnostics[0]["component_acceptance"]["fr_link8"] is False
    assert product.frame_diagnostics[0]["depth_evaluable_pixels"] == 2
    assert product.depth_consistent_track[1, 1, 1]
    assert not product.gripper_track[1, 1, 1]


def test_derived_runner_passes_source_loop_to_plan_and_actual_render_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _integration_config(tmp_path, run_id="authoritative-loop")
    _mock_integration_pipeline(monkeypatch, tmp_path)
    episode = _derived_episode(tmp_path)
    observed: list[ActiveGripperLoop | None] = []

    def load_episode(
        *_args: Any,
        authoritative_loop: ActiveGripperLoop | None = None,
        **_kwargs: Any,
    ) -> Any:
        observed.append(authoritative_loop)
        return episode

    monkeypatch.setattr(render_module, "load_urdf_gripper_episode", load_episode)

    result = run_experiment(config, renderer=object(), fit_config={})

    assert result["status"] == "complete"
    assert observed == [_derived_loop(), _derived_loop()]
    assert result["run_contract"]["episode_plans"][0]["events"] == (
        _derived_loop().to_json()
    )


def test_resume_skips_only_fully_validated_episode_without_creating_renderer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _integration_config(tmp_path)
    _mock_integration_pipeline(monkeypatch, tmp_path)
    first = run_experiment(config, renderer=object(), fit_config={})
    assert first["status"] == "complete"
    resumed = RunConfig(**{**config.__dict__, "resume": True})
    monkeypatch.setattr(
        render_module,
        "create_renderer",
        lambda *_args, **_kwargs: pytest.fail("resume should not create a renderer"),
    )

    result = run_experiment(resumed, fit_config={})

    assert result["status"] == "complete"
    assert result["resume_skipped_episode_count"] == 1


@pytest.mark.parametrize("tamper", ("summary", "loop", "role_artifact"))
def test_resume_rejects_changed_source_lineage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    config = _integration_config(tmp_path, run_id=f"source-change-{tamper}")
    _mock_integration_pipeline(monkeypatch, tmp_path)
    run_experiment(config, renderer=object(), fit_config={})
    episode_dir = (
        config.source_run_dir
        / "move_pillbottle_pad/episode_007152/cam_high"
    )
    if tamper == "summary":
        path = config.source_run_dir / "process_summary.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["test_note"] = "changed"
        path.write_text(json.dumps(payload), encoding="utf-8")
    elif tamper == "loop":
        path = episode_dir / "loop.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["task_text"] = "changed"
        path.write_text(json.dumps(payload), encoding="utf-8")
    else:
        path = episode_dir / "target_0/native_track.npz"
        path.write_bytes(path.read_bytes() + b"-changed")

    resumed = RunConfig(**{**config.__dict__, "resume": True})
    monkeypatch.setattr(
        render_module,
        "create_renderer",
        lambda *_args, **_kwargs: pytest.fail(
            "changed source lineage must fail before renderer creation"
        ),
    )

    with pytest.raises(
        render_module.UrdfMaskRunError,
        match="immutable run contract",
    ):
        run_experiment(resumed, fit_config={})


def test_resume_refuses_to_overwrite_incomplete_published_episode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _integration_config(tmp_path)
    _mock_integration_pipeline(monkeypatch, tmp_path)
    run_experiment(config, renderer=object(), fit_config={})
    masks_path = config.run_dir / "episode_007152/masks.npz"
    masks_path.unlink()
    resumed = RunConfig(**{**config.__dict__, "resume": True})
    monkeypatch.setattr(
        render_module,
        "create_renderer",
        lambda *_args, **_kwargs: pytest.fail("invalid formal output must not be rerendered"),
    )

    with pytest.raises(render_module.UrdfMaskRunError, match="immutable resume validation"):
        run_experiment(resumed, fit_config={})

    assert not masks_path.exists()
    manifest = render_module._load_json_object(
        config.run_dir / "manifest.json", description="test manifest"
    )
    assert manifest["episodes"][0]["status"] == "complete"
    assert manifest["failure_attempt_count"] == 0


def test_failed_episode_is_checkpointed_and_temporary_directory_is_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _integration_config(tmp_path, run_id="failed-run")
    _mock_integration_pipeline(monkeypatch, tmp_path, fail_render=True)

    with pytest.raises(
        render_module.UrdfBatchIncompleteError,
        match="failed for episodes",
    ) as caught:
        run_experiment(config, renderer=object(), fit_config={})

    assert caught.value.result["status"] == "failed"

    manifest = render_module._load_json_object(
        config.run_dir / "manifest.json", description="test manifest"
    )
    assert manifest["status"] == "failed"
    assert manifest["episodes"][0]["error"] == "synthetic render failure"
    assert not (config.run_dir / "episode_007152").exists()
    assert list(config.run_dir.glob(".episode_*.tmp")) == []


def test_runner_closes_only_renderer_it_creates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _integration_config(tmp_path, run_id="owned-renderer")
    _mock_integration_pipeline(monkeypatch, tmp_path)

    class OwnedRenderer:
        closed = False

        def close(self) -> None:
            self.closed = True

    owned = OwnedRenderer()
    monkeypatch.setattr(render_module, "create_renderer", lambda *_args: (owned, {}))

    run_experiment(config, fit_config={})

    assert owned.closed is True

    injected = OwnedRenderer()
    injected_config = RunConfig(
        **{**config.__dict__, "run_id": "injected-renderer"}
    )
    run_experiment(injected_config, renderer=injected, fit_config={})

    assert injected.closed is False


def test_quality_gate_rejects_too_many_empty_eligible_frames(tmp_path: Path) -> None:
    source_path = tmp_path / "source/masks.npz"
    _write_masks(source_path)
    source = load_four_channel_masks(source_path, frame_count=3)
    base = _product()
    visible = base.gripper_track.copy()
    visible[2] = False
    records = [dict(record) for record in base.frame_diagnostics]
    records[1]["visible_pixels"] = 0
    product = UrdfMaskProduct(
        gripper_track=visible,
        rendered_amodal_track=base.rendered_amodal_track,
        depth_evaluable_track=base.depth_evaluable_track,
        depth_consistent_track=base.depth_consistent_track,
        frame_diagnostics=tuple(records),
    )

    with pytest.raises(render_module.UrdfMaskRunError, match="below the run threshold"):
        save_episode_artifacts(
            tmp_path / "output",
            _episode(tmp_path),
            source,
            product,
            tolerance_mm=8.0,
            minimum_eligible_nonempty_fraction=0.90,
        )


def test_resume_complete_anchor_remains_fatal_across_repeated_attempts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _integration_config(tmp_path, run_id="missing-complete")
    _mock_integration_pipeline(monkeypatch, tmp_path)
    run_experiment(config, renderer=object(), fit_config={})
    output_dir = config.run_dir / "episode_007152"
    detached = tmp_path / "detached-episode"
    output_dir.rename(detached)
    resumed = RunConfig(**{**config.__dict__, "resume": True})
    monkeypatch.setattr(
        render_module,
        "create_renderer",
        lambda *_args, **_kwargs: pytest.fail("complete episode must never rerender"),
    )

    for _attempt in range(2):
        with pytest.raises(
            render_module.UrdfMaskRunError,
            match="published directory is missing",
        ):
            run_experiment(resumed, fit_config={})
        manifest = render_module._load_json_object(
            config.run_dir / "manifest.json", description="test manifest"
        )
        assert manifest["episodes"][0]["status"] == "complete"


def test_resume_complete_artifact_identity_change_is_fatal_and_preserves_anchor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _integration_config(tmp_path, run_id="changed-anchor")
    _mock_integration_pipeline(monkeypatch, tmp_path)
    run_experiment(config, renderer=object(), fit_config={})
    manifest_path = config.run_dir / "manifest.json"
    manifest = render_module._load_json_object(
        manifest_path, description="test manifest"
    )
    manifest["episodes"][0]["artifacts"]["masks"]["sha256"] = "0" * 64
    render_module._atomic_write_json(manifest_path, manifest)
    resumed = RunConfig(**{**config.__dict__, "resume": True})

    for _attempt in range(2):
        with pytest.raises(
            render_module.UrdfMaskRunError,
            match="immutable resume validation",
        ):
            run_experiment(resumed, fit_config={})
        preserved = render_module._load_json_object(
            manifest_path, description="test manifest"
        )
        assert preserved["episodes"][0]["status"] == "complete"
        assert preserved["episodes"][0]["artifacts"]["masks"]["sha256"] == "0" * 64


def test_resume_adopts_valid_directory_only_for_pending_crash_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _integration_config(tmp_path, run_id="crash-recovery")
    _mock_integration_pipeline(monkeypatch, tmp_path)
    run_experiment(config, renderer=object(), fit_config={})
    manifest_path = config.run_dir / "manifest.json"
    manifest = render_module._load_json_object(
        manifest_path, description="test manifest"
    )
    manifest["episodes"] = [{"episode_index": 7152, "status": "pending"}]
    render_module._atomic_write_json(manifest_path, manifest)
    resumed = RunConfig(**{**config.__dict__, "resume": True})
    monkeypatch.setattr(
        render_module,
        "create_renderer",
        lambda *_args, **_kwargs: pytest.fail("valid crash output should be adopted"),
    )

    result = run_experiment(resumed, fit_config={})

    assert result["episodes"][0]["resume_action"] == "crash_recovered"
    assert result["episodes"][0]["status"] == "complete"


def test_resume_validation_recomputes_frame_counts_and_quality(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _integration_config(tmp_path, run_id="diagnostics-cross-check")
    _mock_integration_pipeline(monkeypatch, tmp_path)
    run_experiment(config, renderer=object(), fit_config={})
    resumed = RunConfig(**{**config.__dict__, "resume": True})
    plan = render_module.build_plan(resumed)[0]
    diagnostics_path = config.run_dir / "episode_007152/diagnostics.json"
    diagnostics = render_module._load_json_object(
        diagnostics_path, description="test diagnostics"
    )
    diagnostics["frame_diagnostics"][0]["depth_evaluable_pixels"] += 1
    diagnostics["quality"]["eligible_nonempty_fraction"] = 0.0
    render_module._atomic_write_json(diagnostics_path, diagnostics)

    with pytest.raises(render_module.UrdfMaskRunError, match="diagnostics depth_evaluable_pixels"):
        render_module.validate_completed_episode(
            config.run_dir / "episode_007152",
            plan,
            resumed,
        )


def test_probe_video_reports_actual_count_shape_and_rate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "streams": [
            {
                "width": 320,
                "height": 240,
                "avg_frame_rate": "30/1",
                "nb_read_frames": "138",
            }
        ]
    }

    def fake_run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        assert "-count_frames" in command
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=render_module.json.dumps(payload).encode(),
            stderr=b"",
        )

    monkeypatch.setattr(render_module, "_ffprobe", lambda: "/usr/bin/ffprobe")
    monkeypatch.setattr(render_module.subprocess, "run", fake_run)

    probe = render_module._probe_video(tmp_path / "overlay.mp4")

    assert probe == {
        "frame_count": 138,
        "frame_shape_hw": [240, 320],
        "frame_rate": "30/1",
    }
