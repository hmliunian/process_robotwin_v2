from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from robotwin_annotation_v2.urdf_gripper_publisher import (
    UrdfGripperPublishError,
    publish_urdf_episode,
    publisher_implementation_identity,
    validate_derivation_source_episode,
    validate_published_urdf_episode,
    validate_source_episode_completion_receipt,
    validate_source_run_contract,
    write_source_episode_completion_receipt,
    write_source_run_contract,
)

TASK = "place_empty_cup"
CAMERA = "cam_high"
EPISODE_INDEX = 7
RUN_ID = "canonical-urdf-run"
SOURCE_RUN_ID = "frozen-source-run"
FRAME_COUNT = 4
OLD_GRIPPER_MARKER = "stale-sam-gripper-metadata-must-not-survive"
MASK_KEYS = {
    "format_version",
    "frame_count",
    "masks",
    "instance_names",
    "roles",
    "annotation_status",
    "qc_status",
}


@dataclass(frozen=True)
class PublishFixture:
    source_episode_dir: Path
    backend_episode_dir: Path
    destination_dir: Path
    backend_episode_record: dict[str, Any]
    source_masks: np.ndarray
    urdf_track: np.ndarray

    def publish(self, *, resume: bool = False) -> dict[str, Any]:
        return publish_urdf_episode(
            self.source_episode_dir,
            self.backend_episode_dir,
            self.destination_dir,
            run_id=RUN_ID,
            task=TASK,
            camera=CAMERA,
            backend_episode_record=self.backend_episode_record,
            resume=resume,
        )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _identity(path: Path, *, relative_path: str | None = None) -> dict[str, Any]:
    return {
        "path": path.name if relative_path is None else relative_path,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def _source_role(role: str, window: tuple[int, int]) -> dict[str, Any]:
    directory = f"{role}_0"
    return {
        "role": role,
        "status": "ok",
        "seed_frame_id": 0,
        "primary_query": role,
        "output_window": list(window),
        "seed_rgb_path": f"{directory}/seed.rgb.png",
        "seed_mask_path": f"{directory}/seed.mask.png",
        "canonical_envelope_path": f"{directory}/canonical_envelope.png",
        "native_track_path": f"{directory}/native_track.npz",
        "temporal_qc_path": f"{directory}/temporal_qc.json",
        "nonempty_frames": window[1] - window[0] + 1,
        "failure": None,
        "qc_status": "passed",
        "qc_selected_candidate": "A",
        "qc_reason": f"{role} source QC passed",
    }


def _write_source_episode(source: Path, *, target_only: bool = False) -> np.ndarray:
    source.mkdir(parents=True)
    masks = np.zeros((4, FRAME_COUNT, 2, 3), dtype=bool)
    masks[0, :, 0, 0] = True
    if not target_only:
        masks[1, :, 0, 1] = True
    # Both old visual-gripper channels are deliberately populated. Publishing
    # must never leak either of them into the canonical URDF result.
    masks[2, :, 1, 0] = True
    masks[3, :, 1, 1] = True
    np.savez_compressed(
        source / "masks.npz",
        format_version=np.asarray("robotwin_visible_masks_v2"),
        frame_count=np.asarray(FRAME_COUNT, dtype=np.int64),
        masks=masks,
        instance_names=np.asarray(
            ("target_0", "receiver_0", "gripper_left", "gripper_right")
        ),
        roles=np.asarray(("target", "receiver", "gripper", "gripper")),
        annotation_status=np.asarray(
            (
                "valid",
                "not_applicable" if target_only else "valid",
                "valid",
                "valid",
            )
        ),
        qc_status=np.asarray(
            (
                "passed",
                "not_applicable" if target_only else "passed",
                "passed",
                "passed",
            )
        ),
    )

    object_directories = ("target_0",) if target_only else ("target_0", "receiver_0")
    for role in object_directories:
        role_dir = source / role
        role_dir.mkdir()
        (role_dir / "seed.rgb.png").write_bytes(f"{role}-rgb".encode())
        (role_dir / "seed.mask.png").write_bytes(f"{role}-mask".encode())
        (role_dir / "canonical_envelope.png").write_bytes(
            f"{role}-envelope".encode()
        )
        np.savez_compressed(
            role_dir / "native_track.npz",
            masks=np.zeros((FRAME_COUNT, 2, 3), dtype=bool),
        )
        _write_json(role_dir / "temporal_qc.json", {"status": "passed"})

    common_files = {
        "semantic_plan.json": b'{"format_version":"robotwin_semantic_plan_v1"}\n',
        "mask_qc.json": b'{"format_version":"robotwin_mask_qc_v1"}\n',
        "qwen_rendered_prompt.txt": b"source prompt\n",
        "qwen_raw_response.txt": b"source response\n",
    }
    for relative, content in common_files.items():
        (source / relative).write_bytes(content)

    source_run = source.parents[2]
    dataset_root = source_run.parent / "dataset"
    chunk = "chunk-000"
    state_path = dataset_root / f"data/{chunk}/episode_000007.parquet"
    video_path = (
        dataset_root
        / f"videos/{chunk}/observation.images.{CAMERA}/episode_000007.mp4"
    )
    for path in (state_path, video_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    loop_contract = (
        {
            "format_version": "robotwin_loop_context_v2",
            "annotation_mode": "target_only",
            "timeline_kind": "close_hold",
            "required_object_roles": ["target"],
            "events": {
                "active_arm": "right",
                "t_remove_start": 0,
                "t_close_start": 0,
                "t_close_end": 1,
            },
            "windows": {
                "operation": [0, 3],
                "target_0": [0, 1],
                "receiver_0": None,
                "gripper": [0, 3],
            },
        }
        if target_only
        else {
            "format_version": "robotwin_loop_context_v1",
            "events": {
                "active_arm": "right",
                "t_move_start": 0,
                "t_close_start": 0,
                "t_close_done": 1,
                "t_open_start": 2,
                "t_open_done": 3,
            },
            "windows": {
                "loop": [0, 3],
                "target_0": [0, 1],
                "receiver_0": [1, 3],
            },
        }
    )
    _write_json(
        source / "loop.json",
        {
            **loop_contract,
            "episode": {
                "task": TASK,
                "episode_index": EPISODE_INDEX,
                "episode_id": f"{EPISODE_INDEX:06d}",
                "camera": CAMERA,
            },
            "task_text": "place the empty cup",
            "frame_count": FRAME_COUNT,
            "semantic_frames": [],
            "sources": {
                "state": str(state_path.resolve()),
                "video": str(video_path.resolve()),
            },
        },
    )

    # These are excluded source artifacts. The active directory name may be
    # reused, but only for newly generated URDF files.
    (source / "gripper_left").mkdir()
    (source / "gripper_left" / "old_sam_track.npz").write_bytes(b"old-left")
    (source / "gripper_right").mkdir()
    _write_json(
        source / "gripper_right" / "gripper_seed_qc.json",
        {"status": "passed", "marker": OLD_GRIPPER_MARKER},
    )
    (source / "gripper_right" / "native_track.npz").write_bytes(b"old-right")

    _write_json(
        source / "run_manifest.json",
        {
            "format_version": "robotwin_mask_run_v2",
            **(
                {
                    "annotation_mode": "target_only",
                    "required_object_roles": ["target"],
                }
                if target_only
                else {}
            ),
            "run_id": SOURCE_RUN_ID,
            "episode": {
                "task": TASK,
                "episode_index": EPISODE_INDEX,
                "episode_id": f"{EPISODE_INDEX:06d}",
                "camera": CAMERA,
            },
            "frame_count": FRAME_COUNT,
            "roles": [
                _source_role("target", (0, 1)),
                (
                    {
                        "role": "receiver",
                        "status": "not_applicable",
                        "seed_frame_id": None,
                        "primary_query": None,
                        "output_window": None,
                        "seed_rgb_path": None,
                        "seed_mask_path": None,
                        "canonical_envelope_path": None,
                        "native_track_path": None,
                        "temporal_qc_path": None,
                        "nonempty_frames": 0,
                        "failure": None,
                        "qc_status": "not_applicable",
                        "qc_selected_candidate": None,
                        "qc_reason": None,
                    }
                    if target_only
                    else _source_role("receiver", (1, 3))
                ),
                {
                    "role": "gripper_right",
                    "status": "ok",
                    "output_window": [0, 3],
                    "qc_status": "passed",
                    "qc_reason": OLD_GRIPPER_MARKER,
                },
            ],
            "algorithm": {
                "seed": "sam3_text_only_primary_query",
                "propagation": "sam3_native_mask_forward_backward",
                "gripper_stage": {
                    "backend": "sam",
                    "marker": OLD_GRIPPER_MARKER,
                },
                "amodal_completion": False,
            },
            "semantic_prompt_sha256": "source-semantic-hash",
            "gripper_backend": "sam",
            "gripper_qc": {"backend": "sam", "marker": OLD_GRIPPER_MARKER},
        },
    )
    _write_json(
        source / "frame_provenance.json",
        {
            "format_version": "robotwin_frame_provenance_v2",
            **(
                {
                    "annotation_mode": "target_only",
                    "required_object_roles": ["target"],
                }
                if target_only
                else {}
            ),
            "composition": "source SAM tracks",
            "channels": {
                "target_0": {
                    "status": "ok",
                    "qc_status": "passed",
                    "output_window": [0, 1],
                    "source_marker": "keep-target",
                },
                "receiver_0": {
                    **(
                        {
                            "status": "not_applicable",
                            "qc_status": "not_applicable",
                            "reason": "receiver is not required",
                            "nonempty_frame_ids": [],
                        }
                        if target_only
                        else {
                            "status": "ok",
                            "qc_status": "passed",
                            "output_window": [1, 3],
                            "source_marker": "keep-receiver",
                        }
                    ),
                },
                "gripper_left": {
                    "status": "ok",
                    "backend": "sam",
                    "marker": OLD_GRIPPER_MARKER,
                },
                "gripper_right": {
                    "status": "ok",
                    "backend": "sam",
                    "marker": OLD_GRIPPER_MARKER,
                },
            },
        },
    )
    _write_json(
        source_run / "process_summary.json",
        {
            "format_version": "robotwin_process_dataset_summary_v1",
            **(
                {
                    "annotation_mode": "target_only",
                    "required_object_roles": ["target"],
                }
                if target_only
                else {}
            ),
            "run_id": SOURCE_RUN_ID,
            "dataset_root": str(dataset_root.resolve()),
            "task": TASK,
            "camera": CAMERA,
            "dynamic_manifest": {
                "format_version": "robotwin_dataset_manifest_dynamic_v1",
                "task": TASK,
                "camera": CAMERA,
                "dataset_root": str(dataset_root.resolve()),
                "frame_shape_hw": [2, 3],
                "regression_episode_ids": [EPISODE_INDEX],
            },
            "records": [{"episode": EPISODE_INDEX, "status": "completed"}],
        },
    )
    return masks


def _write_backend_episode(
    backend: Path,
    source_masks_path: Path,
    source_masks: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    backend.mkdir(parents=True)
    track = np.zeros((FRAME_COUNT, 2, 3), dtype=bool)
    track[:, 1, 2] = True
    np.savez_compressed(
        backend / "gripper_masks.npz",
        format_version=np.asarray("robotwin_urdf_gripper_masks_v2"),
        frame_count=np.asarray(FRAME_COUNT, dtype=np.int64),
        gripper_track=track,
        rendered_amodal_track=track.copy(),
        depth_evaluable_track=track.copy(),
        depth_consistent_track=track.copy(),
        active_arm=np.asarray("right"),
        active_window=np.asarray((0, 3), dtype=np.int64),
        depth_tolerance_mm=np.asarray(8.0, dtype=np.float64),
    )
    combined = source_masks.copy()
    combined[2:] = False
    combined[3] = track
    np.savez_compressed(
        backend / "masks.npz",
        format_version=np.asarray("robotwin_visible_masks_urdf_gripper_v1"),
        frame_count=np.asarray(FRAME_COUNT, dtype=np.int64),
        masks=combined,
        instance_names=np.asarray(
            ("target_0", "receiver_0", "gripper_left", "gripper_right")
        ),
        roles=np.asarray(("target", "receiver", "gripper", "gripper")),
        annotation_status=np.asarray(
            ("valid", "valid", "not_annotated", "valid")
        ),
        qc_status=np.asarray(("passed", "passed", "not_run", "not_run")),
    )
    quality = {
        "eligible_frame_count": FRAME_COUNT,
        "eligible_nonempty_frame_count": FRAME_COUNT,
        "eligible_nonempty_fraction": 1.0,
    }
    _write_json(
        backend / "diagnostics.json",
        {
            "format_version": "robotwin_urdf_gripper_diagnostics_v2",
            "status": "complete",
            "episode_index": EPISODE_INDEX,
            "frame_count": FRAME_COUNT,
            "active_arm": "right",
            "active_window": [0, 3],
            "quality": quality,
        },
    )
    validated_source = validate_derivation_source_episode(
        source_masks_path.parent,
        task=TASK,
        camera=CAMERA,
        episode_index=EPISODE_INDEX,
        expected_frame_count=FRAME_COUNT,
    )
    record = {
        "episode_index": EPISODE_INDEX,
        "frame_count": FRAME_COUNT,
        "frame_shape_hw": [2, 3],
        "active_arm": "right",
        "active_window": [0, 3],
        "events": validated_source.loop["events"],
        "source_lineage": validated_source.lineage,
        "status": "complete",
        "output_dir": backend.name,
        "quality": quality,
        "inputs": {
            "source_masks": _identity(
                source_masks_path,
                relative_path=str(source_masks_path),
            ),
            "source_loop": _identity(
                source_masks_path.with_name("loop.json"),
                relative_path=str(source_masks_path.with_name("loop.json")),
            ),
        },
        "artifacts": {
            name: _identity(backend / filename, relative_path=filename)
            for name, filename in {
                "gripper_masks": "gripper_masks.npz",
                "masks": "masks.npz",
                "diagnostics": "diagnostics.json",
            }.items()
        },
    }
    publisher_identity = publisher_implementation_identity()
    _write_json(
        backend.parent / "manifest.json",
        {
            "format_version": "robotwin_urdf_gripper_run_v2",
            "run_id": "backend-run",
            "status": "complete",
            "run_contract": {
                "dataset_root": str(
                    validated_source.lineage["source_run"]["dataset_root"]
                ),
                "source_run_dir": str(validated_source.source_run_dir),
                "task": TASK,
                "camera": CAMERA,
                "depth_tolerance_mm": 8.0,
                "minimum_eligible_nonempty_fraction": 0.9,
                "egl_device_id": 3,
                "fit_config": {},
                "episode_plans": [
                    {
                        key: record[key]
                        for key in (
                            "episode_index",
                            "frame_count",
                            "frame_shape_hw",
                            "active_arm",
                            "active_window",
                            "events",
                            "source_lineage",
                            "inputs",
                        )
                    }
                ],
                "implementation": {
                    "git_revision": None,
                    "files": publisher_identity["files"],
                },
            },
            "assets": {"urdf": {"sha256": "test-urdf-sha"}},
            "episodes": [record],
        },
    )
    return track, record


def _fixture(tmp_path: Path) -> PublishFixture:
    source = (
        tmp_path
        / "frozen-source-run"
        / TASK
        / f"episode_{EPISODE_INDEX:06d}"
        / CAMERA
    )
    source_masks = _write_source_episode(source)
    backend = tmp_path / "backend-run" / f"episode_{EPISODE_INDEX:06d}"
    track, record = _write_backend_episode(
        backend,
        source / "masks.npz",
        source_masks,
    )
    destination = (
        tmp_path
        / "public-runs"
        / RUN_ID
        / TASK
        / f"episode_{EPISODE_INDEX:06d}"
        / CAMERA
    )
    return PublishFixture(source, backend, destination, record, source_masks, track)


def _target_only_fixture(tmp_path: Path) -> PublishFixture:
    source = (
        tmp_path
        / "frozen-source-run"
        / TASK
        / f"episode_{EPISODE_INDEX:06d}"
        / CAMERA
    )
    source_masks = _write_source_episode(source, target_only=True)
    backend = tmp_path / "backend-run" / f"episode_{EPISODE_INDEX:06d}"
    track, record = _write_backend_episode(
        backend,
        source / "masks.npz",
        source_masks,
    )
    destination = (
        tmp_path
        / "public-runs"
        / RUN_ID
        / TASK
        / f"episode_{EPISODE_INDEX:06d}"
        / CAMERA
    )
    return PublishFixture(source, backend, destination, record, source_masks, track)


def _incremental_fixture(tmp_path: Path) -> PublishFixture:
    source = (
        tmp_path
        / "frozen-source-run"
        / TASK
        / f"episode_{EPISODE_INDEX:06d}"
        / CAMERA
    )
    source_masks = _write_source_episode(source)
    source_run = source.parents[2]
    summary_path = source_run / "process_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary_path.unlink()
    summary["dynamic_manifest"]["regression_episode_ids"].append(8)
    write_source_run_contract(
        source_run,
        run_id=SOURCE_RUN_ID,
        dataset_root=Path(summary["dataset_root"]),
        task=TASK,
        camera=CAMERA,
        dynamic_manifest=summary["dynamic_manifest"],
        requested_episode_ids=[EPISODE_INDEX],
    )
    write_source_episode_completion_receipt(
        source,
        task=TASK,
        camera=CAMERA,
        episode_index=EPISODE_INDEX,
        expected_frame_count=FRAME_COUNT,
    )
    backend = tmp_path / "backend-run" / f"episode_{EPISODE_INDEX:06d}"
    track, record = _write_backend_episode(
        backend,
        source / "masks.npz",
        source_masks,
    )
    destination = (
        tmp_path
        / "public-runs"
        / RUN_ID
        / TASK
        / f"episode_{EPISODE_INDEX:06d}"
        / CAMERA
    )
    return PublishFixture(source, backend, destination, record, source_masks, track)


def test_incremental_source_can_publish_before_process_summary_exists(
    tmp_path: Path,
) -> None:
    fixture = _incremental_fixture(tmp_path)
    source_run = fixture.source_episode_dir.parents[2]

    assert not (source_run / "process_summary.json").exists()
    contract = validate_source_run_contract(
        source_run,
        run_id=SOURCE_RUN_ID,
        dataset_root=source_run.parent / "dataset",
        task=TASK,
        camera=CAMERA,
        requested_episode_ids=[EPISODE_INDEX],
    )
    receipt = validate_source_episode_completion_receipt(
        fixture.source_episode_dir,
        task=TASK,
        camera=CAMERA,
        episode_index=EPISODE_INDEX,
        expected_frame_count=FRAME_COUNT,
    )
    assert contract["requested_episode_ids"] == [EPISODE_INDEX]
    assert receipt["status"] == "completed"
    validated = validate_derivation_source_episode(
        fixture.source_episode_dir,
        task=TASK,
        camera=CAMERA,
        episode_index=EPISODE_INDEX,
        expected_frame_count=FRAME_COUNT,
    )
    assert validated.lineage["format_version"] == (
        "robotwin_derivation_source_lineage_v2"
    )
    assert validated.summary["format_version"] == "robotwin_source_run_contract_v2"
    assert validated.summary["annotation_mode"] == "pick_place"
    assert validated.summary["required_object_roles"] == ["target", "receiver"]
    assert validated.lineage["source_run"]["source_run_contract"]["path"] == (
        "source_run_contract.json"
    )
    assert validated.lineage["completion_receipt"]["path"].endswith(
        "/completion_receipt.json"
    )
    artifact_paths = {
        identity["path"] for identity in validated.lineage["episode_artifacts"]
    }
    assert any(path.endswith("/masks.npz") for path in artifact_paths)
    assert any(path.endswith("/semantic_plan.json") for path in artifact_paths)

    _write_json(source_run / "process_summary.json", {"mutable_final_summary": True})
    after_summary = validate_derivation_source_episode(
        fixture.source_episode_dir,
        task=TASK,
        camera=CAMERA,
        episode_index=EPISODE_INDEX,
    )
    assert after_summary.lineage == validated.lineage

    record = fixture.publish()
    assert record["status"] == "completed"
    assert record["source_lineage"] == validated.lineage


@pytest.mark.parametrize(
    ("target", "message"),
    (
        ("contract", "source run contract content hash is invalid"),
        ("receipt", "source completion receipt content hash is invalid"),
        ("artifact", "source episode artifacts differ from the completion receipt"),
    ),
)
def test_incremental_source_rejects_mutated_provenance_dependencies(
    tmp_path: Path,
    target: str,
    message: str,
) -> None:
    fixture = _incremental_fixture(tmp_path)
    if target == "contract":
        path = fixture.source_episode_dir.parents[2] / "source_run_contract.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["camera"] = "tampered_camera"
        _write_json(path, payload)
    elif target == "receipt":
        path = fixture.source_episode_dir / "completion_receipt.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["status"] = "skipped_complete"
        _write_json(path, payload)
    else:
        path = fixture.source_episode_dir / "semantic_plan.json"
        path.write_bytes(path.read_bytes() + b"tampered\n")

    with pytest.raises(UrdfGripperPublishError, match=message):
        validate_derivation_source_episode(
            fixture.source_episode_dir,
            task=TASK,
            camera=CAMERA,
            episode_index=EPISODE_INDEX,
        )


def test_legacy_source_without_contract_keeps_process_summary_lineage(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    validated = validate_derivation_source_episode(
        fixture.source_episode_dir,
        task=TASK,
        camera=CAMERA,
        episode_index=EPISODE_INDEX,
    )

    assert validated.lineage["format_version"] == (
        "robotwin_derivation_source_lineage_v1"
    )
    assert set(validated.lineage["source_run"]) == {
        "run_id",
        "path",
        "dataset_root",
        "process_summary",
    }
    assert validated.summary["format_version"] == (
        "robotwin_process_dataset_summary_v1"
    )


def test_publish_writes_canonical_mask_schema_and_only_replaces_gripper(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    record = fixture.publish()

    assert record["status"] == "completed"
    assert record["gripper_backend"] == "urdf"
    lineage = record["source_lineage"]
    assert record["source_lineage_sha256"] == lineage["lineage_sha256"]
    assert lineage["format_version"] == "robotwin_derivation_source_lineage_v1"
    assert set(lineage["control_artifacts"]) == {
        "loop",
        "run_manifest",
        "frame_provenance",
        "masks",
    }
    assert set(lineage["role_artifacts"]) == {"target", "receiver"}
    assert set(lineage["role_artifacts"]["target"]) == {
        "seed_rgb_path",
        "seed_mask_path",
        "canonical_envelope_path",
        "native_track_path",
        "temporal_qc_path",
    }
    identities = [
        lineage["source_run"]["process_summary"],
        *lineage["control_artifacts"].values(),
        *lineage["role_artifacts"]["target"].values(),
        *lineage["role_artifacts"]["receiver"].values(),
    ]
    assert all(set(identity) == {"path", "sha256", "bytes"} for identity in identities)
    assert all(len(identity["sha256"]) == 64 and identity["bytes"] > 0 for identity in identities)
    assert len(lineage["lineage_sha256"]) == 64
    with np.load(fixture.destination_dir / "masks.npz", allow_pickle=False) as archive:
        assert set(archive.files) == MASK_KEYS
        assert archive["format_version"].shape == ()
        assert archive["format_version"].dtype == np.dtype("<U25")
        assert archive["format_version"].item() == "robotwin_visible_masks_v2"
        assert archive["frame_count"].dtype == np.dtype(np.int64)
        assert archive["frame_count"].shape == ()
        assert archive["masks"].dtype == np.dtype(bool)
        assert archive["masks"].shape == (4, FRAME_COUNT, 2, 3)
        assert archive["instance_names"].dtype == np.dtype("<U13")
        assert archive["roles"].dtype == np.dtype("<U8")
        assert archive["annotation_status"].dtype == np.dtype("<U13")
        assert archive["qc_status"].dtype == np.dtype("<U7")
        assert archive["instance_names"].tolist() == [
            "target_0",
            "receiver_0",
            "gripper_left",
            "gripper_right",
        ]
        assert archive["roles"].tolist() == [
            "target",
            "receiver",
            "gripper",
            "gripper",
        ]
        np.testing.assert_array_equal(archive["masks"][:2], fixture.source_masks[:2])
        assert not archive["masks"][2].any()
        np.testing.assert_array_equal(archive["masks"][3], fixture.urdf_track)
        assert archive["annotation_status"].tolist() == [
            "valid",
            "valid",
            "not_annotated",
            "valid",
        ]
        assert archive["qc_status"].tolist() == [
            "passed",
            "passed",
            "not_run",
            "passed",
        ]


def test_target_only_urdf_publish_preserves_not_applicable_receiver(
    tmp_path: Path,
) -> None:
    fixture = _target_only_fixture(tmp_path)
    source_loop = json.loads(
        (fixture.source_episode_dir / "loop.json").read_text(encoding="utf-8")
    )

    fixture.publish()

    with np.load(fixture.destination_dir / "masks.npz", allow_pickle=False) as archive:
        assert not archive["masks"][1].any()
        assert archive["annotation_status"].tolist()[:2] == [
            "valid",
            "not_applicable",
        ]
        assert archive["qc_status"].tolist()[:2] == [
            "passed",
            "not_applicable",
        ]
    manifest = json.loads(
        (fixture.destination_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["annotation_mode"] == "target_only"
    assert manifest["required_object_roles"] == ["target"]
    assert source_loop["events"] == {
        "active_arm": "right",
        "t_remove_start": 0,
        "t_close_start": 0,
        "t_close_end": 1,
    }
    gripper = next(
        item for item in manifest["roles"] if item["role"] == "gripper_right"
    )
    assert gripper["output_window"] == [0, FRAME_COUNT - 1]
    receiver = next(item for item in manifest["roles"] if item["role"] == "receiver")
    assert receiver["status"] == "not_applicable"
    assert not (fixture.destination_dir / "receiver_0").exists()
    provenance = json.loads(
        (fixture.destination_dir / "frame_provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["channels"]["receiver_0"]["status"] == "not_applicable"


def test_target_only_publish_rejects_backend_window_ending_at_close_end(
    tmp_path: Path,
) -> None:
    fixture = _target_only_fixture(tmp_path)
    fixture.backend_episode_record["active_window"] = [0, 1]

    with pytest.raises(
        UrdfGripperPublishError,
        match="active window differs from the authoritative source loop",
    ):
        fixture.publish()


def test_target_only_source_rejects_nonzero_receiver_pixels(tmp_path: Path) -> None:
    fixture = _target_only_fixture(tmp_path)
    path = fixture.source_episode_dir / "masks.npz"
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]).copy() for key in archive.files}
    payload["masks"][1, 0, 0, 0] = True
    np.savez_compressed(path, **payload)

    with pytest.raises(UrdfGripperPublishError, match="receiver must be zero"):
        validate_derivation_source_episode(
            fixture.source_episode_dir,
            task=TASK,
            camera=CAMERA,
            episode_index=EPISODE_INDEX,
        )


def test_publish_preserves_source_material_and_removes_old_sam_gripper_metadata(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    fixture.publish()

    preserved = (
        "loop.json",
        "semantic_plan.json",
        "mask_qc.json",
        "qwen_rendered_prompt.txt",
        "qwen_raw_response.txt",
        "target_0/seed.rgb.png",
        "target_0/seed.mask.png",
        "target_0/canonical_envelope.png",
        "target_0/native_track.npz",
        "target_0/temporal_qc.json",
        "receiver_0/seed.rgb.png",
        "receiver_0/seed.mask.png",
        "receiver_0/canonical_envelope.png",
        "receiver_0/native_track.npz",
        "receiver_0/temporal_qc.json",
    )
    for relative in preserved:
        assert (fixture.destination_dir / relative).read_bytes() == (
            fixture.source_episode_dir / relative
        ).read_bytes()

    assert not (fixture.destination_dir / "gripper_left").exists()
    assert {path.name for path in (fixture.destination_dir / "gripper_right").iterdir()} == {
        "native_track.npz",
        "urdf_product.npz",
        "urdf_diagnostics.json",
    }
    assert not (fixture.destination_dir / "gripper_right/gripper_seed_qc.json").exists()

    manifest = json.loads(
        (fixture.destination_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["format_version"] == "robotwin_mask_run_v2"
    assert manifest["gripper_backend"] == "urdf"
    assert [role["role"] for role in manifest["roles"]] == [
        "target",
        "receiver",
        "gripper_right",
    ]
    gripper = manifest["roles"][2]
    assert gripper["status"] == "ok"
    assert gripper["qc_status"] == "passed"
    assert gripper["output_window"] == [0, 3]
    assert manifest["algorithm"]["gripper_stage"]["backend"] == "urdf"
    assert manifest["gripper_qc"]["backend"] == "urdf"
    assert manifest["gripper_qc"]["status"] == "ok"
    assert manifest["gripper_qc"]["qc_status"] == "passed"
    assert set(manifest["gripper_qc"]) == {
        "backend",
        "status",
        "qc_status",
        "active_arm",
        "selected_candidate",
        "confidence",
        "reason",
        "forced_fallback",
        "nonempty_frames",
        "quality",
    }
    derivation = manifest["derivation"]
    assert derivation["format_version"] == "robotwin_urdf_gripper_derivation_v1"
    assert derivation["source"]["lineage_sha256"] == fixture.backend_episode_record[
        "source_lineage"
    ]["lineage_sha256"]
    assert derivation["publisher"] == publisher_implementation_identity()
    backend_contract = manifest["algorithm"]["gripper_stage"]["backend_provenance"][
        "run_contract"
    ]
    assert backend_contract["egl_device_id"] == 3
    assert OLD_GRIPPER_MARKER not in json.dumps(manifest, sort_keys=True)

    provenance = json.loads(
        (fixture.destination_dir / "frame_provenance.json").read_text(
            encoding="utf-8"
        )
    )
    assert provenance["format_version"] == "robotwin_frame_provenance_v2"
    assert provenance["gripper_backend"] == "urdf"
    assert provenance["derivation"] == derivation
    assert provenance["channels"]["target_0"]["source_marker"] == "keep-target"
    assert (
        provenance["channels"]["receiver_0"]["source_marker"]
        == "keep-receiver"
    )
    assert provenance["channels"]["gripper_left"] == {"status": "not_annotated"}
    assert provenance["channels"]["gripper_right"]["backend"] == "urdf"
    assert provenance["channels"]["gripper_right"]["qc_status"] == "passed"
    assert OLD_GRIPPER_MARKER not in json.dumps(provenance, sort_keys=True)


@pytest.mark.parametrize(
    "tamper",
    ("masks", "materialized_source", "manifest", "provenance"),
)
def test_resume_validates_complete_publication_and_rejects_tampering(
    tmp_path: Path,
    tamper: str,
) -> None:
    fixture = _fixture(tmp_path)
    fixture.publish()

    validated = validate_published_urdf_episode(
        fixture.source_episode_dir,
        fixture.backend_episode_dir,
        fixture.destination_dir,
        run_id=RUN_ID,
        task=TASK,
        camera=CAMERA,
        backend_episode_record=fixture.backend_episode_record,
    )
    resumed = fixture.publish(resume=True)
    assert validated["status"] == "completed"
    assert resumed["status"] == "skipped_complete"

    if tamper == "masks":
        masks_path = fixture.destination_dir / "masks.npz"
        with np.load(masks_path, allow_pickle=False) as archive:
            payload = {key: np.asarray(archive[key]).copy() for key in archive.files}
        payload["masks"][3, 1, 1, 2] = False
        np.savez_compressed(masks_path, **payload)
        message = "public masks payload differs"
    elif tamper == "materialized_source":
        source_loop_before = (fixture.source_episode_dir / "loop.json").read_bytes()
        replacement = fixture.destination_dir / ".tampered-loop.json"
        replacement.write_text('{"tampered":true}\n', encoding="utf-8")
        replacement.replace(fixture.destination_dir / "loop.json")
        assert (fixture.source_episode_dir / "loop.json").read_bytes() == source_loop_before
        message = "materialized source (size|hash) differs"
    else:
        path = fixture.destination_dir / (
            "run_manifest.json" if tamper == "manifest" else "frame_provenance.json"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["tampered"] = True
        path.write_text(json.dumps(payload), encoding="utf-8")
        message = "published (run manifest|frame provenance) differs"

    with pytest.raises(UrdfGripperPublishError, match=message):
        fixture.publish(resume=True)


@pytest.mark.parametrize(
    "source_target",
    (
        "process_summary",
        "loop",
        "run_manifest",
        "frame_provenance",
        "masks",
        "role_artifact",
    ),
)
def test_publish_rejects_any_changed_inherited_source_lineage(
    tmp_path: Path,
    source_target: str,
) -> None:
    fixture = _fixture(tmp_path)
    if source_target == "process_summary":
        path = fixture.source_episode_dir.parents[2] / "process_summary.json"
    elif source_target in {"loop", "run_manifest", "frame_provenance"}:
        path = fixture.source_episode_dir / f"{source_target}.json"
    elif source_target == "masks":
        path = fixture.source_episode_dir / "masks.npz"
        with np.load(path, allow_pickle=False) as archive:
            payload = {key: np.asarray(archive[key]).copy() for key in archive.files}
        payload["masks"][0, 0, 0, 0] = False
        np.savez_compressed(path, **payload)
        path = None
    else:
        path = fixture.source_episode_dir / "target_0/native_track.npz"
        path.write_bytes(path.read_bytes() + b"-changed")
        path = None
    if path is not None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["test_note"] = "changed"
        path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        UrdfGripperPublishError,
        match="backend source lineage differs",
    ):
        fixture.publish()
    assert not fixture.destination_dir.exists()


@pytest.mark.parametrize("mask_contract", ("extra_key", "format_version"))
def test_publish_rejects_noncanonical_source_mask_contract(
    tmp_path: Path,
    mask_contract: str,
) -> None:
    fixture = _fixture(tmp_path)
    path = fixture.source_episode_dir / "masks.npz"
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: np.asarray(archive[key]).copy() for key in archive.files}
    if mask_contract == "extra_key":
        payload["unexpected"] = np.asarray(1, dtype=np.int64)
        message = "exactly the canonical seven keys"
    else:
        payload["format_version"] = np.asarray("unsupported")
        message = "unsupported source masks format"
    np.savez_compressed(path, **payload)

    with pytest.raises(UrdfGripperPublishError, match=message):
        fixture.publish()
    assert not fixture.destination_dir.exists()


def test_publish_rejects_backend_recorded_lineage_mismatch(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture.backend_episode_record["source_lineage"]["lineage_sha256"] = "0" * 64

    with pytest.raises(
        UrdfGripperPublishError,
        match="backend source lineage differs",
    ):
        fixture.publish()
    assert not fixture.destination_dir.exists()


def test_resume_rejects_backend_publisher_identity_change(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture.publish()
    manifest_path = fixture.backend_episode_dir.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    publisher_file = next(
        item
        for item in manifest["run_contract"]["implementation"]["files"]
        if item["path"] == "src/robotwin_annotation_v2/urdf_gripper_publisher.py"
    )
    publisher_file["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        UrdfGripperPublishError,
        match="publisher identity differs",
    ):
        fixture.publish(resume=True)
    assert fixture.destination_dir.is_dir()
