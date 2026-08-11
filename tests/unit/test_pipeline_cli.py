from __future__ import annotations

import json
import runpy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from robotwin_annotation_v2.adapters import ArtifactStore, Sam3Error
from robotwin_annotation_v2.models import EpisodeRef, MaskStatus

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class FakeResidentBackend:
    def __init__(self) -> None:
        self.shutdown_calls = 0

    def shutdown(self) -> None:
        self.shutdown_calls += 1


class FakeRoleResult:
    status = MaskStatus.OK

    def to_json(self) -> dict[str, str]:
        return {"role": "target", "status": "ok"}


def _batch_config(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        output_root=tmp_path / "runs",
        sam3=SimpleNamespace(checkpoint=Path("sam3.pt"), gpus=(2,)),
        dataset=SimpleNamespace(
            regression_episode_ids=(1, 2),
            task="task",
            camera="cam_high",
        ),
        mask=SimpleNamespace(qc_enabled=True),
    )


def test_run_entrypoint_calls_qwen_sam_then_gripper_with_same_run_id() -> None:
    module = runpy.run_path(str(PROJECT_ROOT / "scripts/run_target_receiver.py"))
    calls: list[tuple[str, int, str]] = []

    def fake_qwen(_config: Any, episode_index: int, run_id: str) -> None:
        calls.append(("qwen", episode_index, run_id))

    def fake_sam(_config: Any, episode_index: int, run_id: str) -> None:
        calls.append(("sam", episode_index, run_id))

    def fake_gripper(_config: Any, episode_index: int, run_id: str) -> None:
        calls.append(("gripper", episode_index, run_id))

    module["run_pipeline"].__globals__["run_qwen"] = fake_qwen
    module["run_pipeline"].__globals__["run_sam"] = fake_sam
    module["run_pipeline"].__globals__["run_gripper"] = fake_gripper

    module["run_pipeline"]("config", 7152, "test-run")

    assert calls == [
        ("qwen", 7152, "test-run"),
        ("sam", 7152, "test-run"),
        ("gripper", 7152, "test-run"),
    ]


def test_sam_batch_reuses_one_backend_for_multiple_episodes(tmp_path: Path) -> None:
    module = runpy.run_path(str(PROJECT_ROOT / "scripts/run_target_receiver.py"))
    execution_type = module["SamEpisodeExecution"]
    backend = FakeResidentBackend()
    factory_calls: list[dict[str, Any]] = []
    episode_calls: list[int] = []

    def factory(**kwargs: Any) -> FakeResidentBackend:
        factory_calls.append(kwargs)
        return backend

    def runner(
        _config: Any,
        episode_id: int,
        _run_id: str,
        received_backend: Any,
    ) -> Any:
        assert received_backend is backend
        episode_calls.append(episode_id)
        artifact_dir = tmp_path / f"episode_{episode_id}"
        return execution_type(
            SimpleNamespace(
                roles=(FakeRoleResult(), FakeRoleResult()),
                artifact_dir=str(artifact_dir),
            ),
            artifact_dir / "mask_qc.json",
        )

    module["run_sam_batch"](
        _batch_config(tmp_path),
        (1, 2),
        "batch-test",
        force=True,
        backend_factory=factory,
        episode_runner=runner,
    )

    assert len(factory_calls) == 1
    assert episode_calls == [1, 2]
    assert backend.shutdown_calls == 1
    summary = json.loads(
        (tmp_path / "runs/batch-test/sam_batch_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["resident_sam3"] is True
    assert [record["status"] for record in summary["records"]] == [
        "completed",
        "completed",
    ]


def test_sam_batch_skips_complete_episode(tmp_path: Path) -> None:
    module = runpy.run_path(str(PROJECT_ROOT / "scripts/run_target_receiver.py"))
    config = _batch_config(tmp_path)
    store = ArtifactStore(config.output_root)
    episode_dir = store.episode_dir("batch-test", EpisodeRef("task", 1, "cam_high"))
    episode_dir.mkdir(parents=True)
    (episode_dir / "masks.npz").touch()
    ArtifactStore.write_json(
        episode_dir / "run_manifest.json",
        {
            "roles": [
                {"role": "target", "status": "ok", "qc_status": "passed"},
                {"role": "receiver", "status": "ok", "qc_status": "passed"},
            ]
        },
    )
    ArtifactStore.write_json(
        episode_dir / "mask_qc.json",
        {
            "roles": {
                "target": {"status": "passed"},
                "receiver": {"status": "passed"},
            }
        },
    )
    backend = FakeResidentBackend()
    episode_calls: list[int] = []
    execution_type = module["SamEpisodeExecution"]

    def runner(_config: Any, episode_id: int, _run_id: str, _backend: Any) -> Any:
        episode_calls.append(episode_id)
        artifact_dir = tmp_path / f"episode_{episode_id}"
        return execution_type(
            SimpleNamespace(
                roles=(FakeRoleResult(), FakeRoleResult()),
                artifact_dir=str(artifact_dir),
            ),
            artifact_dir / "mask_qc.json",
        )

    module["run_sam_batch"](
        config,
        (1, 2),
        "batch-test",
        backend_factory=lambda **_kwargs: backend,
        episode_runner=runner,
    )

    assert episode_calls == [2]
    summary = json.loads(
        (tmp_path / "runs/batch-test/sam_batch_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["records"][0] == {"episode": 1, "status": "skipped_complete"}


def test_sam_batch_stops_after_fatal_cuda_error(tmp_path: Path) -> None:
    module = runpy.run_path(str(PROJECT_ROOT / "scripts/run_target_receiver.py"))
    backend = FakeResidentBackend()
    episode_calls: list[int] = []

    def runner(_config: Any, episode_id: int, _run_id: str, _backend: Any) -> Any:
        episode_calls.append(episode_id)
        raise Sam3Error("CUDA error: unspecified launch failure")

    with pytest.raises(SystemExit) as captured:
        module["run_sam_batch"](
            _batch_config(tmp_path),
            (1, 2),
            "batch-test",
            force=True,
            backend_factory=lambda **_kwargs: backend,
            episode_runner=runner,
        )

    assert captured.value.code == 3
    assert episode_calls == [1]
    assert backend.shutdown_calls == 1
    summary = json.loads(
        (tmp_path / "runs/batch-test/sam_batch_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert [record["status"] for record in summary["records"]] == [
        "failed",
        "not_run_after_fatal_cuda",
    ]


def test_gripper_completion_treats_null_stage_as_incomplete(tmp_path: Path) -> None:
    module = runpy.run_path(str(PROJECT_ROOT / "scripts/run_target_receiver.py"))
    config = _batch_config(tmp_path)
    store = ArtifactStore(config.output_root)
    ref = EpisodeRef("task", 1, "cam_high")
    episode_dir = store.episode_dir("batch-test", ref)
    episode_dir.mkdir(parents=True)
    (episode_dir / "masks.npz").touch()
    ArtifactStore.write_json(
        episode_dir / "run_manifest.json",
        {
            "algorithm": {"gripper_stage": None},
            "roles": [
                {"role": "target", "status": "ok", "qc_status": "passed"},
                {"role": "receiver", "status": "failed"},
            ],
        },
    )

    assert not module["_gripper_episode_complete"](
        config, store, "batch-test", ref
    )


def test_gripper_batch_reuses_one_backend_for_multiple_episodes(
    tmp_path: Path,
) -> None:
    module = runpy.run_path(str(PROJECT_ROOT / "scripts/run_target_receiver.py"))
    execution_type = module["GripperEpisodeExecution"]
    backend = FakeResidentBackend()
    episode_calls: list[int] = []

    def runner(
        _config: Any,
        episode_id: int,
        _run_id: str,
        received_backend: Any,
    ) -> Any:
        assert received_backend is backend
        episode_calls.append(episode_id)
        artifact_dir = tmp_path / f"episode_{episode_id}"
        return execution_type(
            mask_run=SimpleNamespace(
                roles=(FakeRoleResult(), FakeRoleResult()),
                artifact_dir=str(artifact_dir),
            ),
            active_arm="right",
            gripper_status="ok",
            selected_candidate="A",
            seed_qc_path=artifact_dir / "gripper_seed_qc.json",
        )

    module["run_gripper_batch"](
        _batch_config(tmp_path),
        (1, 2),
        "batch-test",
        force=True,
        backend_factory=lambda **_kwargs: backend,
        episode_runner=runner,
    )

    assert episode_calls == [1, 2]
    assert backend.shutdown_calls == 1
    summary = json.loads(
        (tmp_path / "runs/batch-test/gripper_batch_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert [record["status"] for record in summary["records"]] == [
        "completed",
        "completed",
    ]


def test_gripper_batch_records_missing_sam_precondition_as_failure(
    tmp_path: Path,
) -> None:
    module = runpy.run_path(str(PROJECT_ROOT / "scripts/run_target_receiver.py"))
    backend = FakeResidentBackend()

    def runner(
        _config: Any,
        _episode_id: int,
        _run_id: str,
        _backend: Any,
    ) -> Any:
        raise ValueError("required SAM artifacts are missing for gripper stage")

    with pytest.raises(SystemExit) as captured:
        module["run_gripper_batch"](
            _batch_config(tmp_path),
            (1,),
            "batch-test",
            force=True,
            backend_factory=lambda **_kwargs: backend,
            episode_runner=runner,
        )

    assert captured.value.code == 6
    summary = json.loads(
        (tmp_path / "runs/batch-test/gripper_batch_summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["records"][0]["status"] == "failed"
    assert "required SAM artifacts are missing" in summary["records"][0]["error"]
