from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import robotwin_annotation_v2.application.dataset_pipeline as pipeline_module
from robotwin_annotation_v2.application.dataset_pipeline import DatasetPipeline
from robotwin_annotation_v2.application.sam_workflow import SamWorkflowHooks
from robotwin_annotation_v2.config import PipelineConfig
from robotwin_annotation_v2.domain import GripperBackend
from robotwin_annotation_v2.models import ProcessRequest
from robotwin_annotation_v2.terminal_ui import ProcessUI


def _coordinator_config(
    default_backend: GripperBackend = GripperBackend.URDF,
) -> PipelineConfig:
    return cast(
        PipelineConfig,
        SimpleNamespace(
            annotation=SimpleNamespace(
                spec=SimpleNamespace(default_gripper_backend=default_backend)
            )
        ),
    )


def _process_request(tmp_path: Path) -> ProcessRequest:
    return ProcessRequest(
        dataset_root=tmp_path / "dataset",
        output_root=tmp_path / "output",
        task="configured-task",
        camera="cam_high",
        run_id="run-1",
        episode_ids=(7,),
        skip_render=True,
    )


def test_dataset_pipeline_uses_annotation_default_urdf_runner(tmp_path: Path) -> None:
    request = _process_request(tmp_path)
    reporter = cast(ProcessUI, object())
    observed: list[tuple[ProcessRequest, ProcessUI | None]] = []

    def run_urdf(
        received: ProcessRequest,
        *,
        reporter: ProcessUI | None = None,
    ) -> dict[str, Any]:
        observed.append((received, reporter))
        return {"gripper_backend": "urdf", "passed": True}

    result = DatasetPipeline(
        _coordinator_config(),
        urdf_runner=run_urdf,
    ).run(request, reporter=reporter)

    assert result == {"gripper_backend": "urdf", "passed": True}
    assert observed == [(request, reporter)]
    assert observed[0][0] is request
    assert observed[0][1] is reporter


def test_dataset_pipeline_explicit_sam_overrides_default_backend(tmp_path: Path) -> None:
    request = _process_request(tmp_path)
    calls: list[ProcessRequest] = []

    def run_sam(
        received: ProcessRequest,
        *,
        reporter: ProcessUI | None = None,
    ) -> dict[str, Any]:
        assert reporter is None
        calls.append(received)
        return {"gripper_backend": "sam", "passed": True}

    result = DatasetPipeline(
        _coordinator_config(),
        sam_runner=run_sam,
    ).run(request, backend=GripperBackend.SAM)

    assert result == {"gripper_backend": "sam", "passed": True}
    assert calls == [request]
    assert calls[0] is request


def test_dataset_pipeline_missing_selected_runner_fails_closed(tmp_path: Path) -> None:
    sam_calls = 0

    def run_sam(
        _request: ProcessRequest,
        *,
        reporter: ProcessUI | None = None,
    ) -> dict[str, Any]:
        del reporter
        nonlocal sam_calls
        sam_calls += 1
        return {"passed": True}

    pipeline = DatasetPipeline(_coordinator_config(), sam_runner=run_sam)

    with pytest.raises(
        RuntimeError,
        match="no dataset runner configured for 'urdf' backend",
    ):
        pipeline.run(_process_request(tmp_path))

    assert sam_calls == 0


def test_dataset_pipeline_never_calls_unselected_runner(tmp_path: Path) -> None:
    request = _process_request(tmp_path)
    sam_calls = 0
    urdf_calls = 0

    def run_sam(
        _request: ProcessRequest,
        *,
        reporter: ProcessUI | None = None,
    ) -> dict[str, Any]:
        del reporter
        nonlocal sam_calls
        sam_calls += 1
        return {"passed": True}

    def run_urdf(
        _request: ProcessRequest,
        *,
        reporter: ProcessUI | None = None,
    ) -> dict[str, Any]:
        del reporter
        nonlocal urdf_calls
        urdf_calls += 1
        return {"passed": True}

    DatasetPipeline(
        _coordinator_config(),
        sam_runner=run_sam,
        urdf_runner=run_urdf,
    ).run(request, backend=GripperBackend.SAM)

    assert sam_calls == 1
    assert urdf_calls == 0


def test_dataset_pipeline_executes_injected_sam_workflow_directly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    config = cast(
        PipelineConfig,
        SimpleNamespace(dataset=SimpleNamespace(task="configured-task", camera="cam_high")),
    )
    hooks = cast(SamWorkflowHooks[Any, Any, Any], object())

    class RecordingWorkflow:
        def __init__(self, observed_config: Any, observed_hooks: Any) -> None:
            assert observed_config is config
            assert observed_hooks is hooks

        def run(self, **kwargs: Any) -> dict[str, Any]:
            calls.append(kwargs)
            return {"call": len(calls)}

    monkeypatch.setattr(pipeline_module, "SamWorkflow", RecordingWorkflow)
    pipeline = DatasetPipeline(config, sam_hooks=hooks)

    source = pipeline.run_object_source(
        dataset_root=tmp_path / "dataset",
        output_root=tmp_path / "output",
        incremental=True,
    )
    full = pipeline.run_sam_dataset(
        dataset_root=tmp_path / "dataset",
        output_root=tmp_path / "output",
        skip_render=True,
    )

    assert source == {"call": 1}
    assert full == {"call": 2}
    assert calls[0]["task"] == "configured-task"
    assert calls[0]["camera"] == "cam_high"
    assert calls[0]["object_source_only"] is True
    assert calls[0]["incremental_source"] is True
    assert calls[1]["object_source_only"] is False
    assert calls[1]["skip_render"] is True


def test_dataset_pipeline_run_sam_allows_task_camera_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}
    config = cast(
        PipelineConfig,
        SimpleNamespace(dataset=SimpleNamespace(task="configured-task", camera="cam_high")),
    )
    hooks = cast(SamWorkflowHooks[Any, Any, Any], object())

    class RecordingWorkflow:
        def __init__(self, _config: Any, _hooks: Any) -> None:
            pass

        def run(self, **kwargs: Any) -> dict[str, Any]:
            observed.update(kwargs)
            return {"passed": True}

    monkeypatch.setattr(pipeline_module, "SamWorkflow", RecordingWorkflow)

    result = DatasetPipeline(config, sam_hooks=hooks).run_sam(
        dataset_root=tmp_path / "dataset",
        output_root=tmp_path / "output",
        task="runtime-task",
        camera="runtime-camera",
        object_source_only=True,
    )

    assert result == {"passed": True}
    assert observed["task"] == "runtime-task"
    assert observed["camera"] == "runtime-camera"
    assert observed["object_source_only"] is True
    assert "target_receiver_only" not in observed


def test_deprecated_source_only_alias_is_not_in_canonical_pipeline_api() -> None:
    assert "target_receiver_only" not in inspect.signature(
        DatasetPipeline.run_sam
    ).parameters
    assert "target_receiver_only" not in inspect.signature(
        pipeline_module.SamWorkflow.run
    ).parameters
