from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from robotwin_annotation_v2.application import streaming, urdf_runtime
from robotwin_annotation_v2.application.sam_workflow import (
    PROCESS_SUMMARY_FORMAT_VERSION,
)
from robotwin_annotation_v2.domain import AnnotationMode, annotation_spec
from robotwin_annotation_v2.terminal_ui import ProcessUI
from robotwin_annotation_v2.urdf_gripper_publisher import UrdfGripperPublishError


class RecordingUI(ProcessUI):
    def __init__(self) -> None:
        super().__init__(emit_json_summary=False, verbose=True)
        self.progress: list[tuple[int, int | None, int | None, str | None]] = []
        self.details: list[str] = []

    def phase_progress(
        self,
        completed: int,
        *,
        total: int | None = None,
        episode_id: int | None = None,
        status: str | None = None,
    ) -> None:
        self.progress.append((completed, total, episode_id, status))

    def detail(self, text: str) -> None:
        self.details.append(text)


def test_import_does_not_load_legacy_runtime_or_urdf_batch() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import robotwin_annotation_v2.application.urdf_runtime; "
                "assert 'robotwin_annotation_v2.application.dataset_runtime' "
                "not in sys.modules; "
                "assert 'robotwin_annotation_v2.application.urdf_batch' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_capture_urdf_json_progress_translates_jsonl_and_preserves_detail(
    capsys: pytest.CaptureFixture[str],
) -> None:
    reporter = RecordingUI()

    with urdf_runtime.capture_urdf_json_progress(reporter):
        print(
            json.dumps(
                {
                    "progress": "2/5",
                    "episode_index": 42,
                    "status": "complete",
                }
            )
        )
        print("renderer diagnostic")

    assert capsys.readouterr().out == ""
    assert reporter.progress == [(2, 5, 42, "complete")]
    assert reporter.details == ["renderer diagnostic"]


def test_select_urdf_egl_device_uses_freest_non_sam_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed = SimpleNamespace(stdout="0, 9000\n1, 12000\n2, 12000\ninvalid\n")
    monkeypatch.setattr(urdf_runtime.subprocess, "run", lambda *_args, **_kwargs: completed)

    assert urdf_runtime.select_urdf_egl_device((0,), None) == 1


@pytest.mark.parametrize("requested", (-1, True, 2))
def test_select_urdf_egl_device_rejects_invalid_or_live_sam_device(
    requested: int,
) -> None:
    with pytest.raises(ValueError):
        urdf_runtime.select_urdf_egl_device((2,), requested)


def test_release_sam_cuda_cache_is_cpu_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(urdf_runtime.gc, "collect", lambda: 7)

    assert urdf_runtime.release_sam_cuda_cache((0, 1)) == {
        "gc_collected": 7,
        "cuda_available": False,
        "gpus": [],
    }


def test_load_urdf_workflow_runtime_adapts_batch_incomplete_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BatchIncompleteError(RuntimeError):
        def __init__(self) -> None:
            super().__init__("incomplete")
            self.result = {"status": "incomplete"}

    runner = SimpleNamespace(
        new_run_id=lambda: "run-id",
        RunConfig=SimpleNamespace,
        run_experiment=lambda _config: {"status": "complete"},
        UrdfBatchIncompleteError=BatchIncompleteError,
    )
    monkeypatch.setattr(urdf_runtime, "load_urdf_runner", lambda: runner)

    runtime = urdf_runtime.load_urdf_workflow_runtime()

    assert runtime.new_run_id() == "run-id"
    assert runtime.batch_incomplete_errors == (BatchIncompleteError,)
    assert runtime.incomplete_result(BatchIncompleteError()) == {
        "status": "incomplete"
    }
    with pytest.raises(TypeError, match="no result mapping"):
        runtime.incomplete_result(RuntimeError("missing"))


class RecordingConnection:
    def __init__(self) -> None:
        self.messages: list[object] = []
        self.closed = False

    def send(self, message: object) -> None:
        self.messages.append(message)

    def close(self) -> None:
        self.closed = True


def test_process_event_sender_emits_typed_lane_and_source_messages() -> None:
    connection = RecordingConnection()
    sender = urdf_runtime.ProcessEventSender(
        connection,
        lane_name="source",
        episode_ids=(7,),
    )

    sender.stage_started(7, "qwen")
    sender.source_episode_terminal(7, "completed")

    decoded = [streaming.decode_message(message) for message in connection.messages]
    assert isinstance(decoded[0], streaming.EventMessage)
    assert decoded[0].method == "stage_started"
    assert isinstance(decoded[-1], streaming.SourceEpisodeMessage)
    assert decoded[-1].episode_id == 7
    assert decoded[-1].status == "completed"


def test_object_source_process_entry_uses_canonical_dataset_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}

    class FakePipeline:
        def __init__(self, config: object) -> None:
            calls["config"] = config

        def run_sam(self, **kwargs: Any) -> dict[str, Any]:
            calls.update(kwargs)
            callback = kwargs["episode_terminal_callback"]
            callback(7, "completed")
            return {"passed": True, "run_id": kwargs["run_id"]}

    monkeypatch.setattr(urdf_runtime, "DatasetPipeline", FakePipeline)
    connection = RecordingConnection()
    config = SimpleNamespace(name="config")

    urdf_runtime.object_source_process_entry(
        connection,
        config,
        dataset_root=tmp_path / "dataset",
        task="task",
        camera="cam_high",
        output_root=tmp_path / "output",
        run_id="source-run",
        episode_ids=(7,),
        incremental=True,
    )

    assert calls["config"] is config
    assert calls["task"] == "task"
    assert calls["camera"] == "cam_high"
    assert calls["object_source_only"] is True
    assert calls["incremental_source"] is True
    assert connection.closed
    decoded = [streaming.decode_message(message) for message in connection.messages]
    assert any(isinstance(message, streaming.SourceEpisodeMessage) for message in decoded)
    result = next(
        message for message in decoded if isinstance(message, streaming.SourceResultMessage)
    )
    assert result.summary == {"passed": True, "run_id": "source-run"}


def test_validate_urdf_run_ownership_accepts_identified_resume(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "urdf-run"
    backend_manifest = run_dir / "_backend" / "urdf" / "manifest.json"
    backend_manifest.parent.mkdir(parents=True)
    backend_manifest.write_text("{}", encoding="utf-8")
    (run_dir / "process_summary.json").write_text(
        json.dumps(
            {
                "format_version": PROCESS_SUMMARY_FORMAT_VERSION,
                "run_id": "urdf-run",
                "gripper_backend": "urdf",
            }
        ),
        encoding="utf-8",
    )

    urdf_runtime.validate_urdf_run_ownership(
        run_dir,
        run_id="urdf-run",
        resume=True,
    )

    with pytest.raises(FileExistsError):
        urdf_runtime.validate_urdf_run_ownership(
            run_dir,
            run_id="urdf-run",
            resume=False,
        )


def test_select_urdf_source_episodes_owns_source_contract_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "dataset"
    source_root = tmp_path / "source-run"
    source_root.mkdir()
    spec = annotation_spec(AnnotationMode.PICK_PLACE)
    (source_root / "process_summary.json").write_text(
        json.dumps(
            {
                "format_version": PROCESS_SUMMARY_FORMAT_VERSION,
                "run_id": source_root.name,
                "dataset_root": str(dataset_root.resolve()),
                "task": "task",
                "camera": "cam_high",
                "annotation_mode": AnnotationMode.PICK_PLACE.value,
                "required_object_roles": list(spec.required_role_names),
            }
        ),
        encoding="utf-8",
    )

    def validate(_episode_dir: Path, **kwargs: Any) -> Any:
        episode_id = int(kwargs["episode_index"])
        if episode_id == 2:
            raise UrdfGripperPublishError("source failed QC")
        return SimpleNamespace(
            annotation_mode=AnnotationMode.PICK_PLACE,
            required_object_roles=spec.required_object_roles,
            lineage={"lineage_sha256": f"sha-{episode_id}"},
        )

    monkeypatch.setattr(urdf_runtime, "validate_derivation_source_episode", validate)

    selection = urdf_runtime.select_urdf_source_episodes(
        source_root,
        dataset_root=dataset_root,
        task="task",
        camera="cam_high",
        discovered_episode_ids=(1, 2),
        expected_frame_counts={1: 4, 2: 4},
    )

    assert selection.episode_ids == (1,)
    assert selection.source_lineages[1]["lineage_sha256"] == "sha-1"
    assert selection.excluded[0]["episode"] == 2
    assert selection.excluded[0]["reason"].endswith("UrdfGripperPublishError")
    with pytest.raises(ValueError, match="requested URDF source episodes"):
        urdf_runtime.select_urdf_source_episodes(
            source_root,
            dataset_root=dataset_root,
            task="task",
            camera="cam_high",
            discovered_episode_ids=(1, 2),
            requested_episode_ids=(2,),
            expected_frame_counts={1: 4, 2: 4},
        )


class ProtocolReceiveConnection:
    def __init__(self) -> None:
        self.messages: list[tuple[Any, ...]] = []

    def poll(self, _timeout: float = 0.0) -> bool:
        return bool(self.messages)

    def recv(self) -> tuple[Any, ...]:
        return self.messages.pop(0)

    def close(self) -> None:
        pass


class ProtocolSendConnection:
    def __init__(self, receiver: ProtocolReceiveConnection) -> None:
        self.receiver = receiver

    def send(self, message: tuple[Any, ...]) -> None:
        self.receiver.messages.append(message)

    def close(self) -> None:
        pass


class ProtocolQueue:
    def __init__(self, maxsize: int) -> None:
        self.maxsize = maxsize
        self.items: list[Any] = []

    def put_nowait(self, value: Any) -> None:
        if len(self.items) >= self.maxsize:
            raise urdf_runtime.Full
        self.items.append(value)

    def close(self) -> None:
        pass

    def cancel_join_thread(self) -> None:
        pass


class ProtocolProcess:
    def __init__(
        self,
        context: ProtocolContext,
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


class ProtocolContext:
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
        self.processes: list[ProtocolProcess] = []

    def Pipe(self, duplex: bool = False) -> tuple[Any, Any]:
        assert duplex is False
        receiver = ProtocolReceiveConnection()
        return receiver, ProtocolSendConnection(receiver)

    def Queue(self, maxsize: int) -> ProtocolQueue:
        return ProtocolQueue(maxsize)

    def Process(self, *, kwargs: dict[str, Any], name: str, **_rest: Any) -> Any:
        process = ProtocolProcess(self, kwargs=kwargs, name=name)
        self.processes.append(process)
        return process


def run_protocol_coordinator(
    monkeypatch: pytest.MonkeyPatch,
    context: ProtocolContext,
) -> tuple[dict[str, Any], dict[str, Any], str | None]:
    monkeypatch.setattr(urdf_runtime.mp, "get_context", lambda _method: context)
    return urdf_runtime.run_streaming_source_urdf_workers(
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
    source, backend, error = run_protocol_coordinator(monkeypatch, ProtocolContext())

    assert source["passed"] is True
    assert backend == {"status": "complete"}
    assert error is None


def test_streaming_coordinator_terminates_peer_on_child_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = ProtocolContext(source_error=True)

    with pytest.raises(RuntimeError, match="SourceError: boom"):
        run_protocol_coordinator(monkeypatch, context)

    urdf_process = next(
        process for process in context.processes if process.name == "robotwin-streaming-urdf"
    )
    assert urdf_process.terminated is True


def test_streaming_coordinator_empty_backend_result_is_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(RuntimeError, match="empty backend"):
        run_protocol_coordinator(
            monkeypatch,
            ProtocolContext(empty_backend=True),
        )


def test_streaming_coordinator_hard_source_exit_is_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = ProtocolContext(source_hard_exit=True)

    with pytest.raises(RuntimeError, match="source process exited without a result"):
        run_protocol_coordinator(monkeypatch, context)

    urdf_process = next(
        process for process in context.processes if process.name == "robotwin-streaming-urdf"
    )
    assert urdf_process.terminated is True
