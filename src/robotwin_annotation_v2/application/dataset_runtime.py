"""Dataset execution primitives used by the readable application pipeline."""

from __future__ import annotations

import argparse
import contextlib
import gc
import importlib
import io
import json
import math
import multiprocessing as mp
import re
import subprocess
import sys
import traceback
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from queue import Full
from typing import Any

import robotwin_annotation_v2.application.discovery as _discovery
import robotwin_annotation_v2.application.streaming as _streaming
from robotwin_annotation_v2.adapters.artifact_store import ArtifactStore
from robotwin_annotation_v2.adapters.robotwin_dataset import RoboTwinDataset
from robotwin_annotation_v2.application.dataset_input import resolve_dataset_input
from robotwin_annotation_v2.application.sam_workflow import (
    PROCESS_SUMMARY_FORMAT_VERSION,
    SamBackend,
    SamRuntime,
    SamWorkflow,
    SamWorkflowHooks,
)
from robotwin_annotation_v2.application.urdf_workflow import (
    DEFAULT_URDF_DEPTH_TOLERANCE_MM,
    DEFAULT_URDF_MINIMUM_ELIGIBLE_NONEMPTY_FRACTION,
    UrdfSourceSelection,
    UrdfWorkflow,
    UrdfWorkflowHooks,
    UrdfWorkflowRuntime,
)
from robotwin_annotation_v2.config import PipelineConfig, load_config
from robotwin_annotation_v2.domain import (
    AnnotationMode,
    annotation_spec,
)
from robotwin_annotation_v2.models import EpisodeRef
from robotwin_annotation_v2.terminal_ui import UI_MODES, ProcessUI, create_process_ui
from robotwin_annotation_v2.urdf_gripper_publisher import (
    UrdfGripperPublishError,
    validate_derivation_source_episode,
    write_source_episode_completion_receipt,
    write_source_run_contract,
)

GRIPPER_BACKENDS = ("sam", "urdf")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BUNDLED_URDF_PATH = (
    PROJECT_ROOT
    / "configs"
    / "assets"
    / "aloha-agilex"
    / "arx5_description_isaac_gripper.urdf"
)
PATH_MODE_CONFIGS = {
    AnnotationMode.PICK_PLACE: PROJECT_ROOT / "configs" / "pilot_move_pillbottle_pad.yaml",
    AnnotationMode.TARGET_ONLY: PROJECT_ROOT / "configs" / "pilot_adjust_bottle_target_only.yaml",
}
DEFAULT_URDF_PIPELINE_BUFFER_SIZE = 2
CHUNK_PATTERN = _discovery.CHUNK_PATTERN
EPISODE_FILE_PATTERN = _discovery.EPISODE_FILE_PATTERN
DiscoveredEpisode = _discovery.DiscoveredEpisode
DiscoveryResult = _discovery.DiscoveryResult
_episode_video_path = _discovery._episode_video_path
_episode_depth_path = _discovery._episode_depth_path
_parquet_frame_count = _discovery._parquet_frame_count
_measure_episode = _discovery._measure_episode


@contextlib.contextmanager
def _captured_stage_output(reporter: ProcessUI | None) -> Iterator[None]:
    """Hide embedded command JSON while preserving it in verbose UI mode."""

    if reporter is None:
        yield
        return
    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured):
            yield
    finally:
        reporter.detail(captured.getvalue().rstrip())


class _JsonProgressWriter(io.TextIOBase):
    """Translate an embedded JSON-lines progress stream into UI events."""

    def __init__(self, reporter: ProcessUI) -> None:
        self._reporter = reporter
        self._buffer = ""

    def writable(self) -> bool:
        return True

    def write(self, value: str) -> int:
        self._buffer += value
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._handle_line(line)
        return len(value)

    def flush(self) -> None:
        return

    def finish(self) -> None:
        if self._buffer:
            self._handle_line(self._buffer)
            self._buffer = ""

    def _handle_line(self, line: str) -> None:
        stripped = line.strip()
        if not stripped:
            return
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            self._reporter.detail(stripped)
            return
        if not isinstance(payload, Mapping):
            self._reporter.detail(stripped)
            return
        progress = str(payload.get("progress", ""))
        match = re.fullmatch(r"(\d+)/(\d+)", progress)
        episode = payload.get("episode_index")
        if match is None or not isinstance(episode, int):
            self._reporter.detail(stripped)
            return
        self._reporter.phase_progress(
            int(match.group(1)),
            total=int(match.group(2)),
            episode_id=episode,
            status=str(payload.get("status", "unknown")),
        )


@contextlib.contextmanager
def _captured_json_progress(reporter: ProcessUI | None) -> Iterator[None]:
    if reporter is None:
        yield
        return
    writer = _JsonProgressWriter(reporter)
    try:
        with contextlib.redirect_stdout(writer):
            yield
    finally:
        writer.finish()


def _load_sam_runtime() -> SamRuntime[SamBackend, Any, Any]:
    """Load SAM/Qwen/OpenCV code only when the SAM backend is executed."""

    from robotwin_annotation_v2.adapters.qwen_client import (
        OpenAICompatibleQwenClient,
    )
    from robotwin_annotation_v2.adapters.sam3_adapter import Sam3Adapter

    runtime = importlib.import_module(
        "robotwin_annotation_v2.application.episode_pipeline"
    )
    return SamRuntime(
        qwen_client_factory=OpenAICompatibleQwenClient,
        backend_factory=Sam3Adapter,
        execution_errors=tuple(runtime.SAM_EXECUTION_ERRORS),
        emit_gripper_result=runtime._emit_gripper_result,
        emit_sam_result=runtime._emit_sam_result,
        execute_gripper_episode=runtime._execute_gripper_episode,
        execute_sam_episode=runtime._execute_sam_episode,
        fatal_cuda_error=runtime._fatal_cuda_error,
        gripper_episode_complete=runtime._gripper_episode_complete,
        sam_episode_complete=runtime._sam_episode_complete,
        run_qwen=runtime.run_qwen,
    )


def _load_urdf_runner() -> Any:
    """Load the package-owned URDF batch engine at the execution boundary."""

    return importlib.import_module("robotwin_annotation_v2.application.urdf_batch")


def _load_urdf_workflow_runtime() -> UrdfWorkflowRuntime:
    """Resolve the frozen-source runner and publisher only when URDF is selected."""

    runner = _load_urdf_runner()
    from robotwin_annotation_v2.urdf_gripper_publisher import (
        publish_urdf_episode,
        validate_published_urdf_episode,
    )

    def incomplete_result(exc: BaseException) -> Mapping[str, Any]:
        result = getattr(exc, "result", None)
        if not isinstance(result, Mapping):
            raise TypeError("URDF batch-incomplete error has no result mapping")
        return result

    return UrdfWorkflowRuntime(
        new_run_id=runner.new_run_id,
        run_config_factory=runner.RunConfig,
        run_experiment=runner.run_experiment,
        batch_incomplete_errors=(runner.UrdfBatchIncompleteError,),
        incomplete_result=incomplete_result,
        publish_episode=publish_urdf_episode,
        validate_episode=validate_published_urdf_episode,
    )


def _release_sam_cuda_cache(gpus: Sequence[int]) -> dict[str, Any]:
    """Release unreachable SAM objects and report the CUDA cache handoff."""

    report: dict[str, Any] = {
        "gc_collected": gc.collect(),
        "cuda_available": False,
        "gpus": [],
    }
    try:
        import torch
    except ImportError:
        return report
    if not torch.cuda.is_available():
        return report
    report["cuda_available"] = True
    device_count = int(torch.cuda.device_count())
    for gpu in dict.fromkeys(int(value) for value in gpus):
        if gpu < 0 or gpu >= device_count:
            continue
        allocated_before = int(torch.cuda.memory_allocated(gpu))
        reserved_before = int(torch.cuda.memory_reserved(gpu))
        with torch.cuda.device(gpu):
            torch.cuda.empty_cache()
        report["gpus"].append(
            {
                "gpu": gpu,
                "allocated_before_bytes": allocated_before,
                "reserved_before_bytes": reserved_before,
                "allocated_after_bytes": int(torch.cuda.memory_allocated(gpu)),
                "reserved_after_bytes": int(torch.cuda.memory_reserved(gpu)),
            }
        )
    return report


def _select_urdf_egl_device(
    sam_gpus: Sequence[int],
    requested: int | None,
) -> int | None:
    """Select a physical EGL GPU that does not host the live SAM worker."""

    sam_devices = {int(value) for value in sam_gpus}
    if requested is not None:
        if isinstance(requested, bool) or requested < 0:
            raise ValueError("URDF EGL device id must be a non-negative integer")
        if requested in sam_devices:
            raise ValueError(
                "streaming URDF requires an EGL GPU different from the SAM GPU"
            )
        return requested

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    candidates: list[tuple[int, int]] = []
    for raw_line in completed.stdout.splitlines():
        parts = [part.strip() for part in raw_line.split(",")]
        if len(parts) != 2:
            continue
        try:
            device_id = int(parts[0])
            free_mib = int(parts[1])
        except ValueError:
            continue
        if device_id not in sam_devices:
            candidates.append((free_mib, device_id))
    if not candidates:
        return None
    highest_free = max(value[0] for value in candidates)
    return min(device_id for free_mib, device_id in candidates if free_mib == highest_free)


class _ProcessEventSender(ProcessUI):
    """Forward child-process UI events to the live pipeline parent."""

    def __init__(
        self,
        connection: Any,
        *,
        lane_name: str | None = None,
        episode_ids: Sequence[int] = (),
    ) -> None:
        super().__init__(emit_json_summary=False, verbose=True)
        self._connection = connection
        self._lane_name = lane_name
        self._lane_completed = 0
        self._lane_total = len(episode_ids)

    def _send(self, method: str, *args: Any, **kwargs: Any) -> None:
        self._connection.send(_streaming.event(method, *args, **kwargs))

    def phase_started(self, label: str, *, total: int | None = None) -> None:
        self._send("phase_started", label, total=total)

    def phase_progress(
        self,
        completed: int,
        *,
        total: int | None = None,
        episode_id: int | None = None,
        status: str | None = None,
    ) -> None:
        self._send(
            "phase_progress",
            completed,
            total=total,
            episode_id=episode_id,
            status=status,
        )

    def phase_finished(
        self,
        label: str,
        *,
        status: str = "completed",
        detail: str | None = None,
    ) -> None:
        self._send("phase_finished", label, status=status, detail=detail)

    def lane_started(
        self,
        name: str,
        label: str,
        total: int | None = None,
    ) -> None:
        self._send("lane_started", name, label, total)

    def lane_progress(
        self,
        name: str,
        completed: int,
        total: int | None = None,
        episode_id: int | None = None,
        status: str | None = None,
        detail: str | None = None,
    ) -> None:
        self._send(
            "lane_progress",
            name,
            completed,
            total,
            episode_id,
            status,
            detail,
        )

    def lane_finished(
        self,
        name: str,
        status: str = "completed",
        detail: str | None = None,
    ) -> None:
        self._send("lane_finished", name, status, detail)

    def stage_started(self, episode_id: int, label: str) -> None:
        self._send("stage_started", episode_id, label)
        if self._lane_name is not None:
            self._send(
                "lane_progress",
                self._lane_name,
                self._lane_completed,
                self._lane_total,
                episode_id,
                label,
            )

    def stage_finished(
        self,
        episode_id: int,
        label: str,
        *,
        status: str = "completed",
        detail: str | None = None,
    ) -> None:
        self._send(
            "stage_finished",
            episode_id,
            label,
            status=status,
            detail=detail,
        )
        if self._lane_name is not None:
            self._send(
                "lane_progress",
                self._lane_name,
                self._lane_completed,
                self._lane_total,
                episode_id,
                status,
                detail or label,
            )

    def source_episode_terminal(self, episode_id: int, status: str) -> None:
        self._lane_completed += 1
        self._send(
            "lane_progress",
            self._lane_name or "source",
            self._lane_completed,
            self._lane_total,
            episode_id,
            status,
        )
        self._connection.send(_streaming.source_episode(episode_id, status))

    def note(self, message: str, *, level: str = "info") -> None:
        self._send("note", message, level=level)

    def detail(self, text: str) -> None:
        self._send("detail", text)


def _object_source_process_entry(
    connection: Any,
    pipeline_config: PipelineConfig,
    *,
    dataset_root: Path,
    task: str,
    camera: str,
    output_root: Path,
    run_id: str,
    episode_ids: tuple[int, ...],
    incremental: bool = False,
) -> None:
    worker_log_path = output_root / "logs" / f"{run_id}.source-worker.log"
    worker_log_path.parent.mkdir(parents=True, exist_ok=True)
    worker_log = worker_log_path.open("a", encoding="utf-8", buffering=1)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = worker_log
    sys.stderr = worker_log
    reporter = _ProcessEventSender(
        connection,
        lane_name="source" if incremental else None,
        episode_ids=episode_ids,
    )
    try:
        summary = process_dataset(
            pipeline_config,
            dataset_root=dataset_root,
            task=task,
            camera=camera,
            output_root=output_root,
            run_id=run_id,
            episode_ids=episode_ids,
            force=False,
            skip_render=True,
            object_source_only=True,
            report_lifecycle=False,
            incremental_source=incremental,
            episode_terminal_callback=(
                reporter.source_episode_terminal if incremental else None
            ),
            reporter=reporter,
        )
    except BaseException as exc:  # noqa: BLE001 - serialize child termination
        connection.send(
            _streaming.error(type(exc).__name__, str(exc), traceback.format_exc())
        )
    else:
        connection.send(_streaming.source_result(summary))
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        worker_log.close()
        connection.close()


def _incremental_urdf_process_entry(
    connection: Any,
    ready_queue: Any,
    run_config: Any,
) -> None:
    """Consume source-ready episode ids with one persistent EGL renderer."""

    worker_log_path = (
        Path(run_config.output_root) / "logs" / f"{run_config.run_id}.worker.log"
    )
    worker_log_path.parent.mkdir(parents=True, exist_ok=True)
    worker_log = worker_log_path.open("a", encoding="utf-8", buffering=1)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = worker_log
    sys.stderr = worker_log
    try:
        urdf_runner = _load_urdf_runner()

        worker: Any | None = None
        completed = 0
        try:
            while True:
                item = ready_queue.get()
                if item is None:
                    break
                ready_episode = _streaming.decode_ready_episode(item)
                episode_id = ready_episode.episode_id
                position = ready_episode.position
                connection.send(
                    _streaming.event(
                        "lane_progress",
                        "urdf",
                        completed,
                        len(run_config.episode_ids),
                        episode_id,
                        "rendering",
                        f"source position {position}",
                    )
                )
                if worker is None:
                    worker = urdf_runner.IncrementalUrdfEpisodeWorker(run_config)
                record = worker.process_episode(episode_id)
                completed += 1
                status = str(record.get("status", "failed"))
                connection.send(
                    _streaming.event(
                        "lane_progress",
                        "urdf",
                        completed,
                        len(run_config.episode_ids),
                        episode_id,
                        status,
                    )
                )
                connection.send(_streaming.urdf_episode(episode_id, record))

            if worker is None:
                connection.send(
                    _streaming.urdf_result(
                        None,
                        "no source episode became ready for the URDF worker",
                    )
                )
            else:
                try:
                    result = worker.finalize()
                except urdf_runner.UrdfBatchIncompleteError as exc:
                    connection.send(
                        _streaming.urdf_result(
                            exc.result,
                            f"{type(exc).__name__}: {exc}",
                        )
                    )
                else:
                    connection.send(_streaming.urdf_result(result, None))
        finally:
            if worker is not None:
                worker.close()
    except BaseException as exc:  # noqa: BLE001 - serialize child termination
        connection.send(
            _streaming.error(type(exc).__name__, str(exc), traceback.format_exc())
        )
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        worker_log.close()
        connection.close()


def _run_streaming_source_urdf_workers(
    pipeline_config: PipelineConfig,
    *,
    dataset_root: Path,
    task: str,
    camera: str,
    source_output_root: Path,
    source_run_id: str,
    episode_ids: tuple[int, ...],
    urdf_run_config: Any,
    buffer_size: int,
    reporter: ProcessUI | None,
) -> tuple[dict[str, Any], dict[str, Any], str | None]:
    """Overlap the source producer and incremental URDF consumer."""

    if buffer_size < 1:
        raise ValueError("URDF pipeline buffer size must be positive")
    context = mp.get_context("spawn")
    source_receive, source_send = context.Pipe(duplex=False)
    urdf_receive, urdf_send = context.Pipe(duplex=False)
    ready_queue = context.Queue(maxsize=buffer_size)
    source_process = context.Process(
        target=_object_source_process_entry,
        kwargs={
            "connection": source_send,
            "pipeline_config": pipeline_config,
            "dataset_root": dataset_root,
            "task": task,
            "camera": camera,
            "output_root": source_output_root,
            "run_id": source_run_id,
            "episode_ids": episode_ids,
            "incremental": True,
        },
        name="robotwin-streaming-source",
    )
    urdf_process = context.Process(
        target=_incremental_urdf_process_entry,
        kwargs={
            "connection": urdf_send,
            "ready_queue": ready_queue,
            "run_config": urdf_run_config,
        },
        name="robotwin-streaming-urdf",
    )

    source_summary: dict[str, Any] | None = None
    backend_result: dict[str, Any] | None = None
    source_result_received = False
    urdf_result_received = False
    backend_error: str | None = None
    child_error: tuple[str, str, str] | None = None
    ready_backlog: deque[_streaming.ReadyEpisode] = deque()
    source_position = 0
    source_open = True
    urdf_open = True
    sentinel_sent = False

    def forward_event(message: _streaming.EventMessage) -> None:
        if reporter is not None:
            getattr(reporter, message.method)(*message.args, **message.kwargs)

    def receive_source() -> None:
        nonlocal child_error, source_open, source_position, source_result_received
        nonlocal source_summary
        try:
            message = source_receive.recv()
        except EOFError:
            source_open = False
            if not source_result_received and child_error is None:
                child_error = (
                    "SourceProcessError",
                    "source process pipe closed without a terminal result",
                    "",
                )
            return
        decoded = _streaming.try_decode_message(message)
        if isinstance(decoded, _streaming.EventMessage):
            forward_event(decoded)
        elif isinstance(decoded, _streaming.SourceEpisodeMessage):
            source_position += 1
            if decoded.status in {"completed", "skipped_complete"}:
                ready_backlog.append(
                    _streaming.ReadyEpisode(int(decoded.episode_id), source_position)
                )
            elif reporter is not None:
                reporter.episode_finished(
                    int(decoded.episode_id), status=str(decoded.status)
                )
        elif isinstance(decoded, _streaming.SourceResultMessage):
            source_result_received = True
            raw_summary = decoded.summary
            if not isinstance(raw_summary, Mapping):
                child_error = (
                    "ProtocolError",
                    "source process returned a non-object summary",
                    "",
                )
            else:
                source_summary = dict(raw_summary)
        elif isinstance(decoded, _streaming.ErrorMessage):
            child_error = (decoded.error_type, decoded.error, decoded.traceback)
        else:
            child_error = ("ProtocolError", "invalid source-process message", "")

    def receive_urdf() -> None:
        nonlocal backend_error, backend_result, child_error, urdf_open
        nonlocal urdf_result_received
        try:
            message = urdf_receive.recv()
        except EOFError:
            urdf_open = False
            if not urdf_result_received and child_error is None:
                child_error = (
                    "UrdfProcessError",
                    "URDF process pipe closed without a terminal result",
                    "",
                )
            return
        decoded = _streaming.try_decode_message(message)
        if isinstance(decoded, _streaming.EventMessage):
            forward_event(decoded)
        elif isinstance(decoded, _streaming.UrdfEpisodeMessage):
            if decoded.record.get("status") != "complete" and reporter is not None:
                reporter.episode_finished(
                    int(decoded.episode_id),
                    status="gripper_incomplete",
                    detail=str(decoded.record.get("error", "URDF episode failed")),
                )
        elif isinstance(decoded, _streaming.UrdfResultMessage):
            urdf_result_received = True
            raw_result, backend_error = decoded.result, decoded.error
            backend_result = None if raw_result is None else dict(raw_result)
        elif isinstance(decoded, _streaming.ErrorMessage):
            child_error = (decoded.error_type, decoded.error, decoded.traceback)
        else:
            child_error = ("ProtocolError", "invalid URDF-process message", "")

    urdf_process.start()
    source_process.start()
    urdf_send.close()
    source_send.close()
    try:
        while not (source_result_received and urdf_result_received) and child_error is None:
            progressed = False
            if ready_backlog:
                try:
                    ready_queue.put_nowait(ready_backlog[0])
                except Full:
                    pass
                else:
                    ready_backlog.popleft()
                    progressed = True

            if not ready_backlog and source_open and source_receive.poll(0.02):
                receive_source()
                progressed = True
            if urdf_open and urdf_receive.poll(0.02):
                receive_urdf()
                progressed = True

            if source_summary is not None and not ready_backlog and not sentinel_sent:
                try:
                    ready_queue.put_nowait(None)
                except Full:
                    pass
                else:
                    sentinel_sent = True
                    progressed = True

            if (
                not source_process.is_alive()
                and not source_result_received
                and source_open
            ):
                while source_open and source_receive.poll():
                    receive_source()
                if not source_result_received and child_error is None:
                    child_error = (
                        "SourceProcessError",
                        f"source process exited without a result: exitcode={source_process.exitcode}",
                        "",
                    )
            if (
                not urdf_process.is_alive()
                and not urdf_result_received
                and urdf_open
            ):
                while urdf_open and urdf_receive.poll():
                    receive_urdf()
                if not urdf_result_received and child_error is None:
                    child_error = (
                        "UrdfProcessError",
                        f"URDF process exited without a result: exitcode={urdf_process.exitcode}",
                        "",
                    )
            if not progressed:
                # Both poll calls above already provide a short cooperative wait.
                continue
    except BaseException:
        for process in (source_process, urdf_process):
            if process.is_alive():
                process.terminate()
        raise
    finally:
        for process in (source_process, urdf_process):
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join()
        source_receive.close()
        urdf_receive.close()
        ready_queue.close()
        ready_queue.cancel_join_thread()

    if child_error is not None:
        error_type, error, child_traceback = child_error
        raise RuntimeError(
            f"streaming pipeline worker failed: {error_type}: {error}\n{child_traceback}"
        )
    if source_summary is None:
        raise RuntimeError("streaming source did not produce a terminal summary")
    if backend_result is None:
        raise RuntimeError(
            "streaming URDF worker produced no backend result: "
            f"{backend_error or 'no source episode became ready'}"
        )
    return source_summary, backend_result, backend_error


def _run_object_source_process(
    pipeline_config: PipelineConfig,
    *,
    dataset_root: Path,
    task: str,
    camera: str,
    output_root: Path,
    run_id: str,
    episode_ids: tuple[int, ...],
    reporter: ProcessUI | None,
) -> dict[str, Any]:
    """Generate a frozen source in a spawned process with its own CUDA lifetime."""

    context = mp.get_context("spawn")
    receive_connection, send_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_object_source_process_entry,
        kwargs={
            "connection": send_connection,
            "pipeline_config": pipeline_config,
            "dataset_root": dataset_root,
            "task": task,
            "camera": camera,
            "output_root": output_root,
            "run_id": run_id,
            "episode_ids": episode_ids,
        },
        name="robotwin-object-source",
    )
    summary: dict[str, Any] | None = None
    child_error: tuple[str, str, str] | None = None
    connection_open = True

    def receive_message() -> None:
        nonlocal child_error, connection_open, summary
        try:
            message = receive_connection.recv()
        except EOFError:
            connection_open = False
            return
        decoded = _streaming.try_decode_message(message)
        if isinstance(decoded, _streaming.EventMessage):
            if reporter is not None:
                getattr(reporter, decoded.method)(*decoded.args, **decoded.kwargs)
        elif isinstance(decoded, _streaming.SourceResultMessage):
            summary = dict(decoded.summary)
        elif isinstance(decoded, _streaming.ErrorMessage):
            child_error = (decoded.error_type, decoded.error, decoded.traceback)
        else:
            child_error = ("ProtocolError", "invalid child-process message", "")

    process.start()
    send_connection.close()
    try:
        while process.is_alive():
            if connection_open and receive_connection.poll(0.1):
                receive_message()
            elif not connection_open:
                process.join(timeout=0.1)
        process.join()
        while connection_open and receive_connection.poll():
            receive_message()
    except BaseException:
        if process.is_alive():
            process.terminate()
            process.join()
        raise
    finally:
        receive_connection.close()
    if child_error is not None:
        error_type, error, child_traceback = child_error
        raise RuntimeError(
            "live object-source subprocess failed: "
            f"{error_type}: {error}\n{child_traceback}"
        )
    if process.exitcode != 0:
        raise RuntimeError(
            "live object-source subprocess exited without a result: "
            f"exitcode={process.exitcode}"
        )
    if summary is None:
        raise RuntimeError("live object-source subprocess returned no summary")
    return summary


def discover_episodes(
    root: Path,
    *,
    camera: str,
    require_depth: bool = False,
) -> DiscoveryResult:
    """Compatibility delegate for the canonical discovery module."""

    return _discovery.discover_episodes(
        root,
        camera=camera,
        require_depth=require_depth,
    )


def _read_json_object(path: Path, *, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{description} is missing: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {description}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must contain one JSON object: {path}")
    return payload


def _validate_run_id(run_id: str) -> str:
    """Validate a public run id before it participates in path construction."""

    if (
        not isinstance(run_id, str)
        or not run_id
        or run_id != run_id.strip()
        or run_id in {".", ".."}
        or "/" in run_id
        or "\\" in run_id
        or ".." in run_id
    ):
        raise ValueError("run_id must be a simple non-empty directory name")
    return run_id


def _summary_gripper_backend(summary: Mapping[str, Any]) -> str | None:
    top_level = summary.get("gripper_backend")
    backend_record = summary.get("backend")
    nested = None
    if isinstance(backend_record, Mapping):
        nested = backend_record.get("gripper", backend_record.get("type"))
    if top_level is not None and nested is not None and top_level != nested:
        raise ValueError("existing process summary has conflicting backend ownership")
    value = top_level if top_level is not None else nested
    return None if value is None else str(value)


def _validate_sam_run_ownership(run_dir: Path, *, run_id: str) -> None:
    """Keep legacy SAM resume, but never adopt an URDF-owned public run."""

    if (run_dir / "_backend" / "urdf").exists():
        raise ValueError(f"existing run is owned by the URDF backend: {run_dir}")
    summary_path = run_dir / "process_summary.json"
    if not summary_path.exists():
        return
    summary = _read_json_object(summary_path, description="existing process summary")
    if summary.get("run_id") != run_id:
        raise ValueError("existing process summary run_id does not match its directory")
    backend = _summary_gripper_backend(summary)
    # Summaries written before the backend discriminator was introduced are SAM runs.
    if backend not in {None, "sam"}:
        raise ValueError(f"existing run is owned by backend {backend!r}, not SAM")


def _validate_urdf_run_ownership(
    run_dir: Path,
    *,
    run_id: str,
    resume: bool,
) -> None:
    """Require a fresh public run, or a positively identified URDF resume run."""

    if not resume:
        if run_dir.exists():
            raise FileExistsError(f"canonical output run already exists: {run_dir}")
        return
    if not run_dir.is_dir():
        raise FileNotFoundError(f"canonical resume run directory is missing: {run_dir}")
    backend_manifest = run_dir / "_backend" / "urdf" / "manifest.json"
    if not backend_manifest.is_file():
        raise ValueError(
            "canonical resume run is not owned by the URDF backend: "
            f"{backend_manifest} is missing"
        )
    summary_path = run_dir / "process_summary.json"
    if not summary_path.exists():
        return
    summary = _read_json_object(summary_path, description="existing process summary")
    if summary.get("format_version") != PROCESS_SUMMARY_FORMAT_VERSION:
        raise ValueError("existing process summary format is not resumable")
    if summary.get("run_id") != run_id:
        raise ValueError("existing process summary run_id does not match its directory")
    if _summary_gripper_backend(summary) != "urdf":
        raise ValueError("canonical resume run is not owned by the URDF backend")


def select_urdf_source_episodes(
    source_run_dir: Path,
    *,
    dataset_root: Path,
    task: str,
    camera: str,
    discovered_episode_ids: tuple[int, ...],
    requested_episode_ids: tuple[int, ...] | None = None,
    expected_frame_counts: Mapping[int, int] | None = None,
) -> UrdfSourceSelection:
    """Select completed source episodes with every mode-required object mask."""

    source_root = source_run_dir.expanduser().resolve()
    summary = _read_json_object(
        source_root / "process_summary.json",
        description="source process summary",
    )
    if summary.get("format_version") != "robotwin_process_dataset_summary_v1":
        raise ValueError(
            "source process summary format must be robotwin_process_dataset_summary_v1"
        )
    if summary.get("run_id") != source_root.name:
        raise ValueError("source process summary run_id does not match its directory")
    if summary.get("task") != task or summary.get("camera") != camera:
        raise ValueError(
            "source process summary task/camera does not match the requested dataset"
        )
    summary_dataset = Path(str(summary.get("dataset_root", ""))).expanduser().resolve()
    if summary_dataset != dataset_root.expanduser().resolve():
        raise ValueError(
            "source process summary dataset_root does not match --dataset-root"
        )
    raw_mode = summary.get("annotation_mode", AnnotationMode.PICK_PLACE.value)
    try:
        source_annotation_mode = AnnotationMode(raw_mode)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"source process summary has unsupported annotation_mode: {raw_mode!r}"
        ) from exc
    source_spec = annotation_spec(source_annotation_mode)
    raw_required_roles = summary.get("required_object_roles")
    if raw_required_roles is None:
        if source_annotation_mode is not AnnotationMode.PICK_PLACE:
            raise ValueError(
                "target_only source process summary must declare required_object_roles"
            )
    elif raw_required_roles != list(source_spec.required_role_names):
        raise ValueError(
            "source process summary required_object_roles differ from annotation_mode"
        )

    discovered = set(discovered_episode_ids)
    selected = (
        discovered_episode_ids
        if requested_episode_ids is None
        else tuple(dict.fromkeys(int(value) for value in requested_episode_ids))
    )
    unknown = sorted(set(selected) - discovered)
    if unknown:
        raise ValueError(f"requested episodes were not discovered: {unknown}")
    if expected_frame_counts is not None:
        missing_frame_counts = sorted(set(selected) - set(expected_frame_counts))
        if missing_frame_counts:
            raise ValueError(
                "expected frame counts are missing for episodes: "
                f"{missing_frame_counts}"
            )
    accepted: list[int] = []
    excluded: list[dict[str, Any]] = []
    source_lineages: dict[int, Mapping[str, Any]] = {}
    for episode_id in selected:
        episode_dir = (
            source_root / task / f"episode_{episode_id:06d}" / camera
        )
        try:
            validated = validate_derivation_source_episode(
                episode_dir,
                task=task,
                camera=camera,
                episode_index=episode_id,
                expected_frame_count=(
                    None
                    if expected_frame_counts is None
                    else expected_frame_counts.get(episode_id)
                ),
                expected_dataset_root=dataset_root,
            )
        except (FileNotFoundError, UrdfGripperPublishError) as exc:
            excluded.append(
                {
                    "episode": episode_id,
                    "status": "source_excluded",
                    "reason": f"source_contract_error:{type(exc).__name__}",
                    "error": str(exc),
                }
            )
            continue
        if (
            validated.annotation_mode is not source_annotation_mode
            or validated.required_object_roles != source_spec.required_object_roles
        ):
            excluded.append(
                {
                    "episode": episode_id,
                    "status": "source_excluded",
                    "reason": "source_contract_error:AnnotationModeMismatch",
                    "error": (
                        "source episode annotation contract differs from the "
                        "source process summary"
                    ),
                }
            )
            continue
        accepted.append(episode_id)
        source_lineages[episode_id] = validated.lineage

    if requested_episode_ids is not None and excluded:
        rendered = ", ".join(
            f"{record['episode']} ({record['reason']})" for record in excluded
        )
        raise ValueError(f"requested URDF source episodes are not publishable: {rendered}")
    if not accepted:
        raise ValueError("source run contains no publishable object-mask episodes")
    return UrdfSourceSelection(
        episode_ids=tuple(accepted),
        excluded=tuple(excluded),
        source_summary=summary,
        source_lineages=source_lineages,
        annotation_mode=source_annotation_mode,
        required_object_roles=source_spec.required_object_roles,
    )


def build_dynamic_manifest(
    root: Path,
    *,
    task: str,
    camera: str,
    episodes: Sequence[DiscoveredEpisode],
) -> dict[str, Any]:
    """Compatibility delegate for the canonical discovery module."""

    return _discovery.build_dynamic_manifest(
        root,
        task=task,
        camera=camera,
        episodes=episodes,
        measure_episode_fn=_measure_episode,
    )


def _dynamic_config(
    config: PipelineConfig,
    *,
    root: Path,
    task: str,
    camera: str,
    manifest: dict[str, Any],
    output_root: Path,
) -> PipelineConfig:
    dataset = replace(
        config.dataset,
        root=root.expanduser().resolve(),
        task=task,
        camera=camera,
        smoke_episode_ids=tuple(manifest["smoke_episode_ids"]),
        regression_episode_ids=tuple(manifest["regression_episode_ids"]),
        manifest_data=manifest,
    )
    return replace(config, dataset=dataset, output_root=output_root.expanduser().resolve())


def _render_processed(
    config: PipelineConfig,
    *,
    run_id: str,
    episode_ids: tuple[int, ...],
    output_dir: Path,
    reporter: ProcessUI | None = None,
) -> dict[str, Any]:
    render = importlib.import_module("robotwin_annotation_v2.adapters.rendering")

    dataset = RoboTwinDataset(
        config.dataset.root,
        task=config.dataset.task,
        camera=config.dataset.camera,
        manifest_path=config.dataset.manifest,
        manifest_data=config.dataset.manifest_data,
    )
    selected = render.select_best_masks(
        config.output_root,
        task=config.dataset.task,
        camera=config.dataset.camera,
        episode_ids=episode_ids,
        run_id=run_id,
    )
    video_dir = output_dir / "rendered_videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for position, episode_id in enumerate(episode_ids, start=1):
        candidate = selected[episode_id]
        artifact = render._load_masks(candidate.path)
        ref = EpisodeRef(
            config.dataset.task,
            episode_id,
            config.dataset.camera,
        )
        video_path = dataset.paths(ref).video
        task_text = dataset.task_text(episode_id)
        output_path = video_dir / render._output_video_name(
            episode_id=episode_id,
            camera=config.dataset.camera,
            task_text=task_text,
            filename_mode="episode",
        )
        video = render.render_video(
            video_path,
            artifact,
            output_path,
            alpha=render.DEFAULT_FILL_ALPHA,
            outline_radius=render.DEFAULT_OUTLINE_RADIUS,
            halo_radius=render.DEFAULT_HALO_RADIUS,
            crf=18,
            preset="medium",
            overwrite=True,
        )
        records.append(
            {
                "episode_index": episode_id,
                "task_text": task_text,
                "run_id": candidate.run_id,
                "source_video": str(video_path),
                "source_masks": str(candidate.path),
                "mask_sha256": render._sha256(candidate.path),
                "mask_format": artifact.format_version,
                "annotation_status": dict(
                    zip(artifact.instance_names, artifact.annotation_status, strict=True)
                ),
                "qc_status": dict(
                    zip(artifact.instance_names, artifact.qc_status, strict=True)
                ),
                "output_video": output_path.name,
                "output_sha256": render._sha256(output_path),
                "output_bytes": output_path.stat().st_size,
                **video,
            }
        )
        if reporter is not None:
            reporter.phase_progress(
                position,
                total=len(episode_ids),
                episode_id=episode_id,
                status="rendered",
            )
    manifest = {
        "format": "robotwin_coverage20_overlay_videos_v3",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "requested_run_id": run_id,
        "config": str(config.config_path),
        "dataset_root": str(config.dataset.root),
        "runs_root": str(config.output_root),
        "task": config.dataset.task,
        "camera": config.dataset.camera,
        "filename_mode": "episode",
        "episode_count": len(records),
        "rendered_roles": [
            *config.annotation.spec.required_role_names,
            "gripper",
        ],
        "alpha": render.DEFAULT_FILL_ALPHA,
        "colors_rgb": {key: list(value) for key, value in render.ROLE_COLORS.items()},
        "episodes": records,
        "review_sheets": [],
    }
    manifest_path = ArtifactStore.write_json(video_dir / "manifest.json", manifest)
    sheets = render.build_sheets(manifest_path, video_dir / "review_sheets")
    manifest["review_sheets"] = [
        str(path.relative_to(video_dir)) for path in sheets
    ]
    ArtifactStore.write_json(manifest_path, manifest)
    return {
        "manifest": str(manifest_path),
        "episode_count": len(records),
        "review_sheets": [str(path) for path in sheets],
    }


def process_urdf_source_run(
    *,
    pipeline_config: PipelineConfig | None = None,
    dataset_root: Path,
    source_run_dir: Path,
    task: str,
    camera: str,
    output_root: Path,
    urdf_path: Path,
    mesh_root: Path | None = None,
    run_id: str | None = None,
    episode_ids: tuple[int, ...] | None = None,
    skip_render: bool = False,
    dry_run: bool = False,
    resume: bool = False,
    depth_tolerance_mm: float = DEFAULT_URDF_DEPTH_TOLERANCE_MM,
    minimum_eligible_nonempty_fraction: float = (
        DEFAULT_URDF_MINIMUM_ELIGIBLE_NONEMPTY_FRACTION
    ),
    fit_config_json: Path | None = None,
    allow_partial_source: bool = False,
    source_mode: str = "frozen_run",
    source_release: Mapping[str, Any] | None = None,
    experiment_runner: Callable[..., Mapping[str, Any]] | None = None,
    episode_publisher: Callable[..., Mapping[str, Any]] | None = None,
    episode_validator: Callable[..., Mapping[str, Any]] | None = None,
    render_builder: Callable[..., dict[str, Any]] | None = None,
    prepared_backend_result: Mapping[str, Any] | None = None,
    prepared_backend_error: str | None = None,
    egl_device_id: int | None = None,
    report_lifecycle: bool = True,
    pipeline_episode_ids: tuple[int, ...] | None = None,
    reporter: ProcessUI | None = None,
) -> dict[str, Any]:
    """Compatibility entry point for the canonical frozen-source workflow."""

    workflow = UrdfWorkflow(
        UrdfWorkflowHooks(
            runtime_loader=_load_urdf_workflow_runtime,
            discover_episodes=discover_episodes,
            parquet_frame_count=_parquet_frame_count,
            select_source_episodes=select_urdf_source_episodes,
            validate_run_id=_validate_run_id,
            validate_run_ownership=_validate_urdf_run_ownership,
            capture_progress=_captured_json_progress,
            validate_source_episode=validate_derivation_source_episode,
            build_dynamic_config=_dynamic_config,
            render_processed=_render_processed,
            summary_format_version=PROCESS_SUMMARY_FORMAT_VERSION,
        )
    )
    return workflow.run(
        pipeline_config=pipeline_config,
        dataset_root=dataset_root,
        source_run_dir=source_run_dir,
        task=task,
        camera=camera,
        output_root=output_root,
        urdf_path=urdf_path,
        mesh_root=mesh_root,
        run_id=run_id,
        episode_ids=episode_ids,
        skip_render=skip_render,
        dry_run=dry_run,
        resume=resume,
        depth_tolerance_mm=depth_tolerance_mm,
        minimum_eligible_nonempty_fraction=minimum_eligible_nonempty_fraction,
        fit_config_json=fit_config_json,
        allow_partial_source=allow_partial_source,
        source_mode=source_mode,
        source_release=source_release,
        experiment_runner=experiment_runner,
        episode_publisher=episode_publisher,
        episode_validator=episode_validator,
        render_builder=render_builder,
        prepared_backend_result=prepared_backend_result,
        prepared_backend_error=prepared_backend_error,
        egl_device_id=egl_device_id,
        report_lifecycle=report_lifecycle,
        pipeline_episode_ids=pipeline_episode_ids,
        reporter=reporter,
    )


def process_dataset(
    config: PipelineConfig,
    *,
    dataset_root: Path,
    task: str,
    camera: str,
    output_root: Path,
    run_id: str | None = None,
    episode_ids: tuple[int, ...] | None = None,
    force: bool = False,
    skip_render: bool = False,
    object_source_only: bool | None = None,
    # Deprecated compatibility alias.  New callers must use object_source_only.
    target_receiver_only: bool = False,
    report_lifecycle: bool = True,
    incremental_source: bool = False,
    episode_terminal_callback: Callable[[int, str], None] | None = None,
    backend_factory: Callable[..., Any] | None = None,
    reporter: ProcessUI | None = None,
) -> dict[str, Any]:
    """Compatibility entry point for the canonical :class:`SamWorkflow`."""

    workflow = SamWorkflow(
        config,
        SamWorkflowHooks(
            runtime_loader=_load_sam_runtime,
            discover_episodes=discover_episodes,
            build_dynamic_manifest=build_dynamic_manifest,
            build_dynamic_config=_dynamic_config,
            capture_stage_output=_captured_stage_output,
            render_processed=_render_processed,
            validate_run_id=_validate_run_id,
            validate_run_ownership=_validate_sam_run_ownership,
            write_source_run_contract=write_source_run_contract,
            write_source_episode_receipt=write_source_episode_completion_receipt,
        ),
    )
    return workflow.run(
        dataset_root=dataset_root,
        task=task,
        camera=camera,
        output_root=output_root,
        run_id=run_id,
        episode_ids=episode_ids,
        force=force,
        skip_render=skip_render,
        object_source_only=object_source_only,
        target_receiver_only=target_receiver_only,
        report_lifecycle=report_lifecycle,
        incremental_source=incremental_source,
        episode_terminal_callback=episode_terminal_callback,
        backend_factory=backend_factory,
        reporter=reporter,
    )


def process_live_urdf_pipeline(
    *,
    pipeline_config: PipelineConfig,
    dataset_root: Path,
    task: str,
    camera: str,
    output_root: Path,
    urdf_path: Path,
    mesh_root: Path | None = None,
    run_id: str | None = None,
    episode_ids: tuple[int, ...] | None = None,
    skip_render: bool = False,
    depth_tolerance_mm: float = DEFAULT_URDF_DEPTH_TOLERANCE_MM,
    minimum_eligible_nonempty_fraction: float = (
        DEFAULT_URDF_MINIMUM_ELIGIBLE_NONEMPTY_FRACTION
    ),
    fit_config_json: Path | None = None,
    allow_partial_source: bool = False,
    urdf_pipeline: bool = True,
    urdf_pipeline_buffer_size: int = DEFAULT_URDF_PIPELINE_BUFFER_SIZE,
    urdf_egl_device_id: int | None = None,
    backend_factory: Callable[..., Any] | None = None,
    reporter: ProcessUI | None = None,
) -> dict[str, Any]:
    """Run the mode-required object source followed by the URDF gripper stage."""

    resolved_dataset_root = dataset_root.expanduser().resolve()
    resolved_output_root = output_root.expanduser().resolve()
    resolved_urdf_path = urdf_path.expanduser().resolve()
    if not resolved_urdf_path.is_file():
        raise FileNotFoundError(f"URDF asset is missing: {resolved_urdf_path}")
    if not math.isfinite(depth_tolerance_mm) or depth_tolerance_mm < 0:
        raise ValueError("URDF depth tolerance must be finite and non-negative")
    if not math.isfinite(minimum_eligible_nonempty_fraction) or not (
        0.0 <= minimum_eligible_nonempty_fraction <= 1.0
    ):
        raise ValueError(
            "URDF minimum eligible nonempty fraction must be finite and in [0, 1]"
        )

    selected_run_id = _validate_run_id(run_id or ArtifactStore.new_run_id())
    canonical_run_dir = resolved_output_root / selected_run_id
    _validate_urdf_run_ownership(
        canonical_run_dir,
        run_id=selected_run_id,
        resume=False,
    )
    source_run_id = _validate_run_id(f"{selected_run_id}-object-source")
    source_output_root = resolved_output_root / "_sources"
    source_run_dir = source_output_root / source_run_id
    if source_run_dir.exists():
        raise FileExistsError(
            f"live object-source run already exists: {source_run_dir}"
        )

    depth_discovery = discover_episodes(
        resolved_dataset_root,
        camera=camera,
        require_depth=True,
    )
    depth_eligible_ids = depth_discovery.episode_ids
    if not depth_eligible_ids:
        raise ValueError(
            f"no complete URDF episodes found under {resolved_dataset_root}"
        )
    requested_ids = (
        None
        if episode_ids is None
        else tuple(dict.fromkeys(int(value) for value in episode_ids))
    )
    if requested_ids is not None:
        if not requested_ids:
            raise ValueError("live URDF pipeline requires at least one selected episode")
        ineligible = sorted(set(requested_ids) - set(depth_eligible_ids))
        if ineligible:
            raise ValueError(
                "requested URDF episodes do not satisfy the dataset/depth contract: "
                f"{ineligible}"
            )
        source_episode_ids = requested_ids
    else:
        if depth_discovery.skipped and not allow_partial_source:
            examples = ", ".join(
                f"{record['episode']} ({','.join(record['missing'])})"
                for record in depth_discovery.skipped[:10]
            )
            suffix = "" if len(depth_discovery.skipped) <= 10 else ", ..."
            raise ValueError(
                "dataset contract excludes episodes before the live source stage: "
                f"{examples}{suffix}; pass --allow-partial-source to process only "
                "the depth-eligible subset"
            )
        source_episode_ids = depth_eligible_ids

    if reporter is not None:
        reporter.run_started(
            backend="urdf",
            dataset_root=str(resolved_dataset_root),
            task=task,
            camera=camera,
        )
        reporter.run_ready(run_id=selected_run_id, episode_ids=source_episode_ids)
        reporter.lane_started("source", "Source (Qwen + SAM)", len(source_episode_ids))
        reporter.lane_started("urdf", "URDF render", len(source_episode_ids))
        reporter.lane_started("publish", "Canonical publish", len(source_episode_ids))
        if not skip_render:
            reporter.lane_started(
                "validation",
                "Canonical validation",
                len(source_episode_ids),
            )
            reporter.lane_started(
                "render",
                "Review render",
                len(source_episode_ids),
            )
        reporter.note(
            f"object-mask source will be frozen at {source_run_dir}"
        )

    selected_egl_device: int | None = None
    if urdf_pipeline and backend_factory is None:
        selected_egl_device = _select_urdf_egl_device(
            pipeline_config.sam3.gpus,
            urdf_egl_device_id,
        )
        if selected_egl_device is None and reporter is not None:
            reporter.note(
                "no independent EGL GPU is available; using the serial URDF path",
                level="warning",
            )
    elif urdf_egl_device_id is not None:
        if isinstance(urdf_egl_device_id, bool) or urdf_egl_device_id < 0:
            raise ValueError("URDF EGL device id must be a non-negative integer")
        # The serial path releases SAM before EGL starts, so sharing is safe.
        selected_egl_device = urdf_egl_device_id

    prepared_backend_result: Mapping[str, Any] | None = None
    prepared_backend_error: str | None = None
    streaming = urdf_pipeline and backend_factory is None and selected_egl_device is not None
    if streaming:
        urdf_runner = _load_urdf_runner()

        streaming_config = urdf_runner.RunConfig(
            dataset_root=resolved_dataset_root,
            source_run_dir=source_run_dir,
            output_root=canonical_run_dir / "_backend",
            run_id="urdf",
            urdf_path=resolved_urdf_path,
            mesh_root=(
                None if mesh_root is None else mesh_root.expanduser().resolve()
            ),
            episode_ids=source_episode_ids,
            task=task,
            camera=camera,
            depth_tolerance_mm=depth_tolerance_mm,
            minimum_eligible_nonempty_fraction=(
                minimum_eligible_nonempty_fraction
            ),
            fit_config_json=(
                None
                if fit_config_json is None
                else fit_config_json.expanduser().resolve()
            ),
            skip_overlay=True,
            dry_run=False,
            resume=False,
            egl_device_id=selected_egl_device,
        )
        if reporter is not None:
            reporter.note(
                "streaming Source -> URDF pipeline enabled: "
                f"sam_gpus={list(pipeline_config.sam3.gpus)} "
                f"egl_gpu={selected_egl_device} buffer={urdf_pipeline_buffer_size}"
            )
        source_summary, prepared_result, prepared_backend_error = (
            _run_streaming_source_urdf_workers(
                pipeline_config,
                dataset_root=resolved_dataset_root,
                task=task,
                camera=camera,
                source_output_root=source_output_root,
                source_run_id=source_run_id,
                episode_ids=source_episode_ids,
                urdf_run_config=streaming_config,
                buffer_size=urdf_pipeline_buffer_size,
                reporter=reporter,
            )
        )
        prepared_backend_result = prepared_result
        if reporter is not None:
            reporter.lane_finished("source", status="completed")
            reporter.lane_finished(
                "urdf",
                status=(
                    "completed" if prepared_backend_error is None else "failed"
                ),
                detail=prepared_backend_error,
            )
    elif backend_factory is None:
        source_summary = _run_object_source_process(
            pipeline_config,
            dataset_root=resolved_dataset_root,
            task=task,
            camera=camera,
            output_root=source_output_root,
            run_id=source_run_id,
            episode_ids=source_episode_ids,
            reporter=reporter,
        )
        if reporter is not None:
            reporter.lane_progress(
                "source", len(source_episode_ids), len(source_episode_ids)
            )
            reporter.lane_finished("source")
    else:
        source_summary = process_dataset(
            pipeline_config,
            dataset_root=resolved_dataset_root,
            task=task,
            camera=camera,
            output_root=source_output_root,
            run_id=source_run_id,
            episode_ids=source_episode_ids,
            force=False,
            skip_render=True,
            object_source_only=True,
            report_lifecycle=False,
            backend_factory=backend_factory,
            reporter=reporter,
        )
        if reporter is not None:
            reporter.lane_progress(
                "source", len(source_episode_ids), len(source_episode_ids)
            )
            reporter.lane_finished("source")
    if reporter is not None:
        reporter.phase_started("sam_backend_release")
    source_release = _release_sam_cuda_cache(pipeline_config.sam3.gpus)
    if reporter is not None:
        gpu_details = "; ".join(
            (
                f"gpu={record['gpu']} "
                f"allocated={record['allocated_before_bytes']}"
                f"->{record['allocated_after_bytes']} "
                f"reserved={record['reserved_before_bytes']}"
                f"->{record['reserved_after_bytes']}"
            )
            for record in source_release["gpus"]
        )
        reporter.phase_finished(
            "sam_backend_release",
            detail=gpu_details or "no CUDA cache was present",
        )
    if not bool(source_summary.get("passed")) and (
        requested_ids is not None or not allow_partial_source
    ):
        raise RuntimeError(
            "live object-source stage did not pass; its frozen diagnostics "
            f"are at {source_run_dir}"
        )

    return process_urdf_source_run(
        pipeline_config=pipeline_config,
        dataset_root=resolved_dataset_root,
        source_run_dir=source_run_dir,
        task=task,
        camera=camera,
        output_root=resolved_output_root,
        urdf_path=resolved_urdf_path,
        mesh_root=mesh_root,
        run_id=selected_run_id,
        episode_ids=requested_ids,
        skip_render=skip_render,
        dry_run=False,
        resume=False,
        depth_tolerance_mm=depth_tolerance_mm,
        minimum_eligible_nonempty_fraction=(
            minimum_eligible_nonempty_fraction
        ),
        fit_config_json=fit_config_json,
        allow_partial_source=allow_partial_source,
        source_mode="live_object_source_stage",
        source_release=source_release,
        prepared_backend_result=prepared_backend_result,
        prepared_backend_error=prepared_backend_error,
        egl_device_id=selected_egl_device,
        report_lifecycle=False,
        pipeline_episode_ids=source_episode_ids,
        reporter=reporter,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument(
        "--data-path",
        "--data_path",
        type=Path,
        help="Single-task dataset or collection root",
    )
    path_mode = parser.add_mutually_exclusive_group()
    path_mode.add_argument(
        "--target-only",
        "--target_only",
        dest="path_mode",
        action="store_const",
        const=AnnotationMode.TARGET_ONLY.value,
    )
    path_mode.add_argument(
        "--pick-place",
        "--pick_place",
        dest="path_mode",
        action="store_const",
        const=AnnotationMode.PICK_PLACE.value,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/pilot_move_pillbottle_pad.yaml"),
    )
    parser.add_argument("--task")
    parser.add_argument("--camera")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--run-id")
    parser.add_argument("--episode-ids", type=int, nargs="*")
    parser.add_argument(
        "--gripper-backend",
        choices=GRIPPER_BACKENDS,
        default="urdf",
        help="Use the SAM gripper stage or generate it from URDF after target/receiver",
    )
    parser.add_argument(
        "--source-run-dir",
        help=(
            "Optional frozen run containing QC-passed target/receiver masks; when "
            "omitted, the URDF pipeline generates a fresh internal source stage"
        ),
    )
    parser.add_argument(
        "--urdf-path",
        help=(
            "RoboTwin Aloha URDF; defaults to the bundled render asset for the "
            "URDF backend"
        ),
    )
    parser.add_argument("--urdf-mesh-root", type=Path)
    parser.add_argument("--urdf-depth-tolerance-mm", type=float)
    parser.add_argument(
        "--urdf-minimum-eligible-nonempty-fraction",
        type=float,
    )
    parser.add_argument("--urdf-fit-config-json", type=Path)
    parser.add_argument(
        "--urdf-egl-device-id",
        type=int,
        help=(
            "Physical GPU for EGL rendering; live URDF mode otherwise selects the "
            "freest GPU not used by SAM"
        ),
    )
    parser.add_argument(
        "--urdf-pipeline-buffer-size",
        type=int,
        default=DEFAULT_URDF_PIPELINE_BUFFER_SIZE,
        help="Maximum source-ready episodes queued ahead of the URDF worker",
    )
    parser.add_argument(
        "--no-urdf-pipeline",
        action="store_true",
        help="Disable Source-to-URDF overlap and use the legacy serial execution path",
    )
    parser.add_argument(
        "--allow-partial-source",
        action="store_true",
        help=(
            "During automatic episode discovery, process only source episodes whose "
            "target and receiver passed QC; explicit --episode-ids remain fail-closed"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument(
        "--ui",
        "--output-format",
        dest="ui",
        choices=UI_MODES,
        default="auto",
        help="Terminal output mode; auto uses Rich on a TTY and plain logs otherwise",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def _optional_cli_path(value: str | None) -> Path | None:
    if value is None or value.strip() in {"", "-"}:
        return None
    return Path(value)


def _path_target_args(
    args: argparse.Namespace,
    *,
    config: Path,
    dataset_root: Path,
    task: str,
    camera: str,
    run_id: str | None,
) -> argparse.Namespace:
    values = vars(args) | {
        "config": config,
        "data_path": None,
        "dataset_root": dataset_root,
        "task": task,
        "camera": camera,
        "path_mode": None,
        "run_id": run_id,
    }
    return argparse.Namespace(**values)


def _run_path_input(args: argparse.Namespace, reporter: ProcessUI) -> dict[str, Any]:
    if args.dataset_root is not None:
        raise ValueError("--data-path and --dataset-root cannot be used together")
    if args.path_mode is None:
        raise ValueError("--data-path requires exactly one of --target-only/--pick-place")
    if _optional_cli_path(args.source_run_dir) is not None:
        raise ValueError("--source-run-dir is not supported with --data-path")
    mode = AnnotationMode(args.path_mode)
    resolved = resolve_dataset_input(args.data_path, mode=mode, task=args.task)
    if args.camera is not None and any(target.camera != args.camera for target in resolved.targets):
        raise ValueError("--camera does not match the dataset extract manifest")
    if resolved.is_collection and args.episode_ids is not None and len(resolved.targets) != 1:
        raise ValueError("collection --episode-ids requires selecting one --task")

    config = PATH_MODE_CONFIGS[mode]
    if not resolved.is_collection:
        target = resolved.targets[0]
        return _run_from_args(
            _path_target_args(
                args,
                config=config,
                dataset_root=target.root,
                task=target.task,
                camera=target.camera,
                run_id=args.run_id,
            ),
            reporter,
        )

    collection_run_id = _validate_run_id(args.run_id or ArtifactStore.new_run_id())
    records: list[dict[str, Any]] = []
    for target in resolved.targets:
        task_run_id = _validate_run_id(f"{collection_run_id}-{target.task}")
        try:
            summary = _run_from_args(
                _path_target_args(
                    args,
                    config=config,
                    dataset_root=target.root,
                    task=target.task,
                    camera=target.camera,
                    run_id=task_run_id,
                ),
                reporter,
            )
        except Exception as exc:  # noqa: BLE001 - one task must not stop a collection
            reporter.note(
                f"collection task {target.task} failed: {type(exc).__name__}: {exc}",
                level="error",
            )
            records.append(
                {
                    "task": target.task,
                    "run_id": task_run_id,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        else:
            records.append(
                {
                    "task": target.task,
                    "run_id": task_run_id,
                    "status": "completed" if summary["passed"] else "failed",
                    "artifact": summary.get("artifact"),
                }
            )
    result: dict[str, Any] = {
        "format_version": "robotwin_process_collection_summary_v1",
        "run_id": collection_run_id,
        "dataset_root": str(resolved.root),
        "annotation_mode": mode.value,
        "records": records,
        "passed": all(record["status"] == "completed" for record in records),
    }
    artifact = ArtifactStore.write_json(
        args.output_dir.expanduser().resolve() / f"{collection_run_id}-collection-summary.json",
        result,
    )
    result["artifact"] = str(artifact)
    return result


def _run_from_args(
    args: argparse.Namespace,
    reporter: ProcessUI,
) -> dict[str, Any]:
    if args.run_id is not None:
        _validate_run_id(args.run_id)
    if args.data_path is not None:
        return _run_path_input(args, reporter)
    if args.path_mode is not None:
        raise ValueError("--target-only/--pick-place require --data-path")
    config = load_config(args.config)
    source_run_dir = _optional_cli_path(args.source_run_dir)
    urdf_path = _optional_cli_path(args.urdf_path)
    if args.gripper_backend == "sam":
        if (
            source_run_dir is not None
            or urdf_path is not None
            or args.urdf_mesh_root is not None
            or args.urdf_depth_tolerance_mm is not None
            or args.urdf_minimum_eligible_nonempty_fraction is not None
            or args.urdf_fit_config_json is not None
            or args.urdf_egl_device_id is not None
            or args.urdf_pipeline_buffer_size != DEFAULT_URDF_PIPELINE_BUFFER_SIZE
            or args.no_urdf_pipeline
            or args.allow_partial_source
        ):
            raise ValueError("URDF-only options require --gripper-backend urdf")
        if args.dry_run or args.resume:
            raise ValueError("--dry-run/--resume are only supported by the URDF backend")
        dataset_root = (
            config.dataset.root if args.dataset_root is None else args.dataset_root
        )
        task = config.dataset.task if args.task is None else args.task
        camera = config.dataset.camera if args.camera is None else args.camera
        summary = process_dataset(
            config,
            dataset_root=dataset_root,
            task=task,
            camera=camera,
            output_root=args.output_dir,
            run_id=args.run_id,
            episode_ids=None if args.episode_ids is None else tuple(args.episode_ids),
            force=args.force,
            skip_render=args.skip_render,
            reporter=reporter,
        )
    else:
        if args.force:
            raise ValueError(
                "--force is not supported by the immutable URDF backend; use a new run id"
            )
        if args.dry_run and args.resume:
            raise ValueError("--dry-run and --resume cannot be used together")
        resolved_urdf_path = (
            DEFAULT_BUNDLED_URDF_PATH if urdf_path is None else urdf_path
        )
        selected_episode_ids = (
            None if args.episode_ids is None else tuple(args.episode_ids)
        )
        depth_tolerance_mm = (
            DEFAULT_URDF_DEPTH_TOLERANCE_MM
            if args.urdf_depth_tolerance_mm is None
            else args.urdf_depth_tolerance_mm
        )
        minimum_eligible_nonempty_fraction = (
            DEFAULT_URDF_MINIMUM_ELIGIBLE_NONEMPTY_FRACTION
            if args.urdf_minimum_eligible_nonempty_fraction is None
            else args.urdf_minimum_eligible_nonempty_fraction
        )
        if source_run_dir is None:
            if args.dry_run or args.resume:
                raise ValueError(
                    "live URDF mode is fresh-only; --dry-run/--resume require "
                    "--source-run-dir"
                )
            dataset_root = (
                config.dataset.root if args.dataset_root is None else args.dataset_root
            )
            task = config.dataset.task if args.task is None else args.task
            camera = config.dataset.camera if args.camera is None else args.camera
            summary = process_live_urdf_pipeline(
                pipeline_config=config,
                dataset_root=dataset_root,
                task=task,
                camera=camera,
                output_root=args.output_dir,
                urdf_path=resolved_urdf_path,
                mesh_root=args.urdf_mesh_root,
                run_id=args.run_id,
                episode_ids=selected_episode_ids,
                skip_render=args.skip_render,
                depth_tolerance_mm=depth_tolerance_mm,
                minimum_eligible_nonempty_fraction=(
                    minimum_eligible_nonempty_fraction
                ),
                fit_config_json=args.urdf_fit_config_json,
                allow_partial_source=args.allow_partial_source,
                urdf_pipeline=not args.no_urdf_pipeline,
                urdf_pipeline_buffer_size=args.urdf_pipeline_buffer_size,
                urdf_egl_device_id=args.urdf_egl_device_id,
                reporter=reporter,
            )
        else:
            if args.resume and not args.run_id:
                raise ValueError("--resume requires an explicit --run-id")
            source_summary = _read_json_object(
                source_run_dir.expanduser().resolve() / "process_summary.json",
                description="source process summary",
            )
            dataset_root = (
                Path(str(source_summary.get("dataset_root", "")))
                if args.dataset_root is None
                else args.dataset_root
            )
            task = (
                str(source_summary.get("task", ""))
                if args.task is None
                else args.task
            )
            camera = (
                str(source_summary.get("camera", ""))
                if args.camera is None
                else args.camera
            )
            if not task or not camera:
                raise ValueError("source process summary does not define task/camera")
            summary = process_urdf_source_run(
                pipeline_config=config,
                dataset_root=dataset_root,
                source_run_dir=source_run_dir,
                task=task,
                camera=camera,
                output_root=args.output_dir,
                urdf_path=resolved_urdf_path,
                mesh_root=args.urdf_mesh_root,
                run_id=args.run_id,
                episode_ids=selected_episode_ids,
                skip_render=args.skip_render,
                dry_run=args.dry_run,
                resume=args.resume,
                depth_tolerance_mm=depth_tolerance_mm,
                minimum_eligible_nonempty_fraction=(
                    minimum_eligible_nonempty_fraction
                ),
                fit_config_json=args.urdf_fit_config_json,
                allow_partial_source=args.allow_partial_source,
                egl_device_id=args.urdf_egl_device_id,
                reporter=reporter,
            )
    return summary


def main() -> None:
    args = _parse_args()
    reporter = create_process_ui(args.ui, verbose=args.verbose)
    try:
        summary = _run_from_args(args, reporter)
    except BaseException as exc:
        reporter.failed(exc)
        raise
    else:
        reporter.finish(summary)
        if reporter.emit_json_summary:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        reporter.close()
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
