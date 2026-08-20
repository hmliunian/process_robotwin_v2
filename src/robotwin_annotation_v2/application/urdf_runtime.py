"""Production runtime dependencies for the dataset-level URDF workflow.

The pure workflow policy lives in :mod:`robotwin_annotation_v2.application.urdf_workflow`.
This module owns the lazy backend imports, GPU handoff, spawned workers, and frozen-source
validation needed to execute that policy without routing through the legacy dataset runtime.
"""

from __future__ import annotations

import contextlib
import gc
import importlib
import io
import json
import multiprocessing as mp
import re
import subprocess
import sys
import traceback
from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from queue import Full
from typing import Any

from robotwin_annotation_v2.application import streaming
from robotwin_annotation_v2.application.dataset_pipeline import DatasetPipeline
from robotwin_annotation_v2.application.sam_workflow import (
    PROCESS_SUMMARY_FORMAT_VERSION,
    read_json_object,
    summary_gripper_backend,
)
from robotwin_annotation_v2.application.urdf_workflow import (
    UrdfSourceSelection,
    UrdfWorkflowRuntime,
)
from robotwin_annotation_v2.config import PipelineConfig
from robotwin_annotation_v2.domain import AnnotationMode, annotation_spec
from robotwin_annotation_v2.terminal_ui import ProcessUI
from robotwin_annotation_v2.urdf_gripper_publisher import (
    UrdfGripperPublishError,
    validate_derivation_source_episode,
)


class JsonProgressWriter(io.TextIOBase):
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
def capture_urdf_json_progress(reporter: ProcessUI | None) -> Iterator[None]:
    """Capture URDF JSONL output and forward structured progress to ``reporter``."""

    if reporter is None:
        yield
        return
    writer = JsonProgressWriter(reporter)
    try:
        with contextlib.redirect_stdout(writer):
            yield
    finally:
        writer.finish()


def load_urdf_runner() -> Any:
    """Load the package-owned URDF batch engine at the execution boundary."""

    return importlib.import_module("robotwin_annotation_v2.application.urdf_batch")


def load_urdf_workflow_runtime() -> UrdfWorkflowRuntime:
    """Resolve the frozen-source runner and publisher only when URDF is selected."""

    runner = load_urdf_runner()
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


def release_sam_cuda_cache(gpus: Sequence[int]) -> dict[str, Any]:
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


def select_urdf_egl_device(
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


class ProcessEventSender(ProcessUI):
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
        self._connection.send(streaming.event(method, *args, **kwargs))

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
        self._connection.send(streaming.source_episode(episode_id, status))

    def note(self, message: str, *, level: str = "info") -> None:
        self._send("note", message, level=level)

    def detail(self, text: str) -> None:
        self._send("detail", text)


def object_source_process_entry(
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
    """Run one object-source pipeline in an isolated, spawn-safe process."""

    worker_log_path = output_root / "logs" / f"{run_id}.source-worker.log"
    worker_log_path.parent.mkdir(parents=True, exist_ok=True)
    worker_log = worker_log_path.open("a", encoding="utf-8", buffering=1)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = worker_log
    sys.stderr = worker_log
    reporter = ProcessEventSender(
        connection,
        lane_name="source" if incremental else None,
        episode_ids=episode_ids,
    )
    try:
        summary = DatasetPipeline(pipeline_config).run_sam(
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
            streaming.error(type(exc).__name__, str(exc), traceback.format_exc())
        )
    else:
        connection.send(streaming.source_result(summary))
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        worker_log.close()
        connection.close()


def incremental_urdf_process_entry(
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
        urdf_runner = load_urdf_runner()

        worker: Any | None = None
        completed = 0
        try:
            while True:
                item = ready_queue.get()
                if item is None:
                    break
                ready_episode = streaming.decode_ready_episode(item)
                episode_id = ready_episode.episode_id
                position = ready_episode.position
                connection.send(
                    streaming.event(
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
                    streaming.event(
                        "lane_progress",
                        "urdf",
                        completed,
                        len(run_config.episode_ids),
                        episode_id,
                        status,
                    )
                )
                connection.send(streaming.urdf_episode(episode_id, record))

            if worker is None:
                connection.send(
                    streaming.urdf_result(
                        None,
                        "no source episode became ready for the URDF worker",
                    )
                )
            else:
                try:
                    result = worker.finalize()
                except urdf_runner.UrdfBatchIncompleteError as exc:
                    connection.send(
                        streaming.urdf_result(
                            exc.result,
                            f"{type(exc).__name__}: {exc}",
                        )
                    )
                else:
                    connection.send(streaming.urdf_result(result, None))
        finally:
            if worker is not None:
                worker.close()
    except BaseException as exc:  # noqa: BLE001 - serialize child termination
        connection.send(
            streaming.error(type(exc).__name__, str(exc), traceback.format_exc())
        )
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        worker_log.close()
        connection.close()


def run_streaming_source_urdf_workers(
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
        target=object_source_process_entry,
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
        target=incremental_urdf_process_entry,
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
    ready_backlog: deque[streaming.ReadyEpisode] = deque()
    source_position = 0
    source_open = True
    urdf_open = True
    sentinel_sent = False

    def forward_event(message: streaming.EventMessage) -> None:
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
        decoded = streaming.try_decode_message(message)
        if isinstance(decoded, streaming.EventMessage):
            forward_event(decoded)
        elif isinstance(decoded, streaming.SourceEpisodeMessage):
            source_position += 1
            if decoded.status in {"completed", "skipped_complete"}:
                ready_backlog.append(
                    streaming.ReadyEpisode(int(decoded.episode_id), source_position)
                )
            elif reporter is not None:
                reporter.episode_finished(
                    int(decoded.episode_id), status=str(decoded.status)
                )
        elif isinstance(decoded, streaming.SourceResultMessage):
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
        elif isinstance(decoded, streaming.ErrorMessage):
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
        decoded = streaming.try_decode_message(message)
        if isinstance(decoded, streaming.EventMessage):
            forward_event(decoded)
        elif isinstance(decoded, streaming.UrdfEpisodeMessage):
            if decoded.record.get("status") != "complete" and reporter is not None:
                reporter.episode_finished(
                    int(decoded.episode_id),
                    status="gripper_incomplete",
                    detail=str(decoded.record.get("error", "URDF episode failed")),
                )
        elif isinstance(decoded, streaming.UrdfResultMessage):
            urdf_result_received = True
            raw_result, backend_error = decoded.result, decoded.error
            backend_result = None if raw_result is None else dict(raw_result)
        elif isinstance(decoded, streaming.ErrorMessage):
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
                        (
                            "source process exited without a result: "
                            f"exitcode={source_process.exitcode}"
                        ),
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
                        (
                            "URDF process exited without a result: "
                            f"exitcode={urdf_process.exitcode}"
                        ),
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


def run_object_source_process(
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
        target=object_source_process_entry,
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
        decoded = streaming.try_decode_message(message)
        if isinstance(decoded, streaming.EventMessage):
            if reporter is not None:
                getattr(reporter, decoded.method)(*decoded.args, **decoded.kwargs)
        elif isinstance(decoded, streaming.SourceResultMessage):
            summary = dict(decoded.summary)
        elif isinstance(decoded, streaming.ErrorMessage):
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


def validate_urdf_run_ownership(
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
    summary = read_json_object(summary_path, description="existing process summary")
    if summary.get("format_version") != PROCESS_SUMMARY_FORMAT_VERSION:
        raise ValueError("existing process summary format is not resumable")
    if summary.get("run_id") != run_id:
        raise ValueError("existing process summary run_id does not match its directory")
    backend = summary_gripper_backend(summary)
    if backend != "urdf":
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
    summary = read_json_object(
        source_root / "process_summary.json",
        description="source process summary",
    )
    if summary.get("format_version") != PROCESS_SUMMARY_FORMAT_VERSION:
        raise ValueError(
            "source process summary format must be "
            f"{PROCESS_SUMMARY_FORMAT_VERSION}"
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
        episode_dir = source_root / task / f"episode_{episode_id:06d}" / camera
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


__all__ = [
    "Full",
    "JsonProgressWriter",
    "ProcessEventSender",
    "capture_urdf_json_progress",
    "incremental_urdf_process_entry",
    "load_urdf_runner",
    "load_urdf_workflow_runtime",
    "mp",
    "object_source_process_entry",
    "release_sam_cuda_cache",
    "run_object_source_process",
    "run_streaming_source_urdf_workers",
    "select_urdf_egl_device",
    "select_urdf_source_episodes",
    "validate_urdf_run_ownership",
]
