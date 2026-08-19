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
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from queue import Full
from typing import Any

import av
import numpy as np
import pandas as pd

from robotwin_annotation_v2.adapters.artifact_store import ArtifactStore
from robotwin_annotation_v2.adapters.robotwin_dataset import RoboTwinDataset
from robotwin_annotation_v2.application.dataset_input import resolve_dataset_input
from robotwin_annotation_v2.config import PipelineConfig, load_config
from robotwin_annotation_v2.domain import (
    AnnotationMode,
    ObjectRole,
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

CHUNK_PATTERN = re.compile(r"chunk-(\d{3})$")
EPISODE_FILE_PATTERN = re.compile(r"episode_(\d+)\.parquet$")
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
DEFAULT_URDF_DEPTH_TOLERANCE_MM = 8.0
DEFAULT_URDF_MINIMUM_ELIGIBLE_NONEMPTY_FRACTION = 0.90
DEFAULT_URDF_PIPELINE_BUFFER_SIZE = 2
PROCESS_SUMMARY_FORMAT_VERSION = "robotwin_process_dataset_summary_v1"


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


@dataclass(frozen=True)
class DiscoveredEpisode:
    episode_id: int
    parquet: Path
    video: Path
    sidecar: Path


@dataclass(frozen=True)
class DiscoveryResult:
    episodes: tuple[DiscoveredEpisode, ...]
    skipped: tuple[dict[str, Any], ...]

    @property
    def episode_ids(self) -> tuple[int, ...]:
        return tuple(episode.episode_id for episode in self.episodes)


@dataclass(frozen=True)
class UrdfSourceSelection:
    """Frozen, QC-passed source masks selected for URDF replacement."""

    episode_ids: tuple[int, ...]
    excluded: tuple[dict[str, Any], ...]
    source_summary: Mapping[str, Any]
    source_lineages: Mapping[int, Mapping[str, Any]]
    annotation_mode: AnnotationMode
    required_object_roles: tuple[ObjectRole, ...]


@dataclass(frozen=True)
class SamRuntime:
    """SAM-only dependencies loaded after the backend has been selected."""

    qwen_client_factory: Callable[..., Any]
    backend_factory: Callable[..., Any]
    execution_errors: tuple[type[BaseException], ...]
    emit_gripper_result: Callable[..., Any]
    emit_sam_result: Callable[..., Any]
    execute_gripper_episode: Callable[..., Any]
    execute_sam_episode: Callable[..., Any]
    fatal_cuda_error: Callable[..., Any]
    gripper_episode_complete: Callable[..., Any]
    sam_episode_complete: Callable[..., Any]
    run_qwen: Callable[..., Any]


def _load_sam_runtime() -> SamRuntime:
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
        self._connection.send(("event", method, args, kwargs))

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
        self._connection.send(("source_episode", episode_id, status))

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
            (
                "error",
                type(exc).__name__,
                str(exc),
                traceback.format_exc(),
            )
        )
    else:
        connection.send(("result", summary))
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
                episode_id, position = item
                connection.send(
                    (
                        "event",
                        "lane_progress",
                        (
                            "urdf",
                            completed,
                            len(run_config.episode_ids),
                            episode_id,
                            "rendering",
                            f"source position {position}",
                        ),
                        {},
                    )
                )
                if worker is None:
                    worker = urdf_runner.IncrementalUrdfEpisodeWorker(run_config)
                record = worker.process_episode(episode_id)
                completed += 1
                status = str(record.get("status", "failed"))
                connection.send(
                    (
                        "event",
                        "lane_progress",
                        (
                            "urdf",
                            completed,
                            len(run_config.episode_ids),
                            episode_id,
                            status,
                        ),
                        {},
                    )
                )
                connection.send(("urdf_episode", episode_id, record))

            if worker is None:
                connection.send(
                    (
                        "result",
                        None,
                        "no source episode became ready for the URDF worker",
                    )
                )
            else:
                try:
                    result = worker.finalize()
                except urdf_runner.UrdfBatchIncompleteError as exc:
                    connection.send(
                        (
                            "result",
                            exc.result,
                            f"{type(exc).__name__}: {exc}",
                        )
                    )
                else:
                    connection.send(("result", result, None))
        finally:
            if worker is not None:
                worker.close()
    except BaseException as exc:  # noqa: BLE001 - serialize child termination
        connection.send(
            (
                "error",
                type(exc).__name__,
                str(exc),
                traceback.format_exc(),
            )
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
    ready_backlog: deque[tuple[int, int]] = deque()
    source_position = 0
    source_open = True
    urdf_open = True
    sentinel_sent = False

    def forward_event(message: tuple[Any, ...]) -> None:
        if reporter is not None:
            _, method, args, kwargs = message
            getattr(reporter, method)(*args, **kwargs)

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
        kind = message[0] if isinstance(message, tuple) and message else None
        if kind == "event":
            forward_event(message)
        elif kind == "source_episode":
            _, episode_id, status = message
            source_position += 1
            if status in {"completed", "skipped_complete"}:
                ready_backlog.append((int(episode_id), source_position))
            elif reporter is not None:
                reporter.episode_finished(int(episode_id), status=str(status))
        elif kind == "result":
            source_result_received = True
            raw_summary = message[1]
            if not isinstance(raw_summary, Mapping):
                child_error = (
                    "ProtocolError",
                    "source process returned a non-object summary",
                    "",
                )
            else:
                source_summary = dict(raw_summary)
        elif kind == "error":
            _, error_type, error, child_traceback = message
            child_error = (error_type, error, child_traceback)
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
        kind = message[0] if isinstance(message, tuple) and message else None
        if kind == "event":
            forward_event(message)
        elif kind == "urdf_episode":
            _, episode_id, record = message
            if record.get("status") != "complete" and reporter is not None:
                reporter.episode_finished(
                    int(episode_id),
                    status="gripper_incomplete",
                    detail=str(record.get("error", "URDF episode failed")),
                )
        elif kind == "result":
            urdf_result_received = True
            raw_result, backend_error = message[1], message[2]
            backend_result = None if raw_result is None else dict(raw_result)
        elif kind == "error":
            _, error_type, error, child_traceback = message
            child_error = (error_type, error, child_traceback)
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
        if not isinstance(message, tuple) or not message:
            child_error = ("ProtocolError", "invalid child-process message", "")
            return
        if message[0] == "event":
            if reporter is not None:
                _, method, args, kwargs = message
                getattr(reporter, method)(*args, **kwargs)
        elif message[0] == "result":
            summary = message[1]
        elif message[0] == "error":
            _, error_type, error, child_traceback = message
            child_error = (error_type, error, child_traceback)

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


def _episode_video_path(root: Path, camera: str, episode_id: int) -> Path:
    chunk = f"chunk-{episode_id // 1000:03d}"
    return (
        root
        / "videos"
        / chunk
        / f"observation.images.{camera}"
        / f"episode_{episode_id:06d}.mp4"
    )


def _episode_depth_path(root: Path, camera: str, episode_id: int) -> Path:
    chunk = f"chunk-{episode_id // 1000:03d}"
    return (
        root
        / "sidecars"
        / "videos"
        / chunk
        / f"observation.depths.{camera}"
        / f"episode_{episode_id:06d}.mkv"
    )


def discover_episodes(
    root: Path,
    *,
    camera: str,
    require_depth: bool = False,
) -> DiscoveryResult:
    """Discover complete dataset inputs by episode id."""

    dataset_root = root.expanduser().resolve()
    data_root = dataset_root / "data"
    if not data_root.is_dir():
        return DiscoveryResult((), ())
    discovered: dict[int, DiscoveredEpisode] = {}
    skipped: list[dict[str, Any]] = []
    for chunk_dir in sorted(data_root.iterdir()):
        if not chunk_dir.is_dir():
            continue
        parquet_files = sorted(chunk_dir.glob("episode_*.parquet"))
        if not parquet_files:
            continue
        match = CHUNK_PATTERN.fullmatch(chunk_dir.name)
        if match is None:
            raise ValueError(f"invalid chunk directory name: {chunk_dir.name}")
        for parquet in parquet_files:
            file_match = EPISODE_FILE_PATTERN.fullmatch(parquet.name)
            if file_match is None:
                raise ValueError(f"invalid episode parquet name: {parquet}")
            episode_id = int(file_match.group(1))
            expected_chunk = f"chunk-{episode_id // 1000:03d}"
            if chunk_dir.name != expected_chunk:
                raise ValueError(
                    f"episode {episode_id} is in {chunk_dir.name}, expected {expected_chunk}"
                )
            if episode_id in discovered:
                raise ValueError(f"duplicate episode id discovered: {episode_id}")
            video = _episode_video_path(dataset_root, camera, episode_id)
            sidecar = dataset_root / "sidecars" / f"episode_{episode_id:06d}.hdf5"
            required_paths = [
                ("video", video),
                ("sidecar", sidecar),
            ]
            if require_depth:
                required_paths.append(
                    (
                        "depth_video",
                        _episode_depth_path(dataset_root, camera, episode_id),
                    )
                )
            missing = [
                name
                for name, path in required_paths
                if not path.is_file()
            ]
            if missing:
                skipped.append(
                    {
                        "episode": episode_id,
                        "status": "discovery_skipped",
                        "missing": missing,
                        "parquet": str(parquet),
                    }
                )
                continue
            discovered[episode_id] = DiscoveredEpisode(
                episode_id=episode_id,
                parquet=parquet,
                video=video,
                sidecar=sidecar,
            )
    return DiscoveryResult(
        tuple(discovered[key] for key in sorted(discovered)),
        tuple(skipped),
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


def _parquet_frame_count(parquet: Path) -> int:
    frame = pd.read_parquet(parquet, columns=["frame_index"])
    if frame.empty:
        raise ValueError(f"episode parquet is empty: {parquet}")
    frame_indices = frame["frame_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(frame_indices, np.arange(len(frame_indices))):
        raise ValueError(f"episode frame_index is not contiguous: {parquet}")
    return len(frame_indices)


def _measure_episode(episode: DiscoveredEpisode) -> tuple[int, tuple[int, int], int]:
    frame_count = _parquet_frame_count(episode.parquet)
    raw_count = 0
    shape: tuple[int, int] | None = None
    with av.open(str(episode.video)) as container:
        for video_frame in container.decode(video=0):
            raw_count += 1
            if shape is None:
                shape = (int(video_frame.height), int(video_frame.width))
    if shape is None:
        raise ValueError(f"episode video contains no frames: {episode.video}")
    return frame_count, shape, raw_count - frame_count


def build_dynamic_manifest(
    root: Path,
    *,
    task: str,
    camera: str,
    episodes: Sequence[DiscoveredEpisode],
) -> dict[str, Any]:
    """Build the manifest contract expected by RoboTwinDataset in memory."""

    if not episodes:
        raise ValueError("cannot build a manifest without discovered episodes")
    frame_count, shape, surplus = _measure_episode(episodes[0])
    if frame_count < 1:
        raise ValueError("first discovered episode has no usable frames")
    return {
        "format_version": "robotwin_dataset_manifest_dynamic_v1",
        "task": task,
        "camera": camera,
        "frame_shape_hw": list(shape),
        "raw_video_frame_surplus": surplus,
        "usable_frame_count_source": "parquet",
        "dataset_root": str(root.expanduser().resolve()),
        "smoke_episode_ids": [episodes[0].episode_id],
        "regression_episode_ids": [episode.episode_id for episode in episodes],
        "required_relative_files": [
            "data/chunk-*/episode_{episode_id}.parquet",
            "videos/chunk-*/observation.images.{camera}/episode_{episode_id}.mp4",
            "sidecars/episode_{episode_id}.hdf5",
        ],
    }


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
    """Replace gripper masks in a frozen process run with canonical URDF output."""

    if dry_run and resume:
        raise ValueError("--dry-run and --resume cannot be used together")
    if run_id is not None:
        _validate_run_id(run_id)
    if reporter is not None and report_lifecycle:
        reporter.run_started(
            backend="urdf",
            dataset_root=str(dataset_root.expanduser().resolve()),
            task=task,
            camera=camera,
        )
        reporter.phase_started("dataset_contract")

    urdf_runner = _load_urdf_runner()
    from robotwin_annotation_v2.urdf_gripper_publisher import (
        publish_urdf_episode,
        validate_published_urdf_episode,
    )

    public_discovery = discover_episodes(dataset_root, camera=camera)
    discovery = discover_episodes(dataset_root, camera=camera, require_depth=True)
    dataset_excluded: list[dict[str, Any]] = [
        {
            "episode": int(record["episode"]),
            "status": "dataset_excluded",
            "reason": "dataset_inputs_missing",
            "missing": list(record["missing"]),
            "parquet": str(record["parquet"]),
        }
        for record in discovery.skipped
    ]
    expected_frame_counts: dict[int, int] = {}
    for discovered_episode in discovery.episodes:
        try:
            expected_frame_counts[discovered_episode.episode_id] = (
                _parquet_frame_count(discovered_episode.parquet)
            )
        except (KeyError, OSError, TypeError, ValueError) as exc:
            dataset_excluded.append(
                {
                    "episode": discovered_episode.episode_id,
                    "status": "dataset_excluded",
                    "reason": "dataset_parquet_invalid",
                    "error": f"{type(exc).__name__}: {exc}",
                    "parquet": str(discovered_episode.parquet),
                }
            )
    if reporter is not None:
        reporter.phase_finished(
            "dataset_contract",
            detail=(
                f"public={len(public_discovery.episodes)} "
                f"depth_eligible={len(expected_frame_counts)} "
                f"excluded={len(dataset_excluded)}"
            ),
        )
    eligible_dataset_ids = tuple(sorted(expected_frame_counts))
    relevant_dataset_excluded = dataset_excluded
    if episode_ids is not None:
        requested = set(episode_ids)
        relevant_dataset_excluded = [
            record for record in dataset_excluded if record["episode"] in requested
        ]
        if relevant_dataset_excluded:
            rendered = ", ".join(
                f"{record['episode']} ({record['reason']})"
                for record in relevant_dataset_excluded
            )
            raise ValueError(
                "requested URDF episodes do not satisfy the dataset contract: "
                f"{rendered}"
            )
    if not eligible_dataset_ids:
        raise ValueError(f"no complete URDF episodes found under {dataset_root}")
    if reporter is not None:
        reporter.phase_started("source_selection", total=len(eligible_dataset_ids))
    selection = select_urdf_source_episodes(
        source_run_dir,
        dataset_root=dataset_root,
        task=task,
        camera=camera,
        discovered_episode_ids=eligible_dataset_ids,
        requested_episode_ids=episode_ids,
        expected_frame_counts=expected_frame_counts,
    )
    if (
        pipeline_config is not None
        and pipeline_config.annotation.mode is not selection.annotation_mode
    ):
        raise ValueError(
            "pipeline config annotation mode differs from the frozen source run: "
            f"{pipeline_config.annotation.mode.value} != {selection.annotation_mode.value}"
        )
    all_excluded = sorted(
        [*relevant_dataset_excluded, *selection.excluded],
        key=lambda record: int(record["episode"]),
    )
    if reporter is not None:
        reporter.phase_finished(
            "source_selection",
            detail=(f"selected={len(selection.episode_ids)} excluded={len(all_excluded)}"),
        )
    if all_excluded and not allow_partial_source:
        examples = ", ".join(
            f"{record['episode']} ({record['reason']})"
            for record in all_excluded[:10]
        )
        suffix = "" if len(all_excluded) <= 10 else ", ..."
        raise ValueError(
            f"dataset/source contracts exclude {len(all_excluded)} episodes: "
            f"{examples}{suffix}; pass --allow-partial-source to process only the "
            "fully eligible subset"
        )
    if not math.isfinite(depth_tolerance_mm) or depth_tolerance_mm < 0:
        raise ValueError("URDF depth tolerance must be finite and non-negative")
    if not math.isfinite(minimum_eligible_nonempty_fraction) or not (
        0.0 <= minimum_eligible_nonempty_fraction <= 1.0
    ):
        raise ValueError(
            "URDF minimum eligible nonempty fraction must be finite and in [0, 1]"
        )
    selected_run_id = _validate_run_id(run_id or urdf_runner.new_run_id())
    resolved_output_root = output_root.expanduser().resolve()
    canonical_run_dir = resolved_output_root / selected_run_id
    _validate_urdf_run_ownership(
        canonical_run_dir,
        run_id=selected_run_id,
        resume=resume or prepared_backend_result is not None,
    )
    if reporter is not None and report_lifecycle:
        reporter.run_ready(run_id=selected_run_id, episode_ids=selection.episode_ids)
        if all_excluded:
            reporter.note(
                f"processing partial source: excluded={len(all_excluded)}",
                level="warning",
            )
    backend_output_root = canonical_run_dir / "_backend"
    config = urdf_runner.RunConfig(
        dataset_root=dataset_root.expanduser().resolve(),
        source_run_dir=source_run_dir.expanduser().resolve(),
        output_root=backend_output_root,
        run_id="urdf",
        urdf_path=urdf_path.expanduser().resolve(),
        mesh_root=None if mesh_root is None else mesh_root.expanduser().resolve(),
        episode_ids=selection.episode_ids,
        task=task,
        camera=camera,
        depth_tolerance_mm=depth_tolerance_mm,
        minimum_eligible_nonempty_fraction=minimum_eligible_nonempty_fraction,
        fit_config_json=(
            None if fit_config_json is None else fit_config_json.expanduser().resolve()
        ),
        # Public overlays are generated from the canonical masks by the shared renderer.
        skip_overlay=True,
        dry_run=dry_run,
        resume=resume,
        egl_device_id=egl_device_id,
    )
    runner = urdf_runner.run_experiment if experiment_runner is None else experiment_runner
    result: Mapping[str, Any]
    batch_error: str | None = prepared_backend_error
    if reporter is not None:
        reporter.phase_started("urdf_backend", total=len(selection.episode_ids))
    if prepared_backend_result is not None:
        result = prepared_backend_result
    else:
        try:
            with _captured_json_progress(reporter):
                result = runner(config)
        except urdf_runner.UrdfBatchIncompleteError as exc:
            result = exc.result
            batch_error = f"{type(exc).__name__}: {exc}"
    if reporter is not None:
        reporter.phase_finished(
            "urdf_backend",
            status="completed" if batch_error is None else "failed",
            detail=batch_error,
        )
        if not report_lifecycle:
            reporter.lane_progress(
                "urdf",
                len(selection.episode_ids),
                len(selection.episode_ids),
                status="completed" if batch_error is None else "failed",
                detail=batch_error,
            )
            reporter.lane_finished(
                "urdf",
                status="completed" if batch_error is None else "failed",
                detail=batch_error,
            )

    source_dynamic_manifest = selection.source_summary.get("dynamic_manifest")
    if not isinstance(source_dynamic_manifest, Mapping):
        raise ValueError("source process summary has no dynamic_manifest object")
    dynamic_manifest = dict(source_dynamic_manifest)
    source_annotation_mode = selection.annotation_mode.value
    source_required_roles = [
        role.value for role in selection.required_object_roles
    ]
    records: list[dict[str, Any]] = list(public_discovery.skipped)
    recorded_episode_ids = {
        int(record["episode"])
        for record in records
        if "episode" in record
    }
    for excluded_record in all_excluded:
        episode_id = int(excluded_record["episode"])
        if episode_id not in recorded_episode_ids:
            records.append(dict(excluded_record))
            recorded_episode_ids.add(episode_id)
    render_report: dict[str, Any] | None = None
    renderable_ids: list[int] = []
    published_source_lineages: dict[int, Mapping[str, Any]] = {}
    published_contexts: dict[int, dict[str, Any]] = {}
    published_episode_statuses: dict[int, str] = {}
    report_pipeline_lanes = reporter is not None and not report_lifecycle
    lane_episode_ids = (
        selection.episode_ids
        if pipeline_episode_ids is None
        else pipeline_episode_ids
    )
    lane_positions = {
        episode_id: position
        for position, episode_id in enumerate(lane_episode_ids, start=1)
    }
    lane_total = len(lane_episode_ids)

    def publish_progress(
        position: int,
        episode_id: int,
        status: str,
        detail: str | None = None,
    ) -> None:
        if report_pipeline_lanes and reporter is not None:
            reporter.lane_progress(
                "publish",
                lane_positions.get(episode_id, position),
                lane_total,
                episode_id,
                status,
                detail,
            )

    if dry_run:
        records.extend(
            {
                "episode": episode_id,
                "status": "planned",
                "gripper_backend": "urdf",
                "source_lineage_sha256": selection.source_lineages[episode_id][
                    "lineage_sha256"
                ],
            }
            for episode_id in selection.episode_ids
        )
        if reporter is not None:
            for episode_id in selection.episode_ids:
                reporter.episode_finished(episode_id, status="planned")
    else:
        raw_episodes = result.get("episodes")
        backend_records = raw_episodes if isinstance(raw_episodes, list) else []
        backend_by_episode: dict[int, Mapping[str, Any]] = {}
        for raw_record in backend_records:
            if not isinstance(raw_record, Mapping) or "episode_index" not in raw_record:
                continue
            backend_by_episode[int(raw_record["episode_index"])] = raw_record

        publisher = publish_urdf_episode if episode_publisher is None else episode_publisher
        for position, episode_id in enumerate(selection.episode_ids, start=1):
            if reporter is not None:
                reporter.episode_started(
                    episode_id,
                    position=position,
                    total=len(selection.episode_ids),
                )
            backend_record = backend_by_episode.get(episode_id)
            if backend_record is None:
                error = "URDF backend manifest has no episode record"
                records.append(
                    {
                        "episode": episode_id,
                        "status": "failed",
                        "gripper_backend": "urdf",
                        "error": error,
                    }
                )
                if reporter is not None:
                    reporter.episode_finished(
                        episode_id,
                        status="failed",
                        detail=error,
                    )
                publish_progress(position, episode_id, "failed", error)
                continue
            if backend_record.get("status") != "complete":
                backend_error = str(
                    backend_record.get("error", "URDF backend episode is incomplete")
                )
                episode_status = (
                    "gripper_incomplete"
                    if "eligible nonempty fraction" in backend_error
                    else "failed"
                )
                records.append(
                    {
                        "episode": episode_id,
                        "status": episode_status,
                        "gripper_backend": "urdf",
                        "error": backend_error,
                        "backend_status": backend_record.get("status"),
                    }
                )
                if reporter is not None:
                    reporter.episode_finished(
                        episode_id,
                        status=episode_status,
                        detail=backend_error,
                    )
                publish_progress(position, episode_id, episode_status, backend_error)
                continue
            frozen_lineage = selection.source_lineages[episode_id]
            if backend_record.get("source_lineage") != frozen_lineage:
                error = (
                    "URDF backend source lineage differs from the "
                    "preflight source contract"
                )
                records.append(
                    {
                        "episode": episode_id,
                        "status": "failed",
                        "gripper_backend": "urdf",
                        "error": error,
                    }
                )
                if reporter is not None:
                    reporter.episode_finished(
                        episode_id,
                        status="failed",
                        detail=error,
                    )
                publish_progress(position, episode_id, "failed", error)
                continue
            source_episode_dir = (
                config.source_run_dir
                / task
                / f"episode_{episode_id:06d}"
                / camera
            )
            backend_episode_dir = config.run_dir / str(
                backend_record.get("output_dir", f"episode_{episode_id:06d}")
            )
            destination_dir = (
                canonical_run_dir
                / task
                / f"episode_{episode_id:06d}"
                / camera
            )
            if reporter is not None:
                reporter.stage_started(episode_id, "canonical_publish")
            try:
                published = publisher(
                    source_episode_dir=source_episode_dir,
                    backend_episode_dir=backend_episode_dir,
                    destination_dir=destination_dir,
                    run_id=selected_run_id,
                    task=task,
                    camera=camera,
                    backend_episode_record=backend_record,
                    resume=resume,
                )
            except Exception as exc:
                error = f"canonical publish failed: {type(exc).__name__}: {exc}"
                records.append(
                    {
                        "episode": episode_id,
                        "status": "failed",
                        "gripper_backend": "urdf",
                        "error": error,
                    }
                )
                if reporter is not None:
                    reporter.stage_finished(
                        episode_id,
                        "canonical_publish",
                        status="failed",
                        detail=error,
                    )
                    reporter.episode_finished(
                        episode_id,
                        status="failed",
                        detail=error,
                    )
                publish_progress(position, episode_id, "failed", error)
                continue
            publish_status = str(published.get("status", "published"))
            episode_status = (
                "skipped_complete"
                if publish_status in {"validated_skip", "skipped_complete"}
                else "completed"
            )
            records.append(
                {
                    "episode": episode_id,
                    "status": episode_status,
                    "gripper_backend": "urdf",
                    "active_arm": backend_record.get("active_arm"),
                    "artifact_dir": str(destination_dir),
                    "backend_output_dir": str(backend_episode_dir),
                    "publish_status": publish_status,
                    "source_lineage_sha256": frozen_lineage["lineage_sha256"],
                }
            )
            if reporter is not None:
                reporter.stage_finished(
                    episode_id,
                    "canonical_publish",
                    status=episode_status,
                )
                if skip_render:
                    reporter.episode_finished(episode_id, status=episode_status)
            publish_progress(position, episode_id, episode_status)
            renderable_ids.append(episode_id)
            published_source_lineages[episode_id] = frozen_lineage
            published_episode_statuses[episode_id] = episode_status
            published_contexts[episode_id] = {
                "source_episode_dir": source_episode_dir,
                "backend_episode_dir": backend_episode_dir,
                "destination_dir": destination_dir,
                "backend_episode_record": backend_record,
            }

        if report_pipeline_lanes and reporter is not None:
            reporter.lane_progress(
                "publish",
                lane_total,
                lane_total,
                status="completed",
            )
            reporter.lane_finished("publish")

        if not skip_render and renderable_ids:
            pre_render_failures: list[dict[str, Any]] = []
            canonical_validator = (
                validate_published_urdf_episode
                if episode_validator is None
                else episode_validator
            )
            if reporter is not None:
                reporter.phase_started(
                    "canonical_validation",
                    total=len(renderable_ids),
                )
            selection_positions = lane_positions
            for phase_position, episode_id in enumerate(renderable_ids, start=1):
                position = selection_positions[episode_id]
                context = published_contexts[episode_id]
                try:
                    current_source = validate_derivation_source_episode(
                        context["source_episode_dir"],
                        task=task,
                        camera=camera,
                        episode_index=episode_id,
                        expected_frame_count=expected_frame_counts[episode_id],
                        expected_dataset_root=config.dataset_root,
                    )
                    if current_source.lineage != published_source_lineages[episode_id]:
                        raise UrdfGripperPublishError(
                            "source lineage differs from the published episode"
                        )
                except (FileNotFoundError, UrdfGripperPublishError) as exc:
                    pre_render_failures.append(
                        {
                            "episode": episode_id,
                            "status": "source_lineage_changed",
                            "gripper_backend": "urdf",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    if reporter is not None:
                        reporter.phase_progress(
                            phase_position,
                            total=len(renderable_ids),
                            episode_id=episode_id,
                            status="source_lineage_changed",
                        )
                        reporter.episode_finished(
                            episode_id,
                            status="source_lineage_changed",
                            detail=f"{type(exc).__name__}: {exc}",
                        )
                        if report_pipeline_lanes:
                            reporter.lane_progress(
                                "validation",
                                position,
                                lane_total,
                                episode_id,
                                "source_lineage_changed",
                                str(exc),
                            )
                    continue
                try:
                    canonical_validator(
                        **context,
                        run_id=selected_run_id,
                        task=task,
                        camera=camera,
                    )
                except Exception as exc:
                    pre_render_failures.append(
                        {
                            "episode": episode_id,
                            "status": "canonical_validation_failed",
                            "gripper_backend": "urdf",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    if reporter is not None:
                        reporter.phase_progress(
                            phase_position,
                            total=len(renderable_ids),
                            episode_id=episode_id,
                            status="canonical_validation_failed",
                        )
                        reporter.episode_finished(
                            episode_id,
                            status="canonical_validation_failed",
                            detail=f"{type(exc).__name__}: {exc}",
                        )
                        if report_pipeline_lanes:
                            reporter.lane_progress(
                                "validation",
                                position,
                                lane_total,
                                episode_id,
                                "canonical_validation_failed",
                                str(exc),
                            )
                    continue
                if reporter is not None:
                    reporter.phase_progress(
                        phase_position,
                        total=len(renderable_ids),
                        episode_id=episode_id,
                        status="validated",
                    )
                    if report_pipeline_lanes:
                        reporter.lane_progress(
                            "validation",
                            position,
                            lane_total,
                            episode_id,
                            "validated",
                        )
                    reporter.episode_finished(
                        episode_id,
                        status=published_episode_statuses[episode_id],
                    )
            if report_pipeline_lanes and reporter is not None:
                reporter.lane_progress(
                    "validation",
                    lane_total,
                    lane_total,
                    status="failed" if pre_render_failures else "completed",
                )
                reporter.lane_finished(
                    "validation",
                    status="failed" if pre_render_failures else "completed",
                    detail=(
                        f"failures={len(pre_render_failures)}"
                        if pre_render_failures
                        else None
                    ),
                )
            if pre_render_failures:
                records.extend(pre_render_failures)
                if reporter is not None:
                    reporter.phase_finished(
                        "canonical_validation",
                        status="failed",
                        detail=f"failures={len(pre_render_failures)}; render blocked",
                    )
                    if report_pipeline_lanes:
                        reporter.lane_progress(
                            "render",
                            lane_total,
                            lane_total,
                            status="skipped",
                            detail="blocked by canonical validation failures",
                        )
                        reporter.lane_finished(
                            "render",
                            status="skipped",
                            detail="blocked by canonical validation failures",
                        )
            else:
                if reporter is not None:
                    reporter.phase_finished("canonical_validation")
                    reporter.phase_started(
                        "canonical_render",
                        total=len(renderable_ids),
                    )
                try:
                    if pipeline_config is None:
                        raise ValueError(
                            "pipeline_config is required to render canonical URDF output"
                        )
                    dynamic = _dynamic_config(
                        pipeline_config,
                        root=config.dataset_root,
                        task=task,
                        camera=camera,
                        manifest=dynamic_manifest,
                        output_root=resolved_output_root,
                    )
                    if render_builder is None:
                        render_report = _render_processed(
                            dynamic,
                            run_id=selected_run_id,
                            episode_ids=tuple(renderable_ids),
                            output_dir=canonical_run_dir,
                            reporter=reporter,
                        )
                    else:
                        render_report = render_builder(
                            dynamic,
                            run_id=selected_run_id,
                            episode_ids=tuple(renderable_ids),
                            output_dir=canonical_run_dir,
                        )
                    if reporter is not None:
                        reporter.phase_finished("canonical_render")
                        if report_pipeline_lanes:
                            reporter.lane_progress(
                                "render",
                                lane_total,
                                lane_total,
                                status="completed",
                            )
                            reporter.lane_finished("render")
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    records.append(
                        {
                            "status": "render_failed",
                            "gripper_backend": "urdf",
                            "error": error,
                        }
                    )
                    if reporter is not None:
                        reporter.phase_finished(
                            "canonical_render",
                            status="render_failed",
                            detail=error,
                        )
                        if report_pipeline_lanes:
                            reporter.lane_progress(
                                "render",
                                lane_total,
                                lane_total,
                                status="failed",
                                detail=error,
                            )
                            reporter.lane_finished(
                                "render", status="failed", detail=error
                            )
        elif reporter is not None:
            reason = "disabled by --skip-render" if skip_render else "no publishable episodes"
            reporter.note(f"canonical_render skipped: {reason}", level="warning")
            if report_pipeline_lanes and not skip_render:
                reporter.lane_progress(
                    "validation",
                    lane_total,
                    lane_total,
                    status="skipped",
                    detail=reason,
                )
                reporter.lane_finished("validation", status="skipped", detail=reason)
                reporter.lane_progress(
                    "render",
                    lane_total,
                    lane_total,
                    status="skipped",
                    detail=reason,
                )
                reporter.lane_finished("render", status="skipped", detail=reason)

    failure_statuses = {
        "failed",
        "gripper_incomplete",
        "render_failed",
        "source_lineage_changed",
        "canonical_validation_failed",
    }
    summary = {
        "format_version": "robotwin_process_dataset_summary_v1",
        "annotation_mode": source_annotation_mode,
        "required_object_roles": list(source_required_roles),
        "gripper_backend": "urdf",
        "run_id": selected_run_id,
        "dataset_root": str(config.dataset_root),
        "task": task,
        "camera": camera,
        "discovered_episode_ids": list(public_discovery.episode_ids),
        "requested_episode_ids": (
            list(public_discovery.episode_ids)
            if episode_ids is None
            else list(episode_ids)
        ),
        "dynamic_manifest": dynamic_manifest,
        "qwen_health": (
            selection.source_summary.get("qwen_health")
            if source_mode in {"live_object_source_stage", "live_target_receiver_stage"}
            else None
        ),
        "records": records,
        "render": render_report,
        "fatal_error": batch_error,
        "backend": {
            "type": "urdf",
            "source_mode": source_mode,
            "source_release": (
                None if source_release is None else dict(source_release)
            ),
            "source_run_dir": str(config.source_run_dir),
            "source_run_id": selection.source_summary["run_id"],
            "source_lineage_sha256_by_episode": {
                str(episode_id): lineage["lineage_sha256"]
                for episode_id, lineage in selection.source_lineages.items()
            },
            "selected_episode_ids": list(selection.episode_ids),
            "dataset_excluded": relevant_dataset_excluded,
            "source_excluded": list(selection.excluded),
            "source_selection_complete": not all_excluded,
            "allow_partial_source": allow_partial_source,
            "run_dir": str(config.run_dir),
            "manifest": None if dry_run else str(config.run_dir / "manifest.json"),
            "status": result.get("status"),
            "error": batch_error,
        },
        "passed": (
            batch_error is None
            and not any(
                record.get("status") in failure_statuses for record in records
            )
        ),
    }
    if dry_run:
        summary["plan"] = result
        return summary
    summary_path = ArtifactStore.write_json(
        canonical_run_dir / "process_summary.json",
        summary,
    )
    summary["artifact"] = str(summary_path)
    return summary


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
    """Run Qwen and object SAM, optionally followed by the SAM gripper stage.

    ``object_source_only`` describes stage scope, not annotation semantics.  The
    semantic mode always comes from ``config.annotation.mode``.  The deprecated
    ``target_receiver_only`` keyword remains accepted for old callers while new
    code uses the role-neutral name.
    """

    if object_source_only is not None and target_receiver_only:
        raise ValueError(
            "object_source_only and deprecated target_receiver_only cannot both be set"
        )
    source_only = (
        target_receiver_only
        if object_source_only is None
        else bool(object_source_only)
    )
    if config.annotation.mode is AnnotationMode.TARGET_ONLY and not source_only:
        raise ValueError(
            "target_only does not support --gripper-backend sam; use the default URDF backend"
        )
    if incremental_source and not source_only:
        raise ValueError("incremental source receipts require object_source_only mode")
    if run_id is not None:
        _validate_run_id(run_id)
    if reporter is not None:
        if report_lifecycle:
            reporter.run_started(
                backend="sam",
                dataset_root=str(dataset_root.expanduser().resolve()),
                task=task,
                camera=camera,
            )
        reporter.phase_started("dataset_discovery")
    runtime = _load_sam_runtime()
    discovery = discover_episodes(dataset_root, camera=camera)
    if not discovery.episodes:
        raise ValueError(f"no complete episodes found under {dataset_root}")
    manifest = build_dynamic_manifest(
        dataset_root,
        task=task,
        camera=camera,
        episodes=discovery.episodes,
    )
    discovered_ids = set(discovery.episode_ids)
    selected_ids = (
        discovery.episode_ids
        if episode_ids is None
        else tuple(dict.fromkeys(int(value) for value in episode_ids))
    )
    if not selected_ids:
        raise ValueError("process_dataset requires at least one selected episode")
    unknown = sorted(set(selected_ids) - discovered_ids)
    if unknown:
        raise ValueError(f"requested episodes were not discovered: {unknown}")
    if reporter is not None:
        reporter.phase_finished(
            "dataset_discovery",
            detail=(
                f"discovered={len(discovery.episodes)} "
                f"selected={len(selected_ids)} skipped_inputs={len(discovery.skipped)}"
            ),
        )
    dynamic = _dynamic_config(
        config,
        root=dataset_root,
        task=task,
        camera=camera,
        manifest=manifest,
        output_root=output_root,
    )
    store = ArtifactStore(dynamic.output_root)
    selected_run_id = _validate_run_id(run_id or store.new_run_id())
    canonical_run_dir = store.run_dir(selected_run_id)
    _validate_sam_run_ownership(canonical_run_dir, run_id=selected_run_id)
    if incremental_source:
        write_source_run_contract(
            canonical_run_dir,
            run_id=selected_run_id,
            dataset_root=dataset_root,
            task=task,
            camera=camera,
            dynamic_manifest=manifest,
            requested_episode_ids=selected_ids,
            annotation_mode=config.annotation.mode.value,
            required_object_roles=config.annotation.spec.required_role_names,
        )

    def episode_terminal(episode_id: int, status: str) -> None:
        if incremental_source and status in {"completed", "skipped_complete"}:
            ref = EpisodeRef(task, episode_id, camera)
            write_source_episode_completion_receipt(
                store.episode_dir(selected_run_id, ref),
                task=task,
                camera=camera,
                episode_index=episode_id,
                status=status,
                expected_dataset_root=dataset_root,
            )
        if episode_terminal_callback is not None:
            episode_terminal_callback(episode_id, status)

    if reporter is not None and report_lifecycle:
        reporter.run_ready(run_id=selected_run_id, episode_ids=selected_ids)
    if reporter is not None:
        reporter.phase_started("qwen_health")
    qwen = runtime.qwen_client_factory(
        endpoint=dynamic.qwen.endpoint,
        model=dynamic.qwen.model,
        timeout_seconds=dynamic.qwen.timeout_seconds,
    )
    health = qwen.health()
    if reporter is not None:
        reporter.phase_finished("qwen_health")
        reporter.phase_started("resume_scan", total=len(selected_ids))
    records: list[dict[str, Any]] = list(discovery.skipped)
    pending: list[int] = []
    completion_check = (
        runtime.sam_episode_complete
        if source_only
        else runtime.gripper_episode_complete
    )
    for position, episode_id in enumerate(selected_ids, start=1):
        ref = EpisodeRef(
            task,
            episode_id,
            camera,
        )
        if not force and completion_check(dynamic, store, selected_run_id, ref):
            records.append({"episode": episode_id, "status": "skipped_complete"})
            episode_terminal(episode_id, "skipped_complete")
            if reporter is not None and report_lifecycle:
                reporter.episode_finished(episode_id, status="skipped_complete")
        else:
            pending.append(episode_id)
        if reporter is not None:
            reporter.phase_progress(
                position,
                total=len(selected_ids),
                episode_id=episode_id,
                status=("pending" if episode_id in pending else "skipped_complete"),
            )
    if reporter is not None:
        reporter.phase_finished(
            "resume_scan",
            detail=f"pending={len(pending)} skipped={len(selected_ids) - len(pending)}",
        )

    factory = runtime.backend_factory if backend_factory is None else backend_factory
    backend: Any | None = None
    fatal_error: BaseException | None = None
    try:
        if pending:
            if reporter is not None:
                reporter.phase_started("sam_backend_load")
            try:
                backend = factory(
                    checkpoint_path=dynamic.sam3.checkpoint,
                    gpus=dynamic.sam3.gpus,
                )
            except Exception as exc:
                if reporter is not None:
                    reporter.phase_finished(
                        "sam_backend_load",
                        status="failed",
                        detail=f"{type(exc).__name__}: {exc}",
                    )
                raise
            if reporter is not None:
                reporter.phase_finished("sam_backend_load")
            for episode_id in pending:
                position = selected_ids.index(episode_id) + 1
                current_stage: str | None = None
                if reporter is not None and report_lifecycle:
                    reporter.episode_started(
                        episode_id,
                        position=position,
                        total=len(selected_ids),
                    )
                try:
                    current_stage = "qwen"
                    if reporter is not None:
                        reporter.stage_started(episode_id, current_stage)
                    with _captured_stage_output(reporter):
                        runtime.run_qwen(dynamic, episode_id, selected_run_id)
                    if reporter is not None:
                        reporter.stage_finished(episode_id, current_stage)
                    current_stage = "object_sam"
                    if reporter is not None:
                        reporter.stage_started(episode_id, current_stage)
                    sam_execution = runtime.execute_sam_episode(
                        dynamic,
                        episode_id,
                        selected_run_id,
                        backend,
                    )
                    with _captured_stage_output(reporter):
                        sam_complete = runtime.emit_sam_result(
                            selected_run_id,
                            sam_execution,
                        )
                    if not sam_complete:
                        records.append({"episode": episode_id, "status": "sam_incomplete"})
                        episode_terminal(episode_id, "sam_incomplete")
                        if reporter is not None:
                            reporter.stage_finished(
                                episode_id,
                                current_stage,
                                status="sam_incomplete",
                            )
                            if report_lifecycle:
                                reporter.episode_finished(
                                    episode_id,
                                    status="sam_incomplete",
                                )
                        current_stage = None
                        continue
                    if reporter is not None:
                        reporter.stage_finished(episode_id, current_stage)
                    if source_only:
                        records.append(
                            {
                                "episode": episode_id,
                                "status": "completed",
                            }
                        )
                        episode_terminal(episode_id, "completed")
                        if reporter is not None and report_lifecycle:
                            reporter.episode_finished(
                                episode_id,
                                status="completed",
                            )
                        current_stage = None
                        continue
                    current_stage = "gripper_sam"
                    if reporter is not None:
                        reporter.stage_started(episode_id, current_stage)
                    gripper_execution = runtime.execute_gripper_episode(
                        dynamic,
                        episode_id,
                        selected_run_id,
                        backend,
                    )
                    with _captured_stage_output(reporter):
                        gripper_complete = runtime.emit_gripper_result(
                            selected_run_id,
                            gripper_execution,
                        )
                    episode_status = "completed" if gripper_complete else "gripper_incomplete"
                    records.append(
                        {
                            "episode": episode_id,
                            "status": episode_status,
                        }
                    )
                    episode_terminal(episode_id, episode_status)
                    if reporter is not None:
                        reporter.stage_finished(
                            episode_id,
                            current_stage,
                            status=episode_status,
                        )
                        if report_lifecycle:
                            reporter.episode_finished(
                                episode_id,
                                status=episode_status,
                            )
                    current_stage = None
                except SystemExit as exc:
                    error = f"stage exited with code {exc.code}"
                    records.append(
                        {
                            "episode": episode_id,
                            "status": "failed",
                            "error": error,
                        }
                    )
                    episode_terminal(episode_id, "failed")
                    if reporter is not None:
                        if current_stage is not None:
                            reporter.stage_finished(
                                episode_id,
                                current_stage,
                                status="failed",
                                detail=error,
                            )
                        if report_lifecycle:
                            reporter.episode_finished(
                                episode_id,
                                status="failed",
                                detail=error,
                            )
                except runtime.execution_errors as exc:
                    error = str(exc)
                    records.append(
                        {
                            "episode": episode_id,
                            "status": "failed",
                            "error": error,
                        }
                    )
                    episode_terminal(episode_id, "failed")
                    if reporter is not None:
                        if current_stage is not None:
                            reporter.stage_finished(
                                episode_id,
                                current_stage,
                                status="failed",
                                detail=error,
                            )
                        if report_lifecycle:
                            reporter.episode_finished(
                                episode_id,
                                status="failed",
                                detail=error,
                            )
                    if runtime.fatal_cuda_error(exc):
                        fatal_error = exc
                        break
    finally:
        if backend is not None:
            backend.shutdown()

    if fatal_error is not None:
        recorded_ids = {
            int(record["episode"])
            for record in records
            if "episode" in record and str(record["episode"]).isdigit()
        }
        records.extend(
            {"episode": episode_id, "status": "not_run_after_fatal_cuda"}
            for episode_id in selected_ids
            if episode_id not in recorded_ids
        )
        for episode_id in selected_ids:
            if episode_id not in recorded_ids:
                episode_terminal(episode_id, "not_run_after_fatal_cuda")
        if reporter is not None and report_lifecycle:
            for episode_id in selected_ids:
                if episode_id not in recorded_ids:
                    reporter.episode_finished(
                        episode_id,
                        status="not_run_after_fatal_cuda",
                        detail=str(fatal_error),
                    )

    render_report: dict[str, Any] | None = None
    renderable_ids = tuple(
        int(record["episode"])
        for record in records
        if record.get("status") in {"completed", "skipped_complete"}
    )
    if (
        not source_only
        and not skip_render
        and fatal_error is None
        and renderable_ids
    ):
        if reporter is not None:
            reporter.phase_started("canonical_render", total=len(renderable_ids))
        try:
            render_report = _render_processed(
                dynamic,
                run_id=selected_run_id,
                episode_ids=renderable_ids,
                output_dir=canonical_run_dir,
                reporter=reporter,
            )
            if reporter is not None:
                reporter.phase_finished("canonical_render")
        except Exception as exc:
            records.append({"status": "render_failed", "error": str(exc)})
            if reporter is not None:
                reporter.phase_finished(
                    "canonical_render",
                    status="render_failed",
                    detail=f"{type(exc).__name__}: {exc}",
                )
    elif reporter is not None:
        reason = (
            "object source stage"
            if source_only
            else "disabled by --skip-render"
            if skip_render
            else "blocked by fatal CUDA error"
            if fatal_error is not None
            else "no renderable episodes"
        )
        reporter.note(f"canonical_render skipped: {reason}", level="warning")

    summary = {
        "format_version": "robotwin_process_dataset_summary_v1",
        "annotation_mode": config.annotation.mode.value,
        "required_object_roles": list(config.annotation.spec.required_role_names),
        "gripper_backend": None if source_only else "sam",
        "run_id": selected_run_id,
        "dataset_root": str(dataset_root.expanduser().resolve()),
        "task": task,
        "camera": camera,
        "discovered_episode_ids": list(discovery.episode_ids),
        "requested_episode_ids": list(selected_ids),
        "dynamic_manifest": manifest,
        "qwen_health": health,
        "records": records,
        "render": render_report,
        "fatal_error": None if fatal_error is None else str(fatal_error),
        "backend": {
            "object_masks": "sam",
            "gripper": None if source_only else "sam",
        },
        "stage_mode": (
            "object_source_only" if source_only else "full_sam"
        ),
    }
    failure_statuses = {
        "failed",
        "sam_incomplete",
        "gripper_incomplete",
        "not_run_after_fatal_cuda",
        "render_failed",
    }
    summary["passed"] = (
        fatal_error is None
        and not any(record.get("status") in failure_statuses for record in records)
    )
    summary_path = store.write_json(
        canonical_run_dir / "process_summary.json",
        summary,
    )
    summary["artifact"] = str(summary_path)
    return summary


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
