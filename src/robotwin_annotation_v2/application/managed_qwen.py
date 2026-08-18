"""Start and stop a local Qwen service around a dataset process."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, BinaryIO, Self
from urllib.parse import urlsplit, urlunsplit

DEFAULT_MIN_FREE_MEMORY_MIB = 60_000
DEFAULT_STARTUP_TIMEOUT_SECONDS = 600.0
DEFAULT_SHUTDOWN_TIMEOUT_SECONDS = 30.0
DEFAULT_PROBE_TIMEOUT_SECONDS = 1.0


class ManagedQwenError(RuntimeError):
    """A local Qwen service could not be reused or managed safely."""


@dataclass(frozen=True)
class GpuStatus:
    """One physical GPU reported by nvidia-smi."""

    index: int
    uuid: str
    free_memory_mib: int
    total_memory_mib: int
    utilization_percent: int


@dataclass(frozen=True)
class ManagedQwenSettings:
    """Inputs needed to launch the repository's standalone Qwen server."""

    endpoint: str
    model_name: str
    python_executable: Path
    server_script: Path
    model_path: Path
    log_directory: Path
    excluded_gpu_ids: tuple[int, ...] = ()
    minimum_free_memory_mib: int = DEFAULT_MIN_FREE_MEMORY_MIB
    startup_timeout_seconds: float = DEFAULT_STARTUP_TIMEOUT_SECONDS
    shutdown_timeout_seconds: float = DEFAULT_SHUTDOWN_TIMEOUT_SECONDS
    probe_timeout_seconds: float = DEFAULT_PROBE_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        if not self.endpoint.strip() or not self.model_name.strip():
            raise ValueError("Qwen endpoint and model name must be non-empty")
        if self.minimum_free_memory_mib < 0:
            raise ValueError("minimum Qwen free memory must be non-negative")
        if (
            self.startup_timeout_seconds <= 0
            or self.shutdown_timeout_seconds <= 0
            or self.probe_timeout_seconds <= 0
        ):
            raise ValueError("Qwen lifecycle timeouts must be positive")
        if any(device_id < 0 for device_id in self.excluded_gpu_ids):
            raise ValueError("excluded Qwen GPU ids must be non-negative")


def parse_gpu_inventory(output: str) -> tuple[GpuStatus, ...]:
    """Parse the stable CSV emitted by the managed nvidia-smi query."""

    inventory: list[GpuStatus] = []
    for line_number, raw_line in enumerate(output.splitlines(), start=1):
        if not raw_line.strip():
            continue
        parts = [part.strip() for part in raw_line.split(",")]
        if len(parts) != 5:
            raise ManagedQwenError(f"invalid nvidia-smi row {line_number}: expected 5 columns")
        try:
            inventory.append(
                GpuStatus(
                    index=int(parts[0]),
                    uuid=parts[1],
                    free_memory_mib=int(parts[2]),
                    total_memory_mib=int(parts[3]),
                    utilization_percent=int(parts[4]),
                )
            )
        except ValueError as exc:
            raise ManagedQwenError(
                f"invalid numeric value in nvidia-smi row {line_number}"
            ) from exc
    if not inventory:
        raise ManagedQwenError("nvidia-smi reported no GPUs")
    return tuple(inventory)


def query_gpu_inventory() -> tuple[GpuStatus, ...]:
    """Return physical GPUs and their current free-memory state."""

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.free,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise ManagedQwenError("nvidia-smi is unavailable; cannot select a Qwen GPU") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        suffix = f": {detail}" if detail else ""
        raise ManagedQwenError(f"nvidia-smi GPU query failed{suffix}") from exc
    return parse_gpu_inventory(completed.stdout)


def select_qwen_gpu(
    inventory: tuple[GpuStatus, ...],
    *,
    excluded_gpu_ids: tuple[int, ...],
    minimum_free_memory_mib: int,
) -> GpuStatus:
    """Select the freest eligible GPU, using utilization and index as tie-breakers."""

    excluded = set(excluded_gpu_ids)
    candidates = [
        gpu
        for gpu in inventory
        if gpu.index not in excluded and gpu.free_memory_mib >= minimum_free_memory_mib
    ]
    if candidates:
        return min(
            candidates,
            key=lambda gpu: (
                -gpu.free_memory_mib,
                gpu.utilization_percent,
                gpu.index,
            ),
        )
    available = ", ".join(
        f"gpu {gpu.index}: {gpu.free_memory_mib} MiB free"
        for gpu in sorted(inventory, key=lambda gpu: gpu.index)
        if gpu.index not in excluded
    )
    excluded_text = ", ".join(str(value) for value in sorted(excluded)) or "none"
    raise ManagedQwenError(
        "no eligible GPU has at least "
        f"{minimum_free_memory_mib} MiB free (excluded: {excluded_text}; "
        f"available: {available or 'none'})"
    )


def qwen_health_endpoint(endpoint: str) -> str:
    """Derive the standalone server health URL from its completion endpoint."""

    parsed = urlsplit(endpoint)
    path = parsed.path.rstrip("/")
    completion_suffix = "/v1/chat/completions"
    health_path = (
        f"{path[: -len(completion_suffix)]}/health"
        if path.endswith(completion_suffix)
        else "/health"
    )
    return urlunsplit((parsed.scheme, parsed.netloc, health_path, "", ""))


def probe_qwen_health(
    endpoint: str,
    *,
    expected_model: str,
    timeout_seconds: float,
) -> dict[str, Any] | None:
    """Return health data, or None only when no service can be reached."""

    try:
        request = urllib.request.Request(qwen_health_endpoint(endpoint), method="GET")
    except ValueError as exc:
        raise ManagedQwenError(f"invalid Qwen endpoint: {endpoint}") from exc
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise ManagedQwenError(f"Qwen health endpoint returned HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError):
        return None
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ManagedQwenError("Qwen health endpoint returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ManagedQwenError("Qwen health endpoint must return a JSON object")
    if payload.get("status") != "ok":
        raise ManagedQwenError(f"Qwen health check is not ok: {payload}")
    actual_model = payload.get("model")
    if actual_model != expected_model:
        raise ManagedQwenError(
            f"Qwen endpoint serves model {actual_model!r}, expected {expected_model!r}"
        )
    return payload


def _local_server_address(endpoint: str) -> tuple[str, int]:
    parsed = urlsplit(endpoint)
    if parsed.scheme != "http":
        raise ManagedQwenError("automatic Qwen startup requires an http endpoint")
    hostname = parsed.hostname
    if hostname not in {"127.0.0.1", "localhost"}:
        raise ManagedQwenError(
            "automatic Qwen startup is limited to 127.0.0.1 or localhost endpoints"
        )
    try:
        port = parsed.port or 80
    except ValueError as exc:
        raise ManagedQwenError(f"invalid Qwen endpoint port: {endpoint}") from exc
    return "127.0.0.1", port


def _log_tail(path: Path, *, limit_bytes: int = 4000) -> str:
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - limit_bytes))
            return stream.read().decode("utf-8", errors="replace").strip()
    except OSError:
        return ""


class ManagedQwenService:
    """Reuse a healthy endpoint or own one local server for a bounded lifetime."""

    def __init__(self, settings: ManagedQwenSettings) -> None:
        self.settings = settings
        self.health: dict[str, Any] | None = None
        self.selected_gpu: GpuStatus | None = None
        self.log_path: Path | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._log_stream: BinaryIO | None = None

    @property
    def owns_process(self) -> bool:
        return self._process is not None

    def __enter__(self) -> Self:
        health = probe_qwen_health(
            self.settings.endpoint,
            expected_model=self.settings.model_name,
            timeout_seconds=self.settings.probe_timeout_seconds,
        )
        if health is not None:
            self.health = health
            print(
                "Qwen service is already healthy; reusing it without taking ownership",
                file=sys.stderr,
                flush=True,
            )
            return self
        self._start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.stop()

    def _start(self) -> None:
        host, port = _local_server_address(self.settings.endpoint)
        if not self.settings.python_executable.is_file() or not os.access(
            self.settings.python_executable, os.X_OK
        ):
            raise ManagedQwenError(
                f"Qwen Python is missing or not executable: {self.settings.python_executable}"
            )
        if not self.settings.server_script.is_file():
            raise ManagedQwenError(f"Qwen server script is missing: {self.settings.server_script}")
        if not self.settings.model_path.is_dir():
            raise ManagedQwenError(f"Qwen model directory is missing: {self.settings.model_path}")
        inventory = query_gpu_inventory()
        selected = select_qwen_gpu(
            inventory,
            excluded_gpu_ids=self.settings.excluded_gpu_ids,
            minimum_free_memory_mib=self.settings.minimum_free_memory_mib,
        )
        self.settings.log_directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        log_path = self.settings.log_directory / f"managed-qwen-{stamp}-{os.getpid()}.log"
        log_stream = log_path.open("ab")
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = selected.uuid
        environment["PYTHONUNBUFFERED"] = "1"
        command = [
            str(self.settings.python_executable),
            str(self.settings.server_script),
            "--model",
            str(self.settings.model_path),
            "--served-model-name",
            self.settings.model_name,
            "--host",
            host,
            "--port",
            str(port),
            "--device",
            "cuda:0",
        ]
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=True,
            )
        except OSError as exc:
            log_stream.close()
            raise ManagedQwenError(f"failed to launch Qwen server: {exc}") from exc
        self.selected_gpu = selected
        self.log_path = log_path
        self._log_stream = log_stream
        self._process = process
        print(
            f"Starting managed Qwen on physical GPU {selected.index} "
            f"({selected.free_memory_mib} MiB free); log: {log_path}",
            file=sys.stderr,
            flush=True,
        )
        try:
            self._wait_until_ready()
        except BaseException:
            self.stop()
            raise

    def _wait_until_ready(self) -> None:
        process = self._process
        if process is None:
            raise AssertionError("managed Qwen process was not started")
        deadline = time.monotonic() + self.settings.startup_timeout_seconds
        while True:
            health = probe_qwen_health(
                self.settings.endpoint,
                expected_model=self.settings.model_name,
                timeout_seconds=self.settings.probe_timeout_seconds,
            )
            if health is not None:
                self.health = health
                print(
                    f"Managed Qwen is ready (pid={process.pid})",
                    file=sys.stderr,
                    flush=True,
                )
                return
            return_code = process.poll()
            if return_code is not None:
                detail = _log_tail(self.log_path) if self.log_path is not None else ""
                suffix = f"\n{detail}" if detail else ""
                raise ManagedQwenError(
                    f"Qwen server exited during startup with code {return_code}{suffix}"
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                detail = _log_tail(self.log_path) if self.log_path is not None else ""
                suffix = f"\n{detail}" if detail else ""
                raise ManagedQwenError(
                    f"Qwen server was not healthy after "
                    f"{self.settings.startup_timeout_seconds:g} seconds{suffix}"
                )
            time.sleep(min(1.0, remaining))

    def wait(self) -> int:
        """Wait for an owned service, as used by the manual serve recipe."""

        if self._process is None:
            return 0
        return self._process.wait()

    def stop(self) -> None:
        """Stop only the server process group launched by this manager."""

        process = self._process
        if process is None:
            return
        if process.poll() is None:
            print(
                f"Stopping managed Qwen (pid={process.pid})",
                file=sys.stderr,
                flush=True,
            )
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=self.settings.shutdown_timeout_seconds)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        if self._log_stream is not None:
            self._log_stream.close()
        self._process = None
        self._log_stream = None
