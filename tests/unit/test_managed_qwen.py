from __future__ import annotations

import socket
import sys
from pathlib import Path
from typing import Any

import pytest

import scripts.manage_qwen_process as manager_script
from robotwin_annotation_v2.application import managed_qwen
from robotwin_annotation_v2.application.managed_qwen import (
    GpuStatus,
    ManagedQwenError,
    ManagedQwenService,
    ManagedQwenSettings,
    parse_gpu_inventory,
    probe_qwen_health,
    qwen_health_endpoint,
    select_qwen_gpu,
)


def _settings(tmp_path: Path) -> ManagedQwenSettings:
    python_executable = tmp_path / "qwen-python"
    python_executable.write_text("", encoding="utf-8")
    python_executable.chmod(0o755)
    server_script = tmp_path / "serve_qwen.py"
    server_script.write_text("", encoding="utf-8")
    model_path = tmp_path / "model"
    model_path.mkdir()
    return ManagedQwenSettings(
        endpoint="http://127.0.0.1:18086/v1/chat/completions",
        model_name="qwen3.5-27b",
        python_executable=python_executable,
        server_script=server_script,
        model_path=model_path,
        log_directory=tmp_path / "logs",
        excluded_gpu_ids=(2,),
    )


def test_select_qwen_gpu_uses_freest_eligible_device() -> None:
    inventory = parse_gpu_inventory(
        "0, GPU-zero, 50000, 80000, 0\n"
        "1, GPU-one, 70000, 80000, 8\n"
        "2, GPU-two, 79000, 80000, 0\n"
        "3, GPU-three, 70000, 80000, 2"
    )

    selected = select_qwen_gpu(
        inventory,
        excluded_gpu_ids=(2,),
        minimum_free_memory_mib=60_000,
    )

    assert selected.index == 3
    assert selected.uuid == "GPU-three"


def test_select_qwen_gpu_reports_memory_and_exclusions() -> None:
    inventory = (
        GpuStatus(0, "GPU-zero", 50_000, 80_000, 0),
        GpuStatus(2, "GPU-two", 79_000, 80_000, 0),
    )

    with pytest.raises(ManagedQwenError, match=r"60000 MiB.*excluded: 2.*gpu 0: 50000"):
        select_qwen_gpu(
            inventory,
            excluded_gpu_ids=(2,),
            minimum_free_memory_mib=60_000,
        )


def test_qwen_health_endpoint_preserves_url_prefix() -> None:
    assert (
        qwen_health_endpoint("http://localhost:18086/api/v1/chat/completions")
        == "http://localhost:18086/api/health"
    )


def test_managed_service_reuses_existing_server_without_ownership(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    health = {"status": "ok", "model": "qwen3.5-27b", "pid": 12}
    monkeypatch.setattr(managed_qwen, "probe_qwen_health", lambda *args, **kwargs: health)

    def unexpected_gpu_query() -> tuple[GpuStatus, ...]:
        raise AssertionError("GPU inventory must not be queried for a healthy service")

    monkeypatch.setattr(managed_qwen, "query_gpu_inventory", unexpected_gpu_query)

    with ManagedQwenService(_settings(tmp_path)) as service:
        assert service.health == health
        assert not service.owns_process


def test_managed_service_starts_selected_gpu_and_stops_owned_group(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    probes: Any = iter(
        (
            None,
            {"status": "ok", "model": "qwen3.5-27b", "pid": 4242},
        )
    )
    monkeypatch.setattr(
        managed_qwen,
        "probe_qwen_health",
        lambda *args, **kwargs: next(probes),
    )
    inventory = (
        GpuStatus(1, "GPU-one", 90_000, 100_000, 0),
        GpuStatus(2, "GPU-two", 99_000, 100_000, 0),
    )
    monkeypatch.setattr(managed_qwen, "query_gpu_inventory", lambda: inventory)

    class FakeProcess:
        pid = 4242
        returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            self.returncode = 0
            return 0

    process = FakeProcess()
    launch: dict[str, Any] = {}

    def fake_popen(command: list[str], **kwargs: Any) -> FakeProcess:
        launch["command"] = command
        launch.update(kwargs)
        return process

    kill_calls: list[tuple[int, int]] = []
    monkeypatch.setattr(managed_qwen.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        managed_qwen.os,
        "killpg",
        lambda pid, signum: kill_calls.append((pid, signum)),
    )

    with ManagedQwenService(_settings(tmp_path)) as service:
        assert service.owns_process
        assert service.selected_gpu is not None
        assert service.selected_gpu.index == 1
        assert launch["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-one"
        assert launch["command"][-2:] == ["--device", "cuda:0"]
        assert launch["start_new_session"] is True

    assert kill_calls == [(4242, managed_qwen.signal.SIGTERM)]
    assert process.returncode == 0
    assert not service.owns_process


def test_managed_service_cleans_up_failed_start(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(managed_qwen, "probe_qwen_health", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        managed_qwen,
        "query_gpu_inventory",
        lambda: (GpuStatus(1, "GPU-one", 90_000, 100_000, 0),),
    )

    class FailedProcess:
        pid = 99

        @staticmethod
        def poll() -> int:
            return 7

    monkeypatch.setattr(
        managed_qwen.subprocess,
        "Popen",
        lambda *args, **kwargs: FailedProcess(),
    )
    service = ManagedQwenService(_settings(tmp_path))

    with pytest.raises(ManagedQwenError, match="exited during startup with code 7"):
        service.__enter__()

    assert not service.owns_process


def test_managed_service_real_subprocess_is_unreachable_after_exit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = int(listener.getsockname()[1])
    server_script = tmp_path / "fake_qwen_server.py"
    server_script.write_text(
        """
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

parser = argparse.ArgumentParser()
parser.add_argument("--served-model-name", required=True)
parser.add_argument("--host", required=True)
parser.add_argument("--port", required=True, type=int)
args, _unknown = parser.parse_known_args()

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"status": "ok", "model": args.served_model_name}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass

ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
""".lstrip(),
        encoding="utf-8",
    )
    model_path = tmp_path / "model"
    model_path.mkdir()
    endpoint = f"http://127.0.0.1:{port}/v1/chat/completions"
    monkeypatch.setattr(
        managed_qwen,
        "query_gpu_inventory",
        lambda: (GpuStatus(4, "GPU-fake", 90_000, 100_000, 0),),
    )
    settings = ManagedQwenSettings(
        endpoint=endpoint,
        model_name="fake-qwen",
        python_executable=Path(sys.executable),
        server_script=server_script,
        model_path=model_path,
        log_directory=tmp_path / "logs",
        minimum_free_memory_mib=1,
        startup_timeout_seconds=5.0,
        shutdown_timeout_seconds=5.0,
        probe_timeout_seconds=0.2,
    )

    with ManagedQwenService(settings) as service:
        assert service.owns_process
        assert probe_qwen_health(
            endpoint,
            expected_model="fake-qwen",
            timeout_seconds=0.2,
        ) == {"status": "ok", "model": "fake-qwen"}

    assert (
        probe_qwen_health(
            endpoint,
            expected_model="fake-qwen",
            timeout_seconds=0.2,
        )
        is None
    )


@pytest.mark.parametrize(
    ("arguments", "expected"),
    (
        (("--dataset-root", "/data"), True),
        (("--source-run-dir", "/frozen/run"), False),
        (("--source-run-dir=/frozen/run",), False),
        (("--dry-run", "--source-run-dir", "/frozen/run"), False),
        (("--help",), False),
    ),
)
def test_process_requires_qwen(arguments: tuple[str, ...], expected: bool) -> None:
    assert manager_script._process_requires_qwen(arguments) is expected


def test_forwarded_config_uses_last_argparse_value() -> None:
    args = manager_script._parse_args(
        (
            "--config",
            "default.yaml",
            "--qwen-python",
            "/qwen/python",
            "--qwen-model-path",
            "/models/qwen",
            "--",
            "--config",
            "first.yaml",
            "--config=last.yaml",
        )
    )

    assert manager_script._effective_config_path(args) == Path("last.yaml")


def test_settings_preserve_symlinked_qwen_virtualenv_python(tmp_path: Path) -> None:
    qwen_python = tmp_path / "qwen-venv" / "bin" / "python"
    qwen_python.parent.mkdir(parents=True)
    qwen_python.symlink_to(sys.executable)
    args = manager_script._parse_args(
        (
            "--config",
            "configs/pilot_move_pillbottle_pad.yaml",
            "--qwen-python",
            str(qwen_python),
            "--qwen-model-path",
            str(tmp_path / "model"),
        )
    )

    settings = manager_script._settings(args, args.config)

    assert settings.python_executable == qwen_python
    assert settings.python_executable.is_symlink()
    assert settings.python_executable.resolve() == Path(sys.executable).resolve()
