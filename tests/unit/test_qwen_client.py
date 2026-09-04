from __future__ import annotations

import base64
import io
import json
import multiprocessing as mp
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Self

import numpy as np
import pytest

from robotwin_annotation_v2.adapters import (
    OpenAICompatibleQwenClient,
    image_data_url,
)
from robotwin_annotation_v2.adapters.qwen_client import (
    FileRequestGate,
    install_qwen_request_gate,
)
from robotwin_annotation_v2.config import QwenConfig


class FakeHTTPResponse(io.BytesIO):
    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _acquire_gate_and_exit(gate: FileRequestGate, connection: Any) -> None:
    assert gate.acquire(timeout=1)
    connection.send(True)
    connection.close()
    os._exit(23)


def test_image_data_url_contains_png() -> None:
    image = np.zeros((3, 5, 3), dtype=np.uint8)

    url = image_data_url(image)

    encoded = url.removeprefix("data:image/png;base64,")
    assert base64.b64decode(encoded).startswith(b"\x89PNG\r\n\x1a\n")


def test_openai_client_health_and_completion(monkeypatch: Any) -> None:
    requests: list[urllib.request.Request] = []

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> FakeHTTPResponse:
        assert timeout == 5
        requests.append(request)
        if request.full_url.endswith("/health"):
            payload = {"status": "ok", "model": "fake-qwen"}
        else:
            payload = {
                "model": "fake-qwen",
                "choices": [{"message": {"content": "{\"target\": {}}"}}],
            }
        return FakeHTTPResponse(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = OpenAICompatibleQwenClient(
        endpoint="http://127.0.0.1:18086/v1/chat/completions",
        model="fake-qwen",
        timeout_seconds=5,
    )

    health = client.health()
    completion = client.complete(
        [{"role": "user", "content": "test"}],
        max_tokens=20,
    )

    assert client.health_endpoint == "http://127.0.0.1:18086/health"
    assert health["status"] == "ok"
    assert completion.model == "fake-qwen"
    assert completion.finish_reason is None
    body = json.loads(requests[1].data)
    assert body["max_tokens"] == 20
    assert body["temperature"] == 0
    assert body["enable_thinking"] is False
    assert "response_format" not in body


def test_api_client_authenticates_models_probe_and_completion(monkeypatch: Any) -> None:
    requests: list[urllib.request.Request] = []
    monkeypatch.setenv("QWEN_TEST_API_KEY", "test-secret")

    def fake_urlopen(request: urllib.request.Request, timeout: float) -> FakeHTTPResponse:
        assert timeout == 9
        assert request.get_header("Authorization") == "Bearer test-secret"
        requests.append(request)
        payload = (
            {"data": [{"id": "qwen3.8-max"}, {"id": "qwen3.8-27b"}]}
            if request.full_url.endswith("/models")
            else {
                "model": "qwen3.8-max",
                "choices": [
                    {"message": {"content": "ok"}, "finish_reason": "stop"}
                ],
            }
        )
        return FakeHTTPResponse(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = OpenAICompatibleQwenClient(
        endpoint="https://maas.example/compatible-mode/v1/chat/completions",
        model="qwen3.8-max",
        timeout_seconds=9,
        api_key_env="QWEN_TEST_API_KEY",
        probe="models",
        temperature=0.25,
        enable_thinking=False,
        json_object_response=True,
    )

    assert client.health() == {
        "status": "ok",
        "model": "qwen3.8-max",
        "probe": "models",
    }
    completion = client.complete([{"role": "user", "content": "test"}], max_tokens=32)
    assert completion.content == "ok"
    assert completion.finish_reason == "stop"
    assert client.models_endpoint == "https://maas.example/compatible-mode/v1/models"
    body = json.loads(requests[1].data)
    assert body["model"] == "qwen3.8-max"
    assert body["temperature"] == 0.25
    assert body["enable_thinking"] is False
    assert body["response_format"] == {"type": "json_object"}


def test_api_config_enables_json_object_response() -> None:
    client = OpenAICompatibleQwenClient.from_config(
        QwenConfig(
            endpoint="https://maas.example/v1/chat/completions",
            model="qwen3.8-max",
            prompt_template=Path("prompt.txt"),
            runtime="api",
            api_key_env="QWEN_TEST_API_KEY",
            probe="models",
        )
    )

    assert client.json_object_response


def test_api_client_fails_before_request_when_credential_is_missing(
    monkeypatch: Any,
) -> None:
    monkeypatch.delenv("QWEN_MISSING_API_KEY", raising=False)
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("request must not be sent without credentials"),
    )
    client = OpenAICompatibleQwenClient(
        endpoint="https://maas.example/v1/chat/completions",
        model="qwen3.8-max",
        timeout_seconds=5,
        api_key_env="QWEN_MISSING_API_KEY",
        probe="models",
    )

    with pytest.raises(RuntimeError, match="QWEN_MISSING_API_KEY"):
        client.health()


@pytest.mark.parametrize(
    ("error", "message"),
    (
        (
            urllib.error.HTTPError(
                "https://maas.example/v1/models",
                401,
                "Unauthorized",
                hdrs=None,
                fp=io.BytesIO(b'{"error":"unauthorized"}'),
            ),
            "HTTP 401",
        ),
        (TimeoutError("timed out"), "request failed"),
    ),
)
def test_api_client_reports_http_and_timeout_failures(
    monkeypatch: Any,
    error: BaseException,
    message: str,
) -> None:
    monkeypatch.setenv("QWEN_TEST_API_KEY", "test-secret")
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
    )
    client = OpenAICompatibleQwenClient(
        endpoint="https://maas.example/v1/chat/completions",
        model="qwen3.8-max",
        timeout_seconds=5,
        api_key_env="QWEN_TEST_API_KEY",
        probe="models",
    )

    with pytest.raises(RuntimeError, match=message):
        client.health()


class _RecordingGate:
    def __init__(self, *, acquired: bool = True) -> None:
        self.acquired = acquired
        self.acquire_timeouts: list[float | None] = []
        self.releases = 0

    def acquire(self, blocking: bool = True, timeout: float | None = None) -> bool:
        del blocking
        self.acquire_timeouts.append(timeout)
        return self.acquired

    def release(self) -> None:
        self.releases += 1


def test_qwen_request_gate_releases_after_transport_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = _RecordingGate()
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(TimeoutError("network timeout")),
    )
    client = OpenAICompatibleQwenClient(
        endpoint="http://127.0.0.1:18086/v1/chat/completions",
        model="fake-qwen",
        timeout_seconds=7,
    )

    install_qwen_request_gate(gate)
    try:
        with pytest.raises(RuntimeError, match="request failed"):
            client.health()
    finally:
        install_qwen_request_gate(None)

    assert gate.acquire_timeouts == [7]
    assert gate.releases == 1


def test_qwen_request_gate_timeout_does_not_send_or_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = _RecordingGate(acquired=False)
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: pytest.fail("request must wait for a permit"),
    )
    client = OpenAICompatibleQwenClient(
        endpoint="http://127.0.0.1:18086/v1/chat/completions",
        model="fake-qwen",
        timeout_seconds=3,
    )

    install_qwen_request_gate(gate)
    try:
        with pytest.raises(RuntimeError, match="concurrency permit"):
            client.health()
    finally:
        install_qwen_request_gate(None)

    assert gate.acquire_timeouts == [3]
    assert gate.releases == 0


class _PeakGate:
    def __init__(self, capacity: int) -> None:
        self._semaphore = threading.BoundedSemaphore(capacity)
        self._lock = threading.Lock()
        self.active = 0
        self.peak = 0

    def acquire(self, blocking: bool = True, timeout: float | None = None) -> bool:
        acquired = self._semaphore.acquire(blocking=blocking, timeout=timeout)
        if acquired:
            with self._lock:
                self.active += 1
                self.peak = max(self.peak, self.active)
        return acquired

    def release(self) -> None:
        with self._lock:
            self.active -= 1
        self._semaphore.release()


def test_qwen_request_gate_bounds_concurrent_http_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = _PeakGate(2)

    def fake_urlopen(
        _request: urllib.request.Request,
        timeout: float,
    ) -> FakeHTTPResponse:
        assert timeout == 5
        time.sleep(0.05)
        return FakeHTTPResponse(b'{"status":"ok","model":"fake-qwen"}')

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = OpenAICompatibleQwenClient(
        endpoint="http://127.0.0.1:18086/v1/chat/completions",
        model="fake-qwen",
        timeout_seconds=5,
    )

    install_qwen_request_gate(gate)
    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = tuple(executor.map(lambda _index: client.health(), range(4)))
    finally:
        install_qwen_request_gate(None)

    assert all(result["status"] == "ok" for result in results)
    assert gate.peak == 2
    assert gate.active == 0


def test_file_request_gate_recovers_token_after_hard_process_exit(
    tmp_path: Path,
) -> None:
    gate = FileRequestGate.create(tmp_path / "gate", capacity=1)
    contender = FileRequestGate(gate.token_paths)
    context = mp.get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_acquire_gate_and_exit,
        args=(gate, child_connection),
    )

    process.start()
    child_connection.close()
    assert parent_connection.recv() is True
    process.join(timeout=5)
    assert process.exitcode == 23
    assert contender.acquire(timeout=1) is True
    contender.release()
    parent_connection.close()
    process.close()
