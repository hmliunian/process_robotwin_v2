from __future__ import annotations

import base64
import io
import json
import urllib.error
import urllib.request
from typing import Any, Self

import numpy as np
import pytest

from robotwin_annotation_v2.adapters import (
    OpenAICompatibleQwenClient,
    image_data_url,
)


class FakeHTTPResponse(io.BytesIO):
    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


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
    body = json.loads(requests[1].data)
    assert body["max_tokens"] == 20
    assert body["temperature"] == 0
    assert body["enable_thinking"] is False


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
                "choices": [{"message": {"content": "ok"}}],
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
    )

    assert client.health() == {
        "status": "ok",
        "model": "qwen3.8-max",
        "probe": "models",
    }
    assert client.complete([{"role": "user", "content": "test"}], max_tokens=32).content == "ok"
    assert client.models_endpoint == "https://maas.example/compatible-mode/v1/models"
    body = json.loads(requests[1].data)
    assert body["model"] == "qwen3.8-max"
    assert body["temperature"] == 0.25
    assert body["enable_thinking"] is False


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
