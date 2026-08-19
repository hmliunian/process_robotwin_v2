from __future__ import annotations

import base64
import io
import json
import urllib.request
from typing import Any, Self

import numpy as np

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
    assert json.loads(requests[1].data)["max_tokens"] == 20
