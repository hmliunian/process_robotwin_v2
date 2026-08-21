"""OpenAI-compatible HTTP client for the standalone Qwen server."""

from __future__ import annotations

import base64
import io
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import numpy as np
from PIL import Image

NDArray = np.ndarray[Any, Any]


class QwenServiceError(RuntimeError):
    """The Qwen service is unavailable or returned an invalid API response."""


@dataclass(frozen=True)
class QwenCompletion:
    content: str
    model: str


def image_data_url(image: Image.Image | NDArray) -> str:
    """Encode one RGB image as an inline PNG data URL."""

    if isinstance(image, Image.Image):
        rgb = image.convert("RGB")
    else:
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"expected RGB image [H,W,3], got {array.shape}")
        if array.dtype != np.uint8:
            raise ValueError(f"expected uint8 RGB image, got {array.dtype}")
        rgb = Image.fromarray(np.ascontiguousarray(array), mode="RGB")
    buffer = io.BytesIO()
    rgb.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


class OpenAICompatibleQwenClient:
    """Small client for ``scripts/serve_qwen.py`` or an equivalent endpoint."""

    def __init__(self, *, endpoint: str, model: str, timeout_seconds: float) -> None:
        if not endpoint.strip():
            raise ValueError("Qwen endpoint must be non-empty")
        if not model.strip():
            raise ValueError("Qwen model must be non-empty")
        if timeout_seconds <= 0:
            raise ValueError("Qwen timeout must be positive")
        self.endpoint = endpoint
        self.model_id = model
        self.timeout_seconds = timeout_seconds

    @property
    def health_endpoint(self) -> str:
        parsed = urlsplit(self.endpoint)
        path = parsed.path.rstrip("/")
        suffix = "/v1/chat/completions"
        health_path = f"{path[:-len(suffix)]}/health" if path.endswith(suffix) else "/health"
        return urlunsplit((parsed.scheme, parsed.netloc, health_path, "", ""))

    def _read_json(self, request: urllib.request.Request) -> dict[str, Any]:
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1000]
            raise QwenServiceError(
                f"Qwen service returned HTTP {exc.code}: {detail}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise QwenServiceError(f"Qwen service request failed: {exc}") from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise QwenServiceError("Qwen service returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise QwenServiceError("Qwen service response must be a JSON object")
        return payload

    def health(self) -> dict[str, Any]:
        request = urllib.request.Request(self.health_endpoint, method="GET")
        payload = self._read_json(request)
        if payload.get("status") != "ok":
            raise QwenServiceError(f"Qwen health check failed: {payload}")
        return payload

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
    ) -> QwenCompletion:
        if not messages:
            raise ValueError("Qwen messages must not be empty")
        if max_tokens < 1:
            raise ValueError("Qwen max_tokens must be positive")
        body = json.dumps(
            {
                "model": self.model_id,
                "messages": messages,
                "max_tokens": max_tokens,
                "stream": False,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        payload = self._read_json(request)
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise QwenServiceError("Qwen response has no assistant content") from exc
        if not isinstance(content, str) or not content.strip():
            raise QwenServiceError("Qwen returned empty assistant content")
        model = payload.get("model", self.model_id)
        if not isinstance(model, str) or not model.strip():
            raise QwenServiceError("Qwen response has an invalid model field")
        return QwenCompletion(content=content.strip(), model=model.strip())
