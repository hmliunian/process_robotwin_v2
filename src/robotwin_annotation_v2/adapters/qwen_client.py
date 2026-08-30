"""OpenAI-compatible HTTP client for the standalone Qwen server."""

from __future__ import annotations

import base64
import fcntl
import io
import json
import math
import os
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol
from urllib.parse import urlsplit, urlunsplit

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from robotwin_annotation_v2.config import QwenConfig

NDArray = np.ndarray[Any, Any]


class RequestGate(Protocol):
    def acquire(self, blocking: bool = True, timeout: float | None = None) -> bool: ...

    def release(self) -> None: ...


@dataclass(slots=True)
class FileRequestGate:
    """Cross-process request tokens whose locks are released by the kernel on exit."""

    token_paths: tuple[Path, ...]
    poll_interval_seconds: float = 0.02
    _handle: io.BufferedRandom | None = field(default=None, init=False, repr=False)

    @classmethod
    def create(cls, directory: Path, *, capacity: int) -> FileRequestGate:
        if capacity < 1:
            raise ValueError("request gate capacity must be positive")
        directory.mkdir(parents=True, exist_ok=True)
        token_paths = tuple(directory / f"token-{index}.lock" for index in range(capacity))
        for path in token_paths:
            path.touch(exist_ok=True)
        return cls(token_paths)

    def acquire(self, blocking: bool = True, timeout: float | None = None) -> bool:
        if self._handle is not None:
            raise RuntimeError("Qwen request gate is already acquired")
        if timeout is not None and timeout < 0:
            raise ValueError("request gate timeout must be non-negative")
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            for path in self.token_paths:
                handle = path.open("a+b")
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close()
                else:
                    self._handle = handle
                    return True
            if not blocking or (deadline is not None and time.monotonic() >= deadline):
                return False
            remaining = (
                self.poll_interval_seconds
                if deadline is None
                else min(self.poll_interval_seconds, max(0.0, deadline - time.monotonic()))
            )
            time.sleep(remaining)

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            raise RuntimeError("Qwen request gate is not acquired")
        self._handle = None
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


_REQUEST_GATE: RequestGate | None = None


def install_qwen_request_gate(gate: RequestGate | None) -> None:
    """Install a process-local concurrency gate for all Qwen HTTP requests."""

    global _REQUEST_GATE
    _REQUEST_GATE = gate


class QwenServiceError(RuntimeError):
    """The Qwen service is unavailable or returned an invalid API response."""


@dataclass(frozen=True)
class QwenCompletion:
    content: str
    model: str
    finish_reason: str | None = None


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

    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        timeout_seconds: float,
        api_key_env: str | None = None,
        probe: str = "health",
        temperature: float = 0.0,
        enable_thinking: bool = False,
        json_object_response: bool = False,
    ) -> None:
        if not endpoint.strip():
            raise ValueError("Qwen endpoint must be non-empty")
        if not model.strip():
            raise ValueError("Qwen model must be non-empty")
        if timeout_seconds <= 0:
            raise ValueError("Qwen timeout must be positive")
        if probe not in {"health", "models"}:
            raise ValueError("Qwen probe must be health or models")
        if probe == "models" and not api_key_env:
            raise ValueError("Qwen models probe requires an API key environment variable")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("Qwen temperature must be finite and non-negative")
        self.endpoint = endpoint
        self.model_id = model
        self.timeout_seconds = timeout_seconds
        self.api_key_env = api_key_env
        self.probe = probe
        self.temperature = temperature
        self.enable_thinking = enable_thinking
        self.json_object_response = json_object_response

    @classmethod
    def from_config(cls, config: QwenConfig) -> OpenAICompatibleQwenClient:
        """Build one client without duplicating the YAML-to-transport mapping."""

        return cls(
            endpoint=config.endpoint,
            model=config.model,
            timeout_seconds=config.timeout_seconds,
            api_key_env=config.api_key_env,
            probe=config.probe,
            temperature=config.temperature,
            enable_thinking=config.enable_thinking,
            json_object_response=config.runtime == "api",
        )

    @property
    def health_endpoint(self) -> str:
        parsed = urlsplit(self.endpoint)
        path = parsed.path.rstrip("/")
        suffix = "/v1/chat/completions"
        health_path = f"{path[: -len(suffix)]}/health" if path.endswith(suffix) else "/health"
        return urlunsplit((parsed.scheme, parsed.netloc, health_path, "", ""))

    @property
    def models_endpoint(self) -> str:
        parsed = urlsplit(self.endpoint)
        path = parsed.path.rstrip("/")
        suffix = "/chat/completions"
        models_path = f"{path[: -len(suffix)]}/models" if path.endswith(suffix) else "/models"
        return urlunsplit((parsed.scheme, parsed.netloc, models_path, "", ""))

    def _headers(self, *, json_content: bool = False) -> dict[str, str]:
        headers = {"Content-Type": "application/json"} if json_content else {}
        if self.api_key_env is not None:
            api_key = os.environ.get(self.api_key_env)
            if not api_key:
                raise QwenServiceError(
                    f"Qwen API credential environment variable {self.api_key_env} is not set"
                )
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    @contextmanager
    def _request_permit(self) -> Iterator[None]:
        gate = _REQUEST_GATE
        if gate is None:
            yield
            return
        if not gate.acquire(timeout=self.timeout_seconds):
            raise QwenServiceError("timed out waiting for a Qwen API concurrency permit")
        try:
            yield
        finally:
            gate.release()

    def _read_json(self, request: urllib.request.Request) -> dict[str, Any]:
        try:
            with self._request_permit():
                try:
                    with urllib.request.urlopen(
                        request,
                        timeout=self.timeout_seconds,
                    ) as response:
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
        endpoint = self.models_endpoint if self.probe == "models" else self.health_endpoint
        request = urllib.request.Request(endpoint, headers=self._headers(), method="GET")
        payload = self._read_json(request)
        if self.probe == "models":
            models = payload.get("data")
            if not isinstance(models, list):
                raise QwenServiceError("Qwen models response has no data list")
            model_ids = {item.get("id") for item in models if isinstance(item, dict)}
            if self.model_id not in model_ids:
                raise QwenServiceError(
                    f"Qwen models endpoint does not expose configured model {self.model_id!r}"
                )
            return {"status": "ok", "model": self.model_id, "probe": "models"}
        if payload.get("status") != "ok":
            raise QwenServiceError(f"Qwen health check failed: {payload}")
        actual_model = payload.get("model")
        if actual_model != self.model_id:
            raise QwenServiceError(
                f"Qwen health endpoint serves model {actual_model!r}, expected {self.model_id!r}"
            )
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
        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "enable_thinking": self.enable_thinking,
            "stream": False,
        }
        if self.json_object_response:
            payload["response_format"] = {"type": "json_object"}
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers=self._headers(json_content=True),
            method="POST",
        )
        payload = self._read_json(request)
        try:
            choice = payload["choices"][0]
        except (KeyError, IndexError, TypeError) as exc:
            raise QwenServiceError("Qwen response has no assistant content") from exc
        try:
            finish_reason = choice.get("finish_reason")
        except AttributeError as exc:
            raise QwenServiceError("Qwen response has no assistant content") from exc
        if finish_reason is not None and not isinstance(finish_reason, str):
            raise QwenServiceError("Qwen response has an invalid finish_reason field")
        try:
            content = choice["message"]["content"]
        except (KeyError, TypeError) as exc:
            if finish_reason not in {None, "stop"}:
                raise QwenServiceError(
                    "Qwen response has no assistant content; "
                    f"finish_reason={finish_reason!r}"
                ) from exc
            raise QwenServiceError("Qwen response has no assistant content") from exc
        if not isinstance(content, str) or not content.strip():
            if finish_reason not in {None, "stop"}:
                raise QwenServiceError(
                    "Qwen returned empty assistant content; "
                    f"finish_reason={finish_reason!r}"
                )
            raise QwenServiceError("Qwen returned empty assistant content")
        model = payload.get("model", self.model_id)
        if not isinstance(model, str) or not model.strip():
            raise QwenServiceError("Qwen response has an invalid model field")
        return QwenCompletion(
            content=content.strip(),
            model=model.strip(),
            finish_reason=finish_reason,
        )
