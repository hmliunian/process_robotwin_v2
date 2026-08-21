#!/usr/bin/env python3
"""Serve a local Qwen3.5-VL model through a small OpenAI-compatible API."""

from __future__ import annotations

import argparse
import base64
import binascii
import io
import json
import os
import signal
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

MAX_REQUEST_BYTES = 32 * 1024 * 1024


def _decode_image(image_url: str) -> Any:
    from PIL import Image

    payload = image_url.partition(",")[2] if image_url.startswith("data:") else image_url
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image_url is not valid base64") from exc
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"image_url is not a valid image: {exc}") from exc


def _normalize_content(content: Any) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        if not content.strip():
            raise ValueError("text content must be non-empty")
        return content
    if not isinstance(content, list) or not content:
        raise ValueError("content must be a non-empty string or list")
    normalized: list[dict[str, Any]] = []
    for index, part in enumerate(content):
        if not isinstance(part, dict):
            raise ValueError(  # noqa: TRY004 - preserve the request validation contract
                f"content[{index}] must be an object"
            )
        part_type = part.get("type")
        if part_type == "text":
            text = part.get("text")
            if not isinstance(text, str) or not text:
                raise ValueError(f"content[{index}].text must be a non-empty string")
            normalized.append({"type": "text", "text": text})
        elif part_type == "image_url":
            image_spec = part.get("image_url")
            image_url = image_spec.get("url") if isinstance(image_spec, dict) else image_spec
            if not isinstance(image_url, str) or not image_url:
                raise ValueError(f"content[{index}].image_url must contain a URL")
            normalized.append({"type": "image", "image": _decode_image(image_url)})
        else:
            raise ValueError(f"unsupported content[{index}].type={part_type!r}")
    return normalized


class LocalQwenGenerator:
    """Minimal Hugging Face backend for the Qwen3.5 model used by this experiment."""

    def __init__(self, model_path: Path, *, device: str, dtype: str) -> None:
        try:
            import torch
            from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "Qwen dependencies are missing; use the dedicated Qwen environment"
            ) from exc
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"requested {device}, but CUDA is unavailable")
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        source = str(model_path.expanduser().resolve())
        self.processor = AutoProcessor.from_pretrained(
            source,
            local_files_only=True,
            use_fast=False,
        )
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            source,
            local_files_only=True,
            torch_dtype=dtype_map[dtype],
            low_cpu_mem_usage=True,
        )
        self.model.to(device).eval()
        self.device = device
        self._torch = torch

    @staticmethod
    def _split_images(
        messages: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[Any]]:
        template_messages: list[dict[str, Any]] = []
        images: list[Any] = []
        for message in messages:
            content = message["content"]
            if isinstance(content, str):
                template_messages.append(message)
                continue
            parts: list[dict[str, str]] = []
            for part in content:
                if part["type"] == "image":
                    images.append(part["image"])
                    parts.append({"type": "image"})
                else:
                    parts.append({"type": "text", "text": part["text"]})
            template_messages.append({"role": message["role"], "content": parts})
        return template_messages, images

    def generate(self, messages: list[dict[str, Any]], *, max_tokens: int) -> str:
        template_messages, images = self._split_images(messages)
        try:
            prompt = self.processor.apply_chat_template(
                template_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = self.processor.apply_chat_template(
                template_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        inputs = self.processor(
            text=[prompt],
            images=images or None,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        with self._torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )
        generated_ids = output_ids[:, inputs.input_ids.shape[1] :]
        return self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()


class QwenService:
    def __init__(
        self,
        generator: LocalQwenGenerator,
        *,
        model_name: str,
        default_max_tokens: int,
        max_tokens_limit: int,
    ) -> None:
        self.generator = generator
        self.model_name = model_name
        self.default_max_tokens = default_max_tokens
        self.max_tokens_limit = max_tokens_limit
        self.started_at = int(time.time())
        self._generation_lock = threading.Lock()

    def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        requested_model = payload.get("model", self.model_name)
        if requested_model != self.model_name:
            raise ValueError(
                f"requested model {requested_model!r} is not served; use {self.model_name!r}"
            )
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")
        normalized: list[dict[str, Any]] = []
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValueError(  # noqa: TRY004 - preserve the request validation contract
                    f"messages[{index}] must be an object"
                )
            role = message.get("role")
            if role not in {"system", "user", "assistant"}:
                raise ValueError(f"messages[{index}].role is unsupported")
            try:
                content = _normalize_content(message.get("content"))
            except ValueError as exc:
                raise ValueError(f"messages[{index}].content: {exc}") from exc
            normalized.append({"role": role, "content": content})

        max_tokens = payload.get("max_tokens", self.default_max_tokens)
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
            raise ValueError(  # noqa: TRY004 - preserve the request validation contract
                "max_tokens must be an integer"
            )
        if not 1 <= max_tokens <= self.max_tokens_limit:
            raise ValueError(f"max_tokens must be between 1 and {self.max_tokens_limit}")
        if payload.get("stream", False):
            raise ValueError("streaming is not supported")

        with self._generation_lock:
            content = self.generator.generate(normalized, max_tokens=max_tokens)
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
        }


def build_handler(service: QwenService) -> type[BaseHTTPRequestHandler]:
    class QwenHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status.value)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_error(self, status: HTTPStatus, message: str) -> None:
            self._send_json(status, {"error": {"message": message}})

        def do_GET(self) -> None:
            if self.path.split("?", 1)[0] != "/health":
                self._send_error(HTTPStatus.NOT_FOUND, "route not found")
                return
            self._send_json(
                HTTPStatus.OK,
                {
                    "status": "ok",
                    "model": service.model_name,
                    "device": service.generator.device,
                    "pid": os.getpid(),
                    "started_at": service.started_at,
                },
            )

        def do_POST(self) -> None:
            if self.path.split("?", 1)[0] != "/v1/chat/completions":
                self._send_error(HTTPStatus.NOT_FOUND, "route not found")
                return
            try:
                content_length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                self._send_error(HTTPStatus.LENGTH_REQUIRED, "Content-Length is required")
                return
            if not 1 <= content_length <= MAX_REQUEST_BYTES:
                self._send_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "invalid request size")
                return
            try:
                payload = json.loads(self.rfile.read(content_length))
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._send_error(HTTPStatus.BAD_REQUEST, "request body must be valid JSON")
                return
            if not isinstance(payload, dict):
                self._send_error(HTTPStatus.BAD_REQUEST, "request body must be a JSON object")
                return
            try:
                response = service.complete(payload)
            except ValueError as exc:
                self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
                return
            except RuntimeError as exc:
                self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
                return
            self._send_json(HTTPStatus.OK, response)

    return QwenHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--served-model-name", default="qwen3.5-27b")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18086)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--default-max-tokens", type=int, default=800)
    parser.add_argument("--max-tokens-limit", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.is_dir():
        raise SystemExit(f"model directory does not exist: {args.model}")
    if not 1 <= args.default_max_tokens <= args.max_tokens_limit:
        raise SystemExit("default max tokens must be within max-tokens-limit")

    print(f"Loading {args.model} on {args.device} ({args.dtype})", flush=True)
    generator = LocalQwenGenerator(args.model, device=args.device, dtype=args.dtype)
    service = QwenService(
        generator,
        model_name=args.served_model_name,
        default_max_tokens=args.default_max_tokens,
        max_tokens_limit=args.max_tokens_limit,
    )
    server = ThreadingHTTPServer((args.host, args.port), build_handler(service))

    def request_shutdown(_signum: int, _frame: Any) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)
    print(
        f"Qwen server ready at http://{args.host}:{args.port} "
        f"(model={args.served_model_name}, pid={os.getpid()})",
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
