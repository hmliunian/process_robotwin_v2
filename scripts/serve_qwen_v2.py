#!/usr/bin/env python3
"""Serve Qwen model with v2 architecture - role-aware grounding support.

改进点：
1. 支持多帧输入（宽窗口选帧）
2. 角色感知 prompt（target vs receiver）
3. 结构化输出（GroundingResult）
4. 支持排除对象（exclusions）
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "process_data"))

from robotwin_annotate.qwen_text_normalize import TransformersQwenGenerator

MAX_REQUEST_BYTES = 48 * 1024 * 1024  # 增加到 48MB 支持多帧


def _decode_image_part(image_url: str):
    """Decode base64 image to PIL Image."""
    from PIL import Image

    payload = image_url
    if image_url.startswith("data:"):
        _, _, payload = image_url.partition(",")

    raw = base64.b64decode(payload, validate=True)
    return Image.open(io.BytesIO(raw)).convert("RGB")


class QwenV2Service:
    """Qwen service with v2 architecture support."""

    def __init__(
        self,
        generator: TransformersQwenGenerator,
        *,
        served_model_name: str,
        default_max_tokens: int,
        max_tokens_limit: int,
    ) -> None:
        self.generator = generator
        self.served_model_name = served_model_name
        self.default_max_tokens = default_max_tokens
        self.max_tokens_limit = max_tokens_limit
        self.started_at = int(time.time())

    def _build_role_aware_prompt(
        self,
        text_query: str,
        role: str,
        exclusions: list[str],
    ) -> str:
        """构造角色感知的 prompt。"""

        if role == "target":
            base = f"""任务：找到将被机械臂抓取并移动的物体。

描述：{text_query}

要求：
1. 选择物体完整可见、未被遮挡的帧
2. 物体应该靠近将要闭合的夹爪
3. 不要选择：{', '.join(exclusions) if exclusions else '无'}
4. 返回 JSON 格式：
{{
  "refined_query": "精确的外观描述",
  "selected_frame": 最佳帧索引（0-based）,
  "bbox": {{"x_min": 0.0-1.0, "y_min": 0.0-1.0, "x_max": 0.0-1.0, "y_max": 0.0-1.0}},
  "rationale": "为什么选择这个物体和这一帧",
  "confidence": 0.0-1.0
}}
"""
        else:  # receiver
            base = f"""任务：找到物体将被放置到的目标位置/承接物。

描述：{text_query}

要求：
1. 选择目标位置完整可见、未被遮挡的帧
2. 目标位置应该在抓取开始前已存在
3. 不要选择：{', '.join(exclusions) if exclusions else '无'}
4. 返回 JSON 格式：
{{
  "refined_query": "精确的外观描述",
  "selected_frame": 最佳帧索引（0-based）,
  "bbox": {{"x_min": 0.0-1.0, "y_min": 0.0-1.0, "x_max": 0.0-1.0, "y_max": 0.0-1.0}},
  "rationale": "为什么选择这个位置和这一帧",
  "confidence": 0.0-1.0
}}
"""
        return base

    def ground(self, payload: dict[str, Any]) -> dict[str, Any]:
        """角色感知 grounding（v2 新增端点）。"""

        # 解析参数
        frames_b64 = payload.get("frames", [])
        if not isinstance(frames_b64, list) or not frames_b64:
            raise ValueError("frames must be a non-empty list")

        text_query = payload.get("text_query", "")
        if not isinstance(text_query, str) or not text_query.strip():
            raise ValueError("text_query must be a non-empty string")

        role = payload.get("role", "target")
        if role not in {"target", "receiver"}:
            raise ValueError("role must be 'target' or 'receiver'")

        exclusions = payload.get("exclusions", [])
        if not isinstance(exclusions, list):
            raise ValueError("exclusions must be a list")

        # 解码图像
        frames = []
        for i, frame_b64 in enumerate(frames_b64):
            try:
                frame = _decode_image_part(frame_b64)
                frames.append(frame)
            except Exception as e:
                raise ValueError(f"frames[{i}]: invalid image: {e}") from e

        # 构造角色感知 prompt
        prompt_text = self._build_role_aware_prompt(text_query, role, exclusions)

        # 构造消息（多帧 + 文本）
        content_parts = []
        for frame in frames:
            content_parts.append({"type": "image", "image": frame})
        content_parts.append({"type": "text", "text": prompt_text})

        messages = [
            {
                "role": "user",
                "content": content_parts,
            }
        ]

        # 调用生成
        max_tokens = payload.get("max_tokens", 512)
        response_text = self.generator.generate(messages, max_new_tokens=max_tokens)

        # 解析 JSON 响应
        try:
            # 尝试提取 JSON（可能被 markdown 包裹）
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()
            else:
                json_str = response_text.strip()

            result = json.loads(json_str)
        except (json.JSONDecodeError, IndexError) as e:
            # 如果解析失败，返回原始响应
            result = {
                "refined_query": text_query,
                "selected_frame": 0,
                "bbox": {"x_min": 0.3, "y_min": 0.3, "x_max": 0.7, "y_max": 0.7},
                "rationale": f"Failed to parse JSON: {response_text[:200]}",
                "confidence": 0.5,
                "raw_response": response_text,
            }

        return result

    def completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        """标准 OpenAI-compatible chat completion（v1 兼容）。"""

        messages = payload.get("messages", [])
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")

        # 简化：直接传递给 generator
        normalized_messages = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            # 处理 content
            if isinstance(content, str):
                normalized_messages.append({"role": role, "content": content})
            elif isinstance(content, list):
                # 多模态内容
                parts = []
                for part in content:
                    if part.get("type") == "text":
                        parts.append({"type": "text", "text": part.get("text", "")})
                    elif part.get("type") in ("image_url", "image"):
                        image_url = part.get("image_url", part).get("url", "")
                        image = _decode_image_part(image_url)
                        parts.append({"type": "image", "image": image})
                normalized_messages.append({"role": role, "content": parts})

        max_tokens = payload.get("max_tokens", self.default_max_tokens)
        response_text = self.generator.generate(
            normalized_messages,
            max_new_tokens=max_tokens,
        )

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.served_model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response_text},
                    "finish_reason": "stop",
                }
            ],
        }


def build_handler(service: QwenV2Service) -> type[BaseHTTPRequestHandler]:
    """创建 HTTP handler。"""

    class QwenV2RequestHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status.value)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def send_error_json(self, status: HTTPStatus, message: str) -> None:
            self.send_json(
                status,
                {"error": {"message": message, "type": "invalid_request_error"}},
            )

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]

            if path == "/health":
                self.send_json(
                    HTTPStatus.OK,
                    {
                        "status": "ok",
                        "model": service.served_model_name,
                        "device": service.generator.device,
                        "pid": os.getpid(),
                        "started_at": service.started_at,
                        "version": "v2",
                    },
                )
                return

            if path == "/v1/models":
                self.send_json(
                    HTTPStatus.OK,
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": service.served_model_name,
                                "object": "model",
                                "created": service.started_at,
                                "owned_by": "local",
                            }
                        ],
                    },
                )
                return

            self.send_error_json(HTTPStatus.NOT_FOUND, "route not found")

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]

            # 读取请求体
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length < 1 or content_length > MAX_REQUEST_BYTES:
                self.send_error_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request too large")
                return

            try:
                payload = json.loads(self.rfile.read(content_length))
            except json.JSONDecodeError:
                self.send_error_json(HTTPStatus.BAD_REQUEST, "invalid JSON")
                return

            # 路由
            if path == "/v2/ground":
                # v2 新增：角色感知 grounding
                try:
                    result = service.ground(payload)
                    self.send_json(HTTPStatus.OK, result)
                except ValueError as e:
                    self.send_error_json(HTTPStatus.BAD_REQUEST, str(e))
                except Exception as e:
                    self.send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))
                return

            if path == "/v1/chat/completions":
                # v1 兼容：标准 chat completion
                try:
                    result = service.completion(payload)
                    self.send_json(HTTPStatus.OK, result)
                except ValueError as e:
                    self.send_error_json(HTTPStatus.BAD_REQUEST, str(e))
                except Exception as e:
                    self.send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))
                return

            self.send_error_json(HTTPStatus.NOT_FOUND, "route not found")

        def log_message(self, format: str, *args: Any) -> None:
            # 简化日志
            sys.stderr.write(f"[{self.log_date_time_string()}] {format % args}\n")

    return QwenV2RequestHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face model path")
    parser.add_argument("--served-model-name", default="qwen3.5-27b")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18086)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--default-max-tokens", type=int, default=512)
    parser.add_argument("--max-tokens-limit", type=int, default=2048)
    parser.add_argument("--pid-file", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading Qwen model from {args.model}...")
    generator = TransformersQwenGenerator(
        model=args.model,
        device=args.device,
        dtype=args.dtype,
    )

    service = QwenV2Service(
        generator,
        served_model_name=args.served_model_name,
        default_max_tokens=args.default_max_tokens,
        max_tokens_limit=args.max_tokens_limit,
    )

    handler_class = build_handler(service)
    server = ThreadingHTTPServer((args.host, args.port), handler_class)

    # 写入 PID 文件
    if args.pid_file:
        args.pid_file.parent.mkdir(parents=True, exist_ok=True)
        args.pid_file.write_text(str(os.getpid()))

    print(f"✅ Qwen v2 service started")
    print(f"   Model: {args.served_model_name}")
    print(f"   Device: {args.device}")
    print(f"   Listening on http://{args.host}:{args.port}")
    print(f"   Endpoints:")
    print(f"     GET  /health")
    print(f"     GET  /v1/models")
    print(f"     POST /v1/chat/completions  (v1 compatible)")
    print(f"     POST /v2/ground            (v2 role-aware grounding)")
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
