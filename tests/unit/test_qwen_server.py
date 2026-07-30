from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any

import numpy as np

from robotwin_annotation_v2.adapters import image_data_url


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class FakeGenerator:
    device = "cpu"

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
    ) -> str:
        assert max_tokens == 20
        content = messages[0]["content"]
        assert content[0] == {"type": "text", "text": "frame 0"}
        assert content[1]["type"] == "image"
        assert content[1]["image"].mode == "RGB"
        return '{"target": {}, "receiver": {}}'


def test_qwen_server_service_decodes_openai_image_parts() -> None:
    module = runpy.run_path(str(PROJECT_ROOT / "scripts/serve_qwen.py"))
    service = module["QwenService"](
        FakeGenerator(),
        model_name="fake-qwen",
        default_max_tokens=10,
        max_tokens_limit=100,
    )
    image_url = image_data_url(np.zeros((2, 3, 3), dtype=np.uint8))

    response = service.complete(
        {
            "model": "fake-qwen",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "frame 0"},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            "max_tokens": 20,
            "stream": False,
        }
    )

    assert response["model"] == "fake-qwen"
    assert response["choices"][0]["message"]["content"].startswith("{")
