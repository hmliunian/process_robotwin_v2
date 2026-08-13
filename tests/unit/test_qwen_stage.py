from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from robotwin_annotation_v2.adapters import QwenCompletion
from robotwin_annotation_v2.config import QwenConfig
from robotwin_annotation_v2.domain import AnnotationMode
from robotwin_annotation_v2.models import (
    EpisodeRef,
    FramePurpose,
    LoopContext,
    LoopEvents,
    SemanticFrame,
)
from robotwin_annotation_v2.pipeline import (
    QwenStageError,
    build_qwen_request,
    parse_semantic_plan,
    run_qwen_stage,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _context() -> LoopContext:
    return LoopContext(
        episode=EpisodeRef("move_pillbottle_pad", 7152, "cam_high"),
        task_text="Move the pill bottle onto the pad.",
        frame_count=20,
        events=LoopEvents("right", 2, 6, 8, 14, 17),
        semantic_frames=(
            SemanticFrame(
                0,
                FramePurpose.PRE_GRASP_SEED_CANDIDATE,
                ("target", "receiver"),
            ),
            SemanticFrame(9, FramePurpose.POST_GRASP_CONTEXT, ("target",)),
            SemanticFrame(15, FramePurpose.PLACE_CONTEXT, ("receiver",)),
        ),
        state_source="episode.parquet",
        video_source="episode.mp4",
    )


def _frames() -> dict[int, Image.Image]:
    return {
        frame_id: Image.fromarray(
            np.full((4, 6, 3), frame_id, dtype=np.uint8),
            mode="RGB",
        )
        for frame_id in (0, 9, 15)
    }


def _target_only_context() -> LoopContext:
    return LoopContext(
        episode=EpisodeRef("move_object", 1, "cam_high"),
        task_text="Pick up the bottle.",
        frame_count=20,
        events=LoopEvents("right", 2, 6, 8, 14, 17),
        semantic_frames=(
            SemanticFrame(
                0,
                FramePurpose.PRE_GRASP_SEED_CANDIDATE,
                ("target",),
            ),
            SemanticFrame(9, FramePurpose.POST_GRASP_CONTEXT, ("target",)),
        ),
        state_source="episode.parquet",
        video_source="episode.mp4",
        annotation_mode=AnnotationMode.TARGET_ONLY,
    )


def _response() -> str:
    return json.dumps(
        {
            "target": {
                "status": "ok",
                "seed_frame_id": 0,
                "category_query": "bottle",
                "color_category_query": "orange bottle",
                "shape_category_query": None,
                "general_fallback_query": "container",
                "recommended_order": [
                    "color_category_query",
                    "category_query",
                    "general_fallback_query",
                ],
                "exclude": ["blue pad"],
                "reason": "该物体随后被抓取并移动。",
            },
            "receiver": {
                "status": "ok",
                "seed_frame_id": 0,
                "category_query": "pad",
                "color_category_query": "blue square pad",
                "shape_category_query": "square pad",
                "general_fallback_query": "mat",
                "recommended_order": [
                    "color_category_query",
                    "shape_category_query",
                    "category_query",
                    "general_fallback_query",
                ],
                "exclude": ["orange bottle"],
                "reason": "该区域是最终放置位置。",
            },
        },
        ensure_ascii=False,
    )


def _target_only_response() -> str:
    return json.dumps({"target": json.loads(_response())["target"]}, ensure_ascii=False)


class FakeQwenClient:
    model_id = "fake-qwen"

    def __init__(self, response: str) -> None:
        self.response = response
        self.messages: list[dict[str, Any]] | None = None

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "model": self.model_id}

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
    ) -> QwenCompletion:
        assert max_tokens == 800
        self.messages = messages
        return QwenCompletion(content=self.response, model=self.model_id)


def test_qwen_request_interleaves_frame_label_and_image() -> None:
    template = (
        "task={task_text}\nmove={move_start}\n"
        "frames:\n{labeled_multimodal_frames}\n"
        'schema={"target": {}, "receiver": {}}'
    )

    request = build_qwen_request(_context(), _frames(), template)

    content = request.messages[0]["content"]
    assert [part["type"] for part in content] == [
        "text",
        "image_url",
        "text",
        "image_url",
        "text",
        "image_url",
        "text",
    ]
    assert "frame_id=0" in content[0]["text"]
    assert "frame_id=9" in content[2]["text"]
    assert "frame_id=15" in content[4]["text"]
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "{labeled_multimodal_frames}" not in request.rendered_prompt
    assert 'schema={"target": {}, "receiver": {}}' in request.rendered_prompt


def test_semantic_prompt_defines_receiver_by_direct_contact() -> None:
    template = (
        PROJECT_ROOT / "configs/prompts/target_receiver_semantic.txt"
    ).read_text(encoding="utf-8")

    request = build_qwen_request(_context(), _frames(), template)
    prompt_text = "\n".join(
        part["text"]
        for part in request.messages[0]["content"]
        if part["type"] == "text"
    )

    assert "任务完成时应与 target 直接接触" in prompt_text
    assert "核心判断依据是二者的直接接触关系" in prompt_text
    assert "不要求位于\n  target 下方或承托 target" in prompt_text
    assert "先用 place_context 确定任务完成时与 target" in prompt_text
    assert "也不得因此返回 no_clear_seed" in prompt_text
    assert "允许在\n   shape_category_query 中给出一个稳定可见的“颜色 + 形状”别名" in prompt_text
    assert "例如 teal white bottle" in prompt_text


def test_parse_semantic_plan_uses_first_qwen_recommendation() -> None:
    plan = parse_semantic_plan(
        _response(),
        context=_context(),
        model="fake-qwen",
        rendered_prompt="rendered prompt",
    )

    assert plan.target.primary_query == "orange bottle"
    assert plan.receiver.primary_query == "blue square pad"
    assert plan.input_frame_ids == (0, 9, 15)
    assert len(plan.prompt_sha256) == 64


def test_target_only_qwen_contract_accepts_exactly_target() -> None:
    context = _target_only_context()
    plan = parse_semantic_plan(
        _target_only_response(),
        context=context,
        model="fake-qwen",
        rendered_prompt="rendered prompt",
    )

    assert plan.annotation_mode is AnnotationMode.TARGET_ONLY
    assert tuple(item.role for item in plan.role_plans) == ("target",)
    assert plan.target.primary_query == "orange bottle"
    with pytest.raises(KeyError, match="not applicable"):
        _ = plan.receiver

    with pytest.raises(QwenStageError, match="exactly"):
        parse_semantic_plan(
            _response(),
            context=context,
            model="fake-qwen",
            rendered_prompt="rendered prompt",
        )


def test_parse_semantic_plan_canonicalizes_exact_duplicate_candidates() -> None:
    payload = json.loads(_response())
    payload["target"]["shape_category_query"] = "orange bottle"
    payload["target"]["recommended_order"].insert(1, "shape_category_query")
    payload["receiver"]["category_query"] = "blue square pad"

    plan = parse_semantic_plan(
        json.dumps(payload),
        context=_context(),
        model="fake-qwen",
        rendered_prompt="rendered prompt",
    )

    assert plan.target.query_bank is not None
    assert plan.target.query_bank.shape_category_query is None
    assert plan.target.query_bank.recommended_order == (
        "color_category_query",
        "category_query",
        "general_fallback_query",
    )
    assert plan.receiver.query_bank is not None
    assert plan.receiver.query_bank.category_query == "blue square pad"
    assert plan.receiver.query_bank.color_category_query is None
    assert plan.receiver.query_bank.recommended_order == (
        "category_query",
        "shape_category_query",
        "general_fallback_query",
    )
    assert plan.receiver.primary_query == "blue square pad"


def test_parse_semantic_plan_deduplicates_exclude_terms_in_order() -> None:
    payload = json.loads(_response())
    payload["target"]["exclude"] = ["blue pad", "bottle", "blue pad", "bottle"]

    plan = parse_semantic_plan(
        json.dumps(payload),
        context=_context(),
        model="fake-qwen",
        rendered_prompt="rendered prompt",
    )

    assert plan.target.exclude == ("blue pad", "bottle")


def test_parse_semantic_plan_completes_omitted_candidate_order_entries() -> None:
    payload = json.loads(_response())
    payload["receiver"]["recommended_order"] = [
        "color_category_query",
        "shape_category_query",
        "category_query",
    ]

    plan = parse_semantic_plan(
        json.dumps(payload),
        context=_context(),
        model="fake-qwen",
        rendered_prompt="rendered prompt",
    )

    assert plan.receiver.query_bank is not None
    assert plan.receiver.query_bank.recommended_order == (
        "color_category_query",
        "shape_category_query",
        "category_query",
        "general_fallback_query",
    )


def test_parse_semantic_plan_rejects_bbox_and_non_candidate_seed() -> None:
    payload = json.loads(_response())
    payload["target"]["bbox"] = [0, 0, 10, 10]
    with pytest.raises(QwenStageError, match="extra=.*bbox"):
        parse_semantic_plan(
            json.dumps(payload),
            context=_context(),
            model="fake-qwen",
            rendered_prompt="prompt",
        )

    payload = json.loads(_response())
    payload["target"]["seed_frame_id"] = 9
    with pytest.raises(QwenStageError, match="not an eligible seed"):
        parse_semantic_plan(
            json.dumps(payload),
            context=_context(),
            model="fake-qwen",
            rendered_prompt="prompt",
        )


def test_run_qwen_stage_with_cpu_fake_response(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text(
        "task: {task_text}\nframes:\n{labeled_multimodal_frames}\nreturn json",
        encoding="utf-8",
    )
    config = QwenConfig(
        endpoint="http://127.0.0.1:18086/v1/chat/completions",
        model="fake-qwen",
        prompt_template=prompt_path,
    )
    client = FakeQwenClient(_response())

    result = run_qwen_stage(_context(), _frames(), config, client)

    assert result.semantic_plan.usable
    assert result.health["status"] == "ok"
    assert client.messages is not None


def test_run_qwen_stage_preserves_invalid_raw_response(tmp_path: Path) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text(
        "{labeled_multimodal_frames}",
        encoding="utf-8",
    )
    config = QwenConfig(
        endpoint="http://127.0.0.1:18086/v1/chat/completions",
        model="fake-qwen",
        prompt_template=prompt_path,
    )

    with pytest.raises(QwenStageError) as captured:
        run_qwen_stage(_context(), _frames(), config, FakeQwenClient("not json"))

    assert captured.value.raw_response == "not json"
    assert captured.value.rendered_prompt is not None
