from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from robotwin_annotation_v2.adapters import QwenCompletion
from robotwin_annotation_v2.adapters.artifact_store import ArtifactStore
from robotwin_annotation_v2.application.episode_pipeline import _load_saved_semantic_plan
from robotwin_annotation_v2.config import QwenConfig
from robotwin_annotation_v2.domain import AnnotationMode, TargetProfile
from robotwin_annotation_v2.models import (
    EpisodeRef,
    FramePurpose,
    LoopContext,
    LoopEvents,
    QueryBank,
    SemanticFrame,
    SemanticPlanError,
    TargetOnlyEvents,
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


def _target_only_context(*, reopen_start: int | None = None) -> LoopContext:
    return LoopContext(
        episode=EpisodeRef("move_object", 1, "cam_high"),
        task_text="Pick up the bottle.",
        frame_count=20,
        events=TargetOnlyEvents("right", 2, 6, 8, reopen_start),
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


def _door_open_response(
    category_query: str,
    *,
    color_query: str | None = None,
    shape_query: str | None = None,
    fallback_query: str | None = None,
) -> str:
    target = json.loads(_target_only_response())["target"]
    target.update(
        {
            "category_query": category_query,
            "color_category_query": color_query,
            "shape_category_query": shape_query,
            "general_fallback_query": fallback_query,
            "recommended_order": [
                field
                for field, value in (
                    ("category_query", category_query),
                    ("color_category_query", color_query),
                    ("shape_category_query", shape_query),
                    ("general_fallback_query", fallback_query),
                )
                if value is not None
            ],
        }
    )
    return json.dumps({"target": target})


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


class TerminatedQwenClient(FakeQwenClient):
    def __init__(self, response: str, finish_reason: str) -> None:
        super().__init__(response)
        self.finish_reason = finish_reason

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
    ) -> QwenCompletion:
        super().complete(messages, max_tokens=max_tokens)
        return QwenCompletion(self.response, self.model_id, self.finish_reason)


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
    assert "seed_eligible_roles=target,receiver; seed_candidate=yes" in content[0]["text"]
    assert "context_only_roles=receiver" in content[4]["text"]
    assert "seed_candidate=no; forbidden_as_seed=yes" in content[4]["text"]
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


def test_open_set_semantic_prompt_requires_category_query_for_ok_roles() -> None:
    template = (
        PROJECT_ROOT / "configs/prompts/target_receiver_semantic_open_set.txt"
    ).read_text(encoding="utf-8")

    request = build_qwen_request(_context(), _frames(), template)
    prompt_text = " ".join(request.rendered_prompt.split())

    assert (
        '对每个角色，当 status="ok" 时，category_query 必须是非空字符串，且 '
        "recommended_order 必须包含 category_query。"
    ) in prompt_text
    assert (
        '"category_query": "required 1-4 lowercase English words for ok; '
        'null only for no_clear_seed"'
    ) in prompt_text
    assert (
        '对每个角色，当 status="ok" 时，seed_frame_id 必须来自该角色标记为 '
        "seed_candidate=yes 的候选；"
    ) in prompt_text
    assert (
        '"seed_frame_id": "integer from this role\'s seed_candidate=yes frames for ok; '
        'null only for no_clear_seed"'
    ) in prompt_text


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


def test_target_only_semantic_prompt_contains_only_target_contract() -> None:
    template = (
        PROJECT_ROOT / "configs/prompts/target_only_semantic.txt"
    ).read_text(encoding="utf-8")
    context = _target_only_context()
    frames = {frame_id: _frames()[frame_id] for frame_id in (0, 9)}

    request = build_qwen_request(context, frames, template)

    assert "本模式只识别 target" in request.rendered_prompt
    assert "不得添加其他角色" in request.rendered_prompt
    assert '"target"' in request.rendered_prompt
    assert "receiver" not in request.rendered_prompt
    assert "active_arm: right" in request.rendered_prompt
    assert "remove_start: 2" in request.rendered_prompt
    assert "close_start: 6" in request.rendered_prompt
    assert "close_end: 8" in request.rendered_prompt
    assert "hold_end: 19" in request.rendered_prompt
    assert "open_start" not in request.rendered_prompt
    assert "open_done" not in request.rendered_prompt


def test_target_only_open_set_semantic_prompt_uses_target_only_timeline() -> None:
    template = (
        PROJECT_ROOT / "configs/prompts/target_only_semantic_open_set.txt"
    ).read_text(encoding="utf-8")
    context = _target_only_context()
    frames = {frame_id: _frames()[frame_id] for frame_id in (0, 9)}

    request = build_qwen_request(context, frames, template)

    assert "开放集语义规划器" in request.rendered_prompt
    assert "white bar" in request.rendered_prompt
    assert "本模式只识别 target" in request.rendered_prompt
    assert "receiver" not in request.rendered_prompt
    assert "active_arm: right" in request.rendered_prompt
    assert "remove_start: 2" in request.rendered_prompt
    assert "close_start: 6" in request.rendered_prompt
    assert "close_end: 8" in request.rendered_prompt
    assert "hold_end: 19" in request.rendered_prompt
    assert "open_start" not in request.rendered_prompt
    assert "open_done" not in request.rendered_prompt


def test_door_open_semantic_prompt_satisfies_multimodal_contract() -> None:
    template = (
        PROJECT_ROOT / "configs/prompts/target_only_door_open_semantic_open_set.txt"
    ).read_text(encoding="utf-8")
    context = _target_only_context()
    frames = {frame_id: _frames()[frame_id] for frame_id in (0, 9)}

    request = build_qwen_request(context, frames, template)

    assert "<image frame_id=0>" in request.rendered_prompt
    assert "<image frame_id=9>" in request.rendered_prompt
    prompt_text = " ".join(request.rendered_prompt.split())

    assert "smallest complete visible functional part" in prompt_text
    assert "compare every frame marked ``seed_candidate=yes``" in prompt_text
    assert "do not default to the latest pre-close frame" in prompt_text
    assert "Keep the word ``handle`` as the semantic head" in prompt_text
    assert "All non-empty candidates must preserve the same physical identity" in prompt_text
    assert "no separable handle exists, not merely because the handle is hard to see" in prompt_text
    assert "S1" in prompt_text and "S2" in prompt_text and "S3" in prompt_text
    assert "downstream last-resort bbox fallback" in prompt_text


@pytest.mark.parametrize(
    "query",
    (
        "vertical bar",
        "horizontal bar",
        "latch",
        "door latch",
        "microwave door",
        "door",
        "panel",
        "body",
    ),
)
def test_door_open_profile_rejects_non_handle_queries(query: str) -> None:
    with pytest.raises(QwenStageError, match="handle as the head noun"):
        parse_semantic_plan(
            _door_open_response(query),
            context=_target_only_context(),
            model="fake-qwen",
            rendered_prompt="rendered prompt",
            target_profile=TargetProfile.DOOR_OPEN,
        )


@pytest.mark.parametrize(
    ("query", "expected"),
    (
        ("microwave handle", "handle"),
        ("microwave door handle", "handle"),
        ("white microwave door handle", "white handle"),
        ("vertical microwave door handle", "vertical handle"),
        ("black door handle", "black handle"),
    ),
)
def test_door_open_profile_canonicalizes_microwave_handle_aliases(
    query: str,
    expected: str,
) -> None:
    plan = parse_semantic_plan(
        _door_open_response(query),
        context=_target_only_context(),
        model="fake-qwen",
        rendered_prompt="rendered prompt",
        target_profile=TargetProfile.DOOR_OPEN,
    )

    assert plan.target.primary_query == expected
    assert plan.target_profile is TargetProfile.DOOR_OPEN
    assert plan.to_json()["target_profile"] == "door_open"


@pytest.mark.parametrize("query", ("handle", "door handle", "black handle", "vertical handle"))
def test_door_open_profile_accepts_handle_head_queries(query: str) -> None:
    plan = parse_semantic_plan(
        _door_open_response(query),
        context=_target_only_context(),
        model="fake-qwen",
        rendered_prompt="rendered prompt",
        target_profile=TargetProfile.DOOR_OPEN,
    )

    assert plan.target.primary_query == query


def test_door_open_profile_forces_category_first_query_ladder() -> None:
    payload = json.loads(_door_open_response(
        "handle",
        color_query="black handle",
        shape_query="vertical handle",
        fallback_query="door handle",
    ))
    payload["target"]["recommended_order"] = [
        "shape_category_query",
        "color_category_query",
        "category_query",
        "general_fallback_query",
    ]

    plan = parse_semantic_plan(
        json.dumps(payload),
        context=_target_only_context(),
        model="fake-qwen",
        rendered_prompt="rendered prompt",
        target_profile=TargetProfile.DOOR_OPEN,
    )

    assert plan.target.query_bank is not None
    assert plan.target.query_bank.recommended_order == (
        "category_query",
        "color_category_query",
        "shape_category_query",
        "general_fallback_query",
    )
    assert plan.target.primary_query == "handle"


@pytest.mark.parametrize("query", ("door panel", "moving door panel"))
def test_door_open_profile_preserves_door_panel_proxy(query: str) -> None:
    plan = parse_semantic_plan(
        _door_open_response(query),
        context=_target_only_context(),
        model="fake-qwen",
        rendered_prompt="rendered prompt",
        target_profile=TargetProfile.DOOR_OPEN,
    )

    assert plan.target.primary_query == query


@pytest.mark.parametrize("query", ("door-panel", "moving-door-panel"))
def test_door_open_profile_canonicalizes_hyphenated_panel_proxy(query: str) -> None:
    plan = parse_semantic_plan(
        _door_open_response(query),
        context=_target_only_context(),
        model="fake-qwen",
        rendered_prompt="rendered prompt",
        target_profile=TargetProfile.DOOR_OPEN,
    )

    assert plan.target.primary_query == query.replace("-", " ")


def test_door_open_profile_rejects_mixed_handle_and_panel_proxy() -> None:
    with pytest.raises(QwenStageError, match="door-panel proxy"):
        parse_semantic_plan(
            _door_open_response("handle", fallback_query="moving door panel"),
            context=_target_only_context(),
            model="fake-qwen",
            rendered_prompt="rendered prompt",
            target_profile=TargetProfile.DOOR_OPEN,
        )


@pytest.mark.parametrize("category_query", ("door panel", "moving door panel"))
def test_door_open_profile_keeps_panel_proxy_in_category_slot_only(
    category_query: str,
) -> None:
    plan = parse_semantic_plan(
        _door_open_response(category_query),
        context=_target_only_context(),
        model="fake-qwen",
        rendered_prompt="rendered prompt",
        target_profile=TargetProfile.DOOR_OPEN,
    )

    assert plan.target.query_bank is not None
    assert plan.target.query_bank.recommended_order == ("category_query",)


def test_door_open_profile_rejects_two_panel_proxy_fields() -> None:
    with pytest.raises(QwenStageError, match="sole category_query"):
        parse_semantic_plan(
            _door_open_response("door panel", fallback_query="moving door panel"),
            context=_target_only_context(),
            model="fake-qwen",
            rendered_prompt="rendered prompt",
            target_profile=TargetProfile.DOOR_OPEN,
        )


def test_door_open_profile_rejects_panel_proxy_mixed_with_handle_candidate() -> None:
    with pytest.raises(QwenStageError, match="sole category_query"):
        parse_semantic_plan(
            _door_open_response("moving door panel", color_query="black handle"),
            context=_target_only_context(),
            model="fake-qwen",
            rendered_prompt="rendered prompt",
            target_profile=TargetProfile.DOOR_OPEN,
        )


def test_query_bank_rejects_panel_proxy_outside_category_slot() -> None:
    with pytest.raises(SemanticPlanError, match="sole category_query"):
        QueryBank(
            category_query="moving door panel",
            general_fallback_query="door panel",
            allow_moving_door_panel=True,
        )


def test_generic_profile_keeps_door_queries_backward_compatible() -> None:
    plan = parse_semantic_plan(
        _door_open_response("vertical bar"),
        context=_target_only_context(),
        model="fake-qwen",
        rendered_prompt="rendered prompt",
    )

    assert plan.target.primary_query == "vertical bar"
    assert "target_profile" not in plan.to_json()


def test_specialized_profile_requires_target_only_context() -> None:
    with pytest.raises(QwenStageError, match="requires target_only"):
        parse_semantic_plan(
            _response(),
            context=_context(),
            model="fake-qwen",
            rendered_prompt="rendered prompt",
            target_profile=TargetProfile.DOOR_OPEN,
        )


def test_saved_door_open_plan_replays_with_the_same_profile(tmp_path: Path) -> None:
    context = _target_only_context()
    raw_response = _door_open_response("microwave door handle")
    prompt = "rendered prompt"
    plan = parse_semantic_plan(
        raw_response,
        context=context,
        model="fake-qwen",
        rendered_prompt=prompt,
        target_profile=TargetProfile.DOOR_OPEN,
    )
    store = ArtifactStore(tmp_path)
    store.save_loop("run", context.episode, context.to_json())
    store.save_semantic_plan(
        "run",
        context.episode,
        plan.to_json(),
        rendered_prompt=prompt,
        raw_response=raw_response,
    )

    replay = _load_saved_semantic_plan(
        store,
        "run",
        context,
        target_profile=TargetProfile.DOOR_OPEN,
    )

    assert replay.to_json() == plan.to_json()

    # An older door-open run may omit the profile from semantic_plan.json;
    # parsing under the configured profile upgrades that envelope while still
    # enforcing the current handle contract.
    legacy_store = ArtifactStore(tmp_path / "legacy")
    legacy_store.save_loop("run", context.episode, context.to_json())
    legacy_store.save_semantic_plan(
        "run",
        context.episode,
        {key: value for key, value in plan.to_json().items() if key != "target_profile"},
        rendered_prompt=prompt,
        raw_response=raw_response,
    )
    upgraded = _load_saved_semantic_plan(
        legacy_store,
        "run",
        context,
        target_profile=TargetProfile.DOOR_OPEN,
    )
    assert upgraded.target_profile is TargetProfile.DOOR_OPEN
    with pytest.raises(ValueError, match="target_profile differs"):
        _load_saved_semantic_plan(
            store,
            "run",
            context,
            target_profile=TargetProfile.CONTACT_PRESS,
        )

    generic_marker_store = ArtifactStore(tmp_path / "generic-marker")
    generic_marker_store.save_loop("run", context.episode, context.to_json())
    generic_marker = plan.to_json()
    generic_marker["target_profile"] = "grasp_manipulation"
    generic_marker_store.save_semantic_plan(
        "run",
        context.episode,
        generic_marker,
        rendered_prompt=prompt,
        raw_response=raw_response,
    )
    with pytest.raises(ValueError, match="target_profile differs"):
        _load_saved_semantic_plan(
            generic_marker_store,
            "run",
            context,
            target_profile=TargetProfile.DOOR_OPEN,
        )


def test_legacy_door_open_plan_replay_canonicalizes_context_words(tmp_path: Path) -> None:
    context = _target_only_context()
    prompt = "rendered prompt"
    raw_response = _door_open_response(
        "microwave door handle",
        color_query="white microwave door handle",
        shape_query="vertical microwave door handle",
        fallback_query="door handle",
    )
    legacy = parse_semantic_plan(
        raw_response,
        context=context,
        model="fake-qwen",
        rendered_prompt=prompt,
    )
    store = ArtifactStore(tmp_path)
    store.save_loop("run", context.episode, context.to_json())
    store.save_semantic_plan(
        "run",
        context.episode,
        legacy.to_json(),
        rendered_prompt=prompt,
        raw_response=raw_response,
    )

    upgraded = _load_saved_semantic_plan(
        store,
        "run",
        context,
        target_profile=TargetProfile.DOOR_OPEN,
    )

    assert upgraded.target_profile is TargetProfile.DOOR_OPEN
    assert upgraded.target.query_bank is not None
    assert upgraded.target.query_bank.category_query == "handle"
    assert upgraded.target.query_bank.color_category_query == "white handle"
    assert upgraded.target.query_bank.shape_category_query == "vertical handle"
    assert upgraded.target.primary_query == "handle"


def test_target_only_semantic_prompt_ends_hold_before_reopen() -> None:
    template = (
        PROJECT_ROOT / "configs/prompts/target_only_semantic_open_set.txt"
    ).read_text(encoding="utf-8")
    context = _target_only_context(reopen_start=15)
    frames = {frame_id: _frames()[frame_id] for frame_id in (0, 9)}

    request = build_qwen_request(context, frames, template)

    assert "hold_end: 14" in request.rendered_prompt
    assert "episode_end:" not in request.rendered_prompt


def test_target_only_rejects_a_pick_place_prompt_before_model_request() -> None:
    template = (
        "open={open_start}\nframes:\n{labeled_multimodal_frames}\n"
        "schema={response_schema}"
    )
    context = _target_only_context()
    frames = {frame_id: _frames()[frame_id] for frame_id in (0, 9)}

    with pytest.raises(QwenStageError, match="unknown prompt template.*open_start"):
        build_qwen_request(context, frames, template)


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


@pytest.mark.parametrize("finish_reason", ("length", "content_filter"))
def test_run_qwen_stage_rejects_abnormal_completion(
    tmp_path: Path,
    finish_reason: str,
) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("{labeled_multimodal_frames}", encoding="utf-8")
    config = QwenConfig(
        endpoint="http://127.0.0.1:18086/v1/chat/completions",
        model="fake-qwen",
        prompt_template=prompt_path,
    )

    with pytest.raises(
        QwenStageError,
        match=rf"finish_reason={finish_reason!r}",
    ) as captured:
        run_qwen_stage(
            _context(),
            _frames(),
            config,
            TerminatedQwenClient('{"target":', finish_reason),
        )

    assert captured.value.raw_response == '{"target":'
