from __future__ import annotations

import hashlib

import pytest

from robotwin_annotation_v2.models import (
    EpisodeRef,
    FramePurpose,
    FrameWindow,
    LoopContext,
    LoopEvents,
    QueryBank,
    RoleSemanticPlan,
    SemanticFrame,
    SemanticPlan,
    SemanticPlanError,
    SemanticStatus,
    normalize_query,
)


def _episode() -> EpisodeRef:
    return EpisodeRef("move_pillbottle_pad", 7152, "cam_high")


def _loop() -> LoopContext:
    return LoopContext(
        episode=_episode(),
        task_text="Move the orange bottle onto the pad",
        frame_count=138,
        events=LoopEvents("right", 4, 56, 68, 123, 136),
        semantic_frames=(
            SemanticFrame(
                0,
                FramePurpose.PRE_GRASP_SEED_CANDIDATE,
                ("target", "receiver"),
            ),
            SemanticFrame(69, FramePurpose.POST_GRASP_CONTEXT, ("target",)),
            SemanticFrame(128, FramePurpose.PLACE_CONTEXT, ("receiver",)),
        ),
        state_source="episode_007152.parquet",
        video_source="episode_007152.mp4",
    )


def test_loop_context_contract() -> None:
    context = _loop()

    assert context.episode.episode_id == "007152"
    assert context.events.target_window == FrameWindow(4, 68)
    assert context.events.receiver_window == FrameWindow(68, 136)
    assert context.seed_candidates("target") == (0,)
    assert context.seed_candidates("receiver") == (0,)
    assert context.to_json()["windows"]["loop"] == [4, 136]


def test_loop_events_reject_invalid_order() -> None:
    with pytest.raises(ValueError, match="not ordered"):
        LoopEvents("right", 4, 56, 68, 60, 136)


@pytest.mark.parametrize(
    "query",
    ["bottle", "orange bottle", "plastic bottle", "blue square pad"],
)
def test_sam3_query_accepts_short_object_phrases(query: str) -> None:
    assert normalize_query(query) == query


@pytest.mark.parametrize(
    "query",
    [
        "the bottle",
        "Bottle",
        "white bottle with label",
        "bottle near gripper",
        "object",
        "white cylindrical plastic medicine bottle",
    ],
)
def test_sam3_query_rejects_non_native_phrases(query: str) -> None:
    with pytest.raises(SemanticPlanError):
        normalize_query(query)


def test_query_bank_uses_qwen_order_without_automatic_fallback() -> None:
    bank = QueryBank(
        category_query="pad",
        color_category_query="blue square pad",
        shape_category_query="square pad",
        general_fallback_query="mat",
        recommended_order=(
            "color_category_query",
            "shape_category_query",
            "category_query",
            "general_fallback_query",
        ),
    )

    assert bank.primary_query == "blue square pad"
    assert bank.to_json()["primary_query"] == "blue square pad"


def test_query_bank_requires_fallback_last() -> None:
    with pytest.raises(SemanticPlanError, match="must be last"):
        QueryBank(
            category_query="bottle",
            general_fallback_query="container",
            recommended_order=("general_fallback_query", "category_query"),
        )


def test_semantic_plan_contains_joint_roles() -> None:
    target = RoleSemanticPlan(
        role="target",
        status=SemanticStatus.OK,
        seed_frame_id=0,
        query_bank=QueryBank(
            category_query="bottle",
            color_category_query="orange bottle",
            recommended_order=("color_category_query", "category_query"),
        ),
        exclude=("blue pad",),
        reason="该物体随后被抓取并移动。",
    )
    receiver = RoleSemanticPlan(
        role="receiver",
        status=SemanticStatus.OK,
        seed_frame_id=0,
        query_bank=QueryBank(
            category_query="pad",
            color_category_query="blue square pad",
            recommended_order=("color_category_query", "category_query"),
        ),
        exclude=("orange bottle",),
        reason="该区域承接被移动物体。",
    )
    plan = SemanticPlan(
        episode=_episode(),
        role_plans=(target, receiver),
        model="qwen3.5-27b",
        prompt_sha256=hashlib.sha256(b"prompt").hexdigest(),
        input_frame_ids=(0, 69, 128),
        raw_response="{}",
    )

    assert plan.usable
    assert plan.target.primary_query == "orange bottle"
    assert plan.receiver.primary_query == "blue square pad"


def test_no_clear_seed_cannot_contain_query() -> None:
    with pytest.raises(SemanticPlanError):
        RoleSemanticPlan(
            role="target",
            status=SemanticStatus.NO_CLEAR_SEED,
            seed_frame_id=0,
            query_bank=QueryBank(category_query="bottle"),
            exclude=(),
            reason="没有清晰 seed。",
        )
