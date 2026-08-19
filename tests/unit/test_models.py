from __future__ import annotations

import hashlib

import pytest

from robotwin_annotation_v2.domain import AnnotationMode
from robotwin_annotation_v2.models import (
    EpisodeRef,
    FramePurpose,
    FrameWindow,
    LoopContext,
    LoopEvents,
    MaskQCStatus,
    MaskStatus,
    QueryBank,
    RoleMaskResult,
    RoleSemanticPlan,
    SemanticFrame,
    SemanticPlan,
    SemanticPlanError,
    SemanticStatus,
    TargetOnlyEvents,
    derive_episode_windows,
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
    assert context.events.target_hold_window == FrameWindow(69, 122)
    assert context.events.target_output_window == FrameWindow(4, 122)
    assert context.events.receiver_window == FrameWindow(68, 136)
    assert context.windows.operation == FrameWindow(4, 136)
    assert context.windows.target == FrameWindow(4, 122)
    assert context.windows.gripper == FrameWindow(4, 136)
    assert context.seed_candidates("target") == (0,)
    assert context.seed_candidates("receiver") == (0,)
    payload = context.to_json()
    assert payload["format_version"] == "robotwin_loop_context_v3"
    assert payload["timeline_kind"] == "pick_place"
    assert payload["windows"] == {
        "operation": [4, 136],
        "target_0": [4, 122],
        "receiver_0": [68, 136],
        "gripper": [4, 136],
    }


def test_loop_events_reject_invalid_order() -> None:
    with pytest.raises(ValueError, match="not ordered"):
        LoopEvents("right", 4, 56, 68, 60, 136)


def test_target_only_events_derive_full_target_and_gripper_hold_windows() -> None:
    events = TargetOnlyEvents("left", 4, 53, 65)

    windows = derive_episode_windows(events, frame_count=139)

    assert events.target_window == FrameWindow(4, 65)
    assert events.target_hold_window(139) == FrameWindow(66, 138)
    assert windows.target == FrameWindow(4, 138)
    assert windows.receiver is None
    assert windows.operation == FrameWindow(4, 138)
    assert windows.gripper == FrameWindow(4, 138)
    assert windows.to_json() == {
        "operation": [4, 138],
        "target_0": [4, 138],
        "receiver_0": None,
        "gripper": [4, 138],
    }


def test_target_only_events_reject_close_before_remove_start() -> None:
    with pytest.raises(ValueError, match="not ordered"):
        TargetOnlyEvents("right", 60, 55, 68)


def test_target_only_context_has_v3_close_hold_contract_without_fake_open() -> None:
    context = LoopContext(
        episode=EpisodeRef("adjust_bottle", 0, "cam_high"),
        task_text="Lift the bottle with the left arm.",
        frame_count=139,
        events=TargetOnlyEvents("left", 4, 53, 65),
        semantic_frames=(
            SemanticFrame(
                0,
                FramePurpose.PRE_GRASP_SEED_CANDIDATE,
                ("target",),
            ),
            SemanticFrame(66, FramePurpose.POST_GRASP_CONTEXT, ("target",)),
            SemanticFrame(102, FramePurpose.POST_GRASP_CONTEXT, ("target",)),
        ),
        state_source="episode_000000.parquet",
        video_source="episode_000000.mp4",
        annotation_mode=AnnotationMode.TARGET_ONLY,
    )

    payload = context.to_json()

    assert context.windows.target == FrameWindow(4, 138)
    assert context.windows.receiver is None
    assert context.windows.gripper == FrameWindow(4, 138)
    assert context.seed_candidates("receiver") == ()
    assert payload["timeline_kind"] == "close_hold"
    assert payload["events"] == {
        "active_arm": "left",
        "t_remove_start": 4,
        "t_close_start": 53,
        "t_close_end": 65,
    }
    assert not ({"t_open_start", "t_open_done"} & payload["events"].keys())
    assert payload["windows"] == {
        "operation": [4, 138],
        "target_0": [4, 138],
        "receiver_0": None,
        "gripper": [4, 138],
    }


@pytest.mark.parametrize(
    ("mode", "events", "expected_name"),
    (
        (
            AnnotationMode.PICK_PLACE,
            TargetOnlyEvents("left", 4, 53, 65),
            "PickPlaceEvents",
        ),
        (
            AnnotationMode.TARGET_ONLY,
            LoopEvents("right", 4, 56, 68, 123, 136),
            "TargetOnlyEvents",
        ),
    ),
)
def test_loop_context_rejects_mode_event_type_mismatch(
    mode: AnnotationMode,
    events: LoopEvents | TargetOnlyEvents,
    expected_name: str,
) -> None:
    with pytest.raises(TypeError, match=expected_name):
        LoopContext(
            episode=_episode(),
            task_text="test task",
            frame_count=138,
            events=events,
            semantic_frames=(
                SemanticFrame(
                    0,
                    FramePurpose.PRE_GRASP_SEED_CANDIDATE,
                    ("target",),
                ),
            ),
            state_source="state.parquet",
            video_source="video.mp4",
            annotation_mode=mode,
        )


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


def test_query_bank_allows_attributed_object_only_as_general_fallback() -> None:
    bank = QueryBank(
        category_query="fan",
        color_category_query="silver fan",
        general_fallback_query="silver object",
    )

    assert bank.general_fallback_query == "silver object"

    with pytest.raises(SemanticPlanError, match="forbidden descriptor"):
        QueryBank(category_query="silver object")
    with pytest.raises(SemanticPlanError, match="forbidden descriptor"):
        QueryBank(category_query="fan", general_fallback_query="object")


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


def test_role_mask_result_keeps_envelope_as_diagnostic_metadata() -> None:
    result = RoleMaskResult(
        role="target",
        status=MaskStatus.OK,
        seed_frame_id=0,
        primary_query="bottle",
        output_window=FrameWindow(1, 4),
        seed_rgb_path="target_0/seed.rgb.png",
        seed_mask_path="target_0/seed.mask.png",
        canonical_envelope_path="target_0/canonical_envelope.png",
        native_track_path="target_0/native_track.npz",
        temporal_qc_path="target_0/temporal_qc.json",
        nonempty_frames=4,
        qc_status=MaskQCStatus.PASSED,
        qc_selected_candidate="query_0_seed_0",
        qc_reason="candidate matches the target",
    )

    payload = result.to_json()

    assert payload["canonical_envelope_path"] == "target_0/canonical_envelope.png"
    assert "canonical_envelope_guard_applied" not in payload
    assert "pre_envelope_guard_temporal_qc_path" not in payload
