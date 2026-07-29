"""Unit tests for domain policies."""

from robotwin_annotation_v2.domain import (
    AnchorKind,
    EpisodeRef,
)
from robotwin_annotation_v2.domain.policies import (
    InteractionTimeline,
    RolePolicyRegistry,
    SemanticPlan,
    StaticReceiverSeedPolicy,
    TargetSeedPolicy,
)


def test_target_seed_policy():
    ref = EpisodeRef(coarse_task="move_pillbottle_pad", episode_id="007152")

    semantic = SemanticPlan(
        episode=ref,
        target_query="white pill bottle",
    )

    timeline = InteractionTimeline(
        episode=ref,
        move_start=10,
        close_start=50,
    )

    policy = TargetSeedPolicy()
    requests = policy.create_requests(semantic, timeline)

    assert len(requests) == 1
    req = requests[0]
    assert req.slot.name == "target_0"
    assert req.anchor_kind == AnchorKind.PRE_GRASP_VISIBLE
    assert req.allowed_window.first == 10
    assert req.allowed_window.last == 49
    assert req.visual_query == "white pill bottle"
    assert "gripper" in "".join(req.exclusions)


def test_target_seed_policy_missing_timeline():
    ref = EpisodeRef(coarse_task="move_pillbottle_pad", episode_id="007152")

    semantic = SemanticPlan(
        episode=ref,
        target_query="white pill bottle",
    )

    # Missing close_start
    timeline = InteractionTimeline(
        episode=ref,
        move_start=10,
    )

    policy = TargetSeedPolicy()
    requests = policy.create_requests(semantic, timeline)

    assert len(requests) == 0


def test_static_receiver_seed_policy():
    ref = EpisodeRef(coarse_task="move_pillbottle_pad", episode_id="007152")

    semantic = SemanticPlan(
        episode=ref,
        target_query="white pill bottle",
        receiver_query="blue square pad",
        has_static_receiver=True,
    )

    timeline = InteractionTimeline(
        episode=ref,
        move_start=30,
    )

    policy = StaticReceiverSeedPolicy()
    requests = policy.create_requests(semantic, timeline)

    assert len(requests) == 1
    req = requests[0]
    assert req.slot.name == "receiver_0"
    assert req.anchor_kind == AnchorKind.STATIC_RECEIVER_VISIBLE
    assert req.allowed_window.first == 0
    assert req.allowed_window.last == 30
    assert req.visual_query == "blue square pad"


def test_static_receiver_policy_no_receiver():
    ref = EpisodeRef(coarse_task="move_pillbottle_pad", episode_id="007152")

    semantic = SemanticPlan(
        episode=ref,
        target_query="white pill bottle",
        has_static_receiver=False,
    )

    timeline = InteractionTimeline(
        episode=ref,
        move_start=30,
    )

    policy = StaticReceiverSeedPolicy()
    requests = policy.create_requests(semantic, timeline)

    assert len(requests) == 0


def test_role_policy_registry():
    ref = EpisodeRef(coarse_task="move_pillbottle_pad", episode_id="007152")

    semantic = SemanticPlan(
        episode=ref,
        target_query="white pill bottle",
        receiver_query="blue square pad",
        has_static_receiver=True,
    )

    timeline = InteractionTimeline(
        episode=ref,
        move_start=10,
        close_start=50,
    )

    registry = RolePolicyRegistry()
    requests = registry.get_requests(semantic, timeline)

    # Should get both target and receiver requests
    assert len(requests) == 2

    slots = {req.slot.name for req in requests}
    assert "target_0" in slots
    assert "receiver_0" in slots
