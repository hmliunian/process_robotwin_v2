"""Test PrepareKeyframes use case with fake adapters."""

from robotwin_annotation_v2.adapters.fake_adapters import (
    FakeArtifactRepository,
    FakeEpisodeRepository,
    FakeFrameSource,
    FakeGroundingService,
    FakeKeyframeSelector,
    FakeSemanticPlanner,
    FakeSingleFrameSegmenter,
    FakeTimelineDetector,
)
from robotwin_annotation_v2.application import PrepareKeyframes
from robotwin_annotation_v2.domain import EpisodeRef
from robotwin_annotation_v2.domain.policies import RolePolicyRegistry


def test_prepare_keyframes_end_to_end():
    """Test complete PrepareKeyframes workflow with fake adapters."""

    # Arrange: create all fake adapters
    episode_repo = FakeEpisodeRepository()
    semantic_planner = FakeSemanticPlanner()
    timeline_detector = FakeTimelineDetector()
    frame_source = FakeFrameSource()
    keyframe_selector = FakeKeyframeSelector()
    grounding_service = FakeGroundingService()
    segmenter = FakeSingleFrameSegmenter()
    artifact_repo = FakeArtifactRepository()
    policy_registry = RolePolicyRegistry()

    # Create use case
    use_case = PrepareKeyframes(
        episode_repo=episode_repo,
        semantic_planner=semantic_planner,
        timeline_detector=timeline_detector,
        frame_source=frame_source,
        keyframe_selector=keyframe_selector,
        grounding_service=grounding_service,
        segmenter=segmenter,
        artifact_repo=artifact_repo,
        policy_registry=policy_registry,
    )

    # Act: execute for one episode
    ref = EpisodeRef(coarse_task="move_pillbottle_pad", episode_id="007152")
    run_id = use_case.execute(ref)

    # Assert: verify run was created
    assert run_id.startswith("fake-run-")
    assert run_id in artifact_repo.runs

    # Verify run config
    run_data = artifact_repo.runs[run_id]
    assert run_data["config"]["episode"] == str(ref)
    assert run_data["config"]["phase"] == "keyframe"
    assert run_data["config"]["video_propagation"] is False

    # Verify requests were created (target + receiver)
    assert len(run_data["requests"]) == 2
    assert "007152_target_0" in run_data["requests"]
    assert "007152_receiver_0" in run_data["requests"]


def test_prepare_keyframes_target_candidates():
    """Test that target request generates correct candidates."""

    # Arrange
    artifact_repo = FakeArtifactRepository()
    use_case = PrepareKeyframes(
        episode_repo=FakeEpisodeRepository(),
        semantic_planner=FakeSemanticPlanner(),
        timeline_detector=FakeTimelineDetector(),
        frame_source=FakeFrameSource(),
        keyframe_selector=FakeKeyframeSelector(),
        grounding_service=FakeGroundingService(),
        segmenter=FakeSingleFrameSegmenter(),
        artifact_repo=artifact_repo,
        policy_registry=RolePolicyRegistry(),
    )

    # Act
    ref = EpisodeRef(coarse_task="move_pillbottle_pad", episode_id="007152")
    run_id = use_case.execute(ref)

    # Assert: check target_0 request
    target_data = artifact_repo.load_request(run_id, "007152", "target_0")

    # Should have request info
    assert target_data["request"]["slot"] == "target_0"
    assert target_data["request"]["anchor_kind"] == "pre_grasp_visible"

    # Should have grounding evidence
    assert "grounding" in target_data
    assert "refined_query" in target_data["grounding"]
    assert "(refined)" in target_data["grounding"]["refined_query"]

    # Should have 3 candidates (text_only, box_only, text_box)
    assert len(target_data["candidates"]) == 3

    methods = {c["method"] for c in target_data["candidates"]}
    assert methods == {"text_only", "box_only", "text_box"}

    # All candidates should have masks with reasonable area
    for candidate in target_data["candidates"]:
        assert 0 < candidate["area_fraction"] < 1


def test_prepare_keyframes_receiver_candidates():
    """Test that receiver request generates correct candidates."""

    # Arrange
    artifact_repo = FakeArtifactRepository()
    use_case = PrepareKeyframes(
        episode_repo=FakeEpisodeRepository(),
        semantic_planner=FakeSemanticPlanner(),
        timeline_detector=FakeTimelineDetector(),
        frame_source=FakeFrameSource(),
        keyframe_selector=FakeKeyframeSelector(),
        grounding_service=FakeGroundingService(),
        segmenter=FakeSingleFrameSegmenter(),
        artifact_repo=artifact_repo,
        policy_registry=RolePolicyRegistry(),
    )

    # Act
    ref = EpisodeRef(coarse_task="move_pillbottle_pad", episode_id="007152")
    run_id = use_case.execute(ref)

    # Assert: check receiver_0 request
    receiver_data = artifact_repo.load_request(run_id, "007152", "receiver_0")

    assert receiver_data["request"]["slot"] == "receiver_0"
    assert receiver_data["request"]["anchor_kind"] == "static_receiver_visible"

    # Receiver should also have 3 candidates
    assert len(receiver_data["candidates"]) == 3


def test_prepare_keyframes_different_windows():
    """Test that target and receiver can select different frames."""

    # Arrange
    artifact_repo = FakeArtifactRepository()
    use_case = PrepareKeyframes(
        episode_repo=FakeEpisodeRepository(),
        semantic_planner=FakeSemanticPlanner(),
        timeline_detector=FakeTimelineDetector(),
        frame_source=FakeFrameSource(),
        keyframe_selector=FakeKeyframeSelector(),
        grounding_service=FakeGroundingService(),
        segmenter=FakeSingleFrameSegmenter(),
        artifact_repo=artifact_repo,
        policy_registry=RolePolicyRegistry(),
    )

    # Act
    ref = EpisodeRef(coarse_task="move_pillbottle_pad", episode_id="007152")
    run_id = use_case.execute(ref)

    # Assert: target and receiver should have different allowed windows
    target_data = artifact_repo.load_request(run_id, "007152", "target_0")
    receiver_data = artifact_repo.load_request(run_id, "007152", "receiver_0")

    target_window = target_data["request"]["allowed_window"]
    receiver_window = receiver_data["request"]["allowed_window"]

    # Target: [move_start, close_start) = [10, 49]
    assert target_window == [10, 49]

    # Receiver: [0, move_start] = [0, 10]
    assert receiver_window == [0, 10]

    # They should be different
    assert target_window != receiver_window
