"""Unit tests for domain models."""

import pytest

from robotwin_annotation_v2.domain import (
    AnchorKind,
    AnnotationRole,
    Box,
    EpisodeRef,
    FrameWindow,
    InstanceSlot,
    KeyframeRequest,
    VisualPrompt,
)


def test_episode_ref():
    ref = EpisodeRef(coarse_task="move_pillbottle_pad", episode_id="007152")
    assert str(ref) == "move_pillbottle_pad/episode_007152/cam_high"


def test_instance_slot_target():
    slot = InstanceSlot(name="target_0", role=AnnotationRole.TARGET)
    assert slot.name == "target_0"
    assert slot.role == AnnotationRole.TARGET
    assert slot.arm is None


def test_instance_slot_gripper_requires_arm():
    with pytest.raises(ValueError, match="must specify arm"):
        InstanceSlot(name="gripper_left", role=AnnotationRole.GRIPPER)


def test_instance_slot_gripper_with_arm():
    slot = InstanceSlot(name="gripper_left", role=AnnotationRole.GRIPPER, arm="left")
    assert slot.arm == "left"


def test_frame_window_valid():
    window = FrameWindow(first=10, last=50)
    assert 10 in window
    assert 50 in window
    assert 30 in window
    assert 9 not in window
    assert 51 not in window
    assert len(window) == 41


def test_frame_window_invalid():
    with pytest.raises(ValueError, match="first.*>.*last"):
        FrameWindow(first=50, last=10)


def test_box_valid():
    box = Box(x_min=0.2, y_min=0.3, x_max=0.8, y_max=0.9)
    assert abs(box.area() - 0.36) < 1e-6


def test_box_invalid_range():
    with pytest.raises(ValueError):
        Box(x_min=-0.1, y_min=0.2, x_max=0.8, y_max=0.9)


def test_box_invalid_order():
    with pytest.raises(ValueError, match="min >= max"):
        Box(x_min=0.8, y_min=0.2, x_max=0.2, y_max=0.9)


def test_visual_prompt_text_only():
    prompt = VisualPrompt(text="white pill bottle")
    assert prompt.text == "white pill bottle"
    assert prompt.bbox is None


def test_visual_prompt_bbox_only():
    box = Box(x_min=0.2, y_min=0.3, x_max=0.8, y_max=0.9)
    prompt = VisualPrompt(bbox=box)
    assert prompt.bbox == box
    assert prompt.text is None


def test_visual_prompt_empty_invalid():
    with pytest.raises(ValueError, match="at least text or bbox"):
        VisualPrompt()


def test_keyframe_request_revision():
    ref = EpisodeRef(coarse_task="move_pillbottle_pad", episode_id="007152")
    slot = InstanceSlot(name="target_0", role=AnnotationRole.TARGET)
    window = FrameWindow(first=0, last=100)

    request = KeyframeRequest(
        request_id="test_req",
        episode=ref,
        slot=slot,
        anchor_kind=AnchorKind.PRE_GRASP_VISIBLE,
        allowed_window=window,
        visual_query="white pill bottle",
        revision=1,
    )

    next_rev = request.next_revision()
    assert next_rev.revision == 2
    assert next_rev.request_id == request.request_id
    assert next_rev.visual_query == request.visual_query
