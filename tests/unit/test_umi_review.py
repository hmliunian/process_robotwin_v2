from __future__ import annotations

import json
from pathlib import Path

import pytest

from robotwin_annotation_v2.adapters.umi_review import UmiPreview, parse_arm_review


def _answer() -> dict[str, object]:
    return {
        "decision": "right",
        "confidence": 0.95,
        "left_hand": "Idle throughout",
        "right_hand": "Holds the spoon",
        "evidence_frames": ["head/right@0ms", "right_wrist/right@100ms"],
        "reason": "Only the right hand touches the spoon or bowl",
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"decision": "both"},
        {"confidence": float("nan")},
        {"confidence": True},
        {"evidence_frames": ["head/right@0ms", "invented@100ms"]},
        {"evidence_frames": ["head/right@0ms", "left_wrist/left@0ms"]},
        {"left_hand": ""},
    ],
)
def test_screen_rejects_ungrounded_or_invalid_answers(changes: dict[str, object]) -> None:
    frames = (
        UmiPreview(Path("head.jpg"), "head/right@0ms", 0),
        UmiPreview(Path("left.jpg"), "left_wrist/left@0ms", 0),
        UmiPreview(Path("right.jpg"), "right_wrist/right@100ms", 100),
    )
    with pytest.raises(ValueError):
        parse_arm_review(json.dumps({**_answer(), **changes}), frames)


def test_screen_preserves_two_hand_evidence_and_bimanual_decision() -> None:
    frames = (
        UmiPreview(Path("head.jpg"), "head/right@0ms", 0),
        UmiPreview(Path("right.jpg"), "right_wrist/right@100ms", 100),
    )
    answer = {**_answer(), "decision": "bimanual", "left_hand": "Steadies the bowl"}
    assert parse_arm_review(json.dumps(answer), frames) == answer
