from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from robotwin_annotation_v2.domain import AnnotationMode
from robotwin_annotation_v2.models.timeline import TargetOnlyEvents
from robotwin_annotation_v2.urdf_gripper_data import (
    ActiveGripperLoop,
    UrdfGripperDataError,
    load_authoritative_loop_context,
)

TASK = "move_pillbottle_pad"
EPISODE_INDEX = 7152
EPISODE_ID = "007152"
CAMERA = "cam_high"
FRAME_COUNT = 20

PICK_EVENTS = {
    "active_arm": "right",
    "t_move_start": 2,
    "t_close_start": 5,
    "t_close_done": 8,
    "t_open_start": 12,
    "t_open_done": 15,
}
TARGET_ONLY_EVENTS = {
    "active_arm": "right",
    "t_remove_start": 3,
    "t_close_start": 6,
    "t_close_end": 9,
}
WINDOW_KEYS = {"operation", "target_0", "receiver_0", "gripper"}
PICK_EVENT_KEYS = set(PICK_EVENTS)
TARGET_ONLY_EVENT_KEYS = set(TARGET_ONLY_EVENTS)
VALID_CASES = (
    ("robotwin_loop_context_v1", "pick_place"),
    ("robotwin_loop_context_v2", "pick_place"),
    ("robotwin_loop_context_v3", "pick_place"),
    ("robotwin_loop_context_v2", "target_only"),
    ("robotwin_loop_context_v3", "target_only"),
)
UNEXPECTED_EVENT_CASES = tuple(
    (*case, "t_move_start" if case[1] == "pick_place" else "t_remove_start")
    for case in VALID_CASES
)
BOOLEAN_EVENT_CASES = tuple(
    (*case, "t_close_done" if case[1] == "pick_place" else "t_close_end")
    for case in VALID_CASES
)


def _payload(format_version: str, mode: str) -> dict[str, Any]:
    is_pick_place = mode == "pick_place"
    events = dict(PICK_EVENTS if is_pick_place else TARGET_ONLY_EVENTS)
    if format_version == "robotwin_loop_context_v1":
        windows = (
            {
                "loop": [2, 15],
                "target_0": [2, 8],
                "receiver_0": [8, 15],
            }
            if is_pick_place
            else {
                "loop": [3, 9],
                "target_0": [3, 9],
                "receiver_0": [9, 15],
            }
        )
    elif is_pick_place:
        windows = {
            "operation": [2, 15],
            "target_0": [2, 11 if format_version == "robotwin_loop_context_v3" else 8],
            "receiver_0": [8, 15],
            "gripper": [2, 15],
        }
    else:
        windows = {
            "operation": [3, 19],
            "target_0": [
                3,
                19 if format_version == "robotwin_loop_context_v3" else 9,
            ],
            "receiver_0": None,
            "gripper": [3, 19],
        }

    payload: dict[str, Any] = {
        "format_version": format_version,
        "episode": {
            "task": TASK,
            "episode_index": EPISODE_INDEX,
            "episode_id": EPISODE_ID,
            "camera": CAMERA,
        },
        "frame_count": FRAME_COUNT,
        "events": events,
        "windows": windows,
    }
    if format_version != "robotwin_loop_context_v1":
        payload.update(
            {
                "annotation_mode": mode,
                "timeline_kind": "pick_place" if is_pick_place else "close_hold",
                "required_object_roles": ["target", "receiver"]
                if is_pick_place
                else ["target"],
            }
        )
    elif not is_pick_place:
        payload["annotation_mode"] = mode
    return payload


def _write_payload(tmp_path: Path, payload: dict[str, Any]) -> Path:
    path = tmp_path / "loop.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("format_version", "mode"),
    VALID_CASES,
    ids=lambda case: "-".join(case) if isinstance(case, tuple) else str(case),
)
def test_loader_preserves_versioned_events_windows_mode_and_kind(
    tmp_path: Path,
    format_version: str,
    mode: str,
) -> None:
    payload = _payload(format_version, mode)
    path = _write_payload(tmp_path, payload)

    context = load_authoritative_loop_context(
        path,
        expected_task=TASK,
        expected_episode_index=EPISODE_INDEX,
        expected_camera=CAMERA,
    )

    if mode == "pick_place":
        expected_events = ActiveGripperLoop("right", 2, 5, 8, 12, 15)
        expected_target_end = 11 if format_version == "robotwin_loop_context_v3" else 8
        expected_windows = {
            "operation": [2, 15],
            "target_0": [2, expected_target_end],
            "receiver_0": [8, 15],
            "gripper": [2, 15],
        }
        expected_event_keys = PICK_EVENT_KEYS
        expected_hold_window = (9, 11)
        expected_kind = "pick_place"
    else:
        expected_events = TargetOnlyEvents("right", 3, 6, 9)
        expected_target_end = 19 if format_version == "robotwin_loop_context_v3" else 9
        expected_windows = {
            "operation": [3, 19],
            "target_0": [3, expected_target_end],
            "receiver_0": None,
            "gripper": [3, 19],
        }
        expected_event_keys = TARGET_ONLY_EVENT_KEYS
        expected_hold_window = (10, 19)
        expected_kind = "close_hold"

    assert context.events == expected_events
    assert set(context.events.to_json()) == expected_event_keys
    assert context.windows.to_json() == expected_windows
    assert set(context.windows.to_json()) == WINDOW_KEYS
    assert context.annotation_mode is AnnotationMode(mode)
    assert context.timeline_kind == expected_kind
    assert context.target_hold_window == expected_hold_window


def test_v1_rejects_target_only_timeline(tmp_path: Path) -> None:
    path = _write_payload(tmp_path, _payload("robotwin_loop_context_v1", "target_only"))

    with pytest.raises(UrdfGripperDataError, match="requires robotwin_loop_context_v2"):
        load_authoritative_loop_context(
            path,
            expected_task=TASK,
            expected_episode_index=EPISODE_INDEX,
            expected_camera=CAMERA,
        )


@pytest.mark.parametrize(
    ("format_version", "mode", "event_key"),
    UNEXPECTED_EVENT_CASES,
    ids=lambda case: "-".join(case) if isinstance(case, tuple) else str(case),
)
def test_loader_rejects_unexpected_event_keys(
    tmp_path: Path,
    format_version: str,
    mode: str,
    event_key: str,
) -> None:
    payload = _payload(format_version, mode)
    payload["events"]["unexpected"] = 0

    with pytest.raises(UrdfGripperDataError, match="source events must contain exactly"):
        load_authoritative_loop_context(
            _write_payload(tmp_path, payload),
            expected_task=TASK,
            expected_episode_index=EPISODE_INDEX,
            expected_camera=CAMERA,
        )


@pytest.mark.parametrize(
    ("format_version", "mode"),
    tuple(case for case in VALID_CASES if case[0] != "robotwin_loop_context_v1"),
    ids=lambda case: "-".join(case) if isinstance(case, tuple) else str(case),
)
def test_v2_and_v3_reject_unexpected_window_keys(
    tmp_path: Path,
    format_version: str,
    mode: str,
) -> None:
    payload = _payload(format_version, mode)
    payload["windows"]["unexpected"] = [0, 0]

    with pytest.raises(UrdfGripperDataError, match="windows must contain exactly"):
        load_authoritative_loop_context(
            _write_payload(tmp_path, payload),
            expected_task=TASK,
            expected_episode_index=EPISODE_INDEX,
            expected_camera=CAMERA,
        )


@pytest.mark.parametrize(
    ("format_version", "mode", "event_key"),
    BOOLEAN_EVENT_CASES,
    ids=lambda case: "-".join(case) if isinstance(case, tuple) else str(case),
)
def test_loader_rejects_boolean_event_frames(
    tmp_path: Path,
    format_version: str,
    mode: str,
    event_key: str,
) -> None:
    payload = _payload(format_version, mode)
    payload["events"][event_key] = True

    with pytest.raises(UrdfGripperDataError, match="must be an integer"):
        load_authoritative_loop_context(
            _write_payload(tmp_path, payload),
            expected_task=TASK,
            expected_episode_index=EPISODE_INDEX,
            expected_camera=CAMERA,
        )


@pytest.mark.parametrize(
    ("format_version", "mode"),
    VALID_CASES,
    ids=lambda case: "-".join(case) if isinstance(case, tuple) else str(case),
)
def test_loader_rejects_boolean_frame_count(
    tmp_path: Path,
    format_version: str,
    mode: str,
) -> None:
    payload = _payload(format_version, mode)
    payload["frame_count"] = True

    with pytest.raises(UrdfGripperDataError, match="frame_count must be an integer"):
        load_authoritative_loop_context(
            _write_payload(tmp_path, payload),
            expected_task=TASK,
            expected_episode_index=EPISODE_INDEX,
            expected_camera=CAMERA,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("task", "other_task"),
        ("episode_index", EPISODE_INDEX + 1),
        ("episode_id", "007153"),
        ("camera", "cam_low"),
    ),
)
@pytest.mark.parametrize(
    ("format_version", "mode"),
    VALID_CASES,
    ids=lambda case: "-".join(case) if isinstance(case, tuple) else str(case),
)
def test_loader_rejects_episode_identity_mismatch(
    tmp_path: Path,
    field: str,
    value: Any,
    format_version: str,
    mode: str,
) -> None:
    payload = _payload(format_version, mode)
    payload["episode"][field] = value

    with pytest.raises(UrdfGripperDataError, match="episode .* mismatch"):
        load_authoritative_loop_context(
            _write_payload(tmp_path, payload),
            expected_task=TASK,
            expected_episode_index=EPISODE_INDEX,
            expected_camera=CAMERA,
        )
