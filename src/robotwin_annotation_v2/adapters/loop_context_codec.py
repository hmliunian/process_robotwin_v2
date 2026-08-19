"""Codec for authoritative Stage-1 loop context artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from ..domain import AnnotationMode, annotation_spec
from ..models.timeline import (
    EpisodeWindows,
    FrameWindow,
    PickPlaceEvents,
    TargetOnlyEvents,
    TimelineEvents,
    derive_target_hold_window,
)

ArmName = Literal["left", "right"]


class LoopContextCodecError(RuntimeError):
    """A loop context artifact does not satisfy its versioned contract."""


@dataclass(frozen=True)
class AuthoritativeLoopContext:
    """Validated Stage-1 source artifact used by a derived run."""

    path: Path
    task: str
    episode_index: int
    camera: str
    frame_count: int
    events: TimelineEvents
    annotation_mode: AnnotationMode
    timeline_kind: str
    windows: EpisodeWindows

    @property
    def active_arm(self) -> ArmName:
        return self.events.active_arm

    @property
    def gripper_window(self) -> tuple[int, int]:
        return (self.windows.gripper.start, self.windows.gripper.end)

    @property
    def target_hold_window(self) -> tuple[int, int] | None:
        """Inclusive frames carrying the held-target encoding."""

        hold = derive_target_hold_window(self.events, frame_count=self.frame_count)
        return None if hold is None else (hold.start, hold.end)


def _validate_episode_index(episode_index: int) -> int:
    if isinstance(episode_index, bool) or not isinstance(episode_index, int):
        raise TypeError("episode_index must be an integer")
    if episode_index < 0:
        raise ValueError("episode_index must be non-negative")
    return episode_index


def _validate_camera(camera: str) -> str:
    if not camera or camera in {".", ".."} or "/" in camera or "\\" in camera:
        raise ValueError(f"invalid camera name: {camera!r}")
    return camera


def _format_episode_id(episode_index: int) -> str:
    """Format an episode index using the dataset's six-digit convention."""

    return f"{_validate_episode_index(episode_index):06d}"


def _required_mapping(
    payload: Mapping[str, Any],
    key: str,
    *,
    description: str,
) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise LoopContextCodecError(f"{description} {key} must be an object")
    return value


def _required_integer(
    payload: Mapping[str, Any],
    key: str,
    *,
    description: str,
) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise LoopContextCodecError(f"{description} {key} must be an integer")
    return value


def _validate_recorded_window(
    windows: Mapping[str, Any],
    key: str,
    expected: tuple[int, int],
) -> None:
    raw = windows.get(key)
    if (
        not isinstance(raw, list)
        or len(raw) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in raw)
        or tuple(raw) != expected
    ):
        raise LoopContextCodecError(
            f"source loop window {key} does not match events: {raw!r} != {list(expected)!r}"
        )


def _frame_window(value: tuple[int, int]) -> FrameWindow:
    return FrameWindow(start=value[0], end=value[1])


def _pick_place_windows(
    events: PickPlaceEvents,
    *,
    include_held_target: bool,
) -> EpisodeWindows:
    operation = _frame_window(events.inclusive_window)
    target_end = events.t_open_start - 1 if include_held_target else events.t_close_done
    return EpisodeWindows(
        operation=operation,
        target=_frame_window((events.t_move_start, target_end)),
        receiver=_frame_window((events.t_close_done, events.t_open_done)),
        gripper=operation,
    )


def _target_only_windows(
    events: TargetOnlyEvents,
    *,
    frame_count: int,
    include_held_target: bool,
) -> EpisodeWindows:
    operation = _frame_window((events.t_remove_start, frame_count - 1))
    return EpisodeWindows(
        operation=operation,
        target=(
            operation
            if include_held_target
            else _frame_window((events.t_remove_start, events.t_close_end))
        ),
        receiver=None,
        gripper=operation,
    )


def _validate_versioned_windows(
    windows: Mapping[str, Any],
    expected: EpisodeWindows,
    *,
    format_version: str,
) -> None:
    expected_keys = {"operation", "target_0", "receiver_0", "gripper"}
    if set(windows) != expected_keys:
        raise LoopContextCodecError(
            f"source loop {format_version} windows must contain exactly "
            f"{sorted(expected_keys)}, got {sorted(str(key) for key in windows)}"
        )
    _validate_recorded_window(
        windows,
        "operation",
        (expected.operation.start, expected.operation.end),
    )
    _validate_recorded_window(
        windows,
        "target_0",
        (expected.target.start, expected.target.end),
    )
    if expected.receiver is None:
        if windows.get("receiver_0") is not None:
            raise LoopContextCodecError(
                "target_only source loop receiver_0 window must be null"
            )
    else:
        _validate_recorded_window(
            windows,
            "receiver_0",
            (expected.receiver.start, expected.receiver.end),
        )
    _validate_recorded_window(
        windows,
        "gripper",
        (expected.gripper.start, expected.gripper.end),
    )


def load_authoritative_loop_context(
    path: Path,
    *,
    expected_task: str,
    expected_episode_index: int,
    expected_camera: str,
) -> AuthoritativeLoopContext:
    """Load one frozen timeline and normalize its downstream windows.

    Legacy v1/v2 artifacts retain their historic short target window.  V3
    extends target publication through the post-close hold interval while
    preserving the same concrete event state machines.
    """

    source = path.expanduser().resolve()
    if not source.is_file():
        raise LoopContextCodecError(f"source loop artifact is missing: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise LoopContextCodecError(
            f"failed to read source loop artifact {source}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise LoopContextCodecError("source loop artifact must contain a JSON object")
    format_version = payload.get("format_version")
    if format_version not in {
        "robotwin_loop_context_v1",
        "robotwin_loop_context_v2",
        "robotwin_loop_context_v3",
    }:
        raise LoopContextCodecError(
            f"unsupported source loop format: {format_version!r}"
        )

    episode = _required_mapping(payload, "episode", description="source loop")
    expected_index = _validate_episode_index(expected_episode_index)
    expected_identity = {
        "task": expected_task,
        "episode_index": expected_index,
        "episode_id": _format_episode_id(expected_index),
        "camera": _validate_camera(expected_camera),
    }
    for key, expected in expected_identity.items():
        if episode.get(key) != expected:
            raise LoopContextCodecError(
                f"source loop episode {key} mismatch: {episode.get(key)!r} != {expected!r}"
            )

    frame_count = _required_integer(
        payload,
        "frame_count",
        description="source loop",
    )
    if frame_count < 1:
        raise LoopContextCodecError("source loop frame_count must be positive")
    raw_mode = payload.get("annotation_mode", AnnotationMode.PICK_PLACE.value)
    try:
        annotation_mode = AnnotationMode(raw_mode)
    except (TypeError, ValueError) as exc:
        raise LoopContextCodecError(
            f"unsupported source loop annotation_mode: {raw_mode!r}"
        ) from exc
    spec = annotation_spec(annotation_mode)
    raw_roles = payload.get("required_object_roles")
    if raw_roles is not None and raw_roles != list(spec.required_role_names):
        raise LoopContextCodecError(
            "source loop required_object_roles differ from annotation_mode"
        )
    if format_version in {
        "robotwin_loop_context_v2",
        "robotwin_loop_context_v3",
    } and raw_roles is None:
        raise LoopContextCodecError(
            f"source loop {format_version} must declare required_object_roles"
        )

    event_payload = _required_mapping(payload, "events", description="source loop")
    active_arm = event_payload.get("active_arm")
    if active_arm not in {"left", "right"}:
        raise LoopContextCodecError(
            f"source loop active_arm must be left or right, got {active_arm!r}"
        )
    windows = _required_mapping(payload, "windows", description="source loop")
    events: TimelineEvents
    if format_version == "robotwin_loop_context_v1":
        if annotation_mode is not AnnotationMode.PICK_PLACE:
            raise LoopContextCodecError(
                "target_only close-and-hold requires robotwin_loop_context_v2"
            )
        event_keys = {
            "active_arm",
            "t_move_start",
            "t_close_start",
            "t_close_done",
            "t_open_start",
            "t_open_done",
        }
        if set(event_payload) != event_keys:
            raise LoopContextCodecError(
                "pick_place source events must contain exactly "
                f"{sorted(event_keys)}"
            )
        event_values = {
            key: _required_integer(
                event_payload,
                key,
                description="source loop events",
            )
            for key in event_keys - {"active_arm"}
        }
        try:
            pick_place_events = PickPlaceEvents(
                active_arm=cast(ArmName, active_arm),
                **event_values,
            )
        except ValueError as exc:
            raise LoopContextCodecError(f"invalid source loop events: {exc}") from exc
        normalized_windows = _pick_place_windows(
            pick_place_events,
            include_held_target=False,
        )
        if pick_place_events.end >= frame_count:
            raise LoopContextCodecError(
                f"source loop active window {pick_place_events.inclusive_window} exceeds frame count "
                f"{frame_count}"
            )
        _validate_recorded_window(windows, "loop", pick_place_events.inclusive_window)
        _validate_recorded_window(
            windows,
            "target_0",
            (pick_place_events.t_move_start, pick_place_events.t_close_done),
        )
        _validate_recorded_window(
            windows,
            "receiver_0",
            (pick_place_events.t_close_done, pick_place_events.t_open_done),
        )
        raw_gripper = windows.get("gripper")
        if raw_gripper is not None:
            _validate_recorded_window(
                windows,
                "gripper",
                pick_place_events.inclusive_window,
            )
        events = pick_place_events
        timeline_kind = "pick_place"
    else:
        raw_timeline_kind = payload.get("timeline_kind")
        expected_kind = (
            "pick_place"
            if annotation_mode is AnnotationMode.PICK_PLACE
            else "close_hold"
        )
        if raw_timeline_kind != expected_kind:
            raise LoopContextCodecError(
                "source loop timeline_kind differs from annotation_mode: "
                f"{raw_timeline_kind!r} != {expected_kind!r}"
            )
        timeline_kind = expected_kind
        if annotation_mode is AnnotationMode.PICK_PLACE:
            event_keys = {
                "active_arm",
                "t_move_start",
                "t_close_start",
                "t_close_done",
                "t_open_start",
                "t_open_done",
            }
            event_values = {
                key: _required_integer(
                    event_payload,
                    key,
                    description="source loop events",
                )
                for key in event_keys - {"active_arm"}
            }
            try:
                events = PickPlaceEvents(
                    active_arm=cast(ArmName, active_arm),
                    **event_values,
                )
            except ValueError as exc:
                raise LoopContextCodecError(
                    f"invalid source loop events: {exc}"
                ) from exc
            normalized_windows = _pick_place_windows(
                events,
                include_held_target=format_version == "robotwin_loop_context_v3",
            )
        else:
            event_keys = {
                "active_arm",
                "t_remove_start",
                "t_close_start",
                "t_close_end",
            }
            event_values = {
                key: _required_integer(
                    event_payload,
                    key,
                    description="source loop events",
                )
                for key in event_keys - {"active_arm"}
            }
            try:
                events = TargetOnlyEvents(
                    active_arm=cast(ArmName, active_arm),
                    **event_values,
                )
            except ValueError as exc:
                raise LoopContextCodecError(
                    f"invalid source loop events: {exc}"
                ) from exc
            if events.t_close_end >= frame_count:
                raise LoopContextCodecError(
                    "target_only close_end exceeds the episode frame range"
                )
            normalized_windows = _target_only_windows(
                events,
                frame_count=frame_count,
                include_held_target=format_version == "robotwin_loop_context_v3",
            )
        if set(event_payload) != event_keys:
            raise LoopContextCodecError(
                f"{annotation_mode.value} source events must contain exactly "
                f"{sorted(event_keys)}"
            )
        _validate_versioned_windows(
            windows,
            normalized_windows,
            format_version=str(format_version),
        )

    return AuthoritativeLoopContext(
        path=source,
        task=expected_task,
        episode_index=expected_index,
        camera=expected_camera,
        frame_count=frame_count,
        events=events,
        annotation_mode=annotation_mode,
        timeline_kind=timeline_kind,
        windows=normalized_windows,
    )


__all__ = [
    "AuthoritativeLoopContext",
    "LoopContextCodecError",
    "load_authoritative_loop_context",
]
