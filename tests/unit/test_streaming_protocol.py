from __future__ import annotations

import pickle
from typing import Any

import pytest

from robotwin_annotation_v2.application.streaming import (
    ErrorMessage,
    EventMessage,
    ReadyEpisode,
    SourceEpisodeMessage,
    SourceResultMessage,
    UrdfEpisodeMessage,
    UrdfResultMessage,
    decode_message,
    decode_ready_episode,
    error,
    event,
    source_episode,
    source_result,
    try_decode_message,
    urdf_episode,
    urdf_result,
)


def test_factories_preserve_existing_wire_tuple_shapes() -> None:
    summary: dict[str, Any] = {"passed": True}
    record: dict[str, Any] = {"status": "complete"}
    messages = (
        (event("note", "hello", level="info"), ("event", "note", ("hello",), {"level": "info"})),
        (error("RuntimeError", "boom", "trace"), ("error", "RuntimeError", "boom", "trace")),
        (source_episode(7, "completed"), ("source_episode", 7, "completed")),
        (source_result(summary), ("result", summary)),
        (urdf_episode(7, record), ("urdf_episode", 7, record)),
        (urdf_result(summary, None), ("result", summary, None)),
    )

    for message, expected in messages:
        assert message == expected
        assert isinstance(message, tuple)

    ready = ReadyEpisode(7, 3)
    assert ready == (7, 3)
    assert decode_ready_episode((7, 3)) == ready


def test_decode_plain_tuples_returns_tagged_variants() -> None:
    values = (
        (("event", "detail", ("text",), {}), EventMessage),
        (("error", "ValueError", "bad", "trace"), ErrorMessage),
        (("source_episode", 7, "completed"), SourceEpisodeMessage),
        (("result", {"run_id": "source"}), SourceResultMessage),
        (("urdf_episode", 7, {"status": "complete"}), UrdfEpisodeMessage),
        (("result", {"status": "complete"}, None), UrdfResultMessage),
    )

    for value, message_type in values:
        decoded = decode_message(value)
        assert isinstance(decoded, message_type)
        assert decoded == value

    assert try_decode_message(("not-a-message",)) is None


@pytest.mark.parametrize(
    "value",
    (
        None,
        (),
        ("event", "missing-args"),
        ("result", {}, "too", "many"),
        ("unknown", 1, 2),
    ),
)
def test_decode_rejects_malformed_messages(value: object) -> None:
    with pytest.raises(ValueError, match="invalid streaming message"):
        decode_message(value)

    assert try_decode_message(value) is None


def test_ready_episode_decoder_rejects_malformed_items() -> None:
    with pytest.raises(ValueError, match="invalid ready episode"):
        decode_ready_episode((7,))


def test_protocol_messages_are_pickle_safe() -> None:
    messages = (
        event("phase_started", "source", total=2),
        error("RuntimeError", "boom", "trace"),
        source_episode(7, "completed"),
        source_result({"passed": True}),
        urdf_episode(7, {"status": "complete"}),
        urdf_result({"status": "complete"}, None),
        ReadyEpisode(7, 1),
    )

    for message in messages:
        assert pickle.loads(pickle.dumps(message)) == message
