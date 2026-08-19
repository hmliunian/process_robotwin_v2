"""Pickle-safe messages exchanged by the streaming dataset workers.

The worker protocol intentionally remains tuple-shaped.  ``NamedTuple`` keeps
the old wire representation (and equality semantics) while giving the
coordinator a small, typed vocabulary for each message variant.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, NamedTuple, cast

type MessageKind = Literal["event", "error", "source_episode", "result", "urdf_episode"]


class EventMessage(NamedTuple):
    """A child-process UI event: ``("event", method, args, kwargs)``."""

    kind: Literal["event"]
    method: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


class ErrorMessage(NamedTuple):
    """A serialized child exception: ``("error", type, text, traceback)``."""

    kind: Literal["error"]
    error_type: str
    error: str
    traceback: str


class SourceEpisodeMessage(NamedTuple):
    """A source episode terminal notification."""

    kind: Literal["source_episode"]
    episode_id: int
    status: str


class SourceResultMessage(NamedTuple):
    """The source worker's terminal summary."""

    kind: Literal["result"]
    summary: Mapping[str, Any]


class UrdfEpisodeMessage(NamedTuple):
    """An incremental URDF episode record."""

    kind: Literal["urdf_episode"]
    episode_id: int
    record: Mapping[str, Any]


class UrdfResultMessage(NamedTuple):
    """The incremental URDF worker's terminal result and optional error."""

    kind: Literal["result"]
    result: Mapping[str, Any] | None
    error: str | None


class ReadyEpisode(NamedTuple):
    """An episode queued from the source worker to the URDF worker."""

    episode_id: int
    position: int


type StreamingMessage = (
    EventMessage
    | ErrorMessage
    | SourceEpisodeMessage
    | SourceResultMessage
    | UrdfEpisodeMessage
    | UrdfResultMessage
)


def event(method: str, *args: Any, **kwargs: Any) -> EventMessage:
    """Build an event message with the historical tuple shape."""

    return EventMessage("event", method, args, kwargs)


def error(
    error_type: str,
    error_text: str,
    traceback_text: str,
) -> ErrorMessage:
    """Build a serialized exception message."""

    return ErrorMessage("error", error_type, error_text, traceback_text)


def source_episode(episode_id: int, status: str) -> SourceEpisodeMessage:
    """Build a source episode terminal message."""

    return SourceEpisodeMessage("source_episode", episode_id, status)


def source_result(summary: Mapping[str, Any]) -> SourceResultMessage:
    """Build a source worker result message."""

    return SourceResultMessage("result", summary)


def urdf_episode(episode_id: int, record: Mapping[str, Any]) -> UrdfEpisodeMessage:
    """Build an incremental URDF episode message."""

    return UrdfEpisodeMessage("urdf_episode", episode_id, record)


def urdf_result(
    result: Mapping[str, Any] | None,
    error_text: str | None,
) -> UrdfResultMessage:
    """Build an incremental URDF worker result message."""

    return UrdfResultMessage("result", result, error_text)


def decode_ready_episode(value: object) -> ReadyEpisode:
    """Decode a queue item while accepting old plain tuples."""

    if isinstance(value, ReadyEpisode):
        return value
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError("invalid ready episode")
    episode_id, position = value
    return ReadyEpisode(cast(int, episode_id), cast(int, position))


def decode_message(value: object) -> StreamingMessage:
    """Decode one wire tuple into its tagged ``NamedTuple`` variant.

    The two ``"result"`` variants are distinguished by their historical
    arity: source results contain a summary only, while URDF results contain a
    result and an error string.
    """

    if isinstance(
        value,
        (
            EventMessage,
            ErrorMessage,
            SourceEpisodeMessage,
            SourceResultMessage,
            UrdfEpisodeMessage,
            UrdfResultMessage,
        ),
    ):
        return value
    if not isinstance(value, tuple) or not value:
        raise ValueError("invalid streaming message")
    kind = value[0]
    if kind == "event" and len(value) == 4:
        return EventMessage(
            "event",
            cast(str, value[1]),
            cast(tuple[Any, ...], value[2]),
            cast(dict[str, Any], value[3]),
        )
    if kind == "error" and len(value) == 4:
        return ErrorMessage(
            "error",
            cast(str, value[1]),
            cast(str, value[2]),
            cast(str, value[3]),
        )
    if kind == "source_episode" and len(value) == 3:
        return SourceEpisodeMessage(
            "source_episode",
            cast(int, value[1]),
            cast(str, value[2]),
        )
    if kind == "result" and len(value) == 2:
        return SourceResultMessage("result", cast(Mapping[str, Any], value[1]))
    if kind == "urdf_episode" and len(value) == 3:
        return UrdfEpisodeMessage(
            "urdf_episode",
            cast(int, value[1]),
            cast(Mapping[str, Any], value[2]),
        )
    if kind == "result" and len(value) == 3:
        return UrdfResultMessage(
            "result",
            cast(Mapping[str, Any] | None, value[1]),
            cast(str | None, value[2]),
        )
    raise ValueError("invalid streaming message")


def try_decode_message(value: object) -> StreamingMessage | None:
    """Return ``None`` for malformed legacy messages instead of raising."""

    try:
        return decode_message(value)
    except ValueError:
        return None


# Short aliases make the protocol vocabulary convenient for callers that do
# not need to distinguish messages from their surrounding transport.
Event = EventMessage
Error = ErrorMessage
SourceEpisode = SourceEpisodeMessage
SourceResult = SourceResultMessage
UrdfEpisode = UrdfEpisodeMessage
UrdfResult = UrdfResultMessage


__all__ = [
    "Error",
    "ErrorMessage",
    "Event",
    "EventMessage",
    "MessageKind",
    "ReadyEpisode",
    "SourceEpisode",
    "SourceEpisodeMessage",
    "SourceResult",
    "SourceResultMessage",
    "StreamingMessage",
    "UrdfEpisode",
    "UrdfEpisodeMessage",
    "UrdfResult",
    "UrdfResultMessage",
    "decode_message",
    "decode_ready_episode",
    "error",
    "event",
    "source_episode",
    "source_result",
    "try_decode_message",
    "urdf_episode",
    "urdf_result",
]
