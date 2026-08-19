"""Concrete adapters with lazy compatibility exports.

Importing the package itself must stay lightweight: dataset/video and model
adapters carry optional dependencies that are only needed by the workflows
using them.  Internal code imports concrete modules directly; these exports
preserve the historical ``from ...adapters import Name`` API.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_GROUPS = {
    ".artifact_store": ("ArtifactStore",),
    ".canonical_masks": (
        "CANONICAL_INSTANCE_NAMES",
        "CANONICAL_ROLES",
        "CanonicalMaskBundle",
        "CanonicalMaskError",
        "read_canonical_masks",
    ),
    ".qwen_client": (
        "OpenAICompatibleQwenClient",
        "QwenCompletion",
        "QwenServiceError",
        "image_data_url",
    ),
    ".robotwin_dataset": (
        "DatasetError",
        "EpisodePaths",
        "EpisodeState",
        "RoboTwinDataset",
    ),
    ".sam3_adapter": ("Sam3Adapter", "Sam3Error", "sam3_video_resource"),
}
_EXPORTS = {
    name: module_name
    for module_name, names in _EXPORT_GROUPS.items()
    for name in names
}

__all__ = [
    "CANONICAL_INSTANCE_NAMES",
    "CANONICAL_ROLES",
    "ArtifactStore",
    "CanonicalMaskBundle",
    "CanonicalMaskError",
    "DatasetError",
    "EpisodePaths",
    "EpisodeState",
    "OpenAICompatibleQwenClient",
    "QwenCompletion",
    "QwenServiceError",
    "RoboTwinDataset",
    "Sam3Adapter",
    "Sam3Error",
    "image_data_url",
    "read_canonical_masks",
    "sam3_video_resource",
]


def __getattr__(name: str) -> Any:
    """Load the adapter module that owns a requested compatibility export."""

    try:
        module_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
