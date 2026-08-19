"""Application-level orchestration with lazy compatibility exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "DatasetPipeline": (".dataset_pipeline", "DatasetPipeline"),
    "EpisodePipeline": (".episode_pipeline_api", "EpisodePipeline"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load public coordinators without importing their runtimes eagerly."""

    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
