#!/usr/bin/env python3
"""Compatibility launcher for the single-episode annotation pipeline."""

from robotwin_annotation_v2.application import episode_pipeline as _pipeline

__all__ = list(_pipeline.__all__)


def __getattr__(name: str) -> object:
    if name in __all__:
        return getattr(_pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    _pipeline.main()
