#!/usr/bin/env python3
"""Launch the readable RoboTwin dataset annotation pipeline.

Production code lives in
:mod:`robotwin_annotation_v2.application.dataset_runtime`.
"""

from __future__ import annotations

from robotwin_annotation_v2.application import dataset_runtime as _runtime


def __getattr__(name: str) -> object:
    return getattr(_runtime, name)


if __name__ == "__main__":
    _runtime.main()
