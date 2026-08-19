#!/usr/bin/env python3
"""Compatibility entry point for the package-owned URDF batch engine."""

from __future__ import annotations

import sys

from robotwin_annotation_v2.application import urdf_batch as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
