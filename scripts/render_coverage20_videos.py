#!/usr/bin/env python3
"""Compatibility entry point for the package-owned canonical mask renderer."""

from __future__ import annotations

import sys

from robotwin_annotation_v2.adapters import rendering as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
