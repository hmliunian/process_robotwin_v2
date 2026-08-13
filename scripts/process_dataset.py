#!/usr/bin/env python3
"""Launch the readable RoboTwin dataset annotation pipeline.

Imports retain the historical ``scripts.process_dataset`` module identity so
downstream launchers and tests can still patch the runtime seam.  Production
code lives in :mod:`robotwin_annotation_v2.application.dataset_runtime`.
"""

from __future__ import annotations

import sys

from robotwin_annotation_v2.application import dataset_runtime as _runtime

if __name__ != "__main__":
    sys.modules[__name__] = _runtime
else:
    _runtime.main()
