from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _run_import_probe(source: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-c", source],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_application_package_does_not_eagerly_load_runtime() -> None:
    _run_import_probe(
        "import sys\n"
        "import robotwin_annotation_v2.application\n"
        "assert 'robotwin_annotation_v2.application.dataset_runtime' not in sys.modules\n"
        "assert 'robotwin_annotation_v2.application.episode_pipeline' not in sys.modules\n"
    )


def test_pipeline_package_loads_lightweight_stage_exports_on_demand() -> None:
    _run_import_probe(
        "import sys\n"
        "import robotwin_annotation_v2.pipeline as pipeline\n"
        "assert 'cv2' not in sys.modules\n"
        "assert 'robotwin_annotation_v2.pipeline.gripper_stage' not in sys.modules\n"
        "assert callable(pipeline.build_loop_context)\n"
        "assert 'robotwin_annotation_v2.pipeline.state_loop' in sys.modules\n"
        "assert 'robotwin_annotation_v2.pipeline.gripper_stage' not in sys.modules\n"
        "assert 'cv2' not in sys.modules\n"
    )
