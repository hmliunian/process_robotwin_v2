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


def test_adapters_package_does_not_eagerly_load_optional_backends() -> None:
    _run_import_probe(
        "import sys\n"
        "import robotwin_annotation_v2.adapters\n"
        "assert not any(name in sys.modules for name in ('cv2', 'torch', 'sam3', 'av'))\n"
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


def test_mask_qc_stage_does_not_load_artifact_store() -> None:
    _run_import_probe(
        "import sys\n"
        "import robotwin_annotation_v2.pipeline.mask_qc\n"
        "assert 'robotwin_annotation_v2.adapters.artifact_store' not in sys.modules\n"
    )


def test_sam_stage_does_not_load_artifact_or_gripper_modules() -> None:
    _run_import_probe(
        "import sys\n"
        "import robotwin_annotation_v2.pipeline.sam_stage\n"
        "assert 'robotwin_annotation_v2.adapters.artifact_store' not in sys.modules\n"
        "assert 'robotwin_annotation_v2.pipeline.gripper_stage' not in sys.modules\n"
        "assert 'cv2' not in sys.modules\n"
    )


def test_gripper_composition_does_not_load_opencv() -> None:
    _run_import_probe(
        "import sys\n"
        "import robotwin_annotation_v2.pipeline.gripper.sam.composition\n"
        "assert 'robotwin_annotation_v2.pipeline.gripper_stage' not in sys.modules\n"
        "assert 'cv2' not in sys.modules\n"
    )


def test_gripper_qc_does_not_load_model_backends() -> None:
    _run_import_probe(
        "import sys\n"
        "import robotwin_annotation_v2.pipeline.gripper.sam.qc\n"
        "assert not any(name in sys.modules for name in ('torch', 'sam3', 'av'))\n"
    )


def test_gripper_annotator_does_not_load_model_backends() -> None:
    _run_import_probe(
        "import sys\n"
        "import robotwin_annotation_v2.pipeline.gripper.sam.annotator\n"
        "assert not any(name in sys.modules for name in ('torch', 'sam3', 'av'))\n"
    )


def test_sam_artifacts_does_not_load_gripper_annotator() -> None:
    _run_import_probe(
        "import sys\n"
        "import robotwin_annotation_v2.application.sam_artifacts\n"
        "assert 'robotwin_annotation_v2.pipeline.gripper.sam.annotator' not in sys.modules\n"
        "assert 'robotwin_annotation_v2.pipeline.gripper_stage' not in sys.modules\n"
        "assert 'cv2' not in sys.modules\n"
    )


def test_urdf_batch_import_does_not_load_optional_backends() -> None:
    _run_import_probe(
        "import sys\n"
        "import robotwin_annotation_v2.application.urdf_batch\n"
        "assert not any(name in sys.modules for name in ('cv2', 'torch', 'sam3', 'av'))\n"
    )
