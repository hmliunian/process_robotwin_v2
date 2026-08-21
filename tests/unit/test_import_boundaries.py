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


def test_dataset_pipeline_does_not_load_legacy_dataset_runtime() -> None:
    _run_import_probe(
        "import sys\n"
        "from robotwin_annotation_v2.application import DatasetPipeline\n"
        "from robotwin_annotation_v2.application.sam_workflow import "
        "default_sam_workflow_hooks\n"
        "assert DatasetPipeline.__module__ == "
        "'robotwin_annotation_v2.application.dataset_pipeline'\n"
        "assert default_sam_workflow_hooks() is not None\n"
        "assert 'robotwin_annotation_v2.application.dataset_runtime' "
        "not in sys.modules\n"
    )


def test_adapters_package_does_not_eagerly_load_optional_backends() -> None:
    _run_import_probe(
        "import sys\n"
        "import robotwin_annotation_v2.adapters\n"
        "assert not any(name in sys.modules for name in ('cv2', 'torch', 'sam3', 'av'))\n"
    )


def test_adapters_canonical_exports_match_their_owner_modules() -> None:
    _run_import_probe(
        "import robotwin_annotation_v2.adapters as adapters\n"
        "from robotwin_annotation_v2.adapters import canonical_masks\n"
        "from robotwin_annotation_v2.adapters import canonical_publication\n"
        "assert adapters.__all__ == list(adapters._EXPORTS)\n"
        "assert adapters._EXPORT_GROUPS['.canonical_masks'] == "
        "tuple(canonical_masks.__all__)\n"
        "assert adapters._EXPORT_GROUPS['.canonical_publication'] == "
        "tuple(canonical_publication.__all__)\n"
        "assert not any(name in __import__('sys').modules for name in "
        "('cv2', 'torch', 'sam3', 'av'))\n"
    )


def test_loop_context_codec_does_not_load_pipeline_or_optional_backends() -> None:
    _run_import_probe(
        "import sys\n"
        "import robotwin_annotation_v2.adapters.loop_context_codec as codec\n"
        "assert callable(codec.load_authoritative_loop_context)\n"
        "assert 'robotwin_annotation_v2.urdf_gripper_data' not in sys.modules\n"
        "assert not any(name == 'robotwin_annotation_v2.pipeline' or "
        "name.startswith('robotwin_annotation_v2.pipeline.') for name in sys.modules)\n"
        "assert not any(name in sys.modules for name in (\n"
        "    'av', 'pandas', 'cv2', 'torch', 'sam3'\n"
        "))\n"
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


def test_public_timeline_detector_does_not_load_stage_or_dataset_runtime() -> None:
    _run_import_probe(
        "import sys\n"
        "import robotwin_annotation_v2.pipeline as pipeline\n"
        "detector = pipeline.detect_loop_events\n"
        "assert detector.__module__ == 'robotwin_annotation_v2.pipeline.timeline_detector'\n"
        "assert 'robotwin_annotation_v2.pipeline.state_loop' not in sys.modules\n"
        "assert 'robotwin_annotation_v2.adapters.robotwin_dataset' not in sys.modules\n"
        "assert not any(name in sys.modules for name in (\n"
        "    'av', 'pandas', 'cv2', 'torch', 'sam3'\n"
        "))\n"
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
        "assert 'robotwin_annotation_v2.adapters.canonical_publication' in sys.modules\n"
        "assert 'robotwin_annotation_v2.urdf_gripper_publisher' not in sys.modules\n"
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


def test_urdf_finger_fit_import_does_not_load_render_backends() -> None:
    _run_import_probe(
        "import sys\n"
        "import robotwin_annotation_v2.adapters.urdf.finger_fit\n"
        "assert not any(name in sys.modules for name in "
        "('pyrender', 'trimesh', 'OpenGL', 'cv2', 'torch', 'sam3'))\n"
    )
