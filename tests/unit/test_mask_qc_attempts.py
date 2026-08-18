from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from robotwin_annotation_v2.adapters import ArtifactStore
from robotwin_annotation_v2.domain import AnnotationMode
from robotwin_annotation_v2.models import (
    EpisodeRef,
    FramePurpose,
    LoopContext,
    MaskCandidateInfo,
    MaskQCAttempt,
    MaskQCResult,
    MaskQCStatus,
    RoleMaskQC,
    SemanticFrame,
    TargetOnlyEvents,
)
from robotwin_annotation_v2.pipeline import save_mask_qc_artifacts

FRAME_SHAPE = (8, 10)


def _context() -> LoopContext:
    return LoopContext(
        episode=EpisodeRef("adjust_bottle", 1, "cam_high"),
        task_text="Adjust the bottle and hold it steady",
        frame_count=20,
        events=TargetOnlyEvents("right", 2, 6, 8),
        semantic_frames=(
            SemanticFrame(
                0,
                FramePurpose.PRE_GRASP_SEED_CANDIDATE,
                ("target",),
            ),
            SemanticFrame(
                3,
                FramePurpose.PRE_GRASP_SEED_CANDIDATE,
                ("target",),
            ),
            SemanticFrame(10, FramePurpose.POST_GRASP_CONTEXT, ("target",)),
        ),
        state_source="state.parquet",
        video_source="video.mp4",
        annotation_mode=AnnotationMode.TARGET_ONLY,
    )


def _candidate(seed_frame_id: int) -> MaskCandidateInfo:
    return MaskCandidateInfo(
        candidate_id="A",
        query_field="category_query",
        query="bottle",
        nonempty=True,
        area_fraction=0.05,
        component_count=1,
        basic_valid=True,
        seed_frame_id=seed_frame_id,
    )


def _result() -> MaskQCResult:
    first_candidate = _candidate(0)
    selected_candidate = _candidate(3)
    first_attempt = MaskQCAttempt(
        seed_frame_id=0,
        status=MaskQCStatus.REJECTED,
        selected_candidate=None,
        selected_query_field=None,
        selected_query=None,
        confidence=0.91,
        reason="candidate belongs to the wrong instance",
        candidates=(first_candidate,),
        model="fake-qwen",
        raw_response='{"decision":"reject_all"}',
        rendered_prompt="seed=0",
    )
    selected_attempt = MaskQCAttempt(
        seed_frame_id=3,
        status=MaskQCStatus.PASSED,
        selected_candidate="A",
        selected_query_field="category_query",
        selected_query="bottle",
        confidence=0.96,
        reason="candidate covers the complete target",
        candidates=(selected_candidate,),
        model="fake-qwen",
        raw_response='{"decision":"accept","selected_candidate":"A"}',
        rendered_prompt="seed=3",
    )
    report = RoleMaskQC(
        role="target",
        status=MaskQCStatus.PASSED,
        selected_candidate="A",
        selected_query_field="category_query",
        selected_query="bottle",
        selected_seed_frame_id=3,
        confidence=0.96,
        reason="candidate covers the complete target",
        candidates=(selected_candidate,),
        model="fake-qwen",
        raw_response=selected_attempt.raw_response,
        rendered_prompt=selected_attempt.rendered_prompt,
        attempts=(first_attempt, selected_attempt),
    )

    first_mask = np.zeros(FRAME_SHAPE, dtype=bool)
    first_mask[1:3, 1:3] = True
    selected_mask = np.zeros(FRAME_SHAPE, dtype=bool)
    selected_mask[4:7, 6:9] = True
    first_panel = Image.fromarray(np.full((*FRAME_SHAPE, 3), 40, dtype=np.uint8))
    selected_panel = Image.fromarray(np.full((*FRAME_SHAPE, 3), 180, dtype=np.uint8))
    return MaskQCResult(
        role_reports=(report,),
        selected_masks={"target": selected_mask},
        health={"status": "ok"},
        candidate_masks={"target": {"A": selected_mask}},
        candidate_panels={"target": {"A": selected_panel}},
        attempt_candidate_masks={
            "target": {
                0: {"A": first_mask},
                3: {"A": selected_mask},
            }
        },
        attempt_candidate_panels={
            "target": {
                0: {"A": first_panel},
                3: {"A": selected_panel},
            }
        },
    )


def test_mask_qc_attempts_validate_seed_and_selected_candidate_consistency() -> None:
    candidate = _candidate(3)

    with pytest.raises(ValueError, match="belong to its seed frame"):
        MaskQCAttempt(
            seed_frame_id=0,
            status=MaskQCStatus.REJECTED,
            selected_candidate=None,
            selected_query_field=None,
            selected_query=None,
            confidence=None,
            reason="wrong seed metadata",
            candidates=(candidate,),
        )

    payload = _result().to_json()
    attempts = payload["roles"]["target"]["attempts"]
    assert [attempt["seed_frame_id"] for attempt in attempts] == [0, 3]
    assert [attempt["status"] for attempt in attempts] == ["rejected", "passed"]


def test_save_mask_qc_artifacts_keeps_flat_final_and_nested_seed_attempts(
    tmp_path: Path,
) -> None:
    result = _result()

    path = save_mask_qc_artifacts(
        ArtifactStore(tmp_path),
        "attempt-audit",
        _context(),
        result,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    artifacts = payload["artifacts"]
    assert artifacts["candidate_masks"]["target"]["A"] == (
        "target/qc_candidates/candidate_A.mask.png"
    )
    attempts = artifacts["attempts"]["target"]
    assert set(attempts) == {"frame_000000", "frame_000003"}
    first_path = path.parent / attempts["frame_000000"]["candidate_masks"]["A"]
    selected_path = path.parent / attempts["frame_000003"]["candidate_masks"]["A"]
    flat_path = path.parent / artifacts["candidate_masks"]["target"]["A"]
    assert first_path.is_file()
    assert selected_path.is_file()
    assert flat_path.is_file()
    assert first_path != selected_path
    assert np.array_equal(np.asarray(Image.open(selected_path)), np.asarray(Image.open(flat_path)))
    assert not np.array_equal(np.asarray(Image.open(first_path)), np.asarray(Image.open(flat_path)))
