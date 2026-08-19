from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

import robotwin_annotation_v2.pipeline as public_pipeline
from robotwin_annotation_v2.adapters import QwenCompletion
from robotwin_annotation_v2.config import GripperRoiConfig
from robotwin_annotation_v2.domain import AnnotationMode, ObjectRole
from robotwin_annotation_v2.models import (
    EpisodeRef,
    FramePurpose,
    LoopContext,
    LoopEvents,
    SemanticFrame,
    TargetOnlyEvents,
)
from robotwin_annotation_v2.pipeline import (
    GripperSeedQualityGateConfig,
    GripperStageError,
    run_gripper_stage,
)
from robotwin_annotation_v2.pipeline import gripper_stage as legacy_stage
from robotwin_annotation_v2.pipeline.gripper.sam import annotator

FRAME_SHAPE = (240, 320)
FRAME_COUNT = 24
POSE = np.asarray(
    [
        -0.0956539511680603,
        0.0383211150765419,
        0.9840234518051147,
        -0.002832249039784074,
        1.265782356262207,
        0.807780385017395,
    ],
    dtype=np.float64,
)


class FakeGripperBackend:
    def __init__(self, *, valid_text_masks: bool = True) -> None:
        self.valid_text_masks = valid_text_masks
        self.text_box_calls: list[int] = []
        self.box_calls: list[int] = []
        self.propagate_calls: list[int] = []
        self.call_order: list[tuple[str, int]] = []

    @staticmethod
    def _box_mask(
        box_xyxy: Sequence[float],
        frame_shape: tuple[int, int],
    ) -> np.ndarray:
        height, width = frame_shape
        x0, y0, x1, y1 = box_xyxy
        left = max(0, min(width, int(np.floor(x0 * width))))
        top = max(0, min(height, int(np.floor(y0 * height))))
        right = max(0, min(width, int(np.ceil(x1 * width))))
        bottom = max(0, min(height, int(np.ceil(y1 * height))))
        mask = np.zeros(frame_shape, dtype=bool)
        mask[top:bottom, left:right] = True
        return mask

    def text_box_mask(
        self,
        _resource_path: Path,
        text: str,
        box_xyxy: Sequence[float],
        *,
        frame_id: int,
        frame_shape: tuple[int, int],
        **_kwargs: Any,
    ) -> np.ndarray:
        assert text == "black robot gripper"
        self.text_box_calls.append(frame_id)
        self.call_order.append(("text_box", frame_id))
        if not self.valid_text_masks:
            return np.zeros(frame_shape, dtype=bool)
        return self._box_mask(box_xyxy, frame_shape)

    def box_mask(
        self,
        _resource_path: Path,
        box_xyxy: Sequence[float],
        *,
        frame_id: int,
        frame_shape: tuple[int, int],
        **_kwargs: Any,
    ) -> np.ndarray:
        self.box_calls.append(frame_id)
        self.call_order.append(("box_only", frame_id))
        return self._box_mask(box_xyxy, frame_shape)

    def propagate_mask(
        self,
        _resource_path: Path,
        seed_mask: np.ndarray,
        *,
        seed_frame: int,
        frame_count: int,
        tracking_window: tuple[int, int],
        **_kwargs: Any,
    ) -> np.ndarray:
        self.propagate_calls.append(seed_frame)
        output = np.zeros((frame_count, *seed_mask.shape), dtype=bool)
        output[tracking_window[0] : tracking_window[1] + 1] = seed_mask
        return output


class FakeQwenClient:
    model_id = "fake-qwen"

    def __init__(self, selected_candidate: str = "A") -> None:
        self.selected_candidate = selected_candidate

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "model": self.model_id}

    def complete(
        self,
        _messages: list[dict[str, Any]],
        *,
        max_tokens: int,
    ) -> QwenCompletion:
        assert max_tokens == 100
        return QwenCompletion(
            content=json.dumps(
                {
                    "decision": "accept",
                    "selected_candidate": self.selected_candidate,
                    "confidence": 0.95,
                    "reason": "Candidate A cleanly covers the visible gripper.",
                }
            ),
            model=self.model_id,
        )


def _write_state(path: Path) -> None:
    state = np.zeros((FRAME_COUNT, 14), dtype=np.float64)
    state[:, 0:6] = POSE
    state[:, 6] = 1.0
    state[:, 7:13] = POSE
    state[:, 13] = 1.0
    pd.DataFrame(
        {
            "frame_index": np.arange(FRAME_COUNT),
            "episode_index": np.full(FRAME_COUNT, 7),
            "observation.state": list(state),
        }
    ).to_parquet(path, index=False)


def _write_resource(path: Path) -> None:
    path.mkdir()
    image = Image.fromarray(np.zeros((*FRAME_SHAPE, 3), dtype=np.uint8), mode="RGB")
    for frame_id in range(FRAME_COUNT):
        image.save(path / f"{frame_id:06d}.jpg", format="JPEG")


def _context(state_path: Path) -> LoopContext:
    return LoopContext(
        episode=EpisodeRef("task", 7),
        task_text="move the bottle",
        frame_count=FRAME_COUNT,
        events=LoopEvents("right", 1, 10, 12, 20, 22),
        semantic_frames=(
            SemanticFrame(
                0,
                FramePurpose.PRE_GRASP_SEED_CANDIDATE,
                ("target", "receiver"),
            ),
            SemanticFrame(12, FramePurpose.POST_GRASP_CONTEXT, ("target",)),
            SemanticFrame(22, FramePurpose.PLACE_CONTEXT, ("receiver",)),
        ),
        state_source=str(state_path),
        video_source="video.mp4",
    )


def _target_only_context(state_path: Path) -> LoopContext:
    return LoopContext(
        episode=EpisodeRef("task", 7),
        task_text="move the bottle",
        frame_count=FRAME_COUNT,
        events=TargetOnlyEvents("right", 1, 10, 12),
        semantic_frames=(
            SemanticFrame(
                0,
                FramePurpose.PRE_GRASP_SEED_CANDIDATE,
                ("target",),
            ),
            SemanticFrame(12, FramePurpose.POST_GRASP_CONTEXT, ("target",)),
        ),
        state_source=str(state_path),
        video_source="video.mp4",
        annotation_mode=AnnotationMode.TARGET_ONLY,
    )


def test_run_gripper_stage_reuses_object_tracks_and_propagates_only_gripper(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "state.parquet"
    resource_path = tmp_path / "frames"
    prompt_path = tmp_path / "prompt.txt"
    _write_state(state_path)
    _write_resource(resource_path)
    prompt_path.write_text(
        "task={task_text}; arm={active_arm}; ids={candidate_ids}\n"
        "{candidate_records}\n{candidate_panels}\n{context_frames}\n"
        "events={move_start},{close_start},{close_done},{open_start},{open_done}",
        encoding="utf-8",
    )
    backend = FakeGripperBackend()
    target = np.zeros((FRAME_COUNT, *FRAME_SHAPE), dtype=bool)
    receiver = np.zeros_like(target)

    result = run_gripper_stage(
        _context(state_path),
        backend=backend,
        resource_path=resource_path,
        frame_shape=FRAME_SHAPE,
        gripper_roi_config=GripperRoiConfig(
            prompt_axial_back_m=0.120,
            prompt_axial_front_m=0.060,
            hard_axial_back_m=0.120,
            hard_axial_front_m=0.045,
            fixed_half_width_m=0.085,
        ),
        object_tracks={
            ObjectRole.TARGET: target,
            ObjectRole.RECEIVER: receiver,
        },
        qc_client=FakeQwenClient(),
        qc_prompt_template=prompt_path,
        qc_max_tokens=100,
        seed_quality_gate=GripperSeedQualityGateConfig(minimum_pixels=4),
    )

    assert result.status == "ok"
    assert result.active_arm == "right"
    assert result.selected_candidate == "A"
    assert len(backend.text_box_calls) >= 1
    assert backend.box_calls == []
    assert len(backend.propagate_calls) == 1
    assert not result.gripper_track[0].any()
    assert result.gripper_track[1:23].any()
    assert not result.gripper_track[23].any()
    assert result.provenance["known_object_tracks"] == "saved_sam_native_track"


def test_run_gripper_stage_finishes_all_text_attempts_before_box_fallback(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "state.parquet"
    resource_path = tmp_path / "frames"
    prompt_path = tmp_path / "prompt.txt"
    _write_state(state_path)
    _write_resource(resource_path)
    prompt_path.write_text(
        "task={task_text}; arm={active_arm}; ids={candidate_ids}\n"
        "{candidate_records}\n{candidate_panels}\n{context_frames}\n"
        "events={move_start},{close_start},{close_done},{open_start},{open_done}",
        encoding="utf-8",
    )
    backend = FakeGripperBackend(valid_text_masks=False)
    target = np.zeros((FRAME_COUNT, *FRAME_SHAPE), dtype=bool)
    receiver = np.zeros_like(target)

    result = run_gripper_stage(
        _context(state_path),
        backend=backend,
        resource_path=resource_path,
        frame_shape=FRAME_SHAPE,
        gripper_roi_config=GripperRoiConfig(
            prompt_axial_back_m=0.120,
            prompt_axial_front_m=0.060,
            hard_axial_back_m=0.120,
            hard_axial_front_m=0.045,
            fixed_half_width_m=0.085,
        ),
        object_tracks={
            ObjectRole.TARGET: target,
            ObjectRole.RECEIVER: receiver,
        },
        qc_client=FakeQwenClient(selected_candidate="H"),
        qc_prompt_template=prompt_path,
        qc_max_tokens=100,
        seed_quality_gate=GripperSeedQualityGateConfig(minimum_pixels=4),
    )

    keyframes = [1, 9, 12, 16, 19, 20, 22]
    assert backend.call_order == [
        *(("text_box", frame_id) for frame_id in keyframes),
        *(("box_only", frame_id) for frame_id in keyframes),
    ]
    assert tuple(candidate.candidate_id for candidate in result.qc_result.candidates) == tuple(
        "ABCDEFGHIJKLMN"
    )
    assert all(not candidate.basic_valid for candidate in result.qc_result.candidates[:7])
    assert all(candidate.basic_valid for candidate in result.qc_result.candidates[7:])
    assert result.selected_candidate == "H"


def test_target_only_gripper_rejects_sam_backend_before_inference(tmp_path: Path) -> None:
    state_path = tmp_path / "state.parquet"
    resource_path = tmp_path / "frames"
    prompt_path = tmp_path / "prompt.txt"
    _write_state(state_path)
    _write_resource(resource_path)
    prompt_path.write_text(
        "task={task_text}; arm={active_arm}; ids={candidate_ids}\n"
        "{candidate_records}\n{candidate_panels}\n{context_frames}\n"
        "events={move_start},{close_start},{close_done},{open_start},{open_done}",
        encoding="utf-8",
    )
    target = np.zeros((FRAME_COUNT, *FRAME_SHAPE), dtype=bool)

    backend = FakeGripperBackend()
    with np.testing.assert_raises_regex(
        GripperStageError,
        "target_only does not support the SAM gripper backend",
    ):
        run_gripper_stage(
            _target_only_context(state_path),
            backend=backend,
            resource_path=resource_path,
            frame_shape=FRAME_SHAPE,
            gripper_roi_config=GripperRoiConfig(
                prompt_axial_back_m=0.120,
                prompt_axial_front_m=0.060,
                hard_axial_back_m=0.120,
                hard_axial_front_m=0.045,
                fixed_half_width_m=0.085,
            ),
            object_tracks={ObjectRole.TARGET: target},
            qc_client=FakeQwenClient(),
            qc_prompt_template=prompt_path,
            qc_max_tokens=100,
            seed_quality_gate=GripperSeedQualityGateConfig(minimum_pixels=4),
        )

    assert backend.text_box_calls == []
    assert backend.propagate_calls == []


def test_legacy_annotator_exports_preserve_canonical_identity() -> None:
    public_names = (
        "GripperStageError",
        "GripperStageResult",
        "gripper_keyframes",
        "run_gripper_stage",
    )
    for name in public_names:
        canonical = getattr(annotator, name)
        assert getattr(legacy_stage, name) is canonical
        assert getattr(public_pipeline, name) is canonical
    assert legacy_stage.GripperSamBackend is annotator.GripperSamBackend
    for name in (
        "_build_gripper_candidates",
        "_build_roi_track",
        "_context_frame_ids",
        "_load_resource_image",
        "_load_state_arrays",
        "_polygon_mask",
        "_roi_geometries",
        "_roi_policy",
        "_track_summary",
    ):
        assert getattr(legacy_stage, name) is getattr(annotator, name)
