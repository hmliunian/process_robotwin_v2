from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from robotwin_annotation_v2.adapters import ArtifactStore
from robotwin_annotation_v2.config import MaskConfig
from robotwin_annotation_v2.domain import AnnotationMode
from robotwin_annotation_v2.mask_schema import FrameEncoding, target_hold_window
from robotwin_annotation_v2.models import (
    EpisodeRef,
    FramePurpose,
    FrameWindow,
    LoopContext,
    LoopEvents,
    MaskQCResult,
    MaskQCStatus,
    MaskStatus,
    QueryBank,
    RoleMaskQC,
    RoleSemanticPlan,
    SemanticFrame,
    SemanticPlan,
    SemanticStatus,
    TargetOnlyEvents,
)
from robotwin_annotation_v2.pipeline import (
    GripperSeedQCResult,
    GripperStageResult,
    SamStageError,
    compose_visible_mask,
    dilate_envelope,
    evaluate_temporal_mask,
    run_sam_stage,
    save_sam_artifacts,
)

FRAME_SHAPE = (5, 6)


def _context() -> LoopContext:
    return LoopContext(
        episode=EpisodeRef("move_pillbottle_pad", 7152, "cam_high"),
        task_text="Move the bottle onto the pad.",
        frame_count=20,
        events=LoopEvents("right", 2, 6, 8, 14, 17),
        semantic_frames=(
            SemanticFrame(
                0,
                FramePurpose.PRE_GRASP_SEED_CANDIDATE,
                ("target", "receiver"),
            ),
            SemanticFrame(9, FramePurpose.POST_GRASP_CONTEXT, ("target",)),
            SemanticFrame(15, FramePurpose.PLACE_CONTEXT, ("receiver",)),
        ),
        state_source="state.parquet",
        video_source="video.mp4",
    )


def _role(role: str, query: str) -> RoleSemanticPlan:
    return RoleSemanticPlan(
        role=role,  # type: ignore[arg-type]
        status=SemanticStatus.OK,
        seed_frame_id=0,
        query_bank=QueryBank(
            category_query="bottle" if role == "target" else "pad",
            color_category_query=query,
            general_fallback_query="container" if role == "target" else "mat",
            recommended_order=(
                "color_category_query",
                "category_query",
                "general_fallback_query",
            ),
        ),
        exclude=(),
        reason=f"{role} reason",
    )


def _plan(*, target: RoleSemanticPlan | None = None) -> SemanticPlan:
    return SemanticPlan(
        episode=_context().episode,
        role_plans=(
            target or _role("target", "orange bottle"),
            _role("receiver", "blue pad"),
        ),
        model="fake-qwen",
        prompt_sha256=hashlib.sha256(b"prompt").hexdigest(),
        input_frame_ids=(0, 9, 15),
        raw_response="{}",
    )


def _target_only_context() -> LoopContext:
    base = _context()
    return LoopContext(
        episode=base.episode,
        task_text=base.task_text,
        frame_count=base.frame_count,
        events=TargetOnlyEvents("right", 2, 6, 8),
        semantic_frames=(
            SemanticFrame(
                0,
                FramePurpose.PRE_GRASP_SEED_CANDIDATE,
                ("target",),
            ),
            SemanticFrame(9, FramePurpose.POST_GRASP_CONTEXT, ("target",)),
        ),
        state_source=base.state_source,
        video_source=base.video_source,
        annotation_mode=AnnotationMode.TARGET_ONLY,
    )


def _target_only_plan() -> SemanticPlan:
    context = _target_only_context()
    return SemanticPlan(
        episode=context.episode,
        role_plans=(_role("target", "orange bottle"),),
        model="fake-qwen",
        prompt_sha256=hashlib.sha256(b"prompt").hexdigest(),
        input_frame_ids=(0, 9),
        raw_response="{}",
        annotation_mode=AnnotationMode.TARGET_ONLY,
    )


class FakeSamBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.tracking_windows: list[tuple[int, int]] = []
        self.propagation_seed_frames: list[int] = []
        target = np.zeros(FRAME_SHAPE, dtype=bool)
        target[1:3, 1:3] = True
        receiver = np.zeros(FRAME_SHAPE, dtype=bool)
        receiver[3:5, 3:6] = True
        self.seed_masks = {
            "bottle": target,
            "orange bottle": target,
            "blue pad": receiver,
        }

    def text_mask(
        self,
        _resource_path: Path,
        text: str,
        **_kwargs: Any,
    ) -> np.ndarray:
        self.calls.append(("seed", text))
        return self.seed_masks[text].copy()

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
        self.calls.append(("track", ""))
        self.propagation_seed_frames.append(seed_frame)
        self.tracking_windows.append(tracking_window)
        output = np.zeros((frame_count, *seed_mask.shape), dtype=bool)
        output[tracking_window[0] : tracking_window[1] + 1] = seed_mask
        return output

    def text_masks(
        self,
        _resource_path: Path,
        text: str,
        *,
        frame_ids: tuple[int, ...],
        **_kwargs: Any,
    ) -> dict[int, np.ndarray]:
        raise AssertionError("production Stage 3 must not run per-frame text masks")


class DiscontinuousSamBackend(FakeSamBackend):
    def propagate_mask(
        self,
        resource_path: Path,
        seed_mask: np.ndarray,
        *,
        frame_count: int,
        tracking_window: tuple[int, int],
        **kwargs: Any,
    ) -> np.ndarray:
        output = super().propagate_mask(
            resource_path,
            seed_mask,
            frame_count=frame_count,
            tracking_window=tracking_window,
            **kwargs,
        )
        if int(seed_mask.sum()) != 4:
            return output
        for frame_id in range(tracking_window[0], tracking_window[1] + 1):
            output[frame_id] = False
            if frame_id % 2:
                output[frame_id, 0, 0] = True
            else:
                output[frame_id, 3:5, 3:6] = True
        return output


class DiscontinuousHoldSamBackend(FakeSamBackend):
    """Keep the grasp prefix stable while simulating large held-object motion."""

    def propagate_mask(
        self,
        resource_path: Path,
        seed_mask: np.ndarray,
        *,
        frame_count: int,
        tracking_window: tuple[int, int],
        **kwargs: Any,
    ) -> np.ndarray:
        output = super().propagate_mask(
            resource_path,
            seed_mask,
            frame_count=frame_count,
            tracking_window=tracking_window,
            **kwargs,
        )
        if int(seed_mask.sum()) != 4:
            return output
        for frame_id in range(9, tracking_window[1] + 1):
            output[frame_id] = False
            if frame_id % 2:
                output[frame_id, 0, 0] = True
            else:
                output[frame_id, 3:5, 3:6] = True
        return output


def test_visible_composition_preserves_native_pixels_and_applies_window() -> None:
    native = np.zeros((4, 3, 3), dtype=bool)
    native[:, 0:2, 1:3] = True

    visible = compose_visible_mask(native, FrameWindow(1, 2))

    assert not visible[0].any() and not visible[3].any()
    assert np.array_equal(visible[1:3], native[1:3])
    assert int(visible.sum()) == 8


def test_temporal_qc_reviews_gaps_without_quarantining_occlusion() -> None:
    masks = np.zeros((4, 5, 5), dtype=bool)
    masks[0, 1:3, 1:3] = True
    masks[2:, 1:3, 1:3] = True

    qc = evaluate_temporal_mask(masks, FrameWindow(0, 3), MaskConfig())

    assert qc.status == "review"
    assert qc.coverage == 0.75
    assert qc.presence_transitions == 2
    assert qc.internal_missing_frames == 1
    assert "internal_missing_frames" in qc.issues


def test_temporal_qc_quarantines_multiple_severe_jump_signals() -> None:
    masks = np.zeros((4, 12, 12), dtype=bool)
    masks[0, 0, 0] = True
    masks[1, 8:12, 8:12] = True
    masks[2, 0, 0] = True
    masks[3, 8:12, 8:12] = True

    qc = evaluate_temporal_mask(masks, FrameWindow(0, 3), MaskConfig())

    assert qc.status == "quarantine"
    assert set(qc.issues) == {
        "low_adjacent_iou_p05",
        "large_centroid_jump_p95",
        "large_area_ratio_jump_p95",
    }


def test_sam_stage_does_not_publish_a_quarantined_track() -> None:
    result = run_sam_stage(
        _context(),
        _plan(),
        DiscontinuousSamBackend(),
        Path("/tmp/fake-resource"),
        frame_shape=FRAME_SHAPE,
        mask_config=MaskConfig(
            0,
            0,
            temporal_qc_min_adjacent_iou_p05=0.9,
            temporal_qc_max_centroid_jump_p95_px=0.1,
            temporal_qc_max_area_ratio_jump_p95=0.01,
        ),
    )

    assert result.target.status is MaskStatus.QUARANTINED
    assert result.target.failure is not None
    assert result.target.failure.startswith("temporal_qc_quarantine:")
    assert not result.target.visible_mask.any()
    assert result.target.native_track.any()
    assert not result.masks[0].any()
    assert result.receiver.status is MaskStatus.OK


@pytest.mark.parametrize(
    ("context", "plan"),
    [
        pytest.param(_context(), _plan(), id="pick_place"),
        pytest.param(
            _target_only_context(),
            _target_only_plan(),
            id="target_only",
        ),
    ],
)
def test_target_temporal_qc_does_not_quarantine_normal_hold_motion(
    context: LoopContext,
    plan: SemanticPlan,
) -> None:
    result = run_sam_stage(
        context,
        plan,
        DiscontinuousHoldSamBackend(),
        Path("/tmp/fake-resource"),
        frame_shape=FRAME_SHAPE,
        mask_config=MaskConfig(
            0,
            0,
            temporal_qc_min_adjacent_iou_p05=0.9,
            temporal_qc_max_centroid_jump_p95_px=0.1,
            temporal_qc_max_area_ratio_jump_p95=0.01,
        ),
    )

    hold = target_hold_window(context.events, frame_count=context.frame_count)
    assert hold is not None
    hold_start, hold_end = hold
    assert result.target.status is MaskStatus.OK
    assert result.target.temporal_qc is not None
    assert result.target.temporal_qc.status == "pass"
    assert result.target.temporal_qc.window == context.events.target_window
    hold_masks = result.target.visible_mask[hold_start : hold_end + 1]
    assert hold_masks.any(axis=(1, 2)).all()
    assert not np.array_equal(hold_masks[0], hold_masks[1])


def test_dilate_envelope_expands_seed_by_configured_radius() -> None:
    seed = np.zeros((5, 5), dtype=bool)
    seed[2, 2] = True

    envelope = dilate_envelope(seed, 1)

    assert int(envelope.sum()) == 5
    assert envelope[2, 2] and envelope[1, 2] and not envelope[1, 1]


def test_sam_stage_uses_only_primary_queries_and_role_windows() -> None:
    backend = FakeSamBackend()

    result = run_sam_stage(
        _context(),
        _plan(),
        backend,
        Path("/tmp/fake-resource"),
        frame_shape=FRAME_SHAPE,
        mask_config=MaskConfig(0, 0),
    )

    assert result.target.status is MaskStatus.OK
    assert result.receiver.status is MaskStatus.OK
    assert result.target.primary_query == "orange bottle"
    assert result.receiver.primary_query == "blue pad"
    submitted_queries = [value for kind, value in backend.calls if value]
    assert submitted_queries == ["orange bottle", "blue pad"]
    assert all(kind != "observe" for kind, _value in backend.calls)
    assert result.target.temporal_qc is not None
    assert result.target.temporal_qc.status == "pass"
    assert not result.target.visible_mask[:2].any()
    assert result.target.visible_mask[9:14].any()
    assert not result.target.visible_mask[14:].any()
    assert not result.receiver.visible_mask[:8].any()
    assert not result.receiver.visible_mask[18:].any()
    assert not result.masks[2:].any()


def test_target_only_sam_tracks_target_and_leaves_receiver_channel_zero() -> None:
    backend = FakeSamBackend()
    context = _target_only_context()

    result = run_sam_stage(
        context,
        _target_only_plan(),
        backend,
        Path("/tmp/fake-resource"),
        frame_shape=FRAME_SHAPE,
        mask_config=MaskConfig(0, 0),
    )

    assert tuple(data.role for data in result.role_masks) == ("target",)
    assert [value for kind, value in backend.calls if kind == "seed"] == [
        "orange bottle"
    ]
    assert result.target.output_window == context.windows.target
    assert backend.tracking_windows == [(0, 19)]
    assert result.target.visible_mask[9:].any()
    assert not result.masks[1].any()
    with pytest.raises(KeyError, match="non-applicable"):
        _ = result.receiver


def test_target_only_sam_propagates_the_qwen_qc_selected_candidate() -> None:
    backend = FakeSamBackend()
    selected_seed = backend.seed_masks["bottle"].copy()
    mask_qc = MaskQCResult(
        role_reports=(
            _qc_report("target", MaskQCStatus.PASSED, query="bottle"),
        ),
        selected_masks={"target": selected_seed},
        health={"status": "ok"},
    )

    result = run_sam_stage(
        _target_only_context(),
        _target_only_plan(),
        backend,
        Path("/tmp/fake-resource"),
        frame_shape=FRAME_SHAPE,
        mask_config=MaskConfig(0, 0),
        mask_qc=mask_qc,
    )

    assert result.target.qc_status is MaskQCStatus.PASSED
    assert result.target.primary_query == "bottle"
    assert not any(kind == "seed" for kind, _value in backend.calls)
    assert backend.tracking_windows == [(0, 19)]
    assert result.target.visible_mask[2:].any()
    assert not result.masks[1].any()


def test_no_clear_seed_skips_target_sam_calls() -> None:
    backend = FakeSamBackend()
    no_target = RoleSemanticPlan(
        role="target",
        status=SemanticStatus.NO_CLEAR_SEED,
        seed_frame_id=None,
        query_bank=None,
        exclude=(),
        reason="no clear target",
    )

    result = run_sam_stage(
        _context(),
        _plan(target=no_target),
        backend,
        Path("/tmp/fake-resource"),
        frame_shape=FRAME_SHAPE,
        mask_config=MaskConfig(0, 0),
    )

    assert result.target.status is MaskStatus.FAILED
    assert result.target.failure == "semantic_plan_no_clear_seed"
    assert all(value != "orange bottle" for _kind, value in backend.calls)


def _qc_report(
    role: str,
    status: MaskQCStatus,
    *,
    query: str | None = None,
    seed_frame_id: int = 0,
) -> RoleMaskQC:
    selected = "A" if status is MaskQCStatus.PASSED else None
    return RoleMaskQC(
        role=role,  # type: ignore[arg-type]
        status=status,
        selected_candidate=selected,
        selected_query_field="category_query" if selected else None,
        selected_query=query if selected else None,
        selected_seed_frame_id=seed_frame_id if selected else None,
        confidence=0.95,
        reason="test QC decision",
    )


def _multi_seed_context() -> LoopContext:
    base = _context()
    return LoopContext(
        episode=base.episode,
        task_text=base.task_text,
        frame_count=base.frame_count,
        events=base.events,
        semantic_frames=(
            SemanticFrame(
                0,
                FramePurpose.PRE_GRASP_SEED_CANDIDATE,
                ("target", "receiver"),
            ),
            SemanticFrame(
                3,
                FramePurpose.PRE_GRASP_SEED_CANDIDATE,
                ("target",),
            ),
            SemanticFrame(9, FramePurpose.POST_GRASP_CONTEXT, ("target",)),
            SemanticFrame(15, FramePurpose.PLACE_CONTEXT, ("receiver",)),
        ),
        state_source=base.state_source,
        video_source=base.video_source,
    )


def test_sam_stage_propagates_from_the_qc_selected_fallback_seed(tmp_path: Path) -> None:
    backend = FakeSamBackend()
    mask_qc = MaskQCResult(
        role_reports=(
            _qc_report(
                "target",
                MaskQCStatus.PASSED,
                query="bottle",
                seed_frame_id=3,
            ),
            _qc_report("receiver", MaskQCStatus.PASSED, query="blue pad"),
        ),
        selected_masks={
            "target": backend.seed_masks["bottle"].copy(),
            "receiver": backend.seed_masks["blue pad"].copy(),
        },
        health={"status": "ok"},
    )

    result = run_sam_stage(
        _multi_seed_context(),
        _plan(),
        backend,
        Path("/tmp/fake-resource"),
        frame_shape=FRAME_SHAPE,
        mask_config=MaskConfig(0, 0),
        mask_qc=mask_qc,
    )

    assert result.target.seed_frame_id == 3
    assert backend.propagation_seed_frames == [3, 0]
    assert backend.tracking_windows[0] == (2, 13)

    seed_image = Image.fromarray(np.zeros((*FRAME_SHAPE, 3), dtype=np.uint8))
    mask_run = save_sam_artifacts(
        ArtifactStore(tmp_path),
        "sam-fallback-test",
        _multi_seed_context(),
        _plan(),
        result,
        seed_images={0: seed_image, 3: seed_image},
    )
    manifest = json.loads((Path(mask_run.artifact_dir) / "run_manifest.json").read_text())
    assert not manifest["algorithm"]["automatic_query_fallback"]
    assert manifest["algorithm"]["mask_qc_fallback_used"]


def test_sam_stage_rejects_a_qc_seed_outside_role_seed_candidates() -> None:
    backend = FakeSamBackend()
    mask_qc = MaskQCResult(
        role_reports=(
            _qc_report(
                "target",
                MaskQCStatus.PASSED,
                query="bottle",
                seed_frame_id=4,
            ),
            _qc_report("receiver", MaskQCStatus.PASSED, query="blue pad"),
        ),
        selected_masks={
            "target": backend.seed_masks["bottle"].copy(),
            "receiver": backend.seed_masks["blue pad"].copy(),
        },
        health={"status": "ok"},
    )

    with pytest.raises(SamStageError, match="QC selected.*ineligible"):
        run_sam_stage(
            _multi_seed_context(),
            _plan(),
            backend,
            Path("/tmp/fake-resource"),
            frame_shape=FRAME_SHAPE,
            mask_config=MaskConfig(0, 0),
            mask_qc=mask_qc,
        )


def test_sam_stage_uses_qc_selected_query_and_cached_seed_mask() -> None:
    backend = FakeSamBackend()
    target_seed = backend.seed_masks["bottle"].copy()
    receiver_seed = backend.seed_masks["blue pad"].copy()
    mask_qc = MaskQCResult(
        role_reports=(
            _qc_report("target", MaskQCStatus.PASSED, query="bottle"),
            _qc_report("receiver", MaskQCStatus.PASSED, query="blue pad"),
        ),
        selected_masks={"target": target_seed, "receiver": receiver_seed},
        health={"status": "ok"},
    )

    result = run_sam_stage(
        _context(),
        _plan(),
        backend,
        Path("/tmp/fake-resource"),
        frame_shape=FRAME_SHAPE,
        mask_config=MaskConfig(0, 0),
        mask_qc=mask_qc,
    )

    assert result.target.primary_query == "bottle"
    assert result.target.qc_status is MaskQCStatus.PASSED
    assert result.receiver.qc_status is MaskQCStatus.PASSED
    assert not any(kind == "seed" for kind, _value in backend.calls)
    assert [kind for kind, _value in backend.calls] == ["track", "track"]


def test_sam_stage_does_not_propagate_rejected_qc_candidate() -> None:
    backend = FakeSamBackend()
    mask_qc = MaskQCResult(
        role_reports=(
            _qc_report("target", MaskQCStatus.REJECTED),
            _qc_report("receiver", MaskQCStatus.PASSED, query="blue pad"),
        ),
        selected_masks={"receiver": backend.seed_masks["blue pad"].copy()},
        health={"status": "ok"},
    )

    result = run_sam_stage(
        _context(),
        _plan(),
        backend,
        Path("/tmp/fake-resource"),
        frame_shape=FRAME_SHAPE,
        mask_config=MaskConfig(0, 0),
        mask_qc=mask_qc,
    )

    assert result.target.status is MaskStatus.FAILED
    assert result.target.failure == "mask_qc_rejected"
    assert not result.target.visible_mask.any()
    assert all(value != "orange bottle" for _kind, value in backend.calls)


def test_save_sam_artifacts_marks_grippers_not_annotated(tmp_path: Path) -> None:
    context = _context()
    plan = _plan()
    result = run_sam_stage(
        context,
        plan,
        FakeSamBackend(),
        Path("/tmp/fake-resource"),
        frame_shape=FRAME_SHAPE,
        mask_config=MaskConfig(0, 0),
    )
    seed_image = Image.fromarray(np.zeros((*FRAME_SHAPE, 3), dtype=np.uint8))

    mask_run = save_sam_artifacts(
        ArtifactStore(tmp_path),
        "sam-test",
        context,
        plan,
        result,
        seed_images={0: seed_image},
    )

    episode_dir = Path(mask_run.artifact_dir)
    with np.load(episode_dir / "masks.npz", allow_pickle=False) as archive:
        assert archive["format_version"].item() == "robotwin_visible_masks_v3"
        assert archive["frame_count"].item() == 20
        assert archive["masks"].shape == (4, 20, *FRAME_SHAPE)
        assert archive["frame_encoding"].shape == (4, 20)
        assert archive["frame_encoding"].dtype == np.uint8
        assert archive["frame_encoding"][0].tolist() == [
            *([FrameEncoding.ABSENT.value] * 2),
            *([FrameEncoding.VISIBLE.value] * 7),
            *([FrameEncoding.TARGET_GRASP_HOLD.value] * 5),
            *([FrameEncoding.ABSENT.value] * 6),
        ]
        assert not archive["masks"][2:].any()
        assert archive["annotation_status"].tolist() == [
            "valid",
            "valid",
            "not_annotated",
            "not_annotated",
        ]
        assert archive["qc_status"].tolist() == [
            "not_run",
            "not_run",
            "not_run",
            "not_run",
        ]
    manifest = json.loads((episode_dir / "run_manifest.json").read_text())
    assert manifest["format_version"] == "robotwin_mask_run_v2"
    assert manifest["mask_format_version"] == "robotwin_visible_masks_v3"
    assert manifest["frame_encoding"]["target_hold_window"] == [9, 13]
    assert manifest["gripper_backend"] == "sam"
    assert not manifest["algorithm"]["per_frame_text_observation"]
    assert manifest["algorithm"]["canonical_envelope_usage"] == "seed_diagnostic_only"
    assert not manifest["algorithm"]["automatic_query_fallback"]
    assert not manifest["algorithm"]["mask_qc_fallback_used"]
    assert manifest["channels"]["gripper_left"] == "not_annotated"
    assert (episode_dir / "target_0/seed.mask.png").is_file()
    assert (episode_dir / "receiver_0/canonical_envelope.png").is_file()
    assert (episode_dir / "target_0/temporal_qc.json").is_file()
    assert not (episode_dir / "target_0/text_observations.npz").exists()
    target_qc = json.loads(
        (episode_dir / "target_0/temporal_qc.json").read_text()
    )
    assert target_qc["window"] == [2, 8]
    assert target_qc["target_hold_coverage"] == {
        "window": [9, 13],
        "window_frames": 5,
        "nonempty_frames": 5,
        "coverage": 1.0,
    }
    provenance = json.loads((episode_dir / "frame_provenance.json").read_text())
    assert provenance["gripper_backend"] == "sam"
    assert provenance["channels"]["target_0"]["target_hold_coverage"] == (
        target_qc["target_hold_coverage"]
    )


def test_target_only_artifacts_publish_receiver_as_not_applicable(
    tmp_path: Path,
) -> None:
    context = _target_only_context()
    plan = _target_only_plan()
    result = run_sam_stage(
        context,
        plan,
        FakeSamBackend(),
        Path("/tmp/fake-resource"),
        frame_shape=FRAME_SHAPE,
        mask_config=MaskConfig(0, 0),
    )
    seed_image = Image.fromarray(np.zeros((*FRAME_SHAPE, 3), dtype=np.uint8))

    mask_run = save_sam_artifacts(
        ArtifactStore(tmp_path),
        "target-only-test",
        context,
        plan,
        result,
        seed_images={0: seed_image},
    )

    episode_dir = Path(mask_run.artifact_dir)
    with np.load(episode_dir / "masks.npz", allow_pickle=False) as archive:
        assert not archive["masks"][1].any()
        assert archive["frame_encoding"][0, 9:].tolist() == [
            FrameEncoding.TARGET_GRASP_HOLD.value
        ] * 11
        assert archive["annotation_status"].tolist()[:2] == [
            "valid",
            "not_applicable",
        ]
        assert archive["qc_status"].tolist()[:2] == [
            "not_run",
            "not_applicable",
        ]
    manifest = json.loads((episode_dir / "run_manifest.json").read_text())
    assert manifest["annotation_mode"] == "target_only"
    assert manifest["required_object_roles"] == ["target"]
    receiver = next(item for item in manifest["roles"] if item["role"] == "receiver")
    assert receiver["status"] == "not_applicable"
    assert receiver["output_window"] is None
    assert receiver["native_track_path"] is None
    assert not (episode_dir / "receiver_0").exists()
    provenance = json.loads((episode_dir / "frame_provenance.json").read_text())
    assert provenance["channels"]["receiver_0"]["status"] == "not_applicable"


def test_save_sam_artifacts_writes_active_gripper_channel(tmp_path: Path) -> None:
    context = _context()
    plan = _plan()
    result = run_sam_stage(
        context,
        plan,
        FakeSamBackend(),
        Path("/tmp/fake-resource"),
        frame_shape=FRAME_SHAPE,
        mask_config=MaskConfig(0, 0),
    )
    seed_image = Image.fromarray(np.zeros((*FRAME_SHAPE, 3), dtype=np.uint8))
    gripper = np.zeros((context.frame_count, *FRAME_SHAPE), dtype=bool)
    gripper[2:18, 0, 0] = True
    seed = gripper[2].copy()
    empty = np.zeros_like(gripper)
    gripper_result = GripperStageResult(
        active_arm="right",
        active_window=FrameWindow(2, 17),
        frame_count=context.frame_count,
        frame_shape=FRAME_SHAPE,
        seed_frame_id=2,
        selected_candidate="A",
        seed_mask=seed,
        native_track=gripper,
        roi_track=np.ones_like(gripper),
        candidate_track=gripper,
        gripper_track=gripper,
        removed_track=empty,
        target_removed_track=empty,
        receiver_removed_track=empty,
        prompt_rois={},
        hard_rois={},
        qc_result=GripperSeedQCResult(
            status=MaskQCStatus.PASSED,
            selected_candidate="A",
            confidence=0.93,
            reason="candidate A is clean",
            candidates=(),
            model="fake-qwen",
        ),
        candidate_panels={},
        roi_policy={"prompt": {}, "hard": {}},
        provenance={"known_object_tracks": "saved_sam_native_track"},
    )

    mask_run = save_sam_artifacts(
        ArtifactStore(tmp_path),
        "sam-gripper-test",
        context,
        plan,
        result,
        seed_images={0: seed_image},
        gripper_result=gripper_result,
    )

    episode_dir = Path(mask_run.artifact_dir)
    with np.load(episode_dir / "masks.npz", allow_pickle=False) as archive:
        assert np.array_equal(archive["masks"][3], gripper)
        assert not archive["masks"][2].any()
        assert archive["annotation_status"].tolist() == [
            "valid",
            "valid",
            "not_annotated",
            "valid",
        ]
        assert archive["qc_status"].tolist()[3] == "passed"
    manifest = json.loads((episode_dir / "run_manifest.json").read_text())
    assert manifest["channels"]["gripper_right"] == 3
    assert manifest["channels"]["gripper_left"] == "not_annotated"
    assert manifest["algorithm"]["gripper_stage"]["backend"] == "sam"
    assert manifest["gripper_qc"]["backend"] == "sam"
    assert manifest["gripper_qc"]["status"] == "ok"
    assert manifest["gripper_qc"]["qc_status"] == "passed"
    assert manifest["gripper_qc"]["selected_candidate"] == "A"
    assert set(manifest["gripper_qc"]) == {
        "backend",
        "status",
        "qc_status",
        "active_arm",
        "selected_candidate",
        "confidence",
        "reason",
        "forced_fallback",
        "nonempty_frames",
        "quality",
    }
    assert manifest["gripper_qc"]["quality"] is None
    assert [role["role"] for role in manifest["roles"]] == [
        "target",
        "receiver",
        "gripper_right",
    ]
    provenance = json.loads((episode_dir / "frame_provenance.json").read_text())
    assert provenance["gripper_backend"] == "sam"
    assert provenance["channels"]["gripper_right"]["backend"] == "sam"
    assert provenance["channels"]["gripper_right"]["active_window"] == [2, 17]
    assert provenance["channels"]["gripper_left"]["status"] == "not_annotated"
    assert (episode_dir / "gripper_right/native_track.npz").is_file()
    assert (episode_dir / "gripper_right/gripper_seed_qc.json").is_file()
