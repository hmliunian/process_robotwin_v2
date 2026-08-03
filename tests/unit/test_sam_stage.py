from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from robotwin_annotation_v2.adapters import ArtifactStore
from robotwin_annotation_v2.config import MaskConfig
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
)
from robotwin_annotation_v2.pipeline import (
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
        target=target or _role("target", "orange bottle"),
        receiver=_role("receiver", "blue pad"),
        model="fake-qwen",
        prompt_sha256=hashlib.sha256(b"prompt").hexdigest(),
        input_frame_ids=(0, 9, 15),
        raw_response="{}",
    )


class FakeSamBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
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
        frame_count: int,
        tracking_window: tuple[int, int],
        **_kwargs: Any,
    ) -> np.ndarray:
        self.calls.append(("track", ""))
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
    assert not result.target.visible_mask[9:].any()
    assert not result.receiver.visible_mask[:8].any()
    assert not result.receiver.visible_mask[18:].any()
    assert not result.masks[2:].any()


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
) -> RoleMaskQC:
    selected = "A" if status is MaskQCStatus.PASSED else None
    return RoleMaskQC(
        role=role,  # type: ignore[arg-type]
        status=status,
        selected_candidate=selected,
        selected_query_field="category_query" if selected else None,
        selected_query=query if selected else None,
        confidence=0.95,
        reason="test QC decision",
    )


def test_sam_stage_uses_qc_selected_query_and_cached_seed_mask() -> None:
    backend = FakeSamBackend()
    target_seed = backend.seed_masks["bottle"].copy()
    receiver_seed = backend.seed_masks["blue pad"].copy()
    mask_qc = MaskQCResult(
        target=_qc_report("target", MaskQCStatus.PASSED, query="bottle"),
        receiver=_qc_report("receiver", MaskQCStatus.PASSED, query="blue pad"),
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
        target=_qc_report("target", MaskQCStatus.REJECTED),
        receiver=_qc_report("receiver", MaskQCStatus.PASSED, query="blue pad"),
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
        assert archive["format_version"].item() == "robotwin_visible_masks_v2"
        assert archive["frame_count"].item() == 20
        assert archive["masks"].shape == (4, 20, *FRAME_SHAPE)
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
    assert not manifest["algorithm"]["per_frame_text_observation"]
    assert manifest["algorithm"]["canonical_envelope_usage"] == "seed_diagnostic_only"
    assert manifest["channels"]["gripper_left"] == "not_annotated"
    assert (episode_dir / "target_0/seed.mask.png").is_file()
    assert (episode_dir / "receiver_0/canonical_envelope.png").is_file()
    assert (episode_dir / "target_0/temporal_qc.json").is_file()
    assert not (episode_dir / "target_0/text_observations.npz").exists()
