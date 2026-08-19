from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

import robotwin_annotation_v2.pipeline as public_pipeline
from robotwin_annotation_v2.adapters import QwenCompletion
from robotwin_annotation_v2.domain import ObjectRole
from robotwin_annotation_v2.models import (
    EpisodeRef,
    FramePurpose,
    LoopContext,
    LoopEvents,
    SemanticFrame,
)
from robotwin_annotation_v2.models.mask_qc import MaskQCStatus
from robotwin_annotation_v2.pipeline import (
    GripperSeedCandidate,
    ProjectedGripperRoi,
    apply_gripper_seed_quality_gate,
    build_gripper_seed_candidate,
    compose_gripper_track,
    gripper_keyframes,
    load_qc_native_object_tracks,
    mark_same_frame_duplicates,
    render_gripper_candidate_panel,
    run_gripper_seed_qc,
)
from robotwin_annotation_v2.pipeline import gripper_stage as legacy_stage
from robotwin_annotation_v2.pipeline.gripper.sam import candidates

SHAPE = (12, 16)


def _roi(x: float = 8.0, y: float = 6.0) -> ProjectedGripperRoi:
    polygon = np.asarray([[3.0, 2.0], [13.0, 2.0], [13.0, 10.0], [3.0, 10.0]])
    return ProjectedGripperRoi(
        eef_pixel_xy=np.asarray([x, y]),
        tcp_pixel_xy=np.asarray([x, y]),
        corner_pixels_xy=polygon,
        hull_pixels_xy=polygon,
        bbox_xyxy=np.asarray([3.0, 2.0, 13.0, 10.0]),
        corner_depths=np.ones(4),
        open_fraction=0.5,
    )


def _events() -> LoopEvents:
    return LoopEvents(
        active_arm="right",
        t_move_start=1,
        t_close_start=10,
        t_close_done=12,
        t_open_start=20,
        t_open_done=22,
    )


def _context() -> LoopContext:
    return LoopContext(
        episode=EpisodeRef("task", 7),
        task_text="move the bottle",
        frame_count=24,
        events=_events(),
        semantic_frames=(
            SemanticFrame(0, FramePurpose.PRE_GRASP_SEED_CANDIDATE, ("target",)),
            SemanticFrame(12, FramePurpose.POST_GRASP_CONTEXT, ("target",)),
            SemanticFrame(22, FramePurpose.PLACE_CONTEXT, ("receiver",)),
        ),
        state_source="state.parquet",
        video_source="video.mp4",
    )


def _write_qc_prompt(path: Path) -> None:
    path.write_text(
        "task={task_text}; arm={active_arm}; ids={candidate_ids}\n"
        "{candidate_records}\n{candidate_panels}\n{context_frames}\n"
        "events={move_start},{close_start},{close_done},{open_start},{open_done}",
        encoding="utf-8",
    )


def _candidate(candidate_id: str, *, frame_id: int = 5) -> GripperSeedCandidate:
    raw = np.zeros(SHAPE, dtype=bool)
    raw[3:9, 5:11] = True
    roi = np.ones(SHAPE, dtype=bool)
    target = np.zeros(SHAPE, dtype=bool)
    target[7:9, 5:11] = True
    receiver = np.zeros(SHAPE, dtype=bool)
    rgb = np.zeros((*SHAPE, 3), dtype=np.uint8)
    return build_gripper_seed_candidate(
        candidate_id=candidate_id,
        frame_id=frame_id,
        events=_events(),
        prompt_mode="box_only",
        prompt_text=None,
        raw_mask=raw,
        roi_mask=roi,
        target_mask=target,
        receiver_mask=receiver,
        rgb=rgb,
        tcp_pixel_xy=np.asarray([8.0, 6.0]),
        minimum_pixels=4,
    )


class FakeQwenClient:
    model_id = "fake-qwen"

    def __init__(self, *, decision: str = "accept") -> None:
        self.messages: list[list[dict[str, Any]]] = []
        self.decision = decision

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "model": self.model_id}

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
    ) -> QwenCompletion:
        self.messages.append(messages)
        assert max_tokens == 100
        if self.decision == "error":
            raise RuntimeError("unavailable")
        if self.decision == "accept":
            payload = {
                "decision": "accept",
                "selected_candidate": "A",
                "confidence": 0.91,
                "reason": "Candidate A cleanly covers the visible gripper.",
            }
        else:
            payload = {
                "decision": self.decision,
                "selected_candidate": None,
                "confidence": 0.12,
                "reason": "Every candidate is imperfect.",
            }
        return QwenCompletion(
            content=json.dumps(payload),
            model=self.model_id,
        )


def test_gripper_keyframes_match_experiment_contract() -> None:
    events = _events()
    rois = {frame_id: _roi() for frame_id in range(1, 23)}

    assert gripper_keyframes(rois, events, frame_shape=SHAPE) == (
        1,
        9,
        12,
        16,
        19,
        20,
        22,
    )


def test_candidate_applies_pose_and_exact_object_exclusion() -> None:
    candidate = _candidate("A")

    assert candidate.cropped_pixels == 36
    assert candidate.target_removed_pixels == 12
    assert candidate.clean_pixels == 24
    assert candidate.basic_valid
    assert candidate.dark_fraction == 1.0
    assert not (candidate.clean_mask & candidate.target_removed).any()


def test_prompt_roi_crops_seed_while_hard_roi_crops_final_track() -> None:
    raw = np.zeros(SHAPE, dtype=bool)
    raw[5, 7:10] = True
    prompt_roi = np.zeros(SHAPE, dtype=bool)
    prompt_roi[5, 7:10] = True
    hard_roi = np.zeros(SHAPE, dtype=bool)
    hard_roi[5, 8:10] = True
    empty = np.zeros(SHAPE, dtype=bool)
    candidate = build_gripper_seed_candidate(
        candidate_id="A",
        frame_id=5,
        events=_events(),
        prompt_mode="text_box",
        prompt_text="black robot gripper",
        raw_mask=raw,
        roi_mask=prompt_roi,
        target_mask=empty,
        receiver_mask=empty,
        rgb=np.zeros((*SHAPE, 3), dtype=np.uint8),
        tcp_pixel_xy=np.asarray([8.0, 5.0]),
        minimum_pixels=1,
    )
    native = np.repeat(candidate.clean_mask[None, ...], 2, axis=0)
    result = compose_gripper_track(
        native,
        np.repeat(hard_roi[None, ...], 2, axis=0),
        np.zeros_like(native),
        np.zeros_like(native),
        active_window=(0, 1),
    )

    assert candidate.clean_pixels == 3
    assert np.array_equal(candidate.clean_mask, prompt_roi)
    assert np.array_equal(result.gripper_mask[0], hard_roi)


def test_black_gripper_quality_gate_rejects_implausible_candidate() -> None:
    candidate = replace(
        _candidate("A"),
        dark_fraction=0.10,
        component_count=6,
        largest_component_fraction=0.50,
        tcp_distance_px=25.0,
    )

    gated = apply_gripper_seed_quality_gate(candidate)

    assert not gated.basic_valid
    assert gated.basic_reason == (
        "quality_gate:low_dark_fraction,too_many_components,"
        "small_largest_component,too_far_from_tcp"
    )


def test_black_gripper_quality_gate_keeps_compact_dark_candidate() -> None:
    candidate = _candidate("A")

    gated = apply_gripper_seed_quality_gate(candidate)

    assert gated is candidate
    assert gated.basic_valid


def test_same_frame_duplicate_is_not_submitted_as_valid() -> None:
    first = _candidate("A")
    second = _candidate("B")

    marked = mark_same_frame_duplicates((first, second))

    assert marked[0].basic_valid
    assert not marked[1].basic_valid
    assert marked[1].duplicate_of == "A"


def test_component_metrics_preserve_opencv_eight_connectivity() -> None:
    diagonal = np.eye(3, dtype=bool)

    assert candidates._component_metrics(diagonal) == (1, 1.0)


def test_legacy_candidate_exports_preserve_canonical_identity() -> None:
    public_names = (
        "GripperSeedCandidate",
        "GripperSeedQualityGateConfig",
        "apply_gripper_seed_quality_gate",
        "build_gripper_seed_candidate",
        "mark_same_frame_duplicates",
        "phase_for_frame",
    )
    for name in public_names:
        canonical = getattr(candidates, name)
        assert getattr(legacy_stage, name) is canonical
        assert getattr(public_pipeline, name) is canonical
    assert legacy_stage._component_metrics is candidates._component_metrics
    assert legacy_stage._tcp_distance is candidates._tcp_distance


def test_gripper_qwen_qc_receives_candidate_and_context_images(tmp_path: Path) -> None:
    candidate = _candidate("A")
    rgb = Image.fromarray(np.zeros((*SHAPE, 3), dtype=np.uint8))
    panel = render_gripper_candidate_panel(rgb, candidate, _roi())
    prompt = tmp_path / "prompt.txt"
    _write_qc_prompt(prompt)
    client = FakeQwenClient()

    result = run_gripper_seed_qc(
        _context(),
        (candidate,),
        {"A": panel},
        {1: rgb, 22: rgb},
        prompt_template_path=prompt,
        client=client,
        max_tokens=100,
    )

    assert result.status is MaskQCStatus.PASSED
    assert result.selected_candidate == "A"
    content = client.messages[0][0]["content"]
    assert sum(item["type"] == "image_url" for item in content) == 3


def test_gripper_qwen_qc_forces_one_candidate_when_qwen_rejects(tmp_path: Path) -> None:
    candidate = _candidate("A")
    rgb = Image.fromarray(np.zeros((*SHAPE, 3), dtype=np.uint8))
    panel = render_gripper_candidate_panel(rgb, candidate, _roi())
    prompt = tmp_path / "prompt.txt"
    _write_qc_prompt(prompt)

    result = run_gripper_seed_qc(
        _context(),
        (candidate,),
        {"A": panel},
        {1: rgb, 22: rgb},
        prompt_template_path=prompt,
        client=FakeQwenClient(decision="reject_all"),
        max_tokens=100,
    )

    assert result.status is MaskQCStatus.PASSED
    assert result.selected_candidate == "A"
    assert result.forced_fallback
    assert "forced fallback candidate A" in result.reason


def test_gripper_qwen_qc_rejects_when_no_fallback_candidate_exists(tmp_path: Path) -> None:
    result = run_gripper_seed_qc(
        _context(),
        (),
        {},
        {},
        prompt_template_path=tmp_path / "missing.txt",
        client=FakeQwenClient(),
        max_tokens=100,
    )

    assert result.status is MaskQCStatus.REJECTED
    assert result.selected_candidate is None
    assert not result.forced_fallback
    assert result.reason == "all_gripper_candidates_failed_basic_checks"


def test_gripper_qwen_qc_retries_then_stably_selects_first_tied_fallback(
    tmp_path: Path,
) -> None:
    first = _candidate("A", frame_id=5)
    second = _candidate("B", frame_id=6)
    rgb = Image.fromarray(np.zeros((*SHAPE, 3), dtype=np.uint8))
    prompt = tmp_path / "prompt.txt"
    _write_qc_prompt(prompt)
    client = FakeQwenClient(decision="error")

    result = run_gripper_seed_qc(
        _context(),
        (first, second),
        {
            "A": render_gripper_candidate_panel(rgb, first, _roi()),
            "B": render_gripper_candidate_panel(rgb, second, _roi()),
        },
        {1: rgb},
        prompt_template_path=prompt,
        client=client,
        max_tokens=100,
        max_attempts=2,
    )

    assert len(client.messages) == 2
    assert result.status is MaskQCStatus.PASSED
    assert result.selected_candidate == "A"
    assert result.forced_fallback
    assert result.reason == (
        "forced fallback candidate A; "
        "gripper QC request failed after 2 attempt(s): unavailable"
    )


def test_load_qc_native_tracks_includes_approved_seed_masks(tmp_path: Path) -> None:
    ref = EpisodeRef("task", 7)
    episode_dir = tmp_path / "task" / "episode_000007" / "cam_high"
    for role in ("target_0", "receiver_0"):
        role_dir = episode_dir / role
        role_dir.mkdir(parents=True)
        np.savez_compressed(role_dir / "native_track.npz", masks=np.ones((3, *SHAPE), bool))
        Image.fromarray(np.ones(SHAPE, dtype=np.uint8) * 255).save(role_dir / "seed.mask.png")
    manifest = {
        "run_id": "qc-run",
        "episode": ref.to_json(),
        "frame_count": 3,
        "roles": [
            {
                "role": "target",
                "status": "ok",
                "qc_status": "passed",
                "seed_frame_id": 0,
                "seed_mask_path": "target_0/seed.mask.png",
                "native_track_path": "target_0/native_track.npz",
            },
            {
                "role": "receiver",
                "status": "ok",
                "qc_status": "passed",
                "seed_frame_id": 0,
                "seed_mask_path": "receiver_0/seed.mask.png",
                "native_track_path": "receiver_0/native_track.npz",
            },
        ],
    }
    episode_dir.mkdir(parents=True, exist_ok=True)
    (episode_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    tracks = load_qc_native_object_tracks(
        tmp_path,
        ref,
        expected_shape=(3, *SHAPE),
    )

    assert tracks.target.shape == (3, *SHAPE)
    assert tracks.target_seed_mask.all()
    assert tracks.receiver_seed_frame == 0
    assert tracks.provenance["run_id"] == "qc-run"


def test_load_qc_native_tracks_target_only_does_not_require_receiver(
    tmp_path: Path,
) -> None:
    ref = EpisodeRef("task", 7)
    episode_dir = tmp_path / "task" / "episode_000007" / "cam_high"
    target_dir = episode_dir / "target_0"
    target_dir.mkdir(parents=True)
    np.savez_compressed(
        target_dir / "native_track.npz",
        masks=np.ones((3, *SHAPE), dtype=bool),
    )
    Image.fromarray(np.ones(SHAPE, dtype=np.uint8) * 255).save(
        target_dir / "seed.mask.png"
    )
    manifest = {
        "run_id": "target-only-run",
        "episode": ref.to_json(),
        "frame_count": 3,
        "roles": [
            {
                "role": "target",
                "status": "ok",
                "qc_status": "passed",
                "seed_frame_id": 0,
                "seed_mask_path": "target_0/seed.mask.png",
                "native_track_path": "target_0/native_track.npz",
            },
            {
                "role": "receiver",
                "status": "not_applicable",
                "qc_status": "not_applicable",
            },
        ],
    }
    (episode_dir / "run_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    tracks = load_qc_native_object_tracks(
        tmp_path,
        ref,
        expected_shape=(3, *SHAPE),
        required_roles=(ObjectRole.TARGET,),
    )

    assert tracks.target.all()
    assert not tracks.receiver.any()
    assert tracks.receiver_seed_frame is None
