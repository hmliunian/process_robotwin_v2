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
    MaskStatus,
    QueryBank,
    RoleSemanticPlan,
    SemanticFrame,
    SemanticPlan,
    SemanticStatus,
)
from robotwin_annotation_v2.pipeline import (
    compose_visible_mask,
    dilate_envelope,
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
        self.calls.append(("observe", text))
        masks = {frame_id: self.seed_masks[text].copy() for frame_id in frame_ids}
        masks[frame_ids[len(frame_ids) // 2]] = np.zeros(FRAME_SHAPE, dtype=bool)
        return masks


def test_visible_composition_is_strict_intersection_and_windowed() -> None:
    native = np.ones((4, 3, 3), dtype=bool)
    observed = np.zeros_like(native)
    observed[:, 1, 1] = True
    envelope = np.zeros((3, 3), dtype=bool)
    envelope[1, 1:] = True

    visible = compose_visible_mask(native, observed, envelope, FrameWindow(1, 2))

    assert not visible[0].any() and not visible[3].any()
    assert visible[1, 1, 1] and visible[2, 1, 1]
    assert int(visible.sum()) == 2


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
    assert submitted_queries == [
        "orange bottle",
        "orange bottle",
        "blue pad",
        "blue pad",
    ]
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
        assert archive["format_version"].item() == "robotwin_visible_masks_v1"
        assert archive["frame_count"].item() == 20
        assert archive["masks"].shape == (4, 20, *FRAME_SHAPE)
        assert not archive["masks"][2:].any()
        assert archive["annotation_status"].tolist() == [
            "valid",
            "valid",
            "not_annotated",
            "not_annotated",
        ]
    manifest = json.loads((episode_dir / "run_manifest.json").read_text())
    assert manifest["channels"]["gripper_left"] == "not_annotated"
    assert (episode_dir / "target_0/seed.mask.png").is_file()
    assert (episode_dir / "receiver_0/canonical_envelope.png").is_file()
