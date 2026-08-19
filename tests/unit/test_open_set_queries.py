from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from robotwin_annotation_v2.config import MaskConfig
from robotwin_annotation_v2.models import (
    EpisodeRef,
    FramePurpose,
    LoopContext,
    LoopEvents,
    QueryBank,
    RoleSemanticPlan,
    SemanticFrame,
    SemanticStatus,
)
from robotwin_annotation_v2.models.semantic_plan import RoleName
from robotwin_annotation_v2.pipeline import curated_query_aliases, mask_qc, open_set_queries
from robotwin_annotation_v2.pipeline.object_mask.planner import plan_role_queries


def _context(task: str) -> LoopContext:
    return LoopContext(
        episode=EpisodeRef(task, 1, "cam_high"),
        task_text=f"Test {task}",
        frame_count=20,
        events=LoopEvents("right", 2, 6, 8, 14, 17),
        semantic_frames=(
            SemanticFrame(
                0,
                FramePurpose.PRE_GRASP_SEED_CANDIDATE,
                ("target", "receiver"),
            ),
            SemanticFrame(7, FramePurpose.POST_GRASP_CONTEXT, ("target",)),
            SemanticFrame(15, FramePurpose.PLACE_CONTEXT, ("receiver",)),
        ),
        state_source="state.parquet",
        video_source="video.mp4",
    )


def _semantic(
    role: RoleName,
    category: str,
    color: str | None = None,
    shape: str | None = None,
    fallback: str | None = None,
) -> RoleSemanticPlan:
    return RoleSemanticPlan(
        role=role,
        status=SemanticStatus.OK,
        seed_frame_id=0,
        query_bank=QueryBank(
            category_query=category,
            color_category_query=color,
            shape_category_query=shape,
            general_fallback_query=fallback,
        ),
        exclude=(),
        reason="clear seed",
    )


@pytest.mark.parametrize(
    ("task", "semantic", "expected"),
    (
        (
            "move_stapler_pad",
            _semantic("target", "stapler", "blue stapler", "curved stapler", "tool"),
            ("desk stapler", "office stapler", "paper stapler"),
        ),
        (
            "place_fan",
            _semantic("target", "fan", "silver fan", "round fan", "device"),
            ("desk fan", "table fan", "electric fan"),
        ),
        (
            "place_container_plate",
            _semantic("target", "bowl", "white bowl", "round bowl", "container"),
            ("small bowl", "small container", "rice bowl"),
        ),
        (
            "place_object_scale",
            _semantic("target", "mouse", "gray mouse", "gray oval", "device"),
            ("computer mouse", "wireless mouse", "optical mouse"),
        ),
        (
            "place_phone_stand",
            _semantic("target", "phone", "black phone", "rectangular phone", "device"),
            ("smartphone", "mobile phone", "cell phone"),
        ),
        (
            "move_stapler_pad",
            _semantic("receiver", "mat", "cyan mat", "cyan square", "pad"),
            ("cyan rectangle", "cyan square pad", "cyan table mat"),
        ),
        (
            "place_empty_cup",
            _semantic("receiver", "coaster", "white coaster", "round coaster", "mat"),
            ("drink coaster", "cup coaster", "beverage coaster"),
        ),
        (
            "place_object_stand",
            _semantic("receiver", "stand", "black stand", "rectangular stand", "frame"),
            ("black box", "box", "display platform"),
        ),
        (
            "place_phone_stand",
            _semantic(
                "receiver",
                "phonestand",
                "light brown phonestand",
                "light brown stand",
                "stand",
            ),
            ("light brown phone holder", "phone holder", "phone dock"),
        ),
    ),
)
def test_curated_query_aliases_cover_known_semantic_groups(
    task: str,
    semantic: RoleSemanticPlan,
    expected: tuple[str, ...],
) -> None:
    aliases = curated_query_aliases(_context(task), semantic.role, semantic)

    assert aliases == expected
    assert len(aliases) <= 3


def test_curated_query_aliases_remove_queries_already_in_bank_and_fill_next_alias() -> None:
    semantic = _semantic(
        "receiver",
        "stand",
        "brown stand",
        "brown rectangle",
        "box",
    )

    aliases = curated_query_aliases(_context("place_object_stand"), "receiver", semantic)

    assert aliases == ("brown box", "display platform", "display base")


def test_curated_query_aliases_normalize_and_deduplicate_existing_aliases() -> None:
    semantic = _semantic("target", "fan", "white fan", "table fan", "device")

    aliases = curated_query_aliases(_context("place_fan"), "target", semantic)

    assert aliases == ("desk fan", "electric fan")


def test_curated_query_aliases_return_empty_for_unknown_semantics() -> None:
    semantic = _semantic("target", "bottle", "orange bottle", "plastic bottle", "container")

    assert curated_query_aliases(_context("move_bottle"), "target", semantic) == ()


def test_curated_query_aliases_return_empty_without_query_bank() -> None:
    semantic = RoleSemanticPlan(
        role="target",
        status=SemanticStatus.NO_CLEAR_SEED,
        seed_frame_id=None,
        query_bank=None,
        exclude=(),
        reason="no clear seed",
    )

    assert curated_query_aliases(_context("place_fan"), "target", semantic) == ()


def test_curated_query_aliases_reject_role_mismatch() -> None:
    semantic = _semantic("target", "fan", "white fan", "round fan", "device")

    with pytest.raises(ValueError, match="does not match"):
        curated_query_aliases(_context("place_fan"), "receiver", semantic)


def test_query_candidates_keep_four_semantic_queries_and_cap_aliases_at_three() -> None:
    semantic = _semantic(
        "receiver",
        "phonestand",
        "blue phonestand",
        "blue rectangular stand",
        "stand",
    )

    candidates = plan_role_queries(
        _context("place_phone_stand"),
        "receiver",
        semantic,
        query_fallback_enabled=True,
    )

    assert tuple(candidate.field for candidate in candidates) == (
        "category_query",
        "color_category_query",
        "shape_category_query",
        "general_fallback_query",
        "curated_alias_1",
        "curated_alias_2",
        "curated_alias_3",
    )


def test_legacy_query_planner_seam_matches_canonical_planner() -> None:
    context = _context("place_phone_stand")
    semantic = _semantic(
        "receiver",
        "phonestand",
        "blue phonestand",
        "blue rectangular stand",
        "stand",
    )
    config = MaskConfig(
        qc_enabled=True,
        qc_prompt_template=Path("unused"),
        qc_max_candidates=8,
        qc_query_fallback_enabled=True,
    )

    assert mask_qc._role_query_candidates(context, "receiver", semantic, config) == (
        plan_role_queries(
            context,
            "receiver",
            semantic,
            query_fallback_enabled=True,
        )
    )


def test_blue_prior_uses_eighth_candidate_slot(monkeypatch: pytest.MonkeyPatch) -> None:
    config = MaskConfig(
        qc_enabled=True,
        qc_prompt_template=Path("unused"),
        qc_max_candidates=8,
        qc_query_fallback_enabled=True,
    )
    semantic = _semantic(
        "receiver",
        "phonestand",
        "blue phonestand",
        "blue rectangular stand",
        "stand",
    )
    query_candidates = mask_qc._role_query_candidates(
        _context("place_phone_stand"), "receiver", semantic, config
    )

    class RecordingBackend:
        texts: tuple[str, ...] = ()

        def text_query_masks(
            self,
            _resource_path: Path,
            texts: tuple[str, ...],
            **_kwargs: Any,
        ) -> dict[str, np.ndarray]:
            self.texts = tuple(texts)
            mask = np.zeros((12, 16), dtype=bool)
            mask[1:3, 1:4] = True
            return {text: mask for text in texts}

    captured: list[Any] = []

    def capture_candidates(*_args: Any, **kwargs: Any) -> object:
        captured.extend(kwargs["candidates"])
        return object()

    monkeypatch.setattr(mask_qc, "_evaluate_candidate_visual_qc", capture_candidates)
    seed = np.zeros((12, 16, 3), dtype=np.uint8)
    seed[7:11, 9:15] = (10, 20, 240)
    backend = RecordingBackend()

    mask_qc._run_role_qc_at_seed(
        _context("place_phone_stand"),
        "receiver",
        seed_frame_id=0,
        query_candidates=query_candidates,
        backend=backend,
        resource_path=Path("unused"),
        seed_image=Image.fromarray(seed),
        context_images={},
        frame_shape=(12, 16),
        mask_config=config,
        client=object(),
    )

    assert len(backend.texts) == 7
    assert len(captured) == config.qc_max_candidates
    assert captured[-1].query_field == "blue_region_prior"


def test_alias_catalog_loader_fails_closed_for_missing_file(tmp_path: Path) -> None:
    open_set_queries._load_rules.cache_clear()
    with pytest.raises(FileNotFoundError):
        open_set_queries._load_rules(tmp_path / "missing.yaml")


@pytest.mark.parametrize(
    ("contents", "error"),
    (
        ("format_version: wrong\nrules: []\n", "invalid open-set alias catalog"),
        (
            "format_version: robotwin_open_set_query_aliases_v1\nrules: {}\n",
            "rules must be a list",
        ),
        (
            "format_version: robotwin_open_set_query_aliases_v1\n"
            + "rules:\n  - role: object\n    tasks: [place_fan]\n    aliases: [fan]\n",
            "invalid role",
        ),
        (
            "format_version: robotwin_open_set_query_aliases_v1\n"
            + "rules:\n  - role: target\n    aliases: [fan]\n",
            "no match condition",
        ),
        (
            "format_version: robotwin_open_set_query_aliases_v1\n"
            + "rules:\n  - role: target\n    tasks: [place_fan]\n    aliases: fan\n",
            "aliases must be a list",
        ),
    ),
)
def test_alias_catalog_loader_fails_closed_for_malformed_yaml(
    tmp_path: Path,
    contents: str,
    error: str,
) -> None:
    catalog = tmp_path / "aliases.yaml"
    catalog.write_text(contents, encoding="utf-8")
    open_set_queries._load_rules.cache_clear()

    with pytest.raises((TypeError, ValueError), match=error):
        open_set_queries._load_rules(catalog)
