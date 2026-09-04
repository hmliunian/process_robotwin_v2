from __future__ import annotations

import json
from pathlib import Path

from robotwin_annotation_v2.domain import AnnotationMode, TargetProfile
from robotwin_annotation_v2.models import (
    EpisodeRef,
    FramePurpose,
    LoopContext,
    SemanticFrame,
    TargetOnlyEvents,
)
from robotwin_annotation_v2.pipeline import parse_semantic_plan
from scripts import materialize_native_tracking_run as materializer


def test_load_semantic_plan_preserves_saved_target_profile(tmp_path: Path) -> None:
    context = LoopContext(
        episode=EpisodeRef("open_microwave", 1, "cam_high"),
        task_text="Open the microwave door.",
        frame_count=20,
        events=TargetOnlyEvents("right", 2, 6, 8, None),
        semantic_frames=(
            SemanticFrame(
                0,
                FramePurpose.PRE_GRASP_SEED_CANDIDATE,
                ("target",),
            ),
        ),
        state_source="episode.parquet",
        video_source="episode.mp4",
        annotation_mode=AnnotationMode.TARGET_ONLY,
    )
    raw_response = json.dumps(
        {
            "target": {
                "status": "ok",
                "seed_frame_id": 0,
                "category_query": "handle",
                "color_category_query": None,
                "shape_category_query": None,
                "general_fallback_query": None,
                "recommended_order": ["category_query"],
                "exclude": [],
                "reason": "The handle is visible before contact.",
            }
        }
    )
    rendered_prompt = "rendered prompt"
    expected = parse_semantic_plan(
        raw_response,
        context=context,
        model="fake-qwen",
        rendered_prompt=rendered_prompt,
        target_profile=TargetProfile.DOOR_OPEN,
    )
    (tmp_path / "semantic_plan.json").write_text(
        json.dumps(expected.to_json()),
        encoding="utf-8",
    )
    (tmp_path / "qwen_rendered_prompt.txt").write_text(
        rendered_prompt,
        encoding="utf-8",
    )
    (tmp_path / "qwen_raw_response.txt").write_text(
        raw_response,
        encoding="utf-8",
    )

    loaded = materializer._load_semantic_plan(tmp_path, context)

    assert loaded == expected
    assert loaded.target_profile is TargetProfile.DOOR_OPEN
