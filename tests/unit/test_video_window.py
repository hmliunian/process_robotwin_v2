from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from PIL import Image

from robotwin_annotation_v2.adapters.loop_context_codec import (
    LoopContextCodecError,
    load_authoritative_loop_context,
)
from robotwin_annotation_v2.adapters.robotwin_dataset import DatasetError, RoboTwinDataset
from robotwin_annotation_v2.config import load_profile
from robotwin_annotation_v2.domain import AnnotationMode, TargetProfile
from robotwin_annotation_v2.models import EpisodeRef, LoopContext, VideoWindowEvents
from robotwin_annotation_v2.models.timeline import derive_target_hold_window
from robotwin_annotation_v2.pipeline.bbox_localization import render_bbox_prompt
from robotwin_annotation_v2.pipeline.prompt_context import timeline_prompt_fields
from robotwin_annotation_v2.pipeline.qwen_stage import build_qwen_request
from robotwin_annotation_v2.pipeline.state_loop import build_loop_context, sample_semantic_frames


@pytest.mark.parametrize("mode", [AnnotationMode.TARGET_ONLY, AnnotationMode.TOOL_USE])
def test_video_window_roundtrip_without_fabricated_grasp(
    tmp_path: Path, mode: AnnotationMode
) -> None:
    events = VideoWindowEvents("left", 0, 19)
    context = LoopContext(
        episode=EpisodeRef("video_task", 0),
        task_text="Act on the object with the held tool.",
        frame_count=20,
        events=events,
        semantic_frames=sample_semantic_frames(events, frame_count=20, annotation_mode=mode),
        state_source="episodes.jsonl",
        video_source="video.mp4",
        annotation_mode=mode,
    )
    assert len(context.seed_candidates("target")) == 8
    assert context.seed_candidates("target")[-1] == 19
    assert bool(context.seed_candidates("receiver")) == (mode is AnnotationMode.TOOL_USE)
    assert context.windows.target.to_json() == [0, 19]
    assert (context.windows.receiver is not None) == (mode is AnnotationMode.TOOL_USE)
    assert derive_target_hold_window(events, frame_count=20) is None
    assert "close_start" not in timeline_prompt_fields(context)
    profile = load_profile(
        Path("configs/process.yaml"),
        mode=mode,
        target_profile=(
            TargetProfile.VIDEO_OBJECT
            if mode is AnnotationMode.TARGET_ONLY
            else TargetProfile.GRASP_MANIPULATION
        ),
    )
    request = build_qwen_request(
        context,
        {frame.frame_id: Image.new("RGB", (8, 8)) for frame in context.semantic_frames},
        profile.qwen.prompt_template.read_text(),
    )
    assert "seed_candidate=yes" in request.rendered_prompt
    assert profile.mask.qc_bbox_prompt_template is not None
    bbox_prompt = render_bbox_prompt(
        profile.mask.qc_bbox_prompt_template.read_text(),
        task="video_task",
        task_text=context.task_text,
        episode_id="000000",
        role="target",
        seed_frame_id=0,
    )
    assert "video_task" in bbox_prompt
    assert "000000" in bbox_prompt
    path = tmp_path / "loop.json"
    path.write_text(json.dumps(context.to_json()))
    loaded = load_authoritative_loop_context(
        path, expected_task="video_task", expected_episode_index=0, expected_camera="cam_high"
    )
    assert loaded.events == events
    assert loaded.windows == context.windows
    assert loaded.target_hold_window is None


@pytest.mark.parametrize(
    "invalid",
    [
        {"active_arm": "both", "t_start": 0, "t_end": 19},
        {"active_arm": "left", "t_start": True, "t_end": 19},
        {"active_arm": "left", "t_start": 10, "t_end": 9},
    ],
)
def test_video_window_rejects_invalid_events(invalid: dict[str, Any]) -> None:
    with pytest.raises((TypeError, ValueError)):
        VideoWindowEvents(**invalid)


def test_video_window_metadata_needs_only_frame_indices(tmp_path: Path) -> None:
    dataset = RoboTwinDataset(
        tmp_path,
        task="tool_task",
        camera="cam_high",
        manifest_path=tmp_path / "EXTRACT_MANIFEST.json",
        manifest_data={
            "dataset_root": str(tmp_path),
            "task": "tool_task",
            "camera": "cam_high",
            "regression_episode_ids": [0],
            "timeline_source": "video_window",
        },
    )
    ref = EpisodeRef("tool_task", 0)
    parquet = dataset.paths(ref).parquet
    parquet.parent.mkdir(parents=True)
    pd.DataFrame({"frame_index": range(20), "episode_index": [0] * 20}).to_parquet(parquet)
    meta = tmp_path / "meta" / "episodes.jsonl"
    meta.parent.mkdir()
    payload: dict[str, Any] = {
        "episode_index": 0,
        "length": 20,
        "tasks": ["Use the held tool."],
        "video_events": {"active_arm": "right", "t_start": 0, "t_end": 19},
    }
    meta.write_text(json.dumps(payload) + "\n")
    context = build_loop_context(dataset, ref, annotation_mode=AnnotationMode.TOOL_USE)
    assert context.timeline_kind == "video_window"
    assert context.windows.receiver == context.windows.target
    dataset._episode_metadata = None
    payload["video_events"]["t_end"] = 20
    meta.write_text(json.dumps(payload) + "\n")
    with pytest.raises(DatasetError, match="exceed"):
        dataset.load_episode_timeline(ref)


def test_video_window_codec_rejects_extra_or_forged_windows(tmp_path: Path) -> None:
    context = LoopContext(
        EpisodeRef("video_task", 0),
        "Operate the object.",
        20,
        VideoWindowEvents("right", 0, 19),
        sample_semantic_frames(
            VideoWindowEvents("right", 0, 19),
            frame_count=20,
            annotation_mode=AnnotationMode.TARGET_ONLY,
        ),
        "episodes.jsonl",
        "video.mp4",
        AnnotationMode.TARGET_ONLY,
    )
    payload = context.to_json()
    payload["windows"]["target_0"] = [0, 18]
    path = tmp_path / "loop.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(LoopContextCodecError, match="does not match events"):
        load_authoritative_loop_context(
            path, expected_task="video_task", expected_episode_index=0, expected_camera="cam_high"
        )
