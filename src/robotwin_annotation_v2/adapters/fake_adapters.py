"""Fake adapters for testing without GPU/external services.

These implementations are for unit testing only.
"""

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..domain import Box, EpisodeRef, FrameWindow
from ..domain.policies import InteractionTimeline, SemanticPlan
from ..ports import (
    ArtifactRepository,
    EpisodeRepository,
    FrameSource,
    GroundingService,
    KeyframeSelector,
    SemanticPlanner,
    SingleFrameSegmenter,
    TimelineDetector,
)


class FakeEpisodeRepository:
    """Fake episode repository returning dummy data."""

    def load_metadata(self, ref: EpisodeRef) -> dict[str, Any]:
        return {
            "length": 100,
            "height": 240,
            "width": 320,
            "fps": 30,
        }

    def load_state(self, ref: EpisodeRef) -> dict[str, Any]:
        # Fake gripper state trajectory
        return {
            "gripper_left": np.ones((100, 2)),  # [T, 2] (position + openness)
            "gripper_right": np.ones((100, 2)),
        }


class FakeSemanticPlanner:
    """Fake semantic planner returning predefined queries."""

    def plan(self, ref: EpisodeRef) -> SemanticPlan:
        return SemanticPlan(
            episode=ref,
            target_query="white pill bottle",
            receiver_query="blue square pad",
            has_static_receiver=True,
        )


class FakeTimelineDetector:
    """Fake timeline detector returning fixed events."""

    def detect(self, ref: EpisodeRef, state: dict[str, Any]) -> InteractionTimeline:
        return InteractionTimeline(
            episode=ref,
            move_start=10,
            move_end=60,
            close_start=50,
            close_end=52,
            open_start=70,
            open_end=72,
            hold_start=52,
            hold_end=70,
        )


class FakeFrameSource:
    """Fake frame source returning solid color images."""

    def read_frame(self, ref: EpisodeRef, frame_index: int) -> Image.Image:
        # Return a solid color image
        return Image.new("RGB", (320, 240), color=(100, 150, 200))

    def get_dimensions(self, ref: EpisodeRef) -> tuple[int, int]:
        return (240, 320)  # (height, width)


class FakeKeyframeSelector:
    """Fake keyframe selector returning evenly spaced frames."""

    def select_candidates(
        self,
        ref: EpisodeRef,
        window: FrameWindow,
        max_candidates: int = 5,
    ) -> list[int]:
        # Return evenly spaced frames within window
        if len(window) < max_candidates:
            return list(range(window.first, window.last + 1))

        step = len(window) // max_candidates
        return [window.first + i * step for i in range(max_candidates)]


class FakeGroundingService:
    """Fake grounding service returning predefined bbox."""

    def ground(
        self,
        frame: Image.Image,
        text_query: str,
    ) -> tuple[str, Box]:
        # Return refined query and a centered box
        refined_query = f"{text_query} (refined)"
        bbox = Box(x_min=0.3, y_min=0.3, x_max=0.7, y_max=0.7)
        return refined_query, bbox


class FakeSingleFrameSegmenter:
    """Fake segmenter returning rectangular masks."""

    def segment(
        self,
        frame: Image.Image,
        prompt: Any,
        method: Any,
    ) -> np.ndarray:
        # Return a rectangular mask centered in the image
        h, w = 240, 320
        mask = np.zeros((h, w), dtype=bool)
        # Center rectangle: 40% to 60% of image
        mask[int(h * 0.4):int(h * 0.6), int(w * 0.4):int(w * 0.6)] = True
        return mask


class FakeArtifactRepository:
    """Fake artifact repository storing in memory."""

    def __init__(self) -> None:
        self.runs: dict[str, dict[str, Any]] = {}
        self.requests: dict[str, dict[str, Any]] = {}

    def create_run(self, config: dict[str, Any]) -> str:
        run_id = f"fake-run-{len(self.runs):03d}"
        self.runs[run_id] = {
            "config": config,
            "requests": {},
        }
        return run_id

    def save_request(
        self,
        run_id: str,
        request: Any,
        data: dict[str, Any],
    ) -> None:
        key = f"{request.episode.episode_id}_{request.slot.name}"
        self.runs[run_id]["requests"][key] = data

    def load_request(
        self,
        run_id: str,
        episode_id: str,
        slot_name: str,
    ) -> dict[str, Any]:
        key = f"{episode_id}_{slot_name}"
        return self.runs[run_id]["requests"][key]

    def save_approved_seed(
        self,
        run_id: str,
        seed: Any,
    ) -> None:
        # Just store in memory
        if "approved_seeds" not in self.runs[run_id]:
            self.runs[run_id]["approved_seeds"] = []
        self.runs[run_id]["approved_seeds"].append(seed)

    def get_run_dir(self, run_id: str) -> Path:
        return Path(f"/tmp/fake_artifacts/{run_id}")
