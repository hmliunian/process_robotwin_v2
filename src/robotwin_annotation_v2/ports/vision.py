"""Port interfaces for vision capabilities."""

from typing import Protocol

import numpy as np
from PIL import Image

from ..domain import Box, EpisodeRef, FrameWindow, SegmentationMethod, VisualPrompt


class FrameSource(Protocol):
    """Read video frames on demand."""

    def read_frame(self, ref: EpisodeRef, frame_index: int) -> Image.Image:
        """Read a single frame as PIL Image."""
        ...

    def get_dimensions(self, ref: EpisodeRef) -> tuple[int, int]:
        """Get (height, width) of frames."""
        ...


class GroundingService(Protocol):
    """Visual grounding (Qwen VLM) to refine queries and get tight bbox."""

    def ground(
        self,
        frame: Image.Image,
        text_query: str,
    ) -> tuple[str, Box]:
        """
        Refine query and return tight bounding box.

        Returns:
            (refined_query, bbox): e.g., ("white pill bottle" -> "white cylindrical bottle", Box(...))
        """
        ...


class SingleFrameSegmenter(Protocol):
    """Single-frame segmentation (SAM3).

    Phase 1 ONLY. Video propagation is in Phase 2.
    """

    def segment(
        self,
        frame: Image.Image,
        prompt: VisualPrompt,
        method: SegmentationMethod,
    ) -> np.ndarray:
        """
        Segment one frame with given prompt.

        Returns:
            Binary mask [H, W] bool
        """
        ...


class KeyframeSelector(Protocol):
    """Rank candidate frames within allowed window."""

    def select_candidates(
        self,
        ref: EpisodeRef,
        window: FrameWindow,
        max_candidates: int = 5,
    ) -> list[int]:
        """
        Return top-N frame indices ranked by clarity/occlusion.

        Simple heuristic: spread across window, avoid motion blur.
        """
        ...
