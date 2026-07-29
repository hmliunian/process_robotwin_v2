"""Application use case: Prepare keyframes for one episode.

This is the main Phase 1 workflow.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from ..domain import (
    Box,
    EpisodeRef,
    KeyframeRequest,
    SegmentationMethod,
    VisualPrompt,
)
from ..domain.policies import RolePolicyRegistry
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


@dataclass
class MaskCandidate:
    """A candidate mask for one frame + method."""
    candidate_id: str
    frame_index: int
    method: SegmentationMethod
    query: str
    bbox: Box | None
    mask: np.ndarray  # [H, W] bool
    area_fraction: float


@dataclass
class KeyframePackage:
    """Complete keyframe preparation result for one request."""
    request: KeyframeRequest
    candidates: list[MaskCandidate]
    grounding_evidence: dict[str, Any]


class PrepareKeyframes:
    """Use case: Generate keyframe candidates for an episode."""

    def __init__(
        self,
        episode_repo: EpisodeRepository,
        semantic_planner: SemanticPlanner,
        timeline_detector: TimelineDetector,
        frame_source: FrameSource,
        keyframe_selector: KeyframeSelector,
        grounding_service: GroundingService,
        segmenter: SingleFrameSegmenter,
        artifact_repo: ArtifactRepository,
        policy_registry: RolePolicyRegistry,
    ) -> None:
        self.episode_repo = episode_repo
        self.semantic_planner = semantic_planner
        self.timeline_detector = timeline_detector
        self.frame_source = frame_source
        self.keyframe_selector = keyframe_selector
        self.grounding_service = grounding_service
        self.segmenter = segmenter
        self.artifact_repo = artifact_repo
        self.policy_registry = policy_registry

    def execute(self, ref: EpisodeRef) -> str:
        """
        Prepare keyframes for all roles in this episode.

        Returns:
            run_id: identifier for this run's artifacts
        """

        # Create run
        config = {
            "episode": str(ref),
            "phase": "keyframe",
            "video_propagation": False,
        }
        run_id = self.artifact_repo.create_run(config)

        # Load episode context
        state = self.episode_repo.load_state(ref)

        # Plan: get roles and queries
        semantic = self.semantic_planner.plan(ref)

        # Detect timeline
        timeline = self.timeline_detector.detect(ref, state)

        # Generate requests for each role
        requests = self.policy_registry.get_requests(semantic, timeline)

        # Process each request
        for request in requests:
            package = self._prepare_one_request(request)

            # Save artifacts
            data = {
                "request": self._serialize_request(request),
                "candidates": [self._serialize_candidate(c) for c in package.candidates],
                "grounding": package.grounding_evidence,
            }
            self.artifact_repo.save_request(run_id, request, data)

        return run_id

    def _prepare_one_request(self, request: KeyframeRequest) -> KeyframePackage:
        """Generate candidates for one keyframe request."""

        # Select candidate frames
        candidate_frames = self.keyframe_selector.select_candidates(
            request.episode,
            request.allowed_window,
            max_candidates=3,
        )

        if not candidate_frames:
            # No valid frames in window
            return KeyframePackage(
                request=request,
                candidates=[],
                grounding_evidence={"reason": "no_valid_frames"},
            )

        # Use the best frame for grounding
        best_frame_idx = candidate_frames[0]
        frame = self.frame_source.read_frame(request.episode, best_frame_idx)

        # Ground: get refined query + bbox
        refined_query, bbox = self.grounding_service.ground(frame, request.visual_query)

        # Generate candidates with different methods
        candidates: list[MaskCandidate] = []

        # Method 1: text only
        if refined_query:
            mask_text = self.segmenter.segment(
                frame,
                VisualPrompt(text=refined_query),
                SegmentationMethod.TEXT_ONLY,
            )
            candidates.append(
                MaskCandidate(
                    candidate_id=f"{request.slot.name}-r{request.revision:03d}-f{best_frame_idx:06d}-text_only",
                    frame_index=best_frame_idx,
                    method=SegmentationMethod.TEXT_ONLY,
                    query=refined_query,
                    bbox=None,
                    mask=mask_text,
                    area_fraction=float(mask_text.sum() / mask_text.size),
                )
            )

        # Method 2: box only
        mask_box = self.segmenter.segment(
            frame,
            VisualPrompt(bbox=bbox),
            SegmentationMethod.BOX_ONLY,
        )
        candidates.append(
            MaskCandidate(
                candidate_id=f"{request.slot.name}-r{request.revision:03d}-f{best_frame_idx:06d}-box_only",
                frame_index=best_frame_idx,
                method=SegmentationMethod.BOX_ONLY,
                query=refined_query,
                bbox=bbox,
                mask=mask_box,
                area_fraction=float(mask_box.sum() / mask_box.size),
            )
        )

        # Method 3: text + box (combined prompt)
        mask_text_box = self.segmenter.segment(
            frame,
            VisualPrompt(text=refined_query, bbox=bbox),
            SegmentationMethod.TEXT_BOX,
        )
        candidates.append(
            MaskCandidate(
                candidate_id=f"{request.slot.name}-r{request.revision:03d}-f{best_frame_idx:06d}-text_box",
                frame_index=best_frame_idx,
                method=SegmentationMethod.TEXT_BOX,
                query=refined_query,
                bbox=bbox,
                mask=mask_text_box,
                area_fraction=float(mask_text_box.sum() / mask_text_box.size),
            )
        )

        return KeyframePackage(
            request=request,
            candidates=candidates,
            grounding_evidence={
                "original_query": request.visual_query,
                "refined_query": refined_query,
                "bbox": (bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max),
                "frame_index": best_frame_idx,
            },
        )

    def _serialize_request(self, request: KeyframeRequest) -> dict[str, Any]:
        """Convert request to JSON-serializable dict."""
        return {
            "request_id": request.request_id,
            "episode": str(request.episode),
            "slot": request.slot.name,
            "anchor_kind": request.anchor_kind.value,
            "allowed_window": [request.allowed_window.first, request.allowed_window.last],
            "visual_query": request.visual_query,
            "revision": request.revision,
        }

    def _serialize_candidate(self, candidate: MaskCandidate) -> dict[str, Any]:
        """Convert candidate to JSON-serializable dict (mask saved separately)."""
        return {
            "candidate_id": candidate.candidate_id,
            "frame_index": candidate.frame_index,
            "method": candidate.method.value,
            "query": candidate.query,
            "bbox": (
                [candidate.bbox.x_min, candidate.bbox.y_min,
                 candidate.bbox.x_max, candidate.bbox.y_max]
                if candidate.bbox else None
            ),
            "area_fraction": candidate.area_fraction,
        }
