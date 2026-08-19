"""Gripper mask pipeline stage: 3-D ROI projection, candidate generation, and Qwen QC.

Merges the former ``experiments.gripper_pose_roi`` and
``experiments.gripper_seed_qc`` modules into a single pipeline stage. The two
modules previously depended on each other (candidate generation used the ROI
projection's object-exclusion helpers); that dependency is now a plain
same-module reference.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw

from ..adapters.qwen_client import QwenCompletion, image_data_url
from ..config import GripperRoiConfig
from ..domain import AnnotationMode, ObjectRole
from ..models.loop_context import EpisodeRef, LoopContext
from ..models.mask_qc import MaskQCStatus
from ..models.timeline import FrameWindow, LoopEvents
from .gripper.sam import candidates as _candidates
from .gripper.sam import composition as _composition
from .gripper.sam import geometry as _geometry
from .mask_qc import MaskQCError, parse_mask_qc_response

GripperSeedCandidate = _candidates.GripperSeedCandidate
GripperSeedQualityGateConfig = _candidates.GripperSeedQualityGateConfig
_component_metrics = _candidates._component_metrics
_tcp_distance = _candidates._tcp_distance
apply_gripper_seed_quality_gate = _candidates.apply_gripper_seed_quality_gate
build_gripper_seed_candidate = _candidates.build_gripper_seed_candidate
mark_same_frame_duplicates = _candidates.mark_same_frame_duplicates
phase_for_frame = _candidates.phase_for_frame
GripperTrackResult = _composition.GripperTrackResult
ObjectExclusionResult = _composition.ObjectExclusionResult
compose_gripper_track = _composition.compose_gripper_track
exclude_known_objects = _composition.exclude_known_objects
CAM_HIGH_CALIBRATION = _geometry.CAM_HIGH_CALIBRATION
DEFAULT_GRIPPER_ROI_GEOMETRY = _geometry.DEFAULT_GRIPPER_ROI_GEOMETRY
CameraCalibration = _geometry.CameraCalibration
GripperRoiGeometry = _geometry.GripperRoiGeometry
ProjectedGripperRoi = _geometry.ProjectedGripperRoi
_convex_hull = _geometry._convex_hull
_project_world_points = _geometry._project_world_points
normalized_roi_box = _geometry.normalized_roi_box
project_gripper_roi = _geometry.project_gripper_roi
rotation_from_rpy = _geometry.rotation_from_rpy


@dataclass(frozen=True)
class GripperStageResult:
    """Stage-level gripper output consumed by artifact persistence."""

    active_arm: str
    active_window: FrameWindow
    frame_count: int
    frame_shape: tuple[int, int]
    seed_frame_id: int | None
    selected_candidate: str | None
    seed_mask: np.ndarray | None
    native_track: np.ndarray
    roi_track: np.ndarray
    candidate_track: np.ndarray
    gripper_track: np.ndarray
    removed_track: np.ndarray
    target_removed_track: np.ndarray
    receiver_removed_track: np.ndarray
    prompt_rois: dict[int, ProjectedGripperRoi]
    hard_rois: dict[int, ProjectedGripperRoi]
    qc_result: GripperSeedQCResult
    candidate_panels: dict[str, Image.Image]
    roi_policy: dict[str, Any]
    provenance: dict[str, Any]
    failure: str | None = None

    def __post_init__(self) -> None:
        if self.active_arm not in {"left", "right"}:
            raise ValueError("active_arm must be left or right")
        stacks = (
            self.native_track,
            self.roi_track,
            self.candidate_track,
            self.gripper_track,
            self.removed_track,
            self.target_removed_track,
            self.receiver_removed_track,
        )
        expected = (self.frame_count, *self.frame_shape)
        if any(np.asarray(stack).shape != expected for stack in stacks):
            raise ValueError(f"gripper tracks must have shape {expected}")
        if (self.qc_result.status is MaskQCStatus.PASSED) != (
            self.seed_frame_id is not None and self.seed_mask is not None
        ):
            raise ValueError("passed gripper stage must carry the selected seed mask")
        if self.seed_mask is not None and np.asarray(self.seed_mask).shape != self.frame_shape:
            raise ValueError("gripper seed mask must match frame_shape")

    @property
    def instance_name(self) -> str:
        return f"gripper_{self.active_arm}"

    @property
    def nonempty_frame_ids(self) -> tuple[int, ...]:
        present = self.gripper_track.reshape(self.gripper_track.shape[0], -1).any(axis=1)
        return tuple(int(value) for value in np.flatnonzero(present))

    @property
    def status(self) -> str:
        if self.failure is not None:
            return "failed"
        if self.qc_result.status is MaskQCStatus.PASSED and self.gripper_track.any():
            return "ok"
        return "failed"


class GripperStageError(RuntimeError):
    """The gripper stage cannot execute its pose/seed/propagation contract."""


CYAN = np.asarray([15, 230, 185], dtype=np.uint8)
ORANGE = np.asarray([255, 126, 35], dtype=np.uint8)
BLUE = np.asarray([70, 105, 255], dtype=np.uint8)
YELLOW_RGB = (255, 240, 68)
RED_RGB = (255, 59, 59)


class GripperQwenClient(Protocol):
    model_id: str

    def health(self) -> dict[str, Any]: ...

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
    ) -> QwenCompletion: ...


class GripperSamBackend(Protocol):
    def box_mask(
        self,
        resource_path: Path,
        box_xyxy: Sequence[float],
        *,
        frame_id: int,
        frame_count: int,
        frame_shape: tuple[int, int],
    ) -> np.ndarray: ...

    def text_box_mask(
        self,
        resource_path: Path,
        text: str,
        box_xyxy: Sequence[float],
        *,
        frame_id: int,
        frame_count: int,
        frame_shape: tuple[int, int],
    ) -> np.ndarray: ...

    def propagate_mask(
        self,
        resource_path: Path,
        seed_mask: np.ndarray,
        *,
        seed_frame: int,
        frame_count: int,
        frame_shape: tuple[int, int],
        tracking_window: tuple[int, int],
        object_id: int = 1,
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class KnownObjectTracks:
    tracks: Mapping[ObjectRole, np.ndarray]
    seed_masks: Mapping[ObjectRole, np.ndarray]
    seed_frames: Mapping[ObjectRole, int]
    provenance: dict[str, Any]

    def track(self, role: ObjectRole, *, empty_like: np.ndarray | None = None) -> np.ndarray:
        value = self.tracks.get(role)
        if value is not None:
            return np.asarray(value, dtype=bool)
        if empty_like is None:
            raise KeyError(f"known object track has no applicable role {role.value}")
        return np.zeros_like(np.asarray(empty_like, dtype=bool))

    @property
    def target(self) -> np.ndarray:
        return self.track(ObjectRole.TARGET)

    @property
    def receiver(self) -> np.ndarray:
        return self.track(ObjectRole.RECEIVER, empty_like=self.target)

    @property
    def target_seed_mask(self) -> np.ndarray:
        return np.asarray(self.seed_masks[ObjectRole.TARGET], dtype=bool)

    @property
    def receiver_seed_mask(self) -> np.ndarray:
        return np.asarray(
            self.seed_masks.get(ObjectRole.RECEIVER, np.zeros_like(self.target_seed_mask)),
            dtype=bool,
        )

    @property
    def target_seed_frame(self) -> int:
        return self.seed_frames[ObjectRole.TARGET]

    @property
    def receiver_seed_frame(self) -> int | None:
        return self.seed_frames.get(ObjectRole.RECEIVER)


@dataclass(frozen=True)
class GripperSeedQCResult:
    status: MaskQCStatus
    selected_candidate: str | None
    confidence: float | None
    reason: str
    candidates: tuple[GripperSeedCandidate, ...]
    model: str | None = None
    raw_response: str | None = None
    rendered_prompt: str | None = None
    health: dict[str, Any] | None = None
    forced_fallback: bool = False

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("gripper QC reason must be non-empty")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("gripper QC confidence must be between 0 and 1")
        if (self.status is MaskQCStatus.PASSED) != (
            self.selected_candidate is not None
        ):
            raise ValueError("only passed gripper QC may select a candidate")
        if self.forced_fallback and self.selected_candidate is None:
            raise ValueError("a forced gripper fallback must select a candidate")

    @property
    def selected(self) -> GripperSeedCandidate | None:
        if self.selected_candidate is None:
            return None
        return next(
            candidate
            for candidate in self.candidates
            if candidate.candidate_id == self.selected_candidate
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "format_version": "robotwin_gripper_seed_qwen_qc_v1",
            "status": self.status.value,
            "selected_candidate": self.selected_candidate,
            "confidence": self.confidence,
            "reason": self.reason,
            "model": self.model,
            "raw_response": self.raw_response,
            "rendered_prompt": self.rendered_prompt,
            "health": self.health,
            "forced_fallback": self.forced_fallback,
            "candidates": [candidate.to_json() for candidate in self.candidates],
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_qc_native_object_tracks(
    run_root: Path,
    ref: EpisodeRef,
    *,
    expected_shape: tuple[int, int, int],
    required_roles: tuple[ObjectRole, ...] = (ObjectRole.TARGET, ObjectRole.RECEIVER),
) -> KnownObjectTracks:
    """Load identity-QC-passed full native tracks by structured role metadata."""

    episode_dir = (
        run_root.expanduser().resolve()
        / ref.task
        / f"episode_{ref.episode_id}"
        / ref.camera
    )
    manifest_path = episode_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"object run manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    episode = manifest.get("episode", {})
    expected_episode = {
        "task": ref.task,
        "episode_index": ref.episode_index,
        "episode_id": ref.episode_id,
        "camera": ref.camera,
    }
    for key, expected in expected_episode.items():
        if episode.get(key) != expected:
            raise ValueError(
                f"object manifest has {key}={episode.get(key)!r}, expected {expected!r}"
            )
    if manifest.get("frame_count") != expected_shape[0]:
        raise ValueError("object manifest frame_count does not match the episode")

    role_records = manifest.get("roles")
    if not isinstance(role_records, list):
        raise ValueError("object manifest roles must be a list")
    tracks: dict[str, np.ndarray] = {}
    seed_masks: dict[str, np.ndarray] = {}
    seed_frames: dict[str, int] = {}
    sources: dict[str, Any] = {}
    root = episode_dir.resolve()
    for object_role in required_roles:
        role = object_role.value
        matches = [record for record in role_records if record.get("role") == role]
        if len(matches) != 1:
            raise ValueError(f"object manifest must contain exactly one {role} role")
        record = matches[0]
        if record.get("status") != "ok" or record.get("qc_status") != "passed":
            raise ValueError(f"object manifest {role} is not status=ok and qc_status=passed")
        relative_path = record.get("native_track_path")
        if not isinstance(relative_path, str) or not relative_path.strip():
            raise ValueError(f"object manifest {role} native_track_path is invalid")
        track_path = (episode_dir / relative_path).resolve()
        try:
            track_path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"object manifest {role} track escapes episode directory") from exc
        if not track_path.is_file():
            raise FileNotFoundError(f"object native track does not exist: {track_path}")
        with np.load(track_path, allow_pickle=False) as archive:
            if "masks" not in archive.files:
                raise ValueError(f"object native track has no masks array: {track_path}")
            track = np.asarray(archive["masks"], dtype=bool)
        if track.shape != expected_shape:
            raise ValueError(
                f"object native track {track_path} has shape {track.shape}, "
                f"expected {expected_shape}"
            )
        tracks[role] = track
        seed_frame = record.get("seed_frame_id")
        seed_relative_path = record.get("seed_mask_path")
        if not isinstance(seed_frame, int) or not 0 <= seed_frame < expected_shape[0]:
            raise ValueError(f"object manifest {role} seed_frame_id is invalid")
        if not isinstance(seed_relative_path, str) or not seed_relative_path.strip():
            raise ValueError(f"object manifest {role} seed_mask_path is invalid")
        seed_path = (episode_dir / seed_relative_path).resolve()
        try:
            seed_path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"object manifest {role} seed escapes episode directory") from exc
        if not seed_path.is_file():
            raise FileNotFoundError(f"object seed mask does not exist: {seed_path}")
        with Image.open(seed_path) as image:
            seed_mask = np.asarray(image.convert("L"), dtype=np.uint8) != 0
        if seed_mask.shape != expected_shape[1:] or not seed_mask.any():
            raise ValueError(f"object {role} seed mask is empty or has the wrong shape")
        seed_masks[role] = seed_mask
        seed_frames[role] = seed_frame
        sources[role] = {
            "native_track_path": str(track_path),
            "native_track_sha256": _sha256(track_path),
            "seed_frame_id": seed_frame,
            "seed_mask_path": str(seed_path),
            "seed_mask_sha256": _sha256(seed_path),
            "seed_pixels": int(seed_mask.sum()),
            "primary_query": record.get("primary_query"),
            "qc_selected_candidate": record.get("qc_selected_candidate"),
            "qc_reason": record.get("qc_reason"),
        }
    return KnownObjectTracks(
        tracks={ObjectRole(role): track for role, track in tracks.items()},
        seed_masks={ObjectRole(role): mask for role, mask in seed_masks.items()},
        seed_frames={ObjectRole(role): frame for role, frame in seed_frames.items()},
        provenance={
            "run_id": manifest.get("run_id"),
            "manifest": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "roles": sources,
        },
    )


def gripper_keyframes(
    rois: Mapping[int, ProjectedGripperRoi],
    events: LoopEvents,
    *,
    frame_shape: tuple[int, int],
) -> tuple[int, ...]:
    """Return the same seven state-derived review frames used by the experiment."""

    height, width = frame_shape
    visible = [
        frame_id
        for frame_id, roi in sorted(rois.items())
        if 0 <= roi.tcp_pixel_xy[0] < width and 0 <= roi.tcp_pixel_xy[1] < height
    ]
    first_tcp = visible[0] if visible else events.t_close_start
    values = (
        first_tcp,
        max(first_tcp, events.t_close_start - 1),
        events.t_close_done,
        (events.t_close_done + events.t_open_start) // 2,
        max(events.t_close_done + 1, events.t_open_start - 1),
        events.t_open_start,
        events.t_open_done,
    )
    return tuple(dict.fromkeys(int(value) for value in values))


def _outline(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    value = np.asarray(mask, dtype=np.uint8)
    if not value.any():
        return value.astype(bool)
    kernel_size = 2 * radius + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    dilated = cv2.dilate(value, kernel, iterations=1)
    eroded = cv2.erode(value, np.ones((3, 3), dtype=np.uint8), iterations=1)
    return (dilated != eroded)


def render_gripper_candidate_panel(
    rgb: np.ndarray | Image.Image,
    candidate: GripperSeedCandidate,
    roi: ProjectedGripperRoi,
) -> Image.Image:
    source = np.asarray(
        rgb.convert("RGB") if isinstance(rgb, Image.Image) else rgb,
        dtype=np.uint8,
    )
    if source.shape != (*candidate.clean_mask.shape, 3):
        raise ValueError("candidate panel RGB shape does not match candidate mask")
    canvas = source.copy()
    canvas[_outline(candidate.target_removed)] = ORANGE
    canvas[_outline(candidate.receiver_removed)] = BLUE
    canvas[_outline(candidate.clean_mask)] = CYAN
    polygon = np.rint(roi.hull_pixels_xy).astype(np.int32).reshape(-1, 1, 2)
    if len(polygon) >= 3:
        cv2.polylines(canvas, [polygon], True, YELLOW_RGB, 1, cv2.LINE_AA)
    x, y = np.rint(roi.tcp_pixel_xy).astype(int)
    cv2.line(canvas, (x - 4, y), (x + 4, y), RED_RGB, 1, cv2.LINE_AA)
    cv2.line(canvas, (x, y - 4), (x, y + 4), RED_RGB, 1, cv2.LINE_AA)
    panel = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(panel)
    draw.rectangle((0, 0, panel.width, 20), fill=(0, 0, 0))
    text = (
        f"{candidate.candidate_id} f{candidate.frame_id} {candidate.phase} "
        f"{candidate.prompt_mode} clean={candidate.clean_pixels}"
    )
    draw.text((4, 4), text, fill=(255, 255, 255))
    return panel


def render_gripper_candidate_sheet(
    candidates: Sequence[GripperSeedCandidate],
    panels: Mapping[str, Image.Image],
    *,
    columns: int = 4,
) -> Image.Image:
    if columns < 1 or not candidates:
        raise ValueError("candidate sheet requires candidates and positive columns")
    panel_width, panel_height = next(iter(panels.values())).size
    rows = math.ceil(len(candidates) / columns)
    sheet = Image.new("RGB", (panel_width * columns, panel_height * rows), "#111111")
    for index, candidate in enumerate(candidates):
        sheet.paste(
            panels[candidate.candidate_id],
            ((index % columns) * panel_width, (index // columns) * panel_height),
        )
    return sheet


def _candidate_record(candidate: GripperSeedCandidate) -> str:
    dark = "null" if candidate.dark_fraction is None else f"{candidate.dark_fraction:.3f}"
    tcp = "null" if candidate.tcp_distance_px is None else f"{candidate.tcp_distance_px:.1f}"
    return (
        f"- {candidate.candidate_id}: frame={candidate.frame_id}, phase={candidate.phase}, "
        f"method={candidate.prompt_mode}, clean_pixels={candidate.clean_pixels}, "
        f"target_removed={candidate.target_removed_pixels}, "
        f"receiver_removed={candidate.receiver_removed_pixels}, "
        f"dark_fraction={dark}, tcp_distance_px={tcp}"
    )


def _fallback_candidate(candidates: Sequence[GripperSeedCandidate]) -> GripperSeedCandidate:
    """Select a deterministic least-bad seed when Qwen cannot decide.

    The experiment intentionally keeps this fallback simple and auditable.  A
    basic-valid candidate wins over an invalid one, then candidates with more
    usable pixels, a larger connected component, and a darker visible region
    are preferred.  The final terms discourage fragmented masks and prefer a
    component close to the projected TCP.  ``max`` is stable for ties, so the
    original candidate order remains the last tie breaker.
    """

    if not candidates:
        raise ValueError("gripper fallback requires at least one candidate")

    def score(candidate: GripperSeedCandidate) -> tuple[float, ...]:
        return (
            float(candidate.basic_valid),
            float(candidate.clean_pixels),
            float(candidate.largest_component_fraction or 0.0),
            float(candidate.dark_fraction or 0.0),
            float(-candidate.component_count),
            float(-(candidate.tcp_distance_px or 1e9)),
        )

    return max(candidates, key=score)


def build_gripper_qwen_request(
    context: LoopContext,
    candidates: Sequence[GripperSeedCandidate],
    panels: Mapping[str, Image.Image],
    context_images: Mapping[int, Image.Image],
    *,
    prompt_template: str,
) -> tuple[str, list[dict[str, Any]]]:
    valid = tuple(candidate for candidate in candidates if candidate.basic_valid)
    if not valid:
        raise ValueError("Qwen request requires at least one valid gripper candidate")
    candidate_marker = "{candidate_panels}"
    context_marker = "{context_frames}"
    if prompt_template.count(candidate_marker) != 1:
        raise ValueError("gripper QC template must contain {candidate_panels} once")
    if prompt_template.count(context_marker) != 1:
        raise ValueError("gripper QC template must contain {context_frames} once")
    records = "\n".join(_candidate_record(candidate) for candidate in valid)
    context_ids = tuple(sorted(context_images))
    replacements = {
        "task_text": context.task_text,
        "active_arm": context.events.active_arm,
        "candidate_ids": ", ".join(candidate.candidate_id for candidate in valid),
        "candidate_records": records,
        "move_start": str(context.events.t_move_start),
        "close_start": str(context.events.t_close_start),
        "close_done": str(context.events.t_close_done),
        "open_start": str(context.events.t_open_start),
        "open_done": str(context.events.t_open_done),
    }
    partial = prompt_template
    for key, value in replacements.items():
        partial = partial.replace(f"{{{key}}}", value)
    candidate_prefix, remainder = partial.split(candidate_marker)
    candidate_suffix, context_suffix = remainder.split(context_marker)
    content: list[dict[str, Any]] = [{"type": "text", "text": candidate_prefix}]
    for candidate in valid:
        content.append(
            {
                "type": "text",
                "text": (
                    f"[candidate {candidate.candidate_id}; frame={candidate.frame_id}; "
                    f"phase={candidate.phase}; method={candidate.prompt_mode}]"
                ),
            }
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_data_url(panels[candidate.candidate_id])},
            }
        )
    content.append({"type": "text", "text": candidate_suffix})
    for frame_id in context_ids:
        content.append(
            {"type": "text", "text": f"[raw context frame_id={frame_id}]"}
        )
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_data_url(context_images[frame_id])},
            }
        )
    content.append({"type": "text", "text": context_suffix})
    rendered = partial.replace(
        candidate_marker,
        "[candidate images: " + replacements["candidate_ids"] + "]",
    ).replace(
        context_marker,
        "[raw context images: " + ", ".join(map(str, context_ids)) + "]",
    )
    return rendered, [{"role": "user", "content": content}]


def run_gripper_seed_qc(
    context: LoopContext,
    candidates: Sequence[GripperSeedCandidate],
    panels: Mapping[str, Image.Image],
    context_images: Mapping[int, Image.Image],
    *,
    prompt_template_path: Path,
    client: GripperQwenClient,
    max_tokens: int = 200,
    max_attempts: int = 2,
    minimum_confidence: float = 0.70,
) -> GripperSeedQCResult:
    candidate_tuple = tuple(candidates)
    valid = tuple(candidate for candidate in candidate_tuple if candidate.basic_valid)

    def forced_result(
        reason: str,
        *,
        confidence: float | None = None,
        model: str | None = None,
        raw_response: str | None = None,
        rendered_prompt: str | None = None,
        health: dict[str, Any] | None = None,
    ) -> GripperSeedQCResult:
        pool = valid or candidate_tuple
        if not pool:
            return GripperSeedQCResult(
                status=MaskQCStatus.REJECTED,
                selected_candidate=None,
                confidence=confidence,
                reason=reason,
                candidates=candidate_tuple,
                model=model,
                raw_response=raw_response,
                rendered_prompt=rendered_prompt,
                health=health,
            )
        selected = _fallback_candidate(pool)
        return GripperSeedQCResult(
            status=MaskQCStatus.PASSED,
            selected_candidate=selected.candidate_id,
            confidence=confidence,
            reason=(
                f"forced fallback candidate {selected.candidate_id}; {reason}"
            ),
            candidates=candidate_tuple,
            model=model,
            raw_response=raw_response,
            rendered_prompt=rendered_prompt,
            health=health,
            forced_fallback=True,
        )

    if not valid:
        return forced_result(
            "all_gripper_candidates_failed_basic_checks",
        )
    if max_tokens < 1 or max_attempts < 1:
        raise ValueError("gripper QC token and attempt limits must be positive")
    if not 0.0 <= minimum_confidence <= 1.0:
        raise ValueError("gripper QC minimum confidence must be in [0, 1]")
    if not prompt_template_path.is_file():
        return forced_result(
            f"gripper QC prompt is missing: {prompt_template_path}",
        )
    try:
        health = client.health()
    except Exception as exc:
        return forced_result(
            f"gripper QC health failed: {exc}",
        )
    rendered, messages = build_gripper_qwen_request(
        context,
        valid,
        panels,
        context_images,
        prompt_template=prompt_template_path.read_text(encoding="utf-8"),
    )
    completion: QwenCompletion | None = None
    request_error: Exception | None = None
    for _attempt in range(max_attempts):
        try:
            completion = client.complete(messages, max_tokens=max_tokens)
            break
        except Exception as exc:
            request_error = exc
    if completion is None:
        assert request_error is not None
        return forced_result(
            f"gripper QC request failed after {max_attempts} attempt(s): {request_error}",
            rendered_prompt=rendered,
            health=health,
        )
    try:
        decision, selected, confidence, reason = parse_mask_qc_response(
            completion.content,
            candidate_ids=tuple(candidate.candidate_id for candidate in valid),
        )
    except MaskQCError as exc:
        return forced_result(
            str(exc),
            model=completion.model,
            raw_response=completion.content,
            rendered_prompt=rendered,
            health=health,
        )
    if decision == "reject_all":
        return forced_result(
            f"Qwen decision was reject_all: {reason}",
            confidence=confidence,
            model=completion.model,
            raw_response=completion.content,
            rendered_prompt=rendered,
            health=health,
        )
    elif decision == "ambiguous":
        return forced_result(
            f"Qwen decision was ambiguous: {reason}",
            confidence=confidence,
            model=completion.model,
            raw_response=completion.content,
            rendered_prompt=rendered,
            health=health,
        )
    elif confidence < minimum_confidence:
        return forced_result(
            f"Qwen confidence {confidence:.3f} is below minimum "
            f"{minimum_confidence:.3f}: {reason}",
            confidence=confidence,
            model=completion.model,
            raw_response=completion.content,
            rendered_prompt=rendered,
            health=health,
        )
    return GripperSeedQCResult(
        status=MaskQCStatus.PASSED,
        selected_candidate=selected,
        confidence=confidence,
        reason=reason,
        candidates=candidate_tuple,
        model=completion.model,
        raw_response=completion.content,
        rendered_prompt=rendered,
        health=health,
        forced_fallback=False,
    )


def _load_state_arrays(context: LoopContext) -> tuple[np.ndarray, np.ndarray]:
    """Load the state columns needed for pose ROI projection.

    Stage 1 deliberately keeps LoopContext JSON-safe, so the gripper stage
    resolves the already-recorded state source here instead of duplicating
    state arrays in every artifact.
    """

    try:
        import pandas as pd

        frame = pd.read_parquet(
            Path(context.state_source),
            columns=["frame_index", "episode_index", "observation.state"],
        )
    except Exception as exc:
        raise GripperStageError(
            f"failed to read gripper state source {context.state_source}: {exc}"
        ) from exc
    if frame.empty:
        raise GripperStageError("gripper state source is empty")
    frame_indices = frame["frame_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(frame_indices, np.arange(len(frame_indices))):
        raise GripperStageError("gripper state frame_index must be contiguous and zero-based")
    episode_indices = frame["episode_index"].to_numpy(dtype=np.int64)
    if not np.all(episode_indices == context.episode.episode_index):
        raise GripperStageError("gripper state episode_index does not match the context")
    state = np.stack(frame["observation.state"].to_numpy()).astype(np.float64)
    if state.shape != (context.frame_count, 14):
        raise GripperStageError(
            f"gripper state must have shape {(context.frame_count, 14)}, got {state.shape}"
        )
    if not np.isfinite(state).all():
        raise GripperStageError("gripper state contains non-finite values")
    gripper_states = state[:, (6, 13)]
    eef_states = np.stack((state[:, 0:6], state[:, 7:13]), axis=1)
    return eef_states, gripper_states


def _roi_geometries(
    config: GripperRoiConfig,
) -> tuple[GripperRoiGeometry, GripperRoiGeometry]:
    width_overrides = {
        "closed_half_width_m": config.fixed_half_width_m,
        "open_half_width_m": config.fixed_half_width_m,
    }
    prompt = replace(
        DEFAULT_GRIPPER_ROI_GEOMETRY,
        axial_back_m=config.prompt_axial_back_m,
        axial_front_m=config.prompt_axial_front_m,
        **width_overrides,
    )
    hard = replace(
        DEFAULT_GRIPPER_ROI_GEOMETRY,
        axial_back_m=config.hard_axial_back_m,
        axial_front_m=config.hard_axial_front_m,
        **width_overrides,
    )
    return prompt, hard


def _roi_policy(config: GripperRoiConfig) -> dict[str, Any]:
    prompt, hard = _roi_geometries(config)
    return {
        "prompt": {
            "geometry": asdict(prompt),
            "usage": "SAM text-box/box-only candidate and selected-seed crop",
        },
        "hard": {
            "geometry": asdict(hard),
            "usage": "propagated native track crop before known-object exclusion",
        },
        "legacy_roi_track_alias": "hard_roi_track",
    }


def _polygon_mask(roi: ProjectedGripperRoi, frame_shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(frame_shape, dtype=np.uint8)
    polygon = np.rint(roi.hull_pixels_xy).astype(np.int32)
    if len(polygon) >= 3:
        cv2.fillConvexPoly(mask, polygon, 1)
    return mask.astype(bool)


def _build_roi_track(
    eef_states: np.ndarray,
    gripper_states: np.ndarray,
    events: LoopEvents,
    *,
    frame_shape: tuple[int, int],
    geometry: GripperRoiGeometry,
) -> tuple[np.ndarray, dict[int, ProjectedGripperRoi]]:
    eef = np.asarray(eef_states, dtype=np.float64)
    gripper = np.asarray(gripper_states, dtype=np.float64)
    if eef.ndim != 3 or eef.shape[1:] != (2, 6):
        raise GripperStageError(f"eef_states must have [T,2,6] shape, got {eef.shape}")
    if gripper.shape != (eef.shape[0], 2):
        raise GripperStageError(
            f"gripper_states must have shape {(eef.shape[0], 2)}, got {gripper.shape}"
        )
    height, width = frame_shape
    if height < 1 or width < 1:
        raise GripperStageError(f"invalid frame_shape: {frame_shape}")
    arm_index = 0 if events.active_arm == "left" else 1
    track = np.zeros((eef.shape[0], height, width), dtype=bool)
    rois: dict[int, ProjectedGripperRoi] = {}
    for frame_id in range(events.t_move_start, events.t_open_done + 1):
        try:
            roi = project_gripper_roi(
                eef[frame_id, arm_index],
                gripper[frame_id, arm_index],
                geometry=geometry,
            )
        except (IndexError, ValueError) as exc:
            raise GripperStageError(
                f"failed to project gripper ROI at frame {frame_id}: {exc}"
            ) from exc
        track[frame_id] = _polygon_mask(roi, frame_shape)
        rois[frame_id] = roi
    return track, rois


def _load_resource_image(resource_path: Path, frame_id: int) -> Image.Image:
    image_path = resource_path / f"{frame_id:06d}.jpg"
    if not image_path.is_file():
        raise GripperStageError(f"decoded SAM3 frame is missing: {image_path}")
    try:
        with Image.open(image_path) as image:
            return image.convert("RGB")
    except OSError as exc:
        raise GripperStageError(f"failed to read decoded frame {image_path}: {exc}") from exc


def _context_frame_ids(
    keyframes: Sequence[int],
    events: LoopEvents,
    frame_count: int,
) -> tuple[int, ...]:
    values = (
        *keyframes,
        events.t_close_done,
        (events.t_close_done + events.t_open_start) // 2,
        events.t_open_done,
    )
    return tuple(
        dict.fromkeys(
            int(value)
            for value in values
            if 0 <= int(value) < frame_count
        )
    )


def _build_gripper_candidates(
    backend: GripperSamBackend,
    resource_path: Path,
    *,
    frame_images: Mapping[int, Image.Image],
    keyframes: Sequence[int],
    prompt_rois: Mapping[int, ProjectedGripperRoi],
    prompt_roi_track: np.ndarray,
    target_track: np.ndarray,
    receiver_track: np.ndarray,
    events: LoopEvents,
    frame_count: int,
    frame_shape: tuple[int, int],
    gripper_text: str,
    gate: GripperSeedQualityGateConfig,
) -> tuple[GripperSeedCandidate, ...]:
    def build_bank(
        prompt_mode: str,
        prompt_text: str | None,
        *,
        first_index: int,
    ) -> list[GripperSeedCandidate]:
        bank: list[GripperSeedCandidate] = []
        for frame_id in keyframes:
            roi = prompt_rois.get(frame_id)
            if roi is None:
                continue
            box = normalized_roi_box(roi, frame_shape)
            if box is None:
                continue
            if prompt_text is None:
                raw = backend.box_mask(
                    resource_path,
                    box,
                    frame_id=frame_id,
                    frame_count=frame_count,
                    frame_shape=frame_shape,
                )
            else:
                raw = backend.text_box_mask(
                    resource_path,
                    prompt_text,
                    box,
                    frame_id=frame_id,
                    frame_count=frame_count,
                    frame_shape=frame_shape,
                )
            candidate = build_gripper_seed_candidate(
                candidate_id=chr(ord("A") + first_index + len(bank)),
                frame_id=frame_id,
                events=events,
                prompt_mode=prompt_mode,
                prompt_text=prompt_text,
                raw_mask=raw,
                roi_mask=prompt_roi_track[frame_id],
                target_mask=target_track[frame_id],
                receiver_mask=receiver_track[frame_id],
                rgb=np.asarray(frame_images[frame_id], dtype=np.uint8),
                tcp_pixel_xy=roi.tcp_pixel_xy,
                minimum_pixels=gate.minimum_pixels,
            )
            bank.append(
                apply_gripper_seed_quality_gate(
                    candidate,
                    minimum_dark_fraction=gate.minimum_dark_fraction,
                    maximum_components=gate.maximum_components,
                    minimum_largest_component_fraction=(
                        gate.minimum_largest_component_fraction
                    ),
                    maximum_tcp_distance_px=gate.maximum_tcp_distance_px,
                )
            )
        return bank

    text_candidates = build_bank(
        "text_box",
        gripper_text,
        first_index=0,
    )
    candidates = list(text_candidates)
    if not any(candidate.basic_valid for candidate in text_candidates):
        candidates.extend(
            build_bank(
                "box_only",
                None,
                first_index=len(text_candidates),
            )
        )
    return mark_same_frame_duplicates(
        candidates,
        iou_threshold=gate.duplicate_iou_threshold,
    )


def _track_summary(track: np.ndarray, window: FrameWindow) -> dict[str, Any]:
    value = np.asarray(track, dtype=bool)[window.start : window.end + 1]
    areas = value.reshape(value.shape[0], -1).sum(axis=1)
    present = areas > 0
    return {
        "window_inclusive": window.to_json(),
        "window_frames": len(value),
        "nonempty_frames": int(present.sum()),
        "coverage": float(present.mean()),
        "pixels_min_nonempty": (
            None if not present.any() else int(areas[present].min())
        ),
        "pixels_median_nonempty": (
            None if not present.any() else float(np.median(areas[present]))
        ),
        "pixels_max": int(areas.max()) if areas.size else 0,
    }


def run_gripper_stage(
    context: LoopContext,
    *,
    backend: GripperSamBackend,
    resource_path: Path,
    frame_shape: tuple[int, int],
    gripper_roi_config: GripperRoiConfig,
    object_tracks: Mapping[ObjectRole, np.ndarray],
    qc_client: GripperQwenClient,
    qc_prompt_template: Path,
    qc_max_tokens: int = 220,
    qc_max_attempts: int = 2,
    qc_min_confidence: float = 0.70,
    seed_quality_gate: GripperSeedQualityGateConfig | None = None,
    gripper_text: str = "black robot gripper",
) -> GripperStageResult:
    """Run pose-ROI candidate selection and one native gripper propagation."""

    if context.annotation_mode is AnnotationMode.TARGET_ONLY:
        raise GripperStageError(
            "target_only does not support the SAM gripper backend; use URDF"
        )
    gate = seed_quality_gate or GripperSeedQualityGateConfig()
    expected_shape = (context.frame_count, *frame_shape)
    expected_roles = context.annotation_spec.required_object_roles
    if set(object_tracks) != set(expected_roles):
        raise GripperStageError(
            "object_tracks roles must exactly match annotation mode: "
            f"expected={[role.value for role in expected_roles]}, "
            f"actual={[role.value for role in object_tracks]}"
        )
    normalized_tracks = {
        role: np.asarray(track, dtype=bool) for role, track in object_tracks.items()
    }
    if any(track.shape != expected_shape for track in normalized_tracks.values()):
        raise GripperStageError(f"every object track must have shape {expected_shape}")
    target = normalized_tracks[ObjectRole.TARGET]
    receiver = normalized_tracks.get(ObjectRole.RECEIVER, np.zeros(expected_shape, dtype=bool))
    if not resource_path.is_dir():
        raise GripperStageError(f"SAM3 resource directory does not exist: {resource_path}")

    eef_states, gripper_states = _load_state_arrays(context)
    _prompt_geometry, hard_geometry = _roi_geometries(gripper_roi_config)
    prompt_track, prompt_rois = _build_roi_track(
        eef_states,
        gripper_states,
        context.events,
        frame_shape=frame_shape,
        geometry=_prompt_geometry,
    )
    hard_track, hard_rois = _build_roi_track(
        eef_states,
        gripper_states,
        context.events,
        frame_shape=frame_shape,
        geometry=hard_geometry,
    )
    keyframes = gripper_keyframes(
        prompt_rois,
        context.events,
        frame_shape=frame_shape,
    )
    context_ids = _context_frame_ids(keyframes, context.events, context.frame_count)
    frame_images = {
        frame_id: _load_resource_image(resource_path, frame_id)
        for frame_id in context_ids
    }
    candidates = _build_gripper_candidates(
        backend,
        resource_path,
        frame_images=frame_images,
        keyframes=keyframes,
        prompt_rois=prompt_rois,
        prompt_roi_track=prompt_track,
        target_track=target,
        receiver_track=receiver,
        events=context.events,
        frame_count=context.frame_count,
        frame_shape=frame_shape,
        gripper_text=gripper_text,
        gate=gate,
    )
    panels = {
        candidate.candidate_id: render_gripper_candidate_panel(
            frame_images[candidate.frame_id],
            candidate,
            prompt_rois[candidate.frame_id],
        )
        for candidate in candidates
    }
    context_images = {
        frame_id: frame_images[frame_id]
        for frame_id in _context_frame_ids(
            keyframes,
            context.events,
            context.frame_count,
        )
    }
    qc = run_gripper_seed_qc(
        context,
        candidates,
        panels,
        context_images,
        prompt_template_path=qc_prompt_template,
        client=qc_client,
        max_tokens=qc_max_tokens,
        max_attempts=qc_max_attempts,
        minimum_confidence=qc_min_confidence,
    )
    active_window = FrameWindow(
        context.events.t_move_start,
        context.events.t_open_done,
    )
    zeros = np.zeros(expected_shape, dtype=bool)
    policy = _roi_policy(gripper_roi_config)
    provenance = {
        "keyframes": list(keyframes),
        "context_frame_ids": list(context_images),
        "candidate_count": len(candidates),
        "known_object_tracks": "saved_sam_native_track",
        "target_track": _track_summary(target, active_window),
        "receiver_track": _track_summary(receiver, active_window),
        "quality_gate": asdict(gate),
    }
    selected = qc.selected
    if qc.status is not MaskQCStatus.PASSED or selected is None:
        return GripperStageResult(
            active_arm=context.events.active_arm,
            active_window=active_window,
            frame_count=context.frame_count,
            frame_shape=frame_shape,
            seed_frame_id=None,
            selected_candidate=None,
            seed_mask=None,
            native_track=zeros,
            roi_track=hard_track,
            candidate_track=zeros,
            gripper_track=zeros,
            removed_track=zeros,
            target_removed_track=zeros,
            receiver_removed_track=zeros,
            prompt_rois=dict(prompt_rois),
            hard_rois=dict(hard_rois),
            qc_result=qc,
            candidate_panels=panels,
            roi_policy=policy,
            provenance=provenance,
            failure=f"gripper_seed_qc_{qc.status.value}:{qc.reason}",
        )
    if not selected.clean_mask.any():
        return GripperStageResult(
            active_arm=context.events.active_arm,
            active_window=active_window,
            frame_count=context.frame_count,
            frame_shape=frame_shape,
            seed_frame_id=selected.frame_id,
            selected_candidate=selected.candidate_id,
            seed_mask=selected.clean_mask,
            native_track=zeros,
            roi_track=hard_track,
            candidate_track=zeros,
            gripper_track=zeros,
            removed_track=zeros,
            target_removed_track=zeros,
            receiver_removed_track=zeros,
            prompt_rois=dict(prompt_rois),
            hard_rois=dict(hard_rois),
            qc_result=qc,
            candidate_panels=panels,
            roi_policy=policy,
            provenance=provenance,
            failure="selected_gripper_seed_is_empty",
        )

    native = np.asarray(
        backend.propagate_mask(
            resource_path,
            selected.clean_mask,
            seed_frame=selected.frame_id,
            frame_count=context.frame_count,
            frame_shape=frame_shape,
            tracking_window=(active_window.start, active_window.end),
        ),
        dtype=bool,
    )
    if native.shape != expected_shape:
        raise GripperStageError(
            f"gripper native track has shape {native.shape}, expected {expected_shape}"
        )
    composed = compose_gripper_track(
        native,
        hard_track,
        target,
        receiver,
        active_window=(active_window.start, active_window.end),
    )
    failure = None if composed.gripper_mask.any() else "gripper_track_empty_after_constraints"
    provenance.update(
        {
            "selected_candidate": selected.candidate_id,
            "seed_frame_id": selected.frame_id,
            "native_track": _track_summary(native, active_window),
            "final_track": _track_summary(composed.gripper_mask, active_window),
        }
    )
    return GripperStageResult(
        active_arm=context.events.active_arm,
        active_window=active_window,
        frame_count=context.frame_count,
        frame_shape=frame_shape,
        seed_frame_id=selected.frame_id,
        selected_candidate=selected.candidate_id,
        seed_mask=selected.clean_mask,
        native_track=native,
        roi_track=composed.roi_mask,
        candidate_track=composed.candidate_mask,
        gripper_track=composed.gripper_mask,
        removed_track=composed.removed_mask,
        target_removed_track=composed.target_removed,
        receiver_removed_track=composed.receiver_removed,
        prompt_rois=dict(prompt_rois),
        hard_rois=dict(hard_rois),
        qc_result=qc,
        candidate_panels=panels,
        roi_policy=policy,
        provenance=provenance,
        failure=failure,
    )
