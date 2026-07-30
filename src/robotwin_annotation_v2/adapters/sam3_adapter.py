"""Minimal SAM3 adapter for text measurements and native-mask tracking."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np


class Sam3Error(RuntimeError):
    """SAM3 cannot load, accept a prompt, or produce a valid mask."""


def extract_video_frames(
    video_path: Path,
    frame_dir: Path,
    *,
    minimum_frame_count: int,
) -> tuple[Path, ...]:
    """Decode RoboTwin AV1 video into the JPEG directory expected by SAM3."""

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise Sam3Error("ffmpeg is required to decode RoboTwin AV1 videos")
    if not video_path.is_file():
        raise Sam3Error(f"video does not exist: {video_path}")
    frame_dir.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-vsync",
        "0",
        "-q:v",
        "2",
        "-start_number",
        "0",
        str(frame_dir / "%06d.jpg"),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise Sam3Error(f"ffmpeg failed: {exc.stderr.strip()}") from exc
    frames = tuple(sorted(frame_dir.glob("*.jpg")))
    if len(frames) < minimum_frame_count:
        raise Sam3Error(
            f"decoded {len(frames)} frames, expected at least {minimum_frame_count}"
        )
    return frames


@contextmanager
def sam3_video_resource(
    video_path: Path,
    *,
    minimum_frame_count: int,
    temp_root: Path | None = None,
) -> Iterator[Path]:
    """Yield a temporary JPEG resource and remove it after SAM3 finishes."""

    with tempfile.TemporaryDirectory(
        prefix=f"robotwin-sam3-{video_path.stem}-",
        dir=None if temp_root is None else str(temp_root),
    ) as temporary:
        frame_dir = Path(temporary)
        extract_video_frames(
            video_path,
            frame_dir,
            minimum_frame_count=minimum_frame_count,
        )
        yield frame_dir


def _primary_mask(outputs: Mapping[str, Any], shape: tuple[int, int]) -> np.ndarray:
    object_ids = np.asarray(outputs.get("out_obj_ids", [])).reshape(-1)
    masks = np.asarray(outputs.get("out_binary_masks", []))
    if object_ids.size == 0 or masks.ndim < 3:
        return np.zeros(shape, dtype=bool)
    probabilities = np.asarray(
        outputs.get("out_probs", np.ones(object_ids.shape)),
    ).reshape(-1)
    if probabilities.size != object_ids.size:
        probabilities = np.ones(object_ids.shape)
    index = int(np.argmax(probabilities))
    if index >= masks.shape[0]:
        return np.zeros(shape, dtype=bool)
    mask = np.asarray(masks[index])
    if mask.shape != shape:
        raise Sam3Error(f"SAM3 mask shape {mask.shape} does not match {shape}")
    return mask.astype(bool, copy=False)


def _mask_for_object(
    outputs: Mapping[str, Any],
    *,
    object_id: int,
    shape: tuple[int, int],
) -> np.ndarray | None:
    object_ids = np.asarray(outputs.get("out_obj_ids", [])).reshape(-1)
    matches = np.flatnonzero(object_ids == object_id)
    masks = np.asarray(outputs.get("out_binary_masks", []))
    if matches.size == 0 or masks.ndim < 3:
        return None
    index = int(matches[0])
    if index >= masks.shape[0]:
        return None
    mask = np.asarray(masks[index])
    if mask.shape != shape:
        raise Sam3Error(f"SAM3 mask shape {mask.shape} does not match {shape}")
    return mask.astype(bool, copy=False)


def _interior_point(mask: np.ndarray) -> list[float]:
    import cv2

    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    row, column = np.unravel_index(int(np.argmax(distance)), distance.shape)
    height, width = mask.shape
    return [
        (float(column) + 0.5) / float(width),
        (float(row) + 0.5) / float(height),
    ]


class Sam3Adapter:
    """One thin wrapper around the pinned SAM3 video predictor."""

    def __init__(
        self,
        *,
        checkpoint_path: Path,
        gpus: Sequence[int],
        predictor: Any | None = None,
    ) -> None:
        if len(gpus) != 1:
            raise ValueError("native mask propagation requires exactly one SAM3 GPU")
        if predictor is None:
            if not checkpoint_path.is_file():
                raise Sam3Error(f"SAM3 checkpoint does not exist: {checkpoint_path}")
            try:
                import torch

                if not torch.cuda.is_available():
                    raise Sam3Error("CUDA is unavailable to the SAM3 process")
                if gpus[0] >= torch.cuda.device_count():
                    raise Sam3Error(
                        f"SAM3 GPU {gpus[0]} is outside visible CUDA devices"
                    )
                from sam3.model_builder import build_sam3_video_predictor
            except ImportError as exc:
                raise Sam3Error("SAM3 is not installed in this environment") from exc
            try:
                predictor = build_sam3_video_predictor(
                    gpus_to_use=list(gpus),
                    checkpoint_path=str(checkpoint_path),
                )
            except (AssertionError, RuntimeError) as exc:
                raise Sam3Error(f"failed to load SAM3: {exc}") from exc
        self.predictor = predictor

    @contextmanager
    def _session(self, resource_path: Path) -> Iterator[str]:
        response = self.predictor.handle_request(
            request={"type": "start_session", "resource_path": str(resource_path)}
        )
        session_id = response["session_id"]
        try:
            yield session_id
        finally:
            self.predictor.handle_request(
                request={"type": "close_session", "session_id": session_id}
            )

    def text_masks(
        self,
        resource_path: Path,
        text: str,
        *,
        frame_ids: Sequence[int],
        frame_count: int,
        frame_shape: tuple[int, int],
    ) -> dict[int, np.ndarray]:
        if not text.strip():
            raise ValueError("SAM3 text must be non-empty")
        frames = tuple(dict.fromkeys(int(value) for value in frame_ids))
        if any(frame < 0 or frame >= frame_count for frame in frames):
            raise ValueError(f"SAM3 text frame is outside [0, {frame_count})")
        if not frames:
            return {}
        with self._session(resource_path) as session_id:
            result: dict[int, np.ndarray] = {}
            for index, frame_id in enumerate(frames):
                if index:
                    self.predictor.handle_request(
                        request={"type": "reset_session", "session_id": session_id}
                    )
                response = self.predictor.handle_request(
                    request={
                        "type": "add_prompt",
                        "session_id": session_id,
                        "frame_index": frame_id,
                        "text": text,
                    }
                )
                result[frame_id] = _primary_mask(
                    response.get("outputs", {}),
                    frame_shape,
                )
            return result

    def text_mask(
        self,
        resource_path: Path,
        text: str,
        *,
        frame_id: int,
        frame_count: int,
        frame_shape: tuple[int, int],
    ) -> np.ndarray:
        return self.text_masks(
            resource_path,
            text,
            frame_ids=(frame_id,),
            frame_count=frame_count,
            frame_shape=frame_shape,
        )[frame_id]

    def _install_native_mask(
        self,
        *,
        session_id: str,
        seed_frame: int,
        object_id: int,
        seed_mask: np.ndarray,
    ) -> None:
        if int(getattr(self.predictor, "world_size", 1)) != 1:
            raise Sam3Error("native mask prompts require a single-GPU predictor")
        model = getattr(self.predictor, "model", None)
        get_session = getattr(self.predictor, "_get_session", None)
        get_states = getattr(model, "_get_tracker_inference_states_by_obj_ids", None)
        tracker = getattr(model, "tracker", None)
        if not callable(get_session) or not callable(get_states):
            raise Sam3Error("SAM3 predictor does not expose native mask prompts")
        if tracker is None or not hasattr(tracker, "add_new_mask"):
            raise Sam3Error("SAM3 tracker does not support native mask prompts")

        import torch

        session = get_session(session_id)
        states = get_states(session["state"], [object_id])
        if len(states) != 1:
            raise Sam3Error(f"expected one tracker state, got {len(states)}")
        frame_id, object_ids, _low_resolution, _video_resolution = tracker.add_new_mask(
            inference_state=states[0],
            frame_idx=seed_frame,
            obj_id=object_id,
            mask=torch.as_tensor(seed_mask, dtype=torch.bool),
        )
        if int(frame_id) != seed_frame or object_id not in {
            int(value) for value in object_ids
        }:
            raise Sam3Error("SAM3 failed to install native mask prompt")
        tracker.propagate_in_video_preflight(states[0], run_mem_encoder=True)

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
    ) -> np.ndarray:
        mask = np.asarray(seed_mask, dtype=bool)
        if mask.shape != frame_shape or not mask.any():
            raise ValueError("seed_mask must be non-empty and match frame_shape")
        window_start, window_end = tracking_window
        if not 0 <= window_start <= seed_frame <= window_end < frame_count:
            raise ValueError("tracking_window must contain seed_frame inside the episode")

        with self._session(resource_path) as session_id:
            response = self.predictor.handle_request(
                request={
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": seed_frame,
                    "points": [_interior_point(mask)],
                    "point_labels": [1],
                    "obj_id": object_id,
                    "rel_coordinates": True,
                }
            )
            registered = _mask_for_object(
                response.get("outputs", {}),
                object_id=object_id,
                shape=frame_shape,
            )
            if registered is None or not registered.any():
                raise Sam3Error("SAM3 failed to register the seed object")
            self._install_native_mask(
                session_id=session_id,
                seed_frame=seed_frame,
                object_id=object_id,
                seed_mask=mask,
            )

            propagated = np.zeros((frame_count, *frame_shape), dtype=bool)

            def collect(direction: str, count: int) -> None:
                if count < 1:
                    return
                for stream_response in self.predictor.handle_stream_request(
                    request={
                        "type": "propagate_in_video",
                        "session_id": session_id,
                        "propagation_direction": direction,
                        "start_frame_index": seed_frame,
                        "max_frame_num_to_track": count,
                    }
                ):
                    frame_id = int(stream_response["frame_index"])
                    if not window_start <= frame_id <= window_end or frame_id == seed_frame:
                        continue
                    if direction == "forward" and frame_id < seed_frame:
                        continue
                    if direction == "backward" and frame_id > seed_frame:
                        continue
                    frame_mask = _mask_for_object(
                        stream_response.get("outputs", {}),
                        object_id=object_id,
                        shape=frame_shape,
                    )
                    if frame_mask is not None:
                        propagated[frame_id] = frame_mask

            collect("forward", window_end - seed_frame)
            collect("backward", seed_frame - window_start)
            propagated[seed_frame] = mask
            return propagated

    def shutdown(self) -> None:
        shutdown = getattr(self.predictor, "shutdown", None)
        if callable(shutdown):
            shutdown()
