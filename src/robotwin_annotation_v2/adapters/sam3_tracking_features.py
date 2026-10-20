"""Opt-in visual-only feature preparation for pinned SAM3 native-mask tracking.

This replaces only the interactive tracker's feature hook for one propagation
call. Detection, prompt resolution and temporal tracking remain separate: image
encoding is batched, while the tracker still updates its memory in frame order.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


@contextmanager
def native_tracking_features(
    predictor: Any,
    *,
    batch_size: int,
    tracking_window: tuple[int, int],
) -> Iterator[None]:
    """Keep at most one chunk of visual features in a single-owner predictor."""
    if batch_size == 0:
        yield
        return
    model = predictor.model
    if int(getattr(predictor, "world_size", 1)) != 1 or not callable(
        getattr(model, "_prepare_backbone_feats", None)
    ):
        raise RuntimeError("visual-only tracking requires the pinned single-GPU SAM3 predictor")

    import torch

    def prepare(inference_state: dict[str, Any], frame_idx: int, reverse: bool) -> None:
        cache = inference_state["feature_cache"]
        if frame_idx in cache:
            return
        # A direction change or jump replaces the chunk. Do not clear metadata
        # in the shared cache, and never retain a feature for every video frame.
        for key in tuple(cache):
            if isinstance(key, int):
                del cache[key]
        lower, upper = tracking_window
        step = -1 if reverse else 1
        stop = (
            max(lower - 1, frame_idx - batch_size)
            if reverse
            else min(upper + 1, frame_idx + batch_size)
        )
        frame_ids = range(frame_idx, stop, step)
        images = inference_state["input_batch"].img_batch
        image_batch = torch.stack([images[index] for index in frame_ids]).to(
            device=model.device, dtype=torch.float32
        )
        features = model.detector.backbone.forward_image(image_batch)["sam2_backbone_out"]
        # Match the original detector's gather buffer, including its BF16 cast
        # before the tracker's projections. Positional encodings keep their dtype.
        fpn = [value.to(torch.bfloat16) for value in features["backbone_fpn"]]
        decoder = model.tracker.sam_mask_decoder
        fpn[0] = decoder.conv_s0(fpn[0])
        fpn[1] = decoder.conv_s1(fpn[1])
        for offset, index in enumerate(frame_ids):
            frame_fpn = [value[offset : offset + 1] for value in fpn]
            positions = [value[offset : offset + 1] for value in features["vision_pos_enc"]]
            cache[index] = (
                images[index],
                {
                    "tracker_backbone_out": {
                        "vision_features": frame_fpn[-1],
                        "vision_pos_enc": positions,
                        "backbone_fpn": frame_fpn,
                    }
                },
            )

    missing = object()
    previous = vars(model).get("_prepare_backbone_feats", missing)
    model._prepare_backbone_feats = prepare
    try:
        yield
    finally:
        if previous is missing:
            del model._prepare_backbone_feats
        else:
            model._prepare_backbone_feats = previous
