from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from robotwin_annotation_v2.adapters.sam3_tracking_features import native_tracking_features


class FeatureModel:
    device = "cpu"

    def __init__(self) -> None:
        self.batches: list[list[float]] = []
        self.detector = SimpleNamespace(backbone=SimpleNamespace(forward_image=self.encode))
        self.tracker = SimpleNamespace(
            sam_mask_decoder=SimpleNamespace(conv_s0=lambda x: x + 10, conv_s1=lambda x: x + 20)
        )

    def encode(self, images: torch.Tensor) -> dict[str, Any]:
        self.batches.append(images[:, 0, 0, 0].tolist())
        return {
            "sam2_backbone_out": {
                "backbone_fpn": [images.clone() for _ in range(3)],
                "vision_pos_enc": [images.clone() + 100 for _ in range(3)],
            }
        }

    def _prepare_backbone_feats(self, *_args: Any) -> None:
        raise AssertionError("detector must not run on the visual-only path")


def test_chunked_features_preserve_frame_identity_and_bound_cache_in_both_directions() -> None:
    model = FeatureModel()
    predictor = SimpleNamespace(model=model, world_size=1)
    images = torch.arange(9).float().reshape(9, 1, 1, 1).expand(-1, 3, 2, 2)
    cache: dict[Any, Any] = {"tracking_bounds": {"sentinel": True}}
    state = {"feature_cache": cache, "input_batch": SimpleNamespace(img_batch=images)}

    with native_tracking_features(predictor, batch_size=3, tracking_window=(1, 7)):
        for reverse, indices in [(False, [4, 5, 6, 7]), (True, [4, 3, 2, 1])]:
            for index in indices:
                model._prepare_backbone_feats(state, index, reverse)
                features = cache[index][1]["tracker_backbone_out"]
                assert features["backbone_fpn"][0].shape == (1, 3, 2, 2)
                assert features["backbone_fpn"][0].dtype == torch.bfloat16
                assert features["backbone_fpn"][0].flatten()[0] == index + 10
                assert features["backbone_fpn"][1].flatten()[0] == index + 20
                assert features["vision_pos_enc"][0].shape == (1, 3, 2, 2)
                assert features["vision_pos_enc"][0].flatten()[0] == index + 100
                assert sum(isinstance(key, int) for key in cache) <= 3
        assert cache["tracking_bounds"] == {"sentinel": True}
    assert model.batches == [[4, 5, 6], [7], [4, 3, 2], [1]]
    assert "_prepare_backbone_feats" not in vars(model)


def test_feature_hook_restores_existing_override_after_failure() -> None:
    model = FeatureModel()
    original = model._prepare_backbone_feats
    model._prepare_backbone_feats = original
    with (
        pytest.raises(RuntimeError, match="failure"),
        native_tracking_features(
            SimpleNamespace(model=model, world_size=1), batch_size=1, tracking_window=(0, 1)
        ),
    ):
        raise RuntimeError("failure")
    assert vars(model)["_prepare_backbone_feats"] is original


def test_visual_only_path_rejects_multi_gpu_predictor() -> None:
    with (
        pytest.raises(RuntimeError, match="single-GPU"),
        native_tracking_features(
            SimpleNamespace(model=FeatureModel(), world_size=2),
            batch_size=1,
            tracking_window=(0, 1),
        ),
    ):
        pass
