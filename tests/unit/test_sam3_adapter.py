from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import numpy as np

import pytest

from robotwin_annotation_v2.adapters import Sam3Adapter


SHAPE = (4, 5)


def _outputs(mask: np.ndarray, *, object_id: int = 1, probability: float = 1.0) -> dict[str, Any]:
    return {
        "out_obj_ids": np.asarray([object_id]),
        "out_probs": np.asarray([probability]),
        "out_binary_masks": np.asarray([mask]),
    }


class FakePredictor:
    world_size = 1

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.mask = np.zeros(SHAPE, dtype=bool)
        self.mask[1:3, 2:4] = True

    def handle_request(self, *, request: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(request)
        if request["type"] == "start_session":
            return {"session_id": "session"}
        if request["type"] == "add_prompt":
            distractor = np.zeros(SHAPE, dtype=bool)
            distractor[0, 0] = True
            return {
                "outputs": {
                    "out_obj_ids": np.asarray([1, 2]),
                    "out_probs": np.asarray([0.1, 0.9]),
                    "out_binary_masks": np.asarray([distractor, self.mask]),
                }
            }
        return {}

    def handle_stream_request(self, *, request: dict[str, Any]) -> Iterator[dict[str, Any]]:
        self.requests.append(request)
        direction = request["propagation_direction"]
        frames = (2, 3) if direction == "forward" else (0,)
        for frame_id in frames:
            yield {"frame_index": frame_id, "outputs": _outputs(self.mask)}


class NativeTestAdapter(Sam3Adapter):
    def _install_native_mask(self, **_kwargs: Any) -> None:
        return


def test_text_masks_select_highest_probability_object() -> None:
    predictor = FakePredictor()
    adapter = Sam3Adapter(
        checkpoint_path=Path("unused.pt"),
        gpus=(0,),
        predictor=predictor,
    )

    result = adapter.text_masks(
        Path("resource"),
        "orange bottle",
        frame_ids=(0, 2),
        frame_count=4,
        frame_shape=SHAPE,
    )

    assert np.array_equal(result[0], predictor.mask)
    assert np.array_equal(result[2], predictor.mask)
    assert any(request["type"] == "reset_session" for request in predictor.requests)


def test_text_query_masks_reuses_one_video_session() -> None:
    predictor = FakePredictor()
    adapter = Sam3Adapter(
        checkpoint_path=Path("unused.pt"),
        gpus=(0,),
        predictor=predictor,
    )

    result = adapter.text_query_masks(
        Path("resource"),
        ("bottle", "orange bottle", "bottle"),
        frame_id=0,
        frame_count=4,
        frame_shape=SHAPE,
    )

    assert tuple(result) == ("bottle", "orange bottle")
    request_types = [request["type"] for request in predictor.requests]
    assert request_types.count("start_session") == 1
    assert request_types.count("reset_session") == 1
    assert request_types.count("close_session") == 1
    assert [
        request["text"]
        for request in predictor.requests
        if request["type"] == "add_prompt"
    ] == ["bottle", "orange bottle"]


def test_visual_box_masks_submit_box_and_optional_text_together() -> None:
    predictor = FakePredictor()
    adapter = Sam3Adapter(
        checkpoint_path=Path("unused.pt"),
        gpus=(0,),
        predictor=predictor,
    )

    box_result = adapter.box_mask(
        Path("resource"),
        (0.1, 0.2, 0.8, 0.9),
        frame_id=1,
        frame_count=4,
        frame_shape=SHAPE,
    )
    text_result = adapter.text_box_mask(
        Path("resource"),
        "  black   robot gripper ",
        (0.1, 0.2, 0.8, 0.9),
        frame_id=2,
        frame_count=4,
        frame_shape=SHAPE,
    )

    assert np.array_equal(box_result, predictor.mask)
    assert np.array_equal(text_result, predictor.mask)
    prompts = [
        request for request in predictor.requests if request["type"] == "add_prompt"
    ]
    assert prompts[0]["bounding_boxes"] == [[0.1, 0.2, 0.7000000000000001, 0.7]]
    assert "text" not in prompts[0]
    assert prompts[1]["text"] == "black robot gripper"
    assert prompts[1]["bounding_boxes"] == [[0.1, 0.2, 0.7000000000000001, 0.7]]
    assert all(prompt["bounding_box_labels"] == [1] for prompt in prompts)
    assert all(prompt["rel_coordinates"] is True for prompt in prompts)


def test_visual_box_masks_validate_normalized_box() -> None:
    adapter = Sam3Adapter(
        checkpoint_path=Path("unused.pt"),
        gpus=(0,),
        predictor=FakePredictor(),
    )

    with pytest.raises(ValueError, match="normalized box"):
        adapter.box_mask(
            Path("resource"),
            (0.4, 0.2, 0.3, 0.9),
            frame_id=1,
            frame_count=4,
            frame_shape=SHAPE,
        )


def test_native_mask_propagation_keeps_exact_seed() -> None:
    predictor = FakePredictor()
    adapter = NativeTestAdapter(
        checkpoint_path=Path("unused.pt"),
        gpus=(0,),
        predictor=predictor,
    )
    seed = predictor.mask.copy()

    track = adapter.propagate_mask(
        Path("resource"),
        seed,
        seed_frame=1,
        frame_count=4,
        frame_shape=SHAPE,
        tracking_window=(0, 3),
    )

    assert np.array_equal(track[1], seed)
    assert track[0].any() and track[2].any() and track[3].any()
    directions = [
        request["propagation_direction"]
        for request in predictor.requests
        if request["type"] == "propagate_in_video"
    ]
    assert directions == ["forward", "backward"]
