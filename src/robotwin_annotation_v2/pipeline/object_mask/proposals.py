"""Pure image proposals used by object-mask resolution."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from .qc import MaskQCError

NDArray = np.ndarray[Any, Any]


def largest_component(mask: NDArray) -> NDArray:
    """Keep the first largest 4-connected component of a binary mask."""

    remaining = np.asarray(mask, dtype=bool).copy()
    if remaining.ndim != 2:
        raise ValueError("mask must be 2-D")
    largest: list[tuple[int, int]] = []
    height, width = remaining.shape
    while remaining.any():
        row, column = np.argwhere(remaining)[0]
        component: list[tuple[int, int]] = []
        stack = [(int(row), int(column))]
        remaining[row, column] = False
        while stack:
            current_row, current_column = stack.pop()
            component.append((current_row, current_column))
            for next_row, next_column in (
                (current_row - 1, current_column),
                (current_row + 1, current_column),
                (current_row, current_column - 1),
                (current_row, current_column + 1),
            ):
                if (
                    0 <= next_row < height
                    and 0 <= next_column < width
                    and remaining[next_row, next_column]
                ):
                    remaining[next_row, next_column] = False
                    stack.append((next_row, next_column))
        if len(component) > len(largest):
            largest = component
    output = np.zeros_like(remaining)
    for row, column in largest:
        output[row, column] = True
    return output


def blue_planar_region(seed_image: Image.Image, frame_shape: tuple[int, int]) -> NDArray:
    """Build the existing coordinate-free proposal for a saturated blue receiver."""

    rgb = np.asarray(seed_image.convert("RGB"), dtype=np.int16)
    if rgb.shape[:2] != frame_shape:
        raise MaskQCError(f"seed RGB shape {rgb.shape[:2]} does not match expected {frame_shape}")
    red, green, blue = (rgb[..., index] for index in range(3))
    saturated_blue = (blue >= 80) & ((blue - red) >= 30) & ((blue - green) >= 20)
    return largest_component(saturated_blue)


__all__ = ["blue_planar_region", "largest_component"]
