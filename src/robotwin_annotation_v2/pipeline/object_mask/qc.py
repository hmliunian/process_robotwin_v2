"""Pure mechanical quality checks for object-mask candidates."""

from __future__ import annotations

from typing import Any

import numpy as np

from ...models import MaskCandidateInfo

NDArray = np.ndarray[Any, Any]


class MaskQCError(RuntimeError):
    """A mask-QC request, response, or candidate violated its contract."""

    def __init__(
        self,
        message: str,
        *,
        rendered_prompt: str | None = None,
        raw_response: str | None = None,
    ) -> None:
        super().__init__(message)
        self.rendered_prompt = rendered_prompt
        self.raw_response = raw_response


def _component_count(mask: NDArray) -> int:
    """Count 4-connected components without requiring OpenCV or SciPy."""

    remaining = np.asarray(mask, dtype=bool).copy()
    if remaining.ndim != 2:
        raise ValueError("mask must be 2-D")
    count = 0
    height, width = remaining.shape
    while remaining.any():
        row, column = np.argwhere(remaining)[0]
        count += 1
        stack = [(int(row), int(column))]
        remaining[row, column] = False
        while stack:
            current_row, current_column = stack.pop()
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
    return count


def candidate_info(
    candidate_id: str,
    query_field: str,
    query: str,
    mask: NDArray,
    *,
    min_area_fraction: float,
    max_area_fraction: float,
    duplicate_of: str | None = None,
    seed_frame_id: int | None = None,
) -> MaskCandidateInfo:
    """Summarize one candidate and apply the shared mechanical gate."""

    value = np.asarray(mask, dtype=bool)
    if value.ndim != 2:
        raise MaskQCError(f"candidate {candidate_id} mask must be 2-D")
    area_fraction = float(value.mean())
    nonempty = bool(value.any())
    components = _component_count(value) if nonempty else 0
    if not nonempty:
        reason = "empty_seed_mask"
    elif area_fraction < min_area_fraction:
        reason = "seed_mask_too_small"
    elif area_fraction > max_area_fraction:
        reason = "seed_mask_too_large"
    elif duplicate_of is not None:
        reason = "duplicate_candidate_mask"
    else:
        reason = None
    return MaskCandidateInfo(
        candidate_id=candidate_id,
        query_field=query_field,
        query=query,
        nonempty=nonempty,
        area_fraction=area_fraction,
        component_count=components,
        basic_valid=reason is None,
        basic_reason=reason,
        duplicate_of=duplicate_of,
        seed_frame_id=seed_frame_id,
    )


def mask_iou(first: NDArray, second: NDArray) -> float:
    """Return binary-mask intersection over union, treating two empty masks as equal."""

    first_mask = np.asarray(first, dtype=bool)
    second_mask = np.asarray(second, dtype=bool)
    union = first_mask | second_mask
    if not union.any():
        return 1.0
    return float((first_mask & second_mask).sum() / union.sum())


__all__ = ["MaskQCError", "candidate_info", "mask_iou"]
