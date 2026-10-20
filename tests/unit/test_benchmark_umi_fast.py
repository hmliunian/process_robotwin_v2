from __future__ import annotations

import numpy as np
import pytest

from scripts.benchmark_umi_fast import agreement


def test_iou_counts_one_sided_empty_frames_and_reports_dual_empty_separately() -> None:
    reference = np.zeros((3, 2, 2), dtype=bool)
    actual = reference.copy()
    reference[0, 0, 0] = True
    actual[1, 0, 0] = True
    result = agreement(reference, actual)
    assert result["frame_iou"] == [0.0, 0.0, 1.0]
    assert result["nonempty_mean_iou"] == 0.0
    assert result["dual_empty_frames"] == 1 and result["presence_changes"] == 2


def test_iou_rejects_broadcasting_or_nonboolean_input() -> None:
    masks = np.ones((10, 2, 2), dtype=bool)
    for other in (masks[:1], masks.astype(np.uint8)):
        with pytest.raises(ValueError, match="equally shaped boolean"):
            agreement(masks, other)
