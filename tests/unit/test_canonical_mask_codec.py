from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from robotwin_annotation_v2.adapters.canonical_masks import (
    CANONICAL_INSTANCE_NAMES,
    CANONICAL_ROLES,
    CanonicalMaskError,
    read_canonical_masks,
)
from robotwin_annotation_v2.mask_schema import MASK_FORMAT_VERSION


def _payload(*, version: str = MASK_FORMAT_VERSION) -> dict[str, np.ndarray]:
    masks = np.zeros((4, 3, 2, 3), dtype=bool)
    masks[0, :, 0, 0] = True
    masks[3, 1, 1, 2] = True
    payload: dict[str, np.ndarray] = {
        "format_version": np.asarray(version),
        "frame_count": np.asarray(3, dtype=np.int64),
        "masks": masks,
        "instance_names": np.asarray(CANONICAL_INSTANCE_NAMES),
        "roles": np.asarray(CANONICAL_ROLES),
        "annotation_status": np.asarray(("valid", "not_applicable", "not_annotated", "valid")),
        "qc_status": np.asarray(("passed", "not_applicable", "not_run", "passed")),
    }
    if version == MASK_FORMAT_VERSION:
        payload["frame_encoding"] = np.where(
            masks.reshape(4, 3, -1).any(axis=2),
            1,
            0,
        ).astype(np.uint8)
    return payload


def _write(path: Path, payload: dict[str, np.ndarray]) -> Path:
    np.savez_compressed(path, **payload)
    return path


def test_reads_v3_and_preserves_validated_encoding(tmp_path: Path) -> None:
    path = _write(tmp_path / "masks.npz", _payload())

    bundle = read_canonical_masks(path)

    assert bundle.path == path.resolve()
    assert bundle.format_version == MASK_FORMAT_VERSION
    assert bundle.masks.dtype == np.bool_
    assert bundle.frame_encoding.dtype == np.uint8
    assert not bundle.masks.flags.writeable
    assert not bundle.frame_encoding.flags.writeable
    np.testing.assert_array_equal(bundle.to_payload()["frame_encoding"], bundle.frame_encoding)


def test_reads_v2_and_synthesizes_encoding(tmp_path: Path) -> None:
    path = _write(tmp_path / "masks-v2.npz", _payload(version="robotwin_visible_masks_v2"))

    bundle = read_canonical_masks(path)

    assert bundle.frame_encoding.tolist() == [
        [1, 1, 1],
        [0, 0, 0],
        [0, 0, 0],
        [0, 1, 0],
    ]
    assert "frame_encoding" not in bundle.to_payload()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("masks", np.zeros((4, 3, 2, 3), dtype=np.uint8)),
        ("instance_names", np.asarray(("a", "b", "c", "d"))),
        ("frame_count", np.asarray(3.0)),
    ],
)
def test_rejects_noncanonical_fields(
    tmp_path: Path,
    field: str,
    value: np.ndarray,
) -> None:
    payload = _payload()
    payload[field] = value

    with pytest.raises(CanonicalMaskError):
        read_canonical_masks(_write(tmp_path / f"{field}.npz", payload))


@pytest.mark.parametrize("extra", ("unexpected", "frame_encoding"))
def test_rejects_wrong_key_set(tmp_path: Path, extra: str) -> None:
    payload = _payload(version="robotwin_visible_masks_v2")
    payload[extra] = np.asarray(1)

    with pytest.raises(CanonicalMaskError, match="keys differ"):
        read_canonical_masks(_write(tmp_path / f"{extra}.npz", payload))


def test_rejects_invalid_v3_encoding(tmp_path: Path) -> None:
    payload = _payload()
    payload["frame_encoding"][0, 0] = 9

    with pytest.raises(CanonicalMaskError, match="invalid frame_encoding"):
        read_canonical_masks(_write(tmp_path / "invalid.npz", payload))
