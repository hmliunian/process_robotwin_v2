from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import robotwin_annotation_v2.urdf_gripper_publisher as publisher
from robotwin_annotation_v2.adapters import rendering
from robotwin_annotation_v2.application import urdf_batch
from robotwin_annotation_v2.domain import ObjectRole
from robotwin_annotation_v2.mask_schema import (
    FrameEncoding,
    default_frame_encoding,
)

INSTANCE_NAMES = ("target_0", "receiver_0", "gripper_left", "gripper_right")
ROLES = ("target", "receiver", "gripper", "gripper")
FRAME_COUNT = 2
FRAME_SHAPE = (2, 3)


def _base_masks(*, dtype: np.dtype[Any] | None = None) -> np.ndarray:
    masks = np.zeros((4, FRAME_COUNT, *FRAME_SHAPE), dtype=bool)
    masks[0, :, 0, 0] = True
    masks[1, 0, 0, 1] = True
    masks[2, 1, 1, 0] = True
    if dtype is None:
        return masks
    return masks.astype(dtype)


def _archive_payload(
    *,
    version: str = "robotwin_visible_masks_v2",
    masks: np.ndarray | None = None,
    include_qc_status: bool = True,
    include_frame_encoding: bool | None = None,
    instance_names: tuple[str, ...] = INSTANCE_NAMES,
    roles: tuple[str, ...] = ROLES,
) -> dict[str, np.ndarray]:
    value = _base_masks() if masks is None else np.asarray(masks)
    payload: dict[str, np.ndarray] = {
        "format_version": np.asarray(version),
        "frame_count": np.asarray(FRAME_COUNT, dtype=np.int64),
        "masks": value,
        "instance_names": np.asarray(instance_names),
        "roles": np.asarray(roles),
        "annotation_status": np.asarray(("valid", "valid", "not_annotated", "not_annotated")),
    }
    if include_qc_status:
        payload["qc_status"] = np.asarray(("passed", "passed", "not_run", "not_run"))
    if include_frame_encoding is None:
        include_frame_encoding = version == "robotwin_visible_masks_v3"
    if include_frame_encoding:
        encoding = default_frame_encoding(_base_masks())
        encoding[0, 1] = FrameEncoding.TARGET_GRASP_HOLD.value
        payload["frame_encoding"] = encoding
    return payload


def _write_archive(path: Path, payload: dict[str, np.ndarray]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return path


def _assert_reader_fields(rendered: Any, urdf: Any) -> None:
    np.testing.assert_array_equal(rendered.masks, urdf.masks)
    np.testing.assert_array_equal(rendered.frame_encoding, urdf.frame_encoding)
    assert rendered.instance_names == tuple(str(item) for item in urdf.payload["instance_names"])
    assert rendered.roles == tuple(str(item) for item in urdf.payload["roles"])
    assert rendered.annotation_status == urdf.annotation_status
    assert rendered.qc_status == urdf.qc_status
    assert rendered.frame_count == urdf.frame_count == FRAME_COUNT
    assert rendered.format_version == str(urdf.payload["format_version"].item())


@pytest.mark.parametrize("version", ("robotwin_visible_masks_v2", "robotwin_visible_masks_v3"))
def test_render_and_urdf_readers_match_on_canonical_v2_v3_inputs(
    tmp_path: Path,
    version: str,
) -> None:
    path = _write_archive(tmp_path / f"{version}.npz", _archive_payload(version=version))

    rendered = rendering._load_masks(path)
    urdf = urdf_batch.load_four_channel_masks(path, frame_count=FRAME_COUNT)

    _assert_reader_fields(rendered, urdf)
    if version == "robotwin_visible_masks_v2":
        assert rendered.frame_encoding[0].tolist() == [1, 1]
    else:
        assert rendered.frame_encoding[0].tolist() == [1, FrameEncoding.TARGET_GRASP_HOLD.value]


def test_render_reader_uses_shared_codec_for_canonical_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_archive(tmp_path / "canonical.npz", _archive_payload(version="robotwin_visible_masks_v3"))
    calls: list[Path] = []
    original = rendering.read_canonical_masks

    def read(path: Path) -> Any:
        calls.append(path)
        return original(path)

    monkeypatch.setattr(rendering, "read_canonical_masks", read)

    rendering._load_masks(path)

    assert calls == [path]


def test_urdf_reader_uses_shared_codec_for_canonical_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_archive(tmp_path / "canonical.npz", _archive_payload(version="robotwin_visible_masks_v3"))
    calls: list[Path] = []
    original = urdf_batch.read_canonical_masks

    def read(path: Path) -> Any:
        calls.append(path)
        return original(path)

    monkeypatch.setattr(urdf_batch, "read_canonical_masks", read)

    urdf_batch.load_four_channel_masks(path, frame_count=FRAME_COUNT)

    assert calls == [path]


def test_publisher_reader_uses_shared_codec_for_canonical_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _write_archive(tmp_path / "canonical.npz", _archive_payload(version="robotwin_visible_masks_v3"))
    calls: list[Path] = []
    original = publisher.read_canonical_masks

    def read(path: Path) -> Any:
        calls.append(path)
        return original(path)

    monkeypatch.setattr(publisher, "read_canonical_masks", read)

    result = publisher._load_source_masks(
        path,
        required_roles=(ObjectRole.TARGET, ObjectRole.RECEIVER),
    )

    assert calls == [path]
    assert result["format_version"] == "robotwin_visible_masks_v3"


def test_render_reader_synthesizes_legacy_encoding_and_qc_defaults(tmp_path: Path) -> None:
    payload = _archive_payload(include_qc_status=False)
    path = _write_archive(tmp_path / "legacy-no-qc.npz", payload)

    artifact = rendering._load_masks(path)

    assert artifact.qc_status == ("not_run",) * 4
    assert artifact.frame_encoding is not None
    assert artifact.frame_encoding[0].tolist() == [1, 1]


def test_urdf_reader_synthesizes_legacy_encoding_and_qc_defaults(tmp_path: Path) -> None:
    payload = _archive_payload(include_qc_status=False)
    path = _write_archive(tmp_path / "legacy-no-qc.npz", payload)

    artifact = urdf_batch.load_four_channel_masks(path, frame_count=FRAME_COUNT)

    assert artifact.qc_status == ("not_run",) * 4
    assert artifact.frame_encoding.tolist()[0] == [1, 1]
    assert "qc_status" not in artifact.payload


def test_render_reader_currently_accepts_extra_keys_and_casts_mask_dtype(
    tmp_path: Path,
) -> None:
    payload = _archive_payload(masks=_base_masks(dtype=np.uint8))
    payload["reader_note"] = np.asarray("ignored")
    path = _write_archive(tmp_path / "render-permissive.npz", payload)

    artifact = rendering._load_masks(path)

    assert artifact.masks.dtype == np.bool_
    assert artifact.masks[0, 0, 0, 0]


def test_urdf_reader_currently_accepts_extra_keys_and_casts_mask_dtype(
    tmp_path: Path,
) -> None:
    payload = _archive_payload(masks=_base_masks(dtype=np.uint8))
    payload["reader_note"] = np.asarray("preserved")
    path = _write_archive(tmp_path / "urdf-permissive.npz", payload)

    artifact = urdf_batch.load_four_channel_masks(path, frame_count=FRAME_COUNT)

    assert artifact.masks.dtype == np.bool_
    assert artifact.masks[0, 0, 0, 0]
    assert artifact.payload["reader_note"].item() == "preserved"


def test_render_reader_currently_accepts_noncanonical_channel_labels(tmp_path: Path) -> None:
    payload = _archive_payload(
        instance_names=("a", "b", "c", "d"),
        roles=("x", "y", "z", "z"),
    )
    path = _write_archive(tmp_path / "render-labels.npz", payload)

    artifact = rendering._load_masks(path)

    assert artifact.instance_names == ("a", "b", "c", "d")
    assert artifact.roles == ("x", "y", "z", "z")


def test_urdf_reader_rejects_noncanonical_channel_labels(tmp_path: Path) -> None:
    payload = _archive_payload(
        instance_names=("a", "b", "c", "d"),
        roles=("x", "y", "z", "z"),
    )
    path = _write_archive(tmp_path / "urdf-labels.npz", payload)

    with pytest.raises(urdf_batch.UrdfMaskRunError, match="channel contract"):
        urdf_batch.load_four_channel_masks(path, frame_count=FRAME_COUNT)


def test_render_reader_rejects_v3_without_frame_encoding(tmp_path: Path) -> None:
    payload = _archive_payload(version="robotwin_visible_masks_v3", include_frame_encoding=False)
    path = _write_archive(tmp_path / "render-v3-missing-encoding.npz", payload)

    with pytest.raises(ValueError, match="missing frame_encoding"):
        rendering._load_masks(path)


def test_urdf_reader_rejects_v3_without_frame_encoding(tmp_path: Path) -> None:
    payload = _archive_payload(version="robotwin_visible_masks_v3", include_frame_encoding=False)
    path = _write_archive(tmp_path / "urdf-v3-missing-encoding.npz", payload)

    with pytest.raises(urdf_batch.UrdfMaskRunError, match="missing frame_encoding"):
        urdf_batch.load_four_channel_masks(path, frame_count=FRAME_COUNT)


def test_render_reader_rejects_unknown_format_but_urdf_reader_currently_accepts(
    tmp_path: Path,
) -> None:
    path = _write_archive(
        tmp_path / "unknown-format.npz",
        _archive_payload(version="robotwin_visible_masks_unknown"),
    )

    with pytest.raises(ValueError, match="unsupported mask format"):
        rendering._load_masks(path)
    urdf = urdf_batch.load_four_channel_masks(path, frame_count=FRAME_COUNT)
    assert urdf.payload["format_version"].item() == "robotwin_visible_masks_unknown"


def test_both_readers_reject_invalid_frame_encoding(tmp_path: Path) -> None:
    payload = _archive_payload(version="robotwin_visible_masks_v3")
    payload["frame_encoding"] = np.asarray([[1, 9], [1, 0], [0, 1], [0, 0]], dtype=np.uint8)
    path = _write_archive(tmp_path / "invalid-encoding.npz", payload)

    with pytest.raises(ValueError, match="invalid frame_encoding"):
        rendering._load_masks(path)
    with pytest.raises(urdf_batch.UrdfMaskRunError, match="invalid"):
        urdf_batch.load_four_channel_masks(path, frame_count=FRAME_COUNT)


@pytest.mark.parametrize("reader", (rendering._load_masks, urdf_batch.load_four_channel_masks))
def test_readers_reject_missing_canonical_base_key(
    tmp_path: Path,
    reader: Callable[..., Any],
) -> None:
    payload = _archive_payload()
    payload.pop("masks")
    path = _write_archive(tmp_path / "missing-masks.npz", payload)

    with pytest.raises((ValueError, urdf_batch.UrdfMaskRunError), match="missing"):
        if reader is rendering._load_masks:
            reader(path)
        else:
            reader(path, frame_count=FRAME_COUNT)
