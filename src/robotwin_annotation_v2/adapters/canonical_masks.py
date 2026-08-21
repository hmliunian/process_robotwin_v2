"""Validated reader and value object for canonical visible-mask archives."""

from __future__ import annotations

import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from ..mask_schema import (
    LEGACY_MASK_FORMAT_VERSION,
    MASK_BASE_KEYS,
    MASK_FORMAT_VERSION,
    MASK_KEYS,
    default_frame_encoding,
    validate_frame_encoding,
)

CANONICAL_INSTANCE_NAMES = (
    "target_0",
    "receiver_0",
    "gripper_left",
    "gripper_right",
)
CANONICAL_ROLES = ("target", "receiver", "gripper", "gripper")

BoolMaskArray = NDArray[np.bool_]
FrameEncodingArray = NDArray[np.uint8]


class CanonicalMaskError(ValueError):
    """A canonical visible-mask archive violates its v2/v3 reader contract."""


def _string_vector(value: Any, *, label: str) -> tuple[str, ...]:
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind != "U":
        raise CanonicalMaskError(
            f"{label} must be a one-dimensional Unicode string array; "
            f"got {array.shape}/{array.dtype}"
        )
    return tuple(str(item) for item in array.tolist())


def _scalar_string(value: Any, *, label: str) -> str:
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind != "U":
        raise CanonicalMaskError(
            f"{label} must be one Unicode string scalar; got {array.shape}/{array.dtype}"
        )
    return str(array.item())


def _scalar_frame_count(value: Any) -> int:
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in {"i", "u"}:
        raise CanonicalMaskError(
            f"frame_count must be one integer scalar; got {array.shape}/{array.dtype}"
        )
    frame_count = int(array.item())
    if frame_count < 1:
        raise CanonicalMaskError("frame_count must be positive")
    return frame_count


def _copy_read_only(value: NDArray[Any]) -> NDArray[Any]:
    copied = np.asarray(value).copy()
    copied.setflags(write=False)
    return copied


def _status_vector(value: Any, *, label: str) -> tuple[str, ...]:
    return _string_vector(value, label=label)


@dataclass(frozen=True)
class CanonicalMaskBundle:
    """Immutable, validated in-memory representation of a v2/v3 mask archive."""

    path: Path
    format_version: str
    frame_count: int
    masks: BoolMaskArray
    instance_names: tuple[str, ...]
    roles: tuple[str, ...]
    annotation_status: tuple[str, ...]
    qc_status: tuple[str, ...]
    frame_encoding: FrameEncodingArray

    def __post_init__(self) -> None:
        if self.format_version not in {
            LEGACY_MASK_FORMAT_VERSION,
            MASK_FORMAT_VERSION,
        }:
            raise CanonicalMaskError(
                f"unsupported mask format {self.format_version!r}: {self.path}"
            )
        masks = np.asarray(self.masks)
        if masks.dtype != np.bool_ or masks.ndim != 4:
            raise CanonicalMaskError(
                "masks must have bool shape [4,T,H,W]; "
                f"got {masks.shape}/{masks.dtype}"
            )
        if masks.shape[0] != len(CANONICAL_INSTANCE_NAMES):
            raise CanonicalMaskError(
                "masks must have four canonical channels; "
                f"got shape {masks.shape}"
            )
        if masks.shape[1] != self.frame_count:
            raise CanonicalMaskError(
                f"masks frame count {masks.shape[1]} differs from {self.frame_count}"
            )
        metadata = (
            ("instance_names", self.instance_names),
            ("roles", self.roles),
            ("annotation_status", self.annotation_status),
            ("qc_status", self.qc_status),
        )
        for label, values in metadata:
            if len(values) != len(CANONICAL_INSTANCE_NAMES):
                raise CanonicalMaskError(
                    f"{label} must contain four entries; got {len(values)}"
                )
        if self.instance_names != CANONICAL_INSTANCE_NAMES:
            raise CanonicalMaskError("mask channel names are not canonical")
        if self.roles != CANONICAL_ROLES:
            raise CanonicalMaskError("mask channel roles are not canonical")
        if self.frame_count < 1:
            raise CanonicalMaskError("frame_count must be positive")
        encoding = np.asarray(self.frame_encoding)
        try:
            validated_encoding = validate_frame_encoding(masks, encoding)
        except ValueError as exc:
            raise CanonicalMaskError(f"invalid frame_encoding: {exc}") from exc
        object.__setattr__(self, "path", Path(self.path))
        object.__setattr__(self, "masks", cast(BoolMaskArray, _copy_read_only(masks)))
        object.__setattr__(
            self,
            "frame_encoding",
            cast(FrameEncodingArray, _copy_read_only(validated_encoding)),
        )

    def to_payload(self) -> dict[str, NDArray[Any]]:
        """Return a detached archive payload using this bundle's format version."""

        payload: dict[str, NDArray[Any]] = {
            "format_version": np.asarray(self.format_version),
            "frame_count": np.asarray(self.frame_count, dtype=np.int64),
            "masks": np.asarray(self.masks).copy(),
            "instance_names": np.asarray(self.instance_names),
            "roles": np.asarray(self.roles),
            "annotation_status": np.asarray(self.annotation_status),
            "qc_status": np.asarray(self.qc_status),
        }
        if self.format_version == MASK_FORMAT_VERSION:
            payload["frame_encoding"] = np.asarray(self.frame_encoding).copy()
        return payload


def build_canonical_mask_bundle(
    path: Path,
    *,
    frame_count: int,
    masks: NDArray[Any],
    annotation_status: Any,
    qc_status: Any,
    frame_encoding: NDArray[Any],
) -> CanonicalMaskBundle:
    """Build the validated v3 bundle used by new canonical writers."""

    return CanonicalMaskBundle(
        path=path,
        format_version=MASK_FORMAT_VERSION,
        frame_count=frame_count,
        masks=cast(BoolMaskArray, np.asarray(masks)),
        instance_names=CANONICAL_INSTANCE_NAMES,
        roles=CANONICAL_ROLES,
        annotation_status=_status_vector(annotation_status, label="annotation_status"),
        qc_status=_status_vector(qc_status, label="qc_status"),
        frame_encoding=cast(FrameEncodingArray, np.asarray(frame_encoding)),
    )


def _bundle_from_payload(path: Path, payload: Mapping[str, Any]) -> CanonicalMaskBundle:
    format_version = _scalar_string(payload["format_version"], label="format_version")
    expected_keys = (
        MASK_KEYS if format_version == MASK_FORMAT_VERSION else MASK_BASE_KEYS
    )
    if format_version not in {
        LEGACY_MASK_FORMAT_VERSION,
        MASK_FORMAT_VERSION,
    }:
        raise CanonicalMaskError(f"unsupported mask format {format_version!r}: {path}")
    actual_keys = set(payload)
    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)
    if missing or extra:
        raise CanonicalMaskError(
            f"mask keys differ from {format_version}: missing={missing}, extra={extra}: {path}"
        )
    frame_count = _scalar_frame_count(payload["frame_count"])
    masks = np.asarray(payload["masks"])
    instance_names = _string_vector(payload["instance_names"], label="instance_names")
    roles = _string_vector(payload["roles"], label="roles")
    annotation_status = _string_vector(
        payload["annotation_status"],
        label="annotation_status",
    )
    qc_status = _string_vector(payload["qc_status"], label="qc_status")
    if format_version == MASK_FORMAT_VERSION:
        frame_encoding = np.asarray(payload["frame_encoding"])
    else:
        frame_encoding = default_frame_encoding(masks)
    return CanonicalMaskBundle(
        path=path,
        format_version=format_version,
        frame_count=frame_count,
        masks=cast(BoolMaskArray, masks),
        instance_names=instance_names,
        roles=roles,
        annotation_status=annotation_status,
        qc_status=qc_status,
        frame_encoding=cast(FrameEncodingArray, frame_encoding),
    )


def read_canonical_masks(path: Path) -> CanonicalMaskBundle:
    """Read and validate one canonical v2/v3 archive without pickle loading."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"canonical masks are missing: {source}")
    try:
        with np.load(source, allow_pickle=False) as archive:
            payload = {key: np.asarray(archive[key]).copy() for key in archive.files}
    except (EOFError, OSError, ValueError, zipfile.BadZipFile) as exc:
        raise CanonicalMaskError(f"cannot load canonical masks {source}: {exc}") from exc
    try:
        return _bundle_from_payload(source, payload)
    except CanonicalMaskError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise CanonicalMaskError(f"invalid canonical masks {source}: {exc}") from exc


# Compatibility shim: remove after external callers migrate to
# CanonicalMaskPublisher and the function seam has remained for one release.
def write_canonical_masks(path: Path, bundle: CanonicalMaskBundle) -> Path:
    """Compatibility entry point for the canonical publication owner."""

    from .canonical_publication import CanonicalMaskPublisher

    return CanonicalMaskPublisher().publish(path, bundle)


__all__ = [
    "CANONICAL_INSTANCE_NAMES",
    "CANONICAL_ROLES",
    "CanonicalMaskBundle",
    "CanonicalMaskError",
    "build_canonical_mask_bundle",
    "read_canonical_masks",
    "write_canonical_masks",
]
