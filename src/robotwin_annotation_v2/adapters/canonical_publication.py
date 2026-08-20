"""Atomic publication owner for canonical visible-mask archives."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..mask_schema import MASK_FORMAT_VERSION
from .canonical_masks import CanonicalMaskBundle, CanonicalMaskError


@dataclass(frozen=True, slots=True)
class CanonicalMaskPublisher:
    """Publish one validated v3 bundle through an atomic file replacement."""

    def publish(self, path: Path, bundle: CanonicalMaskBundle) -> Path:
        """Write ``bundle`` at ``path`` without exposing a partial archive."""

        if bundle.format_version != MASK_FORMAT_VERSION:
            raise CanonicalMaskError(
                "canonical publisher only supports "
                f"{MASK_FORMAT_VERSION}: {bundle.format_version}"
            )
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".npz",
            dir=target.parent,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            np.savez_compressed(temporary, **bundle.to_payload())
            os.replace(temporary, target)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return target


__all__ = ["CanonicalMaskPublisher"]
