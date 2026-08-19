"""Small filesystem artifact writer shared by all three stages."""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..models import EpisodeRef

NDArray = np.ndarray[Any, Any]


class ArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()

    @staticmethod
    def new_run_id() -> str:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        return f"{stamp}-{uuid.uuid4().hex[:8]}"

    def run_dir(self, run_id: str) -> Path:
        if not run_id or "/" in run_id or ".." in run_id:
            raise ValueError("run_id must be a simple non-empty name")
        return self.root / run_id

    def episode_dir(self, run_id: str, ref: EpisodeRef) -> Path:
        return self.run_dir(run_id) / ref.task / f"episode_{ref.episode_id}" / ref.camera

    @staticmethod
    def write_json(path: Path, payload: dict[str, Any]) -> Path:
        return ArtifactStore.write_text(
            path,
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        )

    @staticmethod
    def write_text(path: Path, content: str) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(content)
            os.replace(temporary, path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return path

    @staticmethod
    def write_png(path: Path, array: NDArray, *, rgb: bool = False) -> Path:
        value = np.asarray(array)
        if rgb:
            if value.ndim != 3 or value.shape[2] != 3:
                raise ValueError(f"RGB image must have shape [H,W,3], got {value.shape}")
            image = Image.fromarray(value.astype(np.uint8, copy=False), mode="RGB")
        else:
            if value.ndim != 2:
                raise ValueError(f"mask image must have shape [H,W], got {value.shape}")
            image = Image.fromarray(value.astype(bool).astype(np.uint8) * 255, mode="L")
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".png",
            dir=path.parent,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            image.save(temporary, format="PNG")
            os.replace(temporary, path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return path

    @staticmethod
    def write_npz(path: Path, **arrays: NDArray) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".npz",
            dir=path.parent,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            np.savez_compressed(temporary, **arrays)
            os.replace(temporary, path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return path

    def save_loop(self, run_id: str, ref: EpisodeRef, payload: dict[str, Any]) -> Path:
        return self.write_json(self.episode_dir(run_id, ref) / "loop.json", payload)

    def save_semantic_plan(
        self,
        run_id: str,
        ref: EpisodeRef,
        payload: dict[str, Any],
        *,
        rendered_prompt: str,
        raw_response: str,
    ) -> dict[str, Path]:
        episode_dir = self.episode_dir(run_id, ref)
        return {
            "semantic_plan": self.write_json(episode_dir / "semantic_plan.json", payload),
            "rendered_prompt": self.write_text(
                episode_dir / "qwen_rendered_prompt.txt",
                rendered_prompt,
            ),
            "raw_response": self.write_text(
                episode_dir / "qwen_raw_response.txt",
                raw_response,
            ),
        }

    def save_qwen_failure(
        self,
        run_id: str,
        ref: EpisodeRef,
        *,
        error: str,
        rendered_prompt: str | None,
        raw_response: str | None,
    ) -> dict[str, Path]:
        episode_dir = self.episode_dir(run_id, ref)
        paths = {
            "failure": self.write_json(
                episode_dir / "qwen_failure.json",
                {
                    "format_version": "robotwin_qwen_failure_v1",
                    "error": error,
                },
            )
        }
        if rendered_prompt is not None:
            paths["rendered_prompt"] = self.write_text(
                episode_dir / "qwen_rendered_prompt.txt",
                rendered_prompt,
            )
        if raw_response is not None:
            paths["raw_response"] = self.write_text(
                episode_dir / "qwen_raw_response.txt",
                raw_response,
            )
        return paths

    def save_sam_failure(
        self,
        run_id: str,
        ref: EpisodeRef,
        *,
        error: str,
    ) -> Path:
        return self.write_json(
            self.episode_dir(run_id, ref) / "sam_failure.json",
            {
                "format_version": "robotwin_sam_failure_v1",
                "error": error,
            },
        )
