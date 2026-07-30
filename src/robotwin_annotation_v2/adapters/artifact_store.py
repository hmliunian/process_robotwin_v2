"""Small filesystem artifact writer shared by all three stages."""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..models import EpisodeRef


class ArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()

    @staticmethod
    def new_run_id() -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"{stamp}-{uuid.uuid4().hex[:8]}"

    def run_dir(self, run_id: str) -> Path:
        if not run_id or "/" in run_id or ".." in run_id:
            raise ValueError("run_id must be a simple non-empty name")
        return self.root / run_id

    def episode_dir(self, run_id: str, ref: EpisodeRef) -> Path:
        return self.run_dir(run_id) / ref.task / f"episode_{ref.episode_id}" / ref.camera

    @staticmethod
    def write_json(path: Path, payload: dict[str, Any]) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            os.replace(temporary, path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return path

    def save_loop(self, run_id: str, ref: EpisodeRef, payload: dict[str, Any]) -> Path:
        return self.write_json(self.episode_dir(run_id, ref) / "loop.json", payload)
