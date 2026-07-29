"""Port interfaces for artifact storage and retrieval."""

from pathlib import Path
from typing import Any, Protocol

from ..domain import ApprovedSeed, EpisodeRef, KeyframeRequest


class ArtifactRepository(Protocol):
    """Store and retrieve immutable run artifacts."""

    def create_run(self, config: dict[str, Any]) -> str:
        """
        Create a new run directory.

        Returns:
            run_id: e.g., "kf-20260729-abc123"
        """
        ...

    def save_request(
        self,
        run_id: str,
        request: KeyframeRequest,
        data: dict[str, Any],
    ) -> None:
        """Save keyframe request and its candidates."""
        ...

    def load_request(
        self,
        run_id: str,
        episode_id: str,
        slot_name: str,
    ) -> dict[str, Any]:
        """Load keyframe request data."""
        ...

    def save_approved_seed(
        self,
        run_id: str,
        seed: ApprovedSeed,
    ) -> None:
        """Save approved seed (immutable)."""
        ...

    def get_run_dir(self, run_id: str) -> Path:
        """Get run directory path."""
        ...
