"""Port interfaces for external capabilities.

These are Protocol definitions that adapters must implement.
Application layer depends on these, not on concrete implementations.
"""

from typing import Any, Protocol

from ..domain import EpisodeRef, InteractionTimeline, SemanticPlan


class EpisodeRepository(Protocol):
    """Access to RoboTwin episode data."""

    def load_metadata(self, ref: EpisodeRef) -> dict[str, Any]:
        """Load episode metadata (length, dimensions, etc)."""
        ...

    def load_state(self, ref: EpisodeRef) -> dict[str, Any]:
        """Load gripper state trajectory."""
        ...


class SemanticPlanner(Protocol):
    """Determine roles and text queries from task type."""

    def plan(self, ref: EpisodeRef) -> SemanticPlan:
        """Generate semantic plan (target/receiver queries)."""
        ...


class TimelineDetector(Protocol):
    """Detect action boundaries from state/gripper."""

    def detect(self, ref: EpisodeRef, state: dict[str, Any]) -> InteractionTimeline:
        """Extract timeline (move_start, close_start, etc)."""
        ...
