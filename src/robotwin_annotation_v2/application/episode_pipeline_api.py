"""Public, readable facade for the per-episode annotation pipeline.

The implementation details remain in :mod:`episode_pipeline` for backwards
compatibility with the stage CLI.  This facade is the API that new callers and
debug launchers should use: one method per pipeline stage, in execution order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..config import PipelineConfig
from . import episode_pipeline as runtime


@dataclass(frozen=True)
class EpisodePipeline:
    """Run the shared stages for one configured episode.

    ``target_only`` is not a second algorithm.  The configured annotation
    contract changes the required object role set; all other stages stay on
    this same path.
    """

    config: PipelineConfig

    def preflight(self) -> None:
        """Validate dataset metadata.

        Input: the dataset and regression IDs in ``config``.
        Output: none; a failing contract raises ``ValueError``.
        Side effects: reads dataset metadata only.
        Failure policy: fail closed before any model or mask artifact is written.
        """

        runtime.run_preflight(self.config)

    def build_loop(self, episode_index: int, run_id: str | None = None) -> None:
        """Execute Stage 1 (state loop and sparse semantic frames).

        Input: one configured episode.
        Output: ``loop.json`` and a compact JSON report.
        Side effects: creates or updates the selected run's loop artifact.
        Failure policy: abort the episode when state events are invalid.
        """

        runtime.run_loop(self.config, episode_index, run_id)

    def plan_semantics(self, episode_index: int, run_id: str | None = None) -> None:
        """Execute Stage 2 (Qwen semantic role and query planning).

        Input: Stage-1 loop plus sparse RGB frames and the configured text prompt.
        Output: ``semantic_plan.json`` and prompt/response provenance.
        Side effects: performs one Qwen request and writes stage artifacts.
        Failure policy: preserve the failed request and stop; never invent a role.
        """

        runtime.run_qwen(self.config, episode_index, run_id)

    def annotate_objects(self, episode_index: int, run_id: str) -> None:
        """Execute Stage 3 (Qwen-QC-selected SAM object propagation).

        Input: saved loop and semantic plan.
        Output: canonical object channels, native tracks, and temporal QC artifacts.
        Side effects: loads one SAM3 backend and writes episode mask artifacts.
        Failure policy: fail closed on missing/ambiguous role masks.
        """

        runtime.run_sam(self.config, episode_index, run_id)

    def annotate_gripper(self, episode_index: int, run_id: str) -> None:
        """Execute Stage 4 (configured gripper producer).

        Input: completed object artifacts and state/depth dependencies.
        Output: active gripper channel in the canonical four-channel run.
        Side effects: runs the selected gripper backend and writes QC/provenance.
        Failure policy: preserve diagnostics and reject incomplete gripper output.
        """

        runtime.run_gripper(self.config, episode_index, run_id)

    def run(self, episode_index: int, run_id: str | None = None) -> None:
        """Run the complete single-episode sequence.

        Input: one episode index and optional run ID.
        Output: Stage 1 → Stage 4 artifacts under one run ID.
        Side effects: invokes Qwen/SAM/gripper services and writes all artifacts.
        Failure policy: stop at the first failed stage, retaining its receipt.
        """

        runtime.run_pipeline(self.config, episode_index, run_id)


__all__ = ["EpisodePipeline"]
