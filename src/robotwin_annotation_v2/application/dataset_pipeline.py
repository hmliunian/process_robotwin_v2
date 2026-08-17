"""Public dataset-level orchestration facade.

This module deliberately contains sequencing, not image/model algorithms.  It
is the stable seam for applications, notebooks, and debug tooling; the legacy
``dataset_runtime`` module remains an implementation/compatibility module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..config import PipelineConfig
from ..terminal_ui import ProcessUI
from . import dataset_runtime as runtime


@dataclass(frozen=True)
class DatasetPipeline:
    """Coordinate discovery, object source, gripper backend, and rendering."""

    config: PipelineConfig

    def discover(self, dataset_root: Path, *, require_depth: bool = False) -> Any:
        """Run the input-discovery stage.

        Input: dataset root and camera from ``config``.
        Output: ``DiscoveryResult`` with complete episodes and exclusions.
        Side effects: filesystem reads only.
        Failure policy: malformed paths raise; incomplete episodes are recorded.
        """

        return runtime.discover_episodes(
            dataset_root,
            camera=self.config.dataset.camera,
            require_depth=require_depth,
        )

    def build_manifest(self, dataset_root: Path, discovery: Any) -> dict[str, Any]:
        """Build the immutable-in-run dynamic dataset manifest.

        Input: discovery output.
        Output: manifest mapping used by every downstream stage.
        Side effects: reads Parquet/video metadata; does not modify source data.
        Failure policy: reject empty or inconsistent episode metadata.
        """

        return runtime.build_dynamic_manifest(
            dataset_root,
            task=self.config.dataset.task,
            camera=self.config.dataset.camera,
            episodes=discovery.episodes,
        )

    def run_object_source(
        self,
        *,
        dataset_root: Path,
        output_root: Path,
        run_id: str | None = None,
        episode_ids: tuple[int, ...] | None = None,
        force: bool = False,
        incremental: bool = False,
        reporter: ProcessUI | None = None,
        backend_factory: Callable[..., Any] | None = None,
    ) -> dict[str, Any]:
        """Run the shared Qwen + object-SAM source stages.

        Input: discovered dataset and the task-declared annotation mode.
        Output: frozen object masks with canonical receiver N/A metadata when applicable.
        Side effects: Qwen/SAM inference and source-run artifact writes.
        Failure policy: continue ordinary episode failures, stop on fatal CUDA errors.
        """

        return runtime.process_dataset(
            self.config,
            dataset_root=dataset_root,
            task=self.config.dataset.task,
            camera=self.config.dataset.camera,
            output_root=output_root,
            run_id=run_id,
            episode_ids=episode_ids,
            force=force,
            skip_render=True,
            object_source_only=True,
            incremental_source=incremental,
            reporter=reporter,
            backend_factory=backend_factory,
        )

    def run_sam_dataset(
        self,
        *,
        dataset_root: Path,
        output_root: Path,
        run_id: str | None = None,
        episode_ids: tuple[int, ...] | None = None,
        force: bool = False,
        skip_render: bool = False,
        reporter: ProcessUI | None = None,
        backend_factory: Callable[..., Any] | None = None,
    ) -> dict[str, Any]:
        """Run object masks plus the SAM gripper backend.

        Input: complete dataset episodes.
        Output: full canonical four-channel run and optional review videos.
        Side effects: model inference, artifact writes, and video rendering.
        Failure policy: preserve per-episode failures and report ``passed=false``.
        """

        return runtime.process_dataset(
            self.config,
            dataset_root=dataset_root,
            task=self.config.dataset.task,
            camera=self.config.dataset.camera,
            output_root=output_root,
            run_id=run_id,
            episode_ids=episode_ids,
            force=force,
            skip_render=skip_render,
            object_source_only=False,
            reporter=reporter,
            backend_factory=backend_factory,
        )


__all__ = ["DatasetPipeline"]
