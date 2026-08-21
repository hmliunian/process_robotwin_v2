"""Public dataset-level orchestration coordinator.

The coordinator owns workflow selection and its typed dependencies.  The
legacy :mod:`dataset_runtime` entry points adapt to this class; importing or
using the public pipeline does not depend on that compatibility module.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..config import PipelineConfig
from ..domain import GripperBackend
from ..models import ProcessRequest
from ..terminal_ui import ProcessUI
from .discovery import (
    DiscoveryResult,
    build_dynamic_manifest,
    discover_episodes,
)
from .sam_workflow import (
    SamWorkflow,
    SamWorkflowHooks,
    default_sam_workflow_hooks,
)


class DatasetBackendRunner(Protocol):
    """Execute one typed dataset request for a configured backend.

    Backend-specific policy is deliberately captured when the runner is
    constructed instead of widening :meth:`DatasetPipeline.run` with options
    that only apply to SAM or URDF.
    """

    def __call__(
        self,
        request: ProcessRequest,
        *,
        reporter: ProcessUI | None = None,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class DatasetPipeline:
    """Coordinate discovery, object source, gripper backend, and rendering."""

    config: PipelineConfig
    sam_hooks: SamWorkflowHooks[Any, Any, Any] | None = field(
        default=None,
        repr=False,
    )
    sam_runner: DatasetBackendRunner | None = field(default=None, repr=False)
    urdf_runner: DatasetBackendRunner | None = field(default=None, repr=False)

    def run(
        self,
        request: ProcessRequest,
        *,
        backend: GripperBackend | None = None,
        reporter: ProcessUI | None = None,
    ) -> dict[str, Any]:
        """Dispatch ``request`` to exactly one configured backend runner.

        The annotation spec owns the default backend.  An unavailable selected
        backend fails closed; the coordinator never falls back to the other
        runner because doing so would change the requested mask provenance.
        """

        selected_backend = (
            self.config.annotation.spec.default_gripper_backend
            if backend is None
            else backend
        )
        runner = {
            GripperBackend.SAM: self.sam_runner,
            GripperBackend.URDF: self.urdf_runner,
        }[selected_backend]
        if runner is None:
            raise RuntimeError(
                f"no dataset runner configured for {selected_backend.value!r} backend"
            )
        return runner(request, reporter=reporter)

    def _sam_workflow(self) -> SamWorkflow[Any, Any, Any]:
        hooks = self.sam_hooks
        if hooks is None:
            hooks = default_sam_workflow_hooks()
        return SamWorkflow(self.config, hooks)

    def discover(self, dataset_root: Path, *, require_depth: bool = False) -> DiscoveryResult:
        """Run the input-discovery stage.

        Input: dataset root and camera from ``config``.
        Output: ``DiscoveryResult`` with complete episodes and exclusions.
        Side effects: filesystem reads only.
        Failure policy: malformed paths raise; incomplete episodes are recorded.
        """

        return discover_episodes(
            dataset_root,
            camera=self.config.dataset.camera,
            require_depth=require_depth,
        )

    def build_manifest(
        self,
        dataset_root: Path,
        discovery: DiscoveryResult,
    ) -> dict[str, Any]:
        """Build the immutable-in-run dynamic dataset manifest.

        Input: discovery output.
        Output: manifest mapping used by every downstream stage.
        Side effects: reads Parquet/video metadata; does not modify source data.
        Failure policy: reject empty or inconsistent episode metadata.
        """

        return build_dynamic_manifest(
            dataset_root,
            task=self.config.dataset.task,
            camera=self.config.dataset.camera,
            episodes=discovery.episodes,
        )

    def run_sam(
        self,
        *,
        dataset_root: Path,
        output_root: Path,
        task: str | None = None,
        camera: str | None = None,
        run_id: str | None = None,
        episode_ids: tuple[int, ...] | None = None,
        force: bool = False,
        skip_render: bool = False,
        object_source_only: bool = False,
        report_lifecycle: bool = True,
        incremental_source: bool = False,
        episode_terminal_callback: Callable[[int, str], None] | None = None,
        reporter: ProcessUI | None = None,
        backend_factory: Callable[..., Any] | None = None,
    ) -> dict[str, Any]:
        """Execute the canonical SAM workflow with explicit lifecycle policy.

        New callers normally use :meth:`run_object_source` or
        :meth:`run_sam_dataset` instead.
        """

        return self._sam_workflow().run(
            dataset_root=dataset_root,
            task=self.config.dataset.task if task is None else task,
            camera=self.config.dataset.camera if camera is None else camera,
            output_root=output_root,
            run_id=run_id,
            episode_ids=episode_ids,
            force=force,
            skip_render=skip_render,
            object_source_only=object_source_only,
            report_lifecycle=report_lifecycle,
            incremental_source=incremental_source,
            episode_terminal_callback=episode_terminal_callback,
            reporter=reporter,
            backend_factory=backend_factory,
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
        report_lifecycle: bool = True,
        episode_terminal_callback: Callable[[int, str], None] | None = None,
        reporter: ProcessUI | None = None,
        backend_factory: Callable[..., Any] | None = None,
    ) -> dict[str, Any]:
        """Run the shared Qwen + object-SAM source stages.

        Input: discovered dataset and the task-declared annotation mode.
        Output: frozen object masks with canonical receiver N/A metadata when applicable.
        Side effects: Qwen/SAM inference and source-run artifact writes.
        Failure policy: continue ordinary episode failures, stop on fatal CUDA errors.
        """

        return self.run_sam(
            dataset_root=dataset_root,
            output_root=output_root,
            run_id=run_id,
            episode_ids=episode_ids,
            force=force,
            skip_render=True,
            object_source_only=True,
            incremental_source=incremental,
            report_lifecycle=report_lifecycle,
            episode_terminal_callback=episode_terminal_callback,
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
        report_lifecycle: bool = True,
        episode_terminal_callback: Callable[[int, str], None] | None = None,
        reporter: ProcessUI | None = None,
        backend_factory: Callable[..., Any] | None = None,
    ) -> dict[str, Any]:
        """Run object masks plus the SAM gripper backend.

        Input: complete dataset episodes.
        Output: full canonical four-channel run and optional review videos.
        Side effects: model inference, artifact writes, and video rendering.
        Failure policy: preserve per-episode failures and report ``passed=false``.
        """

        return self.run_sam(
            dataset_root=dataset_root,
            output_root=output_root,
            run_id=run_id,
            episode_ids=episode_ids,
            force=force,
            skip_render=skip_render,
            object_source_only=False,
            report_lifecycle=report_lifecycle,
            episode_terminal_callback=episode_terminal_callback,
            reporter=reporter,
            backend_factory=backend_factory,
        )


__all__ = ["DatasetBackendRunner", "DatasetPipeline"]
