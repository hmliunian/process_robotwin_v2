"""Terminal presentation for the dataset processing entry point."""

from __future__ import annotations

import os
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TextIO

UI_MODES = ("auto", "rich", "plain", "json")

_SUCCESS_STATUSES = {"completed"}
_SKIPPED_STATUSES = {"skipped_complete", "planned"}
_WARNING_STATUSES = {
    "dataset_excluded",
    "discovery_skipped",
    "source_excluded",
}


def _duration(seconds: float) -> str:
    rounded = max(0, int(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _record_counts(summary: Mapping[str, Any]) -> Counter[str]:
    raw_records = summary.get("records")
    if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes)):
        return Counter()
    statuses: list[str] = []
    for record in raw_records:
        if isinstance(record, Mapping) and record.get("status") is not None:
            statuses.append(str(record["status"]))
    return Counter(statuses)


def _status_level(status: str) -> str:
    if status in _SUCCESS_STATUSES:
        return "success"
    if status in _SKIPPED_STATUSES or status in _WARNING_STATUSES:
        return "warning"
    return "error"


class ProcessUI:
    """No-op UI and common interface used by the processing pipeline."""

    def __init__(self, *, emit_json_summary: bool, verbose: bool) -> None:
        self.emit_json_summary = emit_json_summary
        self.verbose = verbose

    def run_started(
        self,
        *,
        backend: str,
        dataset_root: str,
        task: str,
        camera: str,
    ) -> None:
        del backend, dataset_root, task, camera

    def run_ready(self, *, run_id: str, episode_ids: Sequence[int]) -> None:
        del run_id, episode_ids

    def phase_started(self, label: str, *, total: int | None = None) -> None:
        del label, total

    def phase_progress(
        self,
        completed: int,
        *,
        total: int | None = None,
        episode_id: int | None = None,
        status: str | None = None,
    ) -> None:
        del completed, total, episode_id, status

    def phase_finished(
        self,
        label: str,
        *,
        status: str = "completed",
        detail: str | None = None,
    ) -> None:
        del label, status, detail

    def episode_started(self, episode_id: int, *, position: int, total: int) -> None:
        del episode_id, position, total

    def stage_started(self, episode_id: int, label: str) -> None:
        del episode_id, label

    def stage_finished(
        self,
        episode_id: int,
        label: str,
        *,
        status: str = "completed",
        detail: str | None = None,
    ) -> None:
        del episode_id, label, status, detail

    def episode_finished(
        self,
        episode_id: int,
        *,
        status: str,
        detail: str | None = None,
    ) -> None:
        del episode_id, status, detail

    def note(self, message: str, *, level: str = "info") -> None:
        del message, level

    def detail(self, text: str) -> None:
        del text

    def finish(self, summary: Mapping[str, Any]) -> None:
        del summary

    def failed(self, exc: BaseException) -> None:
        del exc

    def close(self) -> None:
        return


class PlainProcessUI(ProcessUI):
    """Stable line-oriented output for logs and non-interactive terminals."""

    def __init__(
        self,
        *,
        stream: TextIO,
        emit_json_summary: bool,
        verbose: bool,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(emit_json_summary=emit_json_summary, verbose=verbose)
        self._stream = stream
        self._clock = clock
        self._run_started_at = clock()
        self._phase_started_at: float | None = None
        self._stage_started_at: dict[tuple[int, str], float] = {}
        self._finished_episodes: set[int] = set()

    def _write(self, message: str) -> None:
        rendered = message.replace("\r", "\\r").replace("\n", "\\n")
        self._stream.write(f"[process] {rendered}\n")
        self._stream.flush()

    def run_started(
        self,
        *,
        backend: str,
        dataset_root: str,
        task: str,
        camera: str,
    ) -> None:
        self._write(f"start backend={backend} task={task} camera={camera} dataset={dataset_root}")

    def run_ready(self, *, run_id: str, episode_ids: Sequence[int]) -> None:
        self._write(f"run={run_id} episodes={len(episode_ids)}")

    def phase_started(self, label: str, *, total: int | None = None) -> None:
        self._phase_started_at = self._clock()
        suffix = "" if total is None else f" total={total}"
        self._write(f"phase={label} status=running{suffix}")

    def phase_progress(
        self,
        completed: int,
        *,
        total: int | None = None,
        episode_id: int | None = None,
        status: str | None = None,
    ) -> None:
        parts = ["phase_progress", f"completed={completed}"]
        if total is not None:
            parts.append(f"total={total}")
        if episode_id is not None:
            parts.append(f"episode={episode_id:06d}")
        if status is not None:
            parts.append(f"status={status}")
        self._write(" ".join(parts))

    def phase_finished(
        self,
        label: str,
        *,
        status: str = "completed",
        detail: str | None = None,
    ) -> None:
        elapsed = 0.0
        if self._phase_started_at is not None:
            elapsed = self._clock() - self._phase_started_at
        message = f"phase={label} status={status} elapsed={_duration(elapsed)}"
        if detail:
            message += f" detail={detail}"
        self._write(message)
        self._phase_started_at = None

    def episode_started(self, episode_id: int, *, position: int, total: int) -> None:
        self._write(f"episode={episode_id:06d} status=running position={position}/{total}")

    def stage_started(self, episode_id: int, label: str) -> None:
        self._stage_started_at[(episode_id, label)] = self._clock()
        self._write(f"episode={episode_id:06d} stage={label} status=running")

    def stage_finished(
        self,
        episode_id: int,
        label: str,
        *,
        status: str = "completed",
        detail: str | None = None,
    ) -> None:
        started_at = self._stage_started_at.pop((episode_id, label), None)
        if started_at is None:
            started_at = self._clock()
        message = (
            f"episode={episode_id:06d} stage={label} status={status} "
            f"elapsed={_duration(self._clock() - started_at)}"
        )
        if detail:
            message += f" detail={detail}"
        self._write(message)

    def episode_finished(
        self,
        episode_id: int,
        *,
        status: str,
        detail: str | None = None,
    ) -> None:
        if episode_id in self._finished_episodes:
            return
        self._finished_episodes.add(episode_id)
        message = f"episode={episode_id:06d} status={status}"
        if detail:
            message += f" detail={detail}"
        self._write(message)

    def note(self, message: str, *, level: str = "info") -> None:
        self._write(f"level={level} {message}")

    def detail(self, text: str) -> None:
        if not self.verbose:
            return
        for line in text.splitlines():
            self._write(f"detail {line}")

    def finish(self, summary: Mapping[str, Any]) -> None:
        counts = _record_counts(summary)
        rendered_counts = ",".join(f"{status}:{count}" for status, count in sorted(counts.items()))
        self._write(
            "finish "
            f"passed={str(bool(summary.get('passed'))).lower()} "
            f"elapsed={_duration(self._clock() - self._run_started_at)} "
            f"results={rendered_counts or 'none'}"
        )
        artifact = summary.get("artifact")
        if artifact:
            self._write(f"artifact={artifact}")

    def failed(self, exc: BaseException) -> None:
        self._write(f"failed error={type(exc).__name__}: {exc}")


class RichProcessUI(ProcessUI):
    """Interactive Rich progress display."""

    def __init__(
        self,
        *,
        stream: TextIO,
        emit_json_summary: bool,
        verbose: bool,
        force_terminal: bool,
        no_color: bool,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(emit_json_summary=emit_json_summary, verbose=verbose)
        from rich.console import Console
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TaskProgressColumn,
            TextColumn,
            TimeElapsedColumn,
        )

        self._console = Console(
            file=stream,
            force_terminal=force_terminal,
            no_color=no_color,
            highlight=False,
        )
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self._console,
            auto_refresh=True,
        )
        self._overall_task = self._progress.add_task("Episodes", total=1, visible=False)
        self._phase_task = self._progress.add_task("Starting", total=None)
        self._clock = clock
        self._run_started_at = clock()
        self._finished_episodes: set[int] = set()
        self._counts: Counter[str] = Counter()
        self._run_id = ""
        self._episode_total = 0
        self._progress_started = False

    def _ensure_progress(self) -> None:
        if not self._progress_started:
            self._progress.start()
            self._progress_started = True

    def _log(self, message: str, *, style: str | None = None) -> None:
        from rich.text import Text

        self._ensure_progress()
        rendered = Text(message) if style is None else Text(message, style=style)
        self._console.print(rendered)

    def run_started(
        self,
        *,
        backend: str,
        dataset_root: str,
        task: str,
        camera: str,
    ) -> None:
        from rich.panel import Panel
        from rich.table import Table

        metadata = Table.grid(padding=(0, 2))
        metadata.add_column(style="bold cyan")
        metadata.add_column()
        metadata.add_row("Backend", backend.upper())
        metadata.add_row("Dataset", dataset_root)
        metadata.add_row("Task", task)
        metadata.add_row("Camera", camera)
        self._console.print(Panel.fit(metadata, title="RoboTwin Process", border_style="cyan"))
        self._ensure_progress()

    def run_ready(self, *, run_id: str, episode_ids: Sequence[int]) -> None:
        total = len(episode_ids)
        self._run_id = run_id
        self._episode_total = total
        self._log(f"Run {run_id} · {total} episodes", style="cyan")
        self._progress.update(
            self._overall_task,
            total=max(total, 1),
            completed=0,
            description=f"Episodes · run {run_id}",
            visible=True,
        )

    def phase_started(self, label: str, *, total: int | None = None) -> None:
        self._ensure_progress()
        self._progress.reset(
            self._phase_task,
            total=total,
            completed=0,
            description=label,
            visible=True,
        )

    def phase_progress(
        self,
        completed: int,
        *,
        total: int | None = None,
        episode_id: int | None = None,
        status: str | None = None,
    ) -> None:
        description_parts: list[str] = []
        if episode_id is not None:
            description_parts.append(f"episode_{episode_id:06d}")
        if status:
            description_parts.append(status)
        description = " · ".join(description_parts) or "Working"
        self._progress.update(
            self._phase_task,
            completed=completed,
            total=total,
            description=description,
        )

    def phase_finished(
        self,
        label: str,
        *,
        status: str = "completed",
        detail: str | None = None,
    ) -> None:
        icon = "✓" if status == "completed" else "!"
        style = "green" if status == "completed" else "red"
        task = self._progress.tasks[self._phase_task]
        total = task.total
        if total is None:
            self._progress.update(self._phase_task, total=1, completed=1)
        else:
            self._progress.update(self._phase_task, completed=total)
        self._progress.update(
            self._phase_task,
            description=f"[{style}]{icon} {label}[/{style}]",
        )
        if detail:
            self._log(detail, style=style)

    def episode_started(self, episode_id: int, *, position: int, total: int) -> None:
        self._progress.update(
            self._overall_task,
            description=f"Episodes · {position}/{total} · episode_{episode_id:06d}",
        )

    def stage_started(self, episode_id: int, label: str) -> None:
        self._progress.reset(
            self._phase_task,
            total=None,
            completed=0,
            description=f"episode_{episode_id:06d} · {label}",
            visible=True,
        )

    def stage_finished(
        self,
        episode_id: int,
        label: str,
        *,
        status: str = "completed",
        detail: str | None = None,
    ) -> None:
        level = _status_level(status)
        style = {"success": "green", "warning": "yellow", "error": "red"}[level]
        icon = {"success": "✓", "warning": "↷", "error": "✗"}[level]
        self._progress.update(
            self._phase_task,
            total=1,
            completed=1,
            description=(
                f"[{style}]{icon} episode_{episode_id:06d} · {label} · {status}[/{style}]"
            ),
        )
        if detail:
            self._log(detail, style=style)

    def episode_finished(
        self,
        episode_id: int,
        *,
        status: str,
        detail: str | None = None,
    ) -> None:
        if episode_id in self._finished_episodes:
            return
        self._finished_episodes.add(episode_id)
        self._counts[status] += 1
        self._progress.advance(self._overall_task)
        succeeded = sum(self._counts[value] for value in _SUCCESS_STATUSES)
        skipped = sum(self._counts[value] for value in _SKIPPED_STATUSES)
        failed = len(self._finished_episodes) - succeeded - skipped
        self._progress.update(
            self._overall_task,
            description=(
                f"Episodes · {len(self._finished_episodes)}/{self._episode_total} "
                f"· ✓ {succeeded} ↷ {skipped} ✗ {failed}"
            ),
        )
        level = _status_level(status)
        if level != "success":
            style = "yellow" if level == "warning" else "red"
            message = f"episode_{episode_id:06d}: {status}"
            if detail:
                message += f" — {detail}"
            self._log(message, style=style)

    def note(self, message: str, *, level: str = "info") -> None:
        style = {"info": "cyan", "warning": "yellow", "error": "red"}.get(level)
        self._log(message, style=style)

    def detail(self, text: str) -> None:
        if self.verbose and text:
            self._log(text, style="dim")

    def finish(self, summary: Mapping[str, Any]) -> None:
        from rich.panel import Panel
        from rich.table import Table

        self.close()
        passed = bool(summary.get("passed"))
        counts = _record_counts(summary)
        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold")
        table.add_column(justify="right")
        table.add_row("Result", "PASSED" if passed else "FAILED")
        run_id = summary.get("run_id") or self._run_id
        if run_id:
            table.add_row("Run", str(run_id))
        table.add_row("Elapsed", _duration(self._clock() - self._run_started_at))
        for status, count in sorted(counts.items()):
            table.add_row(status, str(count))
        artifact = summary.get("artifact")
        if artifact:
            table.add_row("Summary", str(artifact))
        backend = summary.get("backend")
        if isinstance(backend, Mapping) and backend.get("source_selection_complete") is False:
            table.add_row("Source selection", "partial")
        style = "green" if passed else "red"
        self._console.print(Panel.fit(table, title="Process Summary", border_style=style))

    def failed(self, exc: BaseException) -> None:
        from rich.text import Text

        self.close()
        self._console.print(
            Text.assemble(
                ("Process failed: ", "bold red"),
                f"{type(exc).__name__}: {exc}",
            )
        )

    def close(self) -> None:
        if self._progress_started:
            self._progress.stop()
            self._progress_started = False


def create_process_ui(
    mode: str,
    *,
    verbose: bool,
    stderr: TextIO | None = None,
    stdout_is_terminal: bool | None = None,
    stderr_is_terminal: bool | None = None,
    environ: Mapping[str, str] | None = None,
) -> ProcessUI:
    """Create the requested UI while keeping auto mode safe for redirected output."""

    if mode not in UI_MODES:
        raise ValueError(f"unsupported process UI mode: {mode}")
    selected_stderr = sys.stderr if stderr is None else stderr
    selected_environ = os.environ if environ is None else environ
    if stdout_is_terminal is None:
        stdout_is_terminal = bool(getattr(sys.stdout, "isatty", lambda: False)())
    if stderr_is_terminal is None:
        stderr_is_terminal = bool(getattr(selected_stderr, "isatty", lambda: False)())

    if mode == "json":
        return ProcessUI(emit_json_summary=True, verbose=verbose)

    auto_mode = mode == "auto"
    wants_rich = mode == "rich" or (
        auto_mode
        and stderr_is_terminal
        and not selected_environ.get("CI")
        and selected_environ.get("TERM", "") != "dumb"
    )
    emit_json_summary = auto_mode and not stdout_is_terminal
    if wants_rich:
        try:
            return RichProcessUI(
                stream=selected_stderr,
                emit_json_summary=emit_json_summary,
                verbose=verbose,
                force_terminal=True,
                no_color="NO_COLOR" in selected_environ,
            )
        except ImportError:
            if mode == "rich":
                raise

    return PlainProcessUI(
        stream=selected_stderr,
        emit_json_summary=emit_json_summary,
        verbose=verbose,
    )
