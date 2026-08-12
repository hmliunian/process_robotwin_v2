"""Terminal presentation for the dataset processing entry point."""

from __future__ import annotations

import os
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TextIO

UI_MODES = ("auto", "rich", "plain", "json")

_SUCCESS_STATUSES = {
    "complete",
    "completed",
    "ok",
    "passed",
    "rendered",
    "succeeded",
    "success",
    "validated",
}
_SKIPPED_STATUSES = {
    "dataset_excluded",
    "discovery_skipped",
    "planned",
    "skipped",
    "skipped_complete",
    "source_excluded",
}
_WARNING_STATUSES = {
    "dataset_excluded",
    "discovery_skipped",
    "source_excluded",
}
_RUNNING_STATUSES = {"in_progress", "processing", "running"}
_PENDING_STATUSES = {"pending", "queued", "ready", "waiting"}


@dataclass(slots=True)
class _EpisodeState:
    status: str = "pending"
    detail: str | None = None
    position: int | None = None


@dataclass(slots=True)
class _LaneState:
    label: str
    total: int | None = None
    completed: int = 0
    episode_id: int | None = None
    status: str = "running"
    detail: str | None = None
    started_at: float = 0.0


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


def _episode_category(status: str) -> str:
    """Map pipeline-specific terminal values onto the dashboard counters."""

    normalized = status.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in _SUCCESS_STATUSES:
        return "success"
    if normalized in _SKIPPED_STATUSES or normalized.startswith("skipped_"):
        return "skipped"
    if normalized == "not_run" or normalized.startswith("not_run_"):
        return "not_run"
    if normalized in _RUNNING_STATUSES:
        return "running"
    if normalized in _PENDING_STATUSES or not normalized:
        return "pending"
    return "failed"


def _one_line(value: str, *, limit: int = 180) -> str:
    rendered = " ".join(value.split())
    if len(rendered) <= limit:
        return rendered
    return f"{rendered[: max(0, limit - 1)]}…"


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

    def lane_started(
        self,
        name: str,
        label: str,
        total: int | None = None,
    ) -> None:
        del name, label, total

    def lane_progress(
        self,
        name: str,
        completed: int,
        total: int | None = None,
        episode_id: int | None = None,
        status: str | None = None,
        detail: str | None = None,
    ) -> None:
        del name, completed, total, episode_id, status, detail

    def lane_finished(
        self,
        name: str,
        status: str = "completed",
        detail: str | None = None,
    ) -> None:
        del name, status, detail

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
        self._lane_started_at: dict[str, float] = {}
        self._episode_states: dict[int, _EpisodeState] = {}

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
        for episode_id in episode_ids:
            self._episode_states.setdefault(episode_id, _EpisodeState())
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

    def lane_started(
        self,
        name: str,
        label: str,
        total: int | None = None,
    ) -> None:
        self._lane_started_at[name] = self._clock()
        suffix = "" if total is None else f" total={total}"
        self._write(f"lane={name} label={label} status=running{suffix}")

    def lane_progress(
        self,
        name: str,
        completed: int,
        total: int | None = None,
        episode_id: int | None = None,
        status: str | None = None,
        detail: str | None = None,
    ) -> None:
        parts = [f"lane={name}", "status=running", f"completed={completed}"]
        if total is not None:
            parts.append(f"total={total}")
        if episode_id is not None:
            parts.append(f"episode={episode_id:06d}")
            state = self._episode_states.setdefault(episode_id, _EpisodeState())
            if _episode_category(state.status) == "pending":
                state.status = "running"
        if status is not None:
            parts.append(f"item_status={status}")
        if detail:
            parts.append(f"detail={detail}")
        self._write(" ".join(parts))

    def lane_finished(
        self,
        name: str,
        status: str = "completed",
        detail: str | None = None,
    ) -> None:
        started_at = self._lane_started_at.pop(name, None)
        if started_at is None:
            started_at = self._clock()
        message = f"lane={name} status={status} elapsed={_duration(self._clock() - started_at)}"
        if detail:
            message += f" detail={detail}"
        self._write(message)

    def episode_started(self, episode_id: int, *, position: int, total: int) -> None:
        self._episode_states[episode_id] = _EpisodeState(
            status="running",
            position=position,
        )
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
        previous = self._episode_states.get(episode_id)
        if previous is not None and previous.status == status and previous.detail == detail:
            return
        position = None if previous is None else previous.position
        self._episode_states[episode_id] = _EpisodeState(
            status=status,
            detail=detail,
            position=position,
        )
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
    """Interactive, in-place dashboard for sequential and pipelined runs."""

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

        self._console = Console(
            file=stream,
            force_terminal=force_terminal,
            no_color=no_color,
            highlight=False,
        )
        self._clock = clock
        self._run_started_at = clock()
        self._backend = ""
        self._dataset_root = ""
        self._task = ""
        self._camera = ""
        self._run_id = ""
        self._episode_total = 0
        self._episode_states: dict[int, _EpisodeState] = {}
        self._lanes: dict[str, _LaneState] = {}
        self._phase: _LaneState | None = None
        self._current_episode_id: int | None = None
        self._current_position: int | None = None
        self._current_stage = ""
        self._current_stage_status = ""
        self._latest_message = "Starting"
        self._latest_level = "info"
        self._live: Any | None = None

    def _ensure_live(self) -> None:
        if self._live is not None:
            return
        from rich.live import Live

        self._live = Live(
            self._render_dashboard(),
            console=self._console,
            auto_refresh=True,
            refresh_per_second=8,
            transient=True,
            redirect_stdout=False,
            redirect_stderr=False,
            vertical_overflow="visible",
        )
        self._live.start(refresh=True)

    def _refresh(self) -> None:
        self._ensure_live()
        live = self._live
        assert live is not None
        live.update(self._render_dashboard(), refresh=False)

    def _set_message(self, message: str, *, level: str = "info") -> None:
        self._latest_message = _one_line(message)
        self._latest_level = level

    def _mark_episode_running(self, episode_id: int) -> None:
        state = self._episode_states.setdefault(episode_id, _EpisodeState())
        if _episode_category(state.status) == "pending":
            state.status = "running"

    def _episode_stats(self) -> dict[str, int]:
        counts: Counter[str] = Counter(
            _episode_category(state.status) for state in self._episode_states.values()
        )
        total = max(self._episode_total, len(self._episode_states))
        accounted = sum(
            counts[name] for name in ("success", "failed", "skipped", "running", "not_run")
        )
        return {
            "total": total,
            "success": counts["success"],
            "failed": counts["failed"],
            "skipped": counts["skipped"],
            "running": counts["running"],
            "remaining": max(0, total - accounted),
            "not_run": counts["not_run"],
        }

    def _overall_equivalent(self, stats: Mapping[str, int]) -> float:
        finalized = stats["success"] + stats["failed"] + stats["skipped"]
        lane_fractions = [
            min(max(lane.completed / lane.total, 0.0), 1.0)
            for lane in self._lanes.values()
            if lane.total is not None and lane.total > 0
        ]
        if not lane_fractions or stats["total"] <= 0:
            return float(finalized)
        lane_equivalent = stats["total"] * sum(lane_fractions) / len(lane_fractions)
        return max(float(finalized), lane_equivalent)

    @staticmethod
    def _progress_text(
        completed: float,
        total: int | None,
        *,
        status: str | None = None,
    ) -> Any:
        from rich.text import Text

        text = Text()
        if total is None or total <= 0:
            text.append("…", style="cyan")
            text.append(f" {completed}")
        else:
            bounded = min(max(completed, 0), total)
            ratio = bounded / total
            width = 24
            filled = min(width, int(ratio * width))
            text.append("█" * filled, style="green")
            text.append("░" * (width - filled), style="bright_black")
            rendered_completed = (
                str(int(completed)) if float(completed).is_integer() else f"{completed:.1f}"
            )
            text.append(f"  {rendered_completed}/{total}  {ratio:5.1%}")
        if status:
            category = _episode_category(status)
            style = {
                "success": "green",
                "failed": "red",
                "skipped": "yellow",
                "running": "cyan",
                "pending": "bright_black",
                "not_run": "magenta",
            }[category]
            text.append(f"  {status}", style=style)
        return text

    def _lane_text(self, name: str, lane: _LaneState) -> Any:
        from rich.text import Text

        text = Text()
        text.append(name, style="bold")
        if lane.label and lane.label != name:
            text.append(f" · {lane.label}")
        if lane.episode_id is not None:
            text.append(f" · episode_{lane.episode_id:06d}", style="cyan")
        text.append("  ")
        text.append_text(self._progress_text(lane.completed, lane.total, status=lane.status))
        if lane.detail:
            text.append(f" · {_one_line(lane.detail, limit=100)}", style="dim")
        return text

    def _render_dashboard(self) -> Any:
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        stats = self._episode_stats()
        finalized = stats["success"] + stats["failed"] + stats["skipped"]
        completed = self._overall_equivalent(stats)
        table = Table.grid(expand=True, padding=(0, 1))
        table.add_column(width=10, no_wrap=True, style="bold cyan")
        table.add_column(ratio=1, overflow="fold")

        context_parts = []
        if self._run_id:
            context_parts.append(f"run {self._run_id}")
        if self._camera:
            context_parts.append(self._camera)
        if self._dataset_root:
            context_parts.append(self._dataset_root)
        table.add_row("Context", " · ".join(context_parts) or "Preparing run")

        overall = self._progress_text(completed, stats["total"])
        if self._lanes:
            overall.append(f"  equivalent · finalized {finalized}/{stats['total']}")
        overall.append(f"  elapsed {_duration(self._clock() - self._run_started_at)}", style="dim")
        table.add_row("Overall", overall)

        current = Text()
        if self._current_episode_id is None:
            current.append("Waiting for an episode", style="dim")
        else:
            current.append(f"episode_{self._current_episode_id:06d}", style="bold cyan")
            if self._current_position is not None and stats["total"]:
                current.append(f" · {self._current_position}/{stats['total']}")
            if self._current_stage:
                current.append(f" · {self._current_stage}")
            if self._current_stage_status:
                current.append(f" · {self._current_stage_status}", style="dim")
        table.add_row("Current", current)

        if self._phase is not None:
            table.add_row("Phase", self._lane_text("phase", self._phase))
        for name, lane in self._lanes.items():
            table.add_row("Lane", self._lane_text(name, lane))

        stats_text = Text()
        stat_fields = (
            ("Success", "success", "green"),
            ("Failed", "failed", "red"),
            ("Skipped", "skipped", "yellow"),
            ("Running", "running", "cyan"),
            ("Remaining", "remaining", "white"),
            ("Not-run", "not_run", "magenta"),
        )
        for index, (label, key, style) in enumerate(stat_fields):
            if index:
                stats_text.append("  ")
            stats_text.append(f"{label} ", style="dim")
            stats_text.append(str(stats[key]), style=f"bold {style}")
        table.add_row("Stats", stats_text)

        message_style = {
            "info": "cyan",
            "warning": "yellow",
            "error": "red",
        }.get(self._latest_level, "white")
        table.add_row("Message", Text(self._latest_message, style=message_style))

        title_parts = ["RoboTwin Process", "just process"]
        if self._backend:
            title_parts.append(self._backend.upper())
        if self._task:
            title_parts.append(self._task)
        return Panel(
            table,
            title=" · ".join(title_parts),
            border_style="cyan",
            padding=(0, 1),
        )

    def run_started(
        self,
        *,
        backend: str,
        dataset_root: str,
        task: str,
        camera: str,
    ) -> None:
        self._backend = backend
        self._dataset_root = dataset_root
        self._task = task
        self._camera = camera
        self._set_message("Preparing dataset contract")
        self._refresh()

    def run_ready(self, *, run_id: str, episode_ids: Sequence[int]) -> None:
        self._run_id = run_id
        self._episode_total = len(episode_ids)
        for episode_id in episode_ids:
            self._episode_states.setdefault(episode_id, _EpisodeState())
        self._set_message(f"Run {run_id} ready with {len(episode_ids)} episodes")
        self._refresh()

    def phase_started(self, label: str, *, total: int | None = None) -> None:
        self._phase = _LaneState(
            label=label,
            total=total,
            completed=0,
            status="running",
            started_at=self._clock(),
        )
        self._set_message(f"Phase {label} started")
        self._refresh()

    def phase_progress(
        self,
        completed: int,
        *,
        total: int | None = None,
        episode_id: int | None = None,
        status: str | None = None,
    ) -> None:
        if self._phase is None:
            self._phase = _LaneState(label="Working", started_at=self._clock())
        self._phase.completed = completed
        if total is not None:
            self._phase.total = total
        self._phase.episode_id = episode_id
        self._phase.status = status or "running"
        self._refresh()

    def phase_finished(
        self,
        label: str,
        *,
        status: str = "completed",
        detail: str | None = None,
    ) -> None:
        if self._phase is None:
            self._phase = _LaneState(label=label, started_at=self._clock())
        self._phase.label = label
        self._phase.status = status
        self._phase.detail = detail
        if _episode_category(status) == "success":
            if self._phase.total is None:
                self._phase.total = 1
            self._phase.completed = self._phase.total
        level = "info" if _episode_category(status) == "success" else "error"
        self._set_message(detail or f"Phase {label} {status}", level=level)
        self._refresh()

    def lane_started(
        self,
        name: str,
        label: str,
        total: int | None = None,
    ) -> None:
        self._lanes[name] = _LaneState(
            label=label,
            total=total,
            status="running",
            started_at=self._clock(),
        )
        self._set_message(f"Lane {name} started: {label}")
        self._refresh()

    def lane_progress(
        self,
        name: str,
        completed: int,
        total: int | None = None,
        episode_id: int | None = None,
        status: str | None = None,
        detail: str | None = None,
    ) -> None:
        lane = self._lanes.get(name)
        if lane is None:
            lane = _LaneState(label=name, started_at=self._clock())
            self._lanes[name] = lane
        lane.completed = completed
        if total is not None:
            lane.total = total
        lane.episode_id = episode_id
        lane.status = status or "running"
        lane.detail = detail
        if episode_id is not None:
            self._mark_episode_running(episode_id)
        if detail:
            self._set_message(detail)
        self._refresh()

    def lane_finished(
        self,
        name: str,
        status: str = "completed",
        detail: str | None = None,
    ) -> None:
        lane = self._lanes.get(name)
        if lane is None:
            lane = _LaneState(label=name, started_at=self._clock())
            self._lanes[name] = lane
        lane.status = status
        lane.detail = detail
        if _episode_category(status) == "success":
            if lane.total is None:
                lane.total = 1
            lane.completed = lane.total
        level = "info" if _episode_category(status) == "success" else "error"
        self._set_message(detail or f"Lane {name} {status}", level=level)
        self._refresh()

    def episode_started(self, episode_id: int, *, position: int, total: int) -> None:
        previous = self._episode_states.get(episode_id)
        self._episode_states[episode_id] = _EpisodeState(
            status="running",
            detail=None if previous is None else previous.detail,
            position=position,
        )
        self._episode_total = max(self._episode_total, total)
        self._current_episode_id = episode_id
        self._current_position = position
        self._current_stage = ""
        self._current_stage_status = "running"
        self._refresh()

    def stage_started(self, episode_id: int, label: str) -> None:
        self._mark_episode_running(episode_id)
        self._current_episode_id = episode_id
        self._current_stage = label
        self._current_stage_status = "running"
        self._set_message(f"episode_{episode_id:06d} · {label}")
        self._refresh()

    def stage_finished(
        self,
        episode_id: int,
        label: str,
        *,
        status: str = "completed",
        detail: str | None = None,
    ) -> None:
        self._current_episode_id = episode_id
        self._current_stage = label
        self._current_stage_status = status
        level = _status_level(status)
        message = detail or f"episode_{episode_id:06d} · {label} · {status}"
        message_level = {"success": "info", "warning": "warning", "error": "error"}[level]
        self._set_message(message, level=message_level)
        self._refresh()

    def episode_finished(
        self,
        episode_id: int,
        *,
        status: str,
        detail: str | None = None,
    ) -> None:
        previous = self._episode_states.get(episode_id)
        if previous is not None and previous.status == status and previous.detail == detail:
            return
        position = None if previous is None else previous.position
        self._episode_states[episode_id] = _EpisodeState(
            status=status,
            detail=detail,
            position=position,
        )
        self._current_episode_id = episode_id
        self._current_position = position
        self._current_stage_status = status
        level = _status_level(status)
        message = f"episode_{episode_id:06d}: {status}"
        if detail:
            message += f" — {detail}"
        message_level = {"success": "info", "warning": "warning", "error": "error"}[level]
        self._set_message(message, level=message_level)
        self._refresh()

    def note(self, message: str, *, level: str = "info") -> None:
        self._set_message(message, level=level)
        self._refresh()

    def detail(self, text: str) -> None:
        if self.verbose and text:
            self._set_message(text)
            self._refresh()

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
        if self._live is not None:
            self._live.stop()
            self._live = None


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
