from __future__ import annotations

import io
import json
from collections.abc import Mapping
from typing import Any

import pytest

from robotwin_annotation_v2.terminal_ui import (
    PlainProcessUI,
    ProcessUI,
    RichProcessUI,
    create_process_ui,
)
from scripts.process_dataset import _captured_json_progress, _captured_stage_output


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class RecordingUI(ProcessUI):
    def __init__(self, *, verbose: bool = False) -> None:
        super().__init__(emit_json_summary=False, verbose=verbose)
        self.progress: list[tuple[int, int | None, int | None, str | None]] = []
        self.details: list[str] = []

    def phase_progress(
        self,
        completed: int,
        *,
        total: int | None = None,
        episode_id: int | None = None,
        status: str | None = None,
    ) -> None:
        self.progress.append((completed, total, episode_id, status))

    def detail(self, text: str) -> None:
        if self.verbose and text:
            self.details.append(text)


def _summary(*, passed: bool = True) -> Mapping[str, Any]:
    return {
        "passed": passed,
        "artifact": "artifacts/runs/test/process_summary.json",
        "records": [
            {"episode": 7, "status": "completed"},
            {"episode": 8, "status": "skipped_complete"},
            {"episode": 9, "status": "failed"},
            {"status": "render_failed"},
        ],
    }


def test_plain_ui_emits_stable_stage_and_summary_lines() -> None:
    stream = io.StringIO()
    clock = FakeClock()
    ui = PlainProcessUI(
        stream=stream,
        emit_json_summary=True,
        verbose=False,
        clock=clock,
    )

    ui.run_started(
        backend="sam",
        dataset_root="/dataset",
        task="move_object",
        camera="cam_high",
    )
    ui.run_ready(run_id="run-test", episode_ids=(7, 8, 9))
    ui.phase_started("dataset_discovery", total=3)
    clock.advance(2)
    ui.phase_progress(1, total=3, episode_id=7, status="complete")
    ui.phase_finished("dataset_discovery")
    ui.episode_started(7, position=1, total=3)
    ui.stage_started(7, "qwen")
    clock.advance(3)
    ui.stage_finished(7, "qwen")
    ui.episode_finished(7, status="completed")
    ui.episode_finished(8, status="skipped_complete")
    ui.finish(_summary())

    output = stream.getvalue()
    assert "\x1b[" not in output
    assert "\r" not in output
    assert "start backend=sam task=move_object camera=cam_high dataset=/dataset" in output
    assert "phase=dataset_discovery status=completed elapsed=00:02" in output
    assert "episode=000007 stage=qwen status=completed elapsed=00:03" in output
    assert "completed:1" in output
    assert "render_failed:1" in output
    assert "artifact=artifacts/runs/test/process_summary.json" in output
    assert ui.emit_json_summary


def test_plain_ui_only_replays_captured_detail_in_verbose_mode() -> None:
    quiet_stream = io.StringIO()
    quiet = PlainProcessUI(
        stream=quiet_stream,
        emit_json_summary=False,
        verbose=False,
    )
    quiet.detail('{"roles": ["large payload"]}')

    verbose_stream = io.StringIO()
    verbose = PlainProcessUI(
        stream=verbose_stream,
        emit_json_summary=False,
        verbose=True,
    )
    verbose.detail('{"roles": ["large payload"]}')

    assert "large payload" not in quiet_stream.getvalue()
    assert "large payload" in verbose_stream.getvalue()


def test_auto_non_tty_uses_plain_ui_and_preserves_json_summary() -> None:
    ui = create_process_ui(
        "auto",
        verbose=False,
        stderr=io.StringIO(),
        stdout_is_terminal=False,
        stderr_is_terminal=False,
        environ={"TERM": "xterm-256color"},
    )

    assert isinstance(ui, PlainProcessUI)
    assert ui.emit_json_summary


@pytest.mark.parametrize(
    ("stdout_is_terminal", "emit_json_summary"),
    ((True, False), (False, True)),
)
def test_auto_interactive_stderr_uses_rich_and_preserves_redirected_json(
    stdout_is_terminal: bool,
    emit_json_summary: bool,
) -> None:
    pytest.importorskip("rich")
    ui = create_process_ui(
        "auto",
        verbose=False,
        stderr=io.StringIO(),
        stdout_is_terminal=stdout_is_terminal,
        stderr_is_terminal=True,
        environ={"TERM": "xterm-256color"},
    )

    assert isinstance(ui, RichProcessUI)
    assert ui.emit_json_summary is emit_json_summary
    ui.close()


@pytest.mark.parametrize(
    "environment",
    (
        {"TERM": "dumb"},
        {"TERM": "xterm-256color", "CI": "1"},
    ),
)
def test_auto_uses_plain_ui_for_noninteractive_environments(
    environment: Mapping[str, str],
) -> None:
    ui = create_process_ui(
        "auto",
        verbose=False,
        stderr=io.StringIO(),
        stdout_is_terminal=True,
        stderr_is_terminal=True,
        environ=environment,
    )

    assert isinstance(ui, PlainProcessUI)
    assert not ui.emit_json_summary


def test_json_ui_is_silent_and_requests_one_final_object() -> None:
    stream = io.StringIO()
    ui = create_process_ui(
        "json",
        verbose=True,
        stderr=stream,
        stdout_is_terminal=True,
        stderr_is_terminal=True,
    )

    ui.run_started(backend="sam", dataset_root="/dataset", task="task", camera="cam")
    ui.note("hidden")
    ui.finish(_summary())

    assert type(ui) is ProcessUI
    assert ui.emit_json_summary
    assert stream.getvalue() == ""


def test_embedded_stage_json_is_captured_and_replayed_only_as_detail(
    capsys: pytest.CaptureFixture[str],
) -> None:
    ui = RecordingUI(verbose=True)

    with _captured_stage_output(ui):
        print(json.dumps({"stage": "qwen", "roles": ["target", "receiver"]}))

    captured = capsys.readouterr()
    assert captured.out == ""
    assert len(ui.details) == 1
    assert '"stage": "qwen"' in ui.details[0]


def test_embedded_urdf_json_lines_become_progress_events(
    capsys: pytest.CaptureFixture[str],
) -> None:
    ui = RecordingUI(verbose=True)

    with _captured_json_progress(ui):
        print(
            json.dumps(
                {
                    "progress": "2/5",
                    "episode_index": 42,
                    "status": "complete",
                }
            )
        )
        print("renderer diagnostic")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert ui.progress == [(2, 5, 42, "complete")]
    assert ui.details == ["renderer diagnostic"]


def test_rich_ui_renders_and_closes_without_color_codes_when_requested() -> None:
    pytest.importorskip("rich")
    stream = io.StringIO()
    ui = create_process_ui(
        "rich",
        verbose=False,
        stderr=stream,
        stdout_is_terminal=True,
        stderr_is_terminal=True,
        environ={"TERM": "xterm-256color", "NO_COLOR": ""},
    )

    assert isinstance(ui, RichProcessUI)
    ui.run_started(backend="urdf", dataset_root="/dataset", task="task", camera="cam")
    ui.run_ready(run_id="rich-run", episode_ids=(1,))
    ui.episode_started(1, position=1, total=1)
    ui.stage_started(1, "canonical_publish")
    ui.stage_finished(1, "canonical_publish")
    ui.episode_finished(1, status="completed")
    ui.finish(
        {
            "passed": True,
            "records": [{"episode": 1, "status": "completed"}],
        }
    )
    ui.close()

    output = stream.getvalue()
    assert "RoboTwin Process" in output
    assert "Process Summary" in output
    assert "rich-run" in output
    assert "\x1b[31m" not in output
    assert "\x1b[32m" not in output
