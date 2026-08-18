#!/usr/bin/env python3
"""Run the dataset processor with an automatically managed local Qwen service."""

from __future__ import annotations

import argparse
import contextlib
import os
import signal
import subprocess
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from types import FrameType
from typing import Any

from robotwin_annotation_v2.application.managed_qwen import (
    DEFAULT_MIN_FREE_MEMORY_MIB,
    DEFAULT_STARTUP_TIMEOUT_SECONDS,
    ManagedQwenError,
    ManagedQwenService,
    ManagedQwenSettings,
)
from robotwin_annotation_v2.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class _TerminationRequested(BaseException):
    def __init__(self, signum: int) -> None:
        super().__init__(signum)
        self.signum = signum


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--qwen-python", type=Path, required=True)
    parser.add_argument("--qwen-model-path", type=Path, required=True)
    parser.add_argument(
        "--qwen-min-free-memory-mib",
        type=int,
        default=DEFAULT_MIN_FREE_MEMORY_MIB,
    )
    parser.add_argument(
        "--qwen-startup-timeout",
        type=float,
        default=DEFAULT_STARTUP_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--serve-only",
        action="store_true",
        help="Start or reuse Qwen and wait instead of launching process_dataset.py",
    )
    parser.add_argument("process_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.process_args[:1] == ["--"]:
        args.process_args = args.process_args[1:]
    return args


def _option_value(arguments: Sequence[str], option: str) -> str | None:
    selected: str | None = None
    for index, argument in enumerate(arguments):
        if argument == option:
            selected = arguments[index + 1] if index + 1 < len(arguments) else None
        prefix = f"{option}="
        if argument.startswith(prefix):
            selected = argument[len(prefix) :]
    return selected


def _process_requires_qwen(arguments: Sequence[str]) -> bool:
    if "--help" in arguments or "-h" in arguments:
        return False
    if "--dry-run" in arguments or "--resume" in arguments:
        return False
    source_run_dir = _option_value(arguments, "--source-run-dir")
    return source_run_dir is None or source_run_dir.strip() in {"", "-"}


def _explicit_egl_gpu(arguments: Sequence[str]) -> int | None:
    value = _option_value(arguments, "--urdf-egl-device-id")
    if value is None:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ManagedQwenError(f"invalid --urdf-egl-device-id: {value!r}") from exc


def _effective_config_path(args: argparse.Namespace) -> Path:
    process_config = _option_value(args.process_args, "--config")
    return args.config if process_config is None else Path(process_config)


def _settings(args: argparse.Namespace, config_path: Path) -> ManagedQwenSettings:
    pipeline_config = load_config(config_path)
    excluded = set(pipeline_config.sam3.gpus)
    egl_gpu = _explicit_egl_gpu(args.process_args)
    if egl_gpu is not None:
        excluded.add(egl_gpu)
    return ManagedQwenSettings(
        endpoint=pipeline_config.qwen.endpoint,
        model_name=pipeline_config.qwen.model,
        python_executable=args.qwen_python.expanduser().resolve(),
        server_script=PROJECT_ROOT / "scripts" / "serve_qwen.py",
        model_path=args.qwen_model_path.expanduser().resolve(),
        log_directory=PROJECT_ROOT / "artifacts" / "qwen-services",
        excluded_gpu_ids=tuple(sorted(excluded)),
        minimum_free_memory_mib=args.qwen_min_free_memory_mib,
        startup_timeout_seconds=args.qwen_startup_timeout,
    )


def _signal_process_group(
    process: subprocess.Popen[bytes],
    signum: int,
    *,
    timeout_seconds: float = 15.0,
) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _run_process(arguments: Sequence[str], config_path: Path) -> int:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "process_dataset.py"),
        "--config",
        str(config_path),
        *arguments,
    ]
    process = subprocess.Popen(command, start_new_session=True)
    try:
        return process.wait()
    except KeyboardInterrupt:
        _signal_process_group(process, signal.SIGINT)
        raise
    except _TerminationRequested as exc:
        _signal_process_group(process, exc.signum)
        raise


@contextlib.contextmanager
def _termination_handlers() -> Iterator[None]:
    previous: dict[signal.Signals, Any] = {}

    def request_termination(signum: int, _frame: FrameType | None) -> None:
        raise _TerminationRequested(signum)

    for signum in (signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, request_termination)
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _run(args: argparse.Namespace) -> int:
    if not args.serve_only and not _process_requires_qwen(args.process_args):
        return _run_process(args.process_args, args.config)
    service = ManagedQwenService(_settings(args, _effective_config_path(args)))
    with service:
        if args.serve_only:
            if not service.owns_process:
                return 0
            return int(service.wait())
        return _run_process(args.process_args, args.config)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        with _termination_handlers():
            return _run(args)
    except KeyboardInterrupt:
        return 130
    except _TerminationRequested as exc:
        return 128 + exc.signum
    except ManagedQwenError as exc:
        print(f"managed Qwen error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
