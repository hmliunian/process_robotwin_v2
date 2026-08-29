#!/usr/bin/env python3
"""Run the dataset processor with configured API or managed local Qwen."""

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

from robotwin_annotation_v2.adapters.qwen_client import (
    OpenAICompatibleQwenClient,
    QwenServiceError,
)
from robotwin_annotation_v2.application.managed_qwen import (
    DEFAULT_MIN_FREE_MEMORY_MIB,
    DEFAULT_STARTUP_TIMEOUT_SECONDS,
    ManagedQwenError,
    ManagedQwenService,
    ManagedQwenSettings,
)
from robotwin_annotation_v2.config import (
    ConfigError,
    PipelineConfig,
    PipelineProfile,
    has_dataset_block,
    load_config,
    load_profile,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QWEN_API_KEY_FILE = PROJECT_ROOT / "secrets" / "qwen_api_key.txt"
QWEN_API_KEY_FILE_ENV = "QWEN_API_KEY_FILE"


class _TerminationRequested(BaseException):
    def __init__(self, signum: int) -> None:
        super().__init__(signum)
        self.signum = signum


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--qwen-python", type=Path)
    parser.add_argument("--qwen-model-path", type=Path)
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


def _load_qwen_api_key_file(pipeline_config: PipelineConfig | PipelineProfile) -> None:
    """Load one local API key without putting it on the command line or in YAML."""

    if pipeline_config.qwen.runtime != "api":
        return
    api_key_env = pipeline_config.qwen.api_key_env
    if not api_key_env or os.environ.get(api_key_env):
        return
    configured_path = os.environ.get(QWEN_API_KEY_FILE_ENV)
    key_path = Path(configured_path).expanduser() if configured_path else DEFAULT_QWEN_API_KEY_FILE
    if not key_path.is_absolute():
        key_path = PROJECT_ROOT / key_path
    try:
        value = key_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise ManagedQwenError(f"cannot read Qwen API key file {key_path}: {exc}") from exc
    if not value:
        return
    if any(character.isspace() for character in value):
        raise ManagedQwenError("Qwen API key file must contain exactly one non-empty line")
    os.environ[api_key_env] = value


def _settings(
    args: argparse.Namespace,
    pipeline_config: PipelineConfig | PipelineProfile,
) -> ManagedQwenSettings:
    if pipeline_config.qwen.runtime != "local":
        raise ManagedQwenError("managed Qwen startup requires qwen.runtime=local")
    if args.qwen_python is None or args.qwen_model_path is None:
        raise ManagedQwenError("local Qwen runtime requires --qwen-python and --qwen-model-path")
    excluded = set(pipeline_config.sam3.gpus)
    egl_gpu = _explicit_egl_gpu(args.process_args)
    if egl_gpu is not None:
        excluded.add(egl_gpu)
    return ManagedQwenSettings(
        endpoint=pipeline_config.qwen.endpoint,
        model_name=pipeline_config.qwen.model,
        # Keep the executable inside its virtual environment. Resolving this path follows
        # ``bin/python`` to the base interpreter, which bypasses the venv's pyvenv.cfg and
        # site-packages when the subprocess starts.
        python_executable=Path(os.path.abspath(args.qwen_python.expanduser())),
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


def _process_mode(arguments: Sequence[str]) -> str:
    """Return the mode needed to inspect a shared profile for service setup."""

    mode = _option_value(arguments, "--mode")
    if mode in {"pick_place", "target_only", "contact_press"}:
        return mode
    if mode is None:
        if "--target-only" in arguments or "--target_only" in arguments:
            return "target_only"
        if "--pick-place" in arguments or "--pick_place" in arguments:
            return "pick_place"
    # Runtime ownership is shared across modes; pick-place is the conservative
    # default when service management cannot infer a path mode.
    return "pick_place"


def _load_pipeline_config(
    config_path: Path,
    arguments: Sequence[str],
) -> PipelineConfig | PipelineProfile:
    """Load a reusable profile and retain a narrow legacy compatibility path."""

    mode = _process_mode(arguments)
    try:
        return load_profile(config_path, mode=mode)
    except ConfigError as profile_error:
        # Legacy fallback is valid only for an explicitly task-bound YAML.
        # Preserve field-level errors from malformed shared profiles.
        if config_path.expanduser().resolve().is_file() and not has_dataset_block(config_path):
            raise
        try:
            return load_config(config_path)
        except ConfigError as legacy_error:
            if "dataset block" in str(profile_error):
                raise legacy_error from profile_error
            raise profile_error from legacy_error


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
    config_path = _effective_config_path(args)
    pipeline_config = _load_pipeline_config(config_path, args.process_args)
    print(
        f"Process config: {pipeline_config.config_path} "
        f"(qwen.runtime={pipeline_config.qwen.runtime}, "
        f"qwen.model={pipeline_config.qwen.model})",
        file=sys.stderr,
        flush=True,
    )
    if not args.serve_only and not _process_requires_qwen(args.process_args):
        return _run_process(args.process_args, config_path)
    if pipeline_config.qwen.runtime == "api":
        if args.serve_only:
            raise ManagedQwenError("--serve-only requires qwen.runtime=local")
        _load_qwen_api_key_file(pipeline_config)
        try:
            OpenAICompatibleQwenClient.from_config(pipeline_config.qwen).health()
        except QwenServiceError as exc:
            raise ManagedQwenError(str(exc)) from exc
        return _run_process(args.process_args, config_path)
    service = ManagedQwenService(_settings(args, pipeline_config))
    with service:
        if args.serve_only:
            if not service.owns_process:
                return 0
            return int(service.wait())
        return _run_process(args.process_args, config_path)


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
