# Repository Guidelines

## Project Structure & Module Organization

Core Python code lives in `src/robotwin_annotation_v2/`. Keep domain models in `models/` and `domain/`, orchestration in `application/` and `pipeline/`, and integrations with datasets, Qwen, SAM3, or artifact storage in `adapters/`. Executable workflows belong in `scripts/`; YAML configuration, prompt templates, dataset manifests, and bundled URDF assets belong in `configs/`. Tests mirror behavior under `tests/unit/` and `tests/integration/`. Consult `docs/architecture.md` before changing stage contracts or the four-channel `masks.npz` format.

## Build, Test, and Development Commands

This project requires Python 3.13 and expects its environment at `.venv`.

- `just test` runs the fast CPU unit suite.
- `just test-all` runs unit and integration tests.
- `just lint` checks `src`, `tests`, and `scripts` with Ruff.
- `just format` applies Ruff formatting.
- `.venv/bin/python -m mypy src` performs strict type checking.
- `just preflight` validates the configured external dataset contract.
- `just loop 7152` exercises Stage 1 without model services.
- `just process <dataset_root> --ui plain` runs the complete dataset pipeline; start `just serve-qwen` separately when Qwen-backed stages are needed.

## Coding Style & Naming Conventions

Use four-space indentation, Ruff’s 100-character line limit, and Python 3.13 syntax. Add type annotations to new APIs; mypy runs in strict mode. Use `snake_case` for modules, functions, variables, and YAML keys, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants. Keep task-specific object names in prompts or data, not hard-coded in Python.

## Testing Guidelines

Pytest discovers `test_*.py` under `tests/`. Place isolated logic tests in `tests/unit/`; reserve `tests/integration/` for filesystem and dataset-contract workflows. Add a focused regression test for every bug fix and cover failure paths. No coverage threshold is enforced; changed behavior should run without GPU services in the unit suite.

## Data, Configuration & Artifacts

Do not commit external videos, Parquet/HDF5 datasets, checkpoints, or generated `artifacts/` and `run/` output. Commit reproducible manifests under `configs/datasets/`. Never include credentials or machine-specific absolute paths in configuration.

## Commit & Pull Request Guidelines

History follows concise Conventional Commit subjects such as `feat: encode held targets until release`, `fix: resume partial streaming URDF runs`, and `docs: record ...`. Keep each commit scoped and imperative. Pull requests should explain the pipeline contract affected, list validation commands, link the issue or experiment, and identify configuration or schema changes. For mask or rendering changes, attach representative overlay or review-sheet evidence and note the tested task, camera, and episode IDs.
