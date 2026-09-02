"""Show, validate, or advance the project version and changelog."""

from __future__ import annotations

import argparse
import re
import tomllib
from datetime import UTC, date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = ROOT / "src/robotwin_annotation_v2/_version.py"
CHANGELOG_FILE = ROOT / "CHANGELOG.md"
PYPROJECT_FILE = ROOT / "pyproject.toml"
VERSION_RE = re.compile(r'^__version__ = "(\d+\.\d+\.\d+)"$', re.MULTILINE)


def read_version(path: Path = VERSION_FILE) -> str:
    match = VERSION_RE.search(path.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"missing semantic version in {path}")
    return match.group(1)


def next_version(current: str, change: str) -> str:
    major, minor, patch = (int(part) for part in current.split("."))
    if change == "major":
        return f"{major + 1}.0.0"
    if change == "minor":
        return f"{major}.{minor + 1}.0"
    if change == "patch":
        return f"{major}.{minor}.{patch + 1}"
    if re.fullmatch(r"\d+\.\d+\.\d+", change) is None:
        raise ValueError("change must be major, minor, patch, or X.Y.Z")
    if tuple(map(int, change.split("."))) <= (major, minor, patch):
        raise ValueError("new version must be greater than the current version")
    return change


def bump_version(
    change: str,
    *,
    version_file: Path = VERSION_FILE,
    changelog_file: Path = CHANGELOG_FILE,
    release_date: date | None = None,
) -> str:
    current = read_version(version_file)
    updated = next_version(current, change)
    version_text, count = VERSION_RE.subn(
        f'__version__ = "{updated}"',
        version_file.read_text(encoding="utf-8"),
    )
    marker = "## [Unreleased]\n"
    changelog_text = changelog_file.read_text(encoding="utf-8")
    if count != 1 or changelog_text.count(marker) != 1:
        raise ValueError("version file or changelog does not match the release template")
    released = release_date or datetime.now(UTC).date()
    changelog_text = changelog_text.replace(
        marker,
        f"{marker}\n## [{updated}] - {released.isoformat()}\n",
        1,
    )
    version_file.write_text(version_text, encoding="utf-8")
    changelog_file.write_text(changelog_text, encoding="utf-8")
    return updated


def check_version() -> str:
    version = read_version()
    project = tomllib.loads(PYPROJECT_FILE.read_text(encoding="utf-8"))
    if "version" not in project["project"].get("dynamic", []):
        raise ValueError("pyproject version must be dynamic")
    attr = project["tool"]["setuptools"]["dynamic"]["version"].get("attr")
    if attr != "robotwin_annotation_v2._version.__version__":
        raise ValueError("pyproject does not use the package version source")
    if f"## [{version}]" not in CHANGELOG_FILE.read_text(encoding="utf-8"):
        raise ValueError(f"CHANGELOG.md has no {version} release")
    return version


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("change", nargs="?", help="major, minor, patch, or X.Y.Z")
    parser.add_argument("--check", action="store_true", help="validate version wiring")
    args = parser.parse_args()
    if args.check and args.change:
        parser.error("--check cannot be combined with a version change")
    version = check_version() if args.check else (
        bump_version(args.change) if args.change else read_version()
    )
    print(version)


if __name__ == "__main__":
    main()
