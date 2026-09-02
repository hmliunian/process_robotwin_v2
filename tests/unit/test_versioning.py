from __future__ import annotations

import tomllib
from datetime import date
from pathlib import Path

import pytest

import robotwin_annotation_v2
from scripts.bump_version import bump_version, next_version

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_package_has_one_semantic_version_source() -> None:
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert robotwin_annotation_v2.__version__ == "3.0.0"
    assert project["project"]["dynamic"] == ["version"]
    assert project["tool"]["setuptools"]["dynamic"]["version"] == {
        "attr": "robotwin_annotation_v2._version.__version__"
    }


@pytest.mark.parametrize(
    ("change", "expected"),
    (("major", "4.0.0"), ("minor", "3.1.0"), ("patch", "3.0.1"), ("3.2.1", "3.2.1")),
)
def test_next_version_uses_semver(change: str, expected: str) -> None:
    assert next_version("3.0.0", change) == expected


def test_bump_version_updates_source_and_changelog(tmp_path: Path) -> None:
    version_file = tmp_path / "_version.py"
    changelog_file = tmp_path / "CHANGELOG.md"
    version_file.write_text('__version__ = "3.0.0"\n', encoding="utf-8")
    changelog_file.write_text(
        "# Changelog\n\n## [Unreleased]\n\n### Added\n\n- pending\n",
        encoding="utf-8",
    )

    version = bump_version(
        "patch",
        version_file=version_file,
        changelog_file=changelog_file,
        release_date=date(2026, 9, 3),
    )

    assert version == "3.0.1"
    assert '__version__ = "3.0.1"' in version_file.read_text(encoding="utf-8")
    assert "## [3.0.1] - 2026-09-03" in changelog_file.read_text(encoding="utf-8")
