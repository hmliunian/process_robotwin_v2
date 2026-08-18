"""Config-driven curated aliases for open-set query fallback."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import yaml

from ..models import CANDIDATE_FIELDS, LoopContext, RoleSemanticPlan, normalize_query
from ..models.semantic_plan import RoleName

_CATALOG_PATH = Path(__file__).resolve().parents[3] / "configs/open_set_query_aliases.yaml"
_CATALOG_FORMAT = "robotwin_open_set_query_aliases_v1"
_COLOR_PHRASES = (
    "light brown",
    "dark brown",
    "light gray",
    "dark gray",
    "light green",
    "dark green",
    "light blue",
    "dark blue",
)
_COLOR_WORDS = frozenset(
    {
        "beige",
        "black",
        "blue",
        "brown",
        "cyan",
        "gray",
        "green",
        "grey",
        "lime",
        "magenta",
        "maroon",
        "olive",
        "orange",
        "pink",
        "purple",
        "red",
        "silver",
        "teal",
        "white",
        "yellow",
    }
)


@dataclass(frozen=True)
class _AliasRule:
    role: RoleName
    tasks: frozenset[str]
    query_words: frozenset[str]
    aliases: tuple[str, ...]


def _strings(value: Any, *, field: str, allow_empty: bool = True) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"{field} must be a list of non-empty strings")
    result = tuple(item.strip() for item in value)
    if not allow_empty and not result:
        raise ValueError(f"{field} must not be empty")
    return result


@cache
def _load_rules(path: Path = _CATALOG_PATH) -> tuple[_AliasRule, ...]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping) or raw.get("format_version") != _CATALOG_FORMAT:
        raise ValueError(f"invalid open-set alias catalog: {path}")
    entries = raw.get("rules")
    if not isinstance(entries, list):
        raise TypeError("open-set alias catalog rules must be a list")
    rules: list[_AliasRule] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise TypeError(f"open-set alias rule {index} must be a mapping")
        role = entry.get("role")
        if role not in {"target", "receiver"}:
            raise ValueError(f"open-set alias rule {index} has invalid role")
        tasks = _strings(entry.get("tasks", []), field=f"rules[{index}].tasks")
        query_words = _strings(
            entry.get("query_words", []),
            field=f"rules[{index}].query_words",
        )
        aliases = _strings(
            entry.get("aliases"),
            field=f"rules[{index}].aliases",
            allow_empty=False,
        )
        if not tasks and not query_words:
            raise ValueError(f"open-set alias rule {index} has no match condition")
        rules.append(
            _AliasRule(
                role=role,
                tasks=frozenset(tasks),
                query_words=frozenset(query_words),
                aliases=aliases,
            )
        )
    return tuple(rules)


def _query_values(semantic: RoleSemanticPlan) -> tuple[str, ...]:
    if semantic.query_bank is None:
        return ()
    return tuple(
        value
        for field in CANDIDATE_FIELDS
        if (value := getattr(semantic.query_bank, field)) is not None
    )


def _query_color(queries: Iterable[str]) -> str | None:
    values = tuple(queries)
    for phrase in _COLOR_PHRASES:
        if any(phrase in query for query in values):
            return phrase
    return next(
        (word for query in values for word in query.split() if word in _COLOR_WORDS),
        None,
    )


def _unique_aliases(aliases: Iterable[str], *, existing: Iterable[str]) -> tuple[str, ...]:
    seen = {normalize_query(query, allow_visual_object=True) for query in existing}
    result: list[str] = []
    for alias in aliases:
        normalized = normalize_query(alias, field="curated alias", allow_visual_object=True)
        if normalized in seen:
            continue
        result.append(normalized)
        seen.add(normalized)
        if len(result) == 3:
            break
    return tuple(result)


def curated_query_aliases(
    context: LoopContext,
    role: RoleName,
    semantic: RoleSemanticPlan,
) -> tuple[str, ...]:
    """Return up to three configured aliases not already present in the query bank."""

    if semantic.role != role:
        raise ValueError(f"semantic role {semantic.role!r} does not match requested role {role!r}")
    queries = _query_values(semantic)
    if not queries:
        return ()
    words = frozenset(word for query in queries for word in query.split())
    rule = next(
        (
            item
            for item in _load_rules()
            if item.role == role
            and (context.episode.task in item.tasks or bool(words & item.query_words))
        ),
        None,
    )
    if rule is None:
        return ()
    color = _query_color(queries)
    aliases = (
        alias.format(color=color)
        for alias in rule.aliases
        if "{color}" not in alias or color is not None
    )
    return _unique_aliases(aliases, existing=queries)


__all__ = ["curated_query_aliases"]
