"""Pure ordered query planning for object-mask resolution."""

from __future__ import annotations

from dataclasses import dataclass

from ...models import LoopContext, RoleSemanticPlan
from ...models.loop_context import RoleName
from ..open_set_queries import curated_query_aliases


@dataclass(frozen=True)
class QueryCandidate:
    """One ordered semantic or curated text query."""

    field: str
    query: str


def plan_role_queries(
    context: LoopContext,
    role: RoleName,
    semantic: RoleSemanticPlan,
    *,
    query_fallback_enabled: bool,
) -> tuple[QueryCandidate, ...]:
    """Preserve semantic order, then append normalized unique curated aliases."""

    if semantic.query_bank is None:
        raise ValueError("role query planning requires a semantic query bank")
    candidates = [
        QueryCandidate(field, query)
        for field in semantic.query_bank.recommended_order
        if (query := getattr(semantic.query_bank, field)) is not None
    ]
    if query_fallback_enabled:
        candidates.extend(
            QueryCandidate(f"curated_alias_{index}", query)
            for index, query in enumerate(
                curated_query_aliases(context, role, semantic),
                start=1,
            )
        )

    output: list[QueryCandidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        query = " ".join(candidate.query.split())
        key = query.casefold()
        if not query or key in seen:
            continue
        seen.add(key)
        output.append(QueryCandidate(candidate.field, query))
    return tuple(output)


__all__ = ["QueryCandidate", "plan_role_queries"]
