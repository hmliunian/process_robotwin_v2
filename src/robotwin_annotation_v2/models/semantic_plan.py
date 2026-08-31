"""Stage-2 Qwen semantic plan and SAM3-native query contracts."""

from __future__ import annotations

import hashlib
import re
from dataclasses import InitVar, dataclass
from enum import StrEnum
from typing import Any, Literal, cast

from ..domain import (
    AnnotationMode,
    AnnotationSpec,
    TargetProfile,
    annotation_spec,
)
from .loop_context import EpisodeRef

RoleName = Literal["target", "receiver"]
CANDIDATE_FIELDS = (
    "category_query",
    "color_category_query",
    "shape_category_query",
    "general_fallback_query",
)
MAX_QUERY_WORDS = 4
_WORD = r"[a-z]+(?:-[a-z]+)*"
_QUERY_PATTERN = re.compile(rf"{_WORD}(?: {_WORD}){{0,{MAX_QUERY_WORDS - 1}}}")
_DOOR_OPEN_PANEL_PROXIES = frozenset({"door panel", "moving door panel"})
_DOOR_OPEN_REJECT_WORDS = frozenset({"bar", "body", "housing", "latch", "panel"})
_FORBIDDEN_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "my",
        "your",
        "this",
        "that",
        "with",
        "without",
        "inside",
        "outside",
        "near",
        "beside",
        "between",
        "behind",
        "under",
        "over",
        "above",
        "below",
        "left",
        "right",
        "foreground",
        "background",
        "grab",
        "grasp",
        "hold",
        "holding",
        "held",
        "lift",
        "move",
        "moving",
        "pick",
        "place",
        "put",
        "scan",
        "stack",
        "take",
        "transfer",
        "arm",
        "hand",
        "gripper",
        "item",
        "object",
        "stuff",
        "thing",
        "bigger",
        "biggest",
        "smaller",
        "smallest",
        "larger",
        "largest",
        "nearest",
        "farthest",
    }
)


class SemanticPlanError(ValueError):
    """Raised when a Qwen semantic response violates the stage contract."""


def canonical_target_profile(
    value: TargetProfile | str | None,
) -> TargetProfile | None:
    """Normalize a target profile, omitting the default grasp profile."""

    if value is None:
        return None
    if isinstance(value, TargetProfile):
        profile = value
    elif isinstance(value, str) and value.strip():
        normalized = value.strip().lower().replace("-", "_")
        aliases = {
            "origin": TargetProfile.GRASP_MANIPULATION.value,
            "graspmanipulation": TargetProfile.GRASP_MANIPULATION.value,
            "contactpress": TargetProfile.CONTACT_PRESS.value,
            "dooropen": TargetProfile.DOOR_OPEN.value,
        }
        try:
            profile = TargetProfile(aliases.get(normalized, normalized))
        except ValueError as exc:
            choices = ", ".join(item.value for item in TargetProfile)
            raise ValueError(
                f"target_profile must be one of: {choices}"
            ) from exc
    else:
        raise ValueError("target_profile must be a non-empty string or null")
    if profile is TargetProfile.GRASP_MANIPULATION:
        return None
    return profile


def normalize_query(
    value: Any,
    *,
    field: str = "query",
    allow_visual_object: bool = False,
    allow_moving_door_panel: bool = False,
) -> str:
    """Validate a direct SAM3 query's mechanically checkable constraints."""

    if not isinstance(value, str) or not value.strip():
        raise SemanticPlanError(f"{field} must be a non-empty string")
    normalized = " ".join(value.split())
    if normalized != normalized.lower():
        raise SemanticPlanError(f"{field} must be lowercase")
    if _QUERY_PATTERN.fullmatch(normalized) is None:
        raise SemanticPlanError(
            f"{field} must contain 1-{MAX_QUERY_WORDS} lowercase English words"
        )
    words = tuple(normalized.replace("-", " ").split())
    forbidden_words = set(words) & _FORBIDDEN_WORDS
    if allow_moving_door_panel and normalized == "moving door panel":
        forbidden_words.discard("moving")
    if allow_visual_object and len(words) > 1 and words[-1] == "object":
        forbidden_words.discard("object")
    forbidden = sorted(forbidden_words)
    if forbidden:
        raise SemanticPlanError(
            f"{field} contains forbidden descriptor(s): {', '.join(forbidden)}"
        )
    return normalized


def normalize_profile_query(
    value: Any,
    *,
    target_profile: TargetProfile | str | None,
    field: str = "query",
    allow_visual_object: bool = False,
) -> str:
    """Validate a query and enforce specialized target-profile semantics."""

    profile = canonical_target_profile(target_profile)
    if profile is not TargetProfile.DOOR_OPEN:
        return normalize_query(
            value,
            field=field,
            allow_visual_object=allow_visual_object,
        )
    if not isinstance(value, str) or not value.strip():
        raise SemanticPlanError(f"{field} must be a non-empty string")
    normalized = " ".join(value.split())
    panel_phrase = normalized.replace("-", " ")
    if panel_phrase in _DOOR_OPEN_PANEL_PROXIES:
        return normalize_query(
            panel_phrase,
            field=field,
            allow_visual_object=allow_visual_object,
            allow_moving_door_panel=True,
        )
    normalized = normalize_query(
        normalized,
        field=field,
        allow_visual_object=allow_visual_object,
    )
    words = normalized.replace("-", " ").split()
    if words[-1] != "handle":
        raise SemanticPlanError(
            f"{field} must use handle as the head noun or be an explicit door panel proxy"
        )
    if set(words) & _DOOR_OPEN_REJECT_WORDS:
        rejected = sorted(set(words) & _DOOR_OPEN_REJECT_WORDS)
        raise SemanticPlanError(
            f"{field} contains a non-handle door descriptor: {', '.join(rejected)}"
        )
    context_words = {"microwave"}
    if "microwave" in words:
        context_words.add("door")
    canonical_words = [word for word in words if word not in context_words]
    if len(canonical_words) > 2 and canonical_words[-2:] == ["door", "handle"]:
        canonical_words.pop(-2)
    canonical = " ".join(canonical_words)
    if not canonical:
        return "handle"
    # Keep the canonical form within the same mechanical query contract after
    # removing appliance context words (e.g. ``white microwave door handle``).
    return normalize_query(canonical, field=field)


@dataclass(frozen=True)
class QueryBank:
    """Ordered short queries generated by Qwen for one complete object."""

    category_query: str
    color_category_query: str | None = None
    shape_category_query: str | None = None
    general_fallback_query: str | None = None
    recommended_order: tuple[str, ...] = ()
    allow_moving_door_panel: InitVar[bool] = False

    def __post_init__(self, allow_moving_door_panel: bool) -> None:
        normalized: dict[str, str | None] = {}
        for field in CANDIDATE_FIELDS:
            value = getattr(self, field)
            if value is None:
                normalized[field] = None
            else:
                if allow_moving_door_panel:
                    normalized[field] = normalize_profile_query(
                        value,
                        target_profile=TargetProfile.DOOR_OPEN,
                        field=field,
                        allow_visual_object=field == "general_fallback_query",
                    )
                else:
                    normalized[field] = normalize_query(
                        value,
                        field=field,
                        allow_visual_object=field == "general_fallback_query",
                    )
        if normalized["category_query"] is None:
            raise SemanticPlanError("category_query is required")
        if len({value for value in normalized.values() if value is not None}) != len(
            [value for value in normalized.values() if value is not None]
        ):
            raise SemanticPlanError("query candidates must be distinct")
        panel_fields = tuple(
            field
            for field, value in normalized.items()
            if value in _DOOR_OPEN_PANEL_PROXIES
        )
        if allow_moving_door_panel and panel_fields:
            has_other_candidate = any(
                normalized[field] is not None for field in CANDIDATE_FIELDS if field != "category_query"
            )
            if panel_fields != ("category_query",) or has_other_candidate:
                raise SemanticPlanError(
                    "door-panel proxy must be the sole category_query"
                )

        order = tuple(self.recommended_order)
        expected = tuple(field for field in CANDIDATE_FIELDS if normalized[field] is not None)
        if not order:
            order = expected
        if len(order) != len(set(order)) or set(order) != set(expected):
            raise SemanticPlanError(
                "recommended_order must contain every non-null candidate exactly once"
            )
        if "general_fallback_query" in order and order[-1] != "general_fallback_query":
            raise SemanticPlanError("general_fallback_query must be last")
        for field, value in normalized.items():
            object.__setattr__(self, field, value)
        object.__setattr__(self, "recommended_order", order)

    @property
    def primary_query(self) -> str:
        value = getattr(self, self.recommended_order[0])
        assert value is not None
        return cast(str, value)

    def to_json(self) -> dict[str, Any]:
        return {
            "category_query": self.category_query,
            "color_category_query": self.color_category_query,
            "shape_category_query": self.shape_category_query,
            "general_fallback_query": self.general_fallback_query,
            "recommended_order": list(self.recommended_order),
            "primary_query": self.primary_query,
        }


class SemanticStatus(StrEnum):
    OK = "ok"
    NO_CLEAR_SEED = "no_clear_seed"


@dataclass(frozen=True)
class RoleSemanticPlan:
    """Qwen's semantic result for target or receiver."""

    role: RoleName
    status: SemanticStatus
    seed_frame_id: int | None
    query_bank: QueryBank | None
    exclude: tuple[str, ...]
    reason: str

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise ValueError("semantic reason must be non-empty")
        if self.status is SemanticStatus.OK:
            if self.seed_frame_id is None or self.query_bank is None:
                raise SemanticPlanError("ok role requires seed_frame_id and query_bank")
        elif self.seed_frame_id is not None or self.query_bank is not None:
            raise SemanticPlanError("no_clear_seed role cannot contain seed or queries")
        if self.seed_frame_id is not None and self.seed_frame_id < 0:
            raise ValueError("seed_frame_id must be non-negative")

    @property
    def primary_query(self) -> str | None:
        return None if self.query_bank is None else self.query_bank.primary_query

    def to_json(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "status": self.status.value,
            "seed_frame_id": self.seed_frame_id,
            **(self.query_bank.to_json() if self.query_bank is not None else {
                "category_query": None,
                "color_category_query": None,
                "shape_category_query": None,
                "general_fallback_query": None,
                "recommended_order": [],
                "primary_query": None,
            }),
            "exclude": list(self.exclude),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class SemanticPlan:
    """Exact role collection passed from semantic planning to object-mask stages."""

    episode: EpisodeRef
    role_plans: tuple[RoleSemanticPlan, ...]
    model: str
    prompt_sha256: str
    input_frame_ids: tuple[int, ...]
    raw_response: str
    annotation_mode: AnnotationMode = AnnotationMode.PICK_PLACE
    prompt_version: str = "object_roles_semantic_v2"
    # Keep this new optional field after the historical optional fields so
    # positional construction of SemanticPlan continues to interpret its
    # eighth argument as ``prompt_version``.
    target_profile: TargetProfile | None = None

    def __post_init__(self) -> None:
        expected = self.annotation_spec.required_role_names
        actual = tuple(plan.role for plan in self.role_plans)
        if actual != expected:
            raise ValueError(
                "SemanticPlan roles must exactly match annotation mode: "
                f"expected={expected}, actual={actual}"
            )
        if not self.model.strip():
            raise ValueError("model must be non-empty")
        if len(self.prompt_sha256) != 64:
            raise ValueError("prompt_sha256 must be a SHA-256 hex digest")
        if not self.input_frame_ids:
            raise ValueError("input_frame_ids must not be empty")
        try:
            target_profile = canonical_target_profile(self.target_profile)
        except ValueError as exc:
            raise ValueError("SemanticPlan target_profile is invalid") from exc
        if (
            target_profile in {TargetProfile.CONTACT_PRESS, TargetProfile.DOOR_OPEN}
            and self.annotation_mode is not AnnotationMode.TARGET_ONLY
        ):
            raise ValueError(
                f"{target_profile.value} target profile requires target_only mode"
            )
        object.__setattr__(self, "target_profile", target_profile)

    @property
    def usable(self) -> bool:
        return all(plan.primary_query is not None for plan in self.role_plans)

    @property
    def annotation_spec(self) -> AnnotationSpec:
        return annotation_spec(self.annotation_mode)

    def for_role(self, role: RoleName) -> RoleSemanticPlan:
        """Return one applicable role, failing clearly for a non-applicable role."""

        for plan in self.role_plans:
            if plan.role == role:
                return plan
        raise KeyError(f"role {role!r} is not applicable in {self.annotation_mode.value} mode")

    @property
    def target(self) -> RoleSemanticPlan:
        """Compatibility accessor for existing pick-place callers."""

        return self.for_role("target")

    @property
    def receiver(self) -> RoleSemanticPlan:
        """Compatibility accessor; target-only callers should iterate ``role_plans``."""

        return self.for_role("receiver")

    @staticmethod
    def prompt_hash(rendered_prompt: str) -> str:
        return hashlib.sha256(rendered_prompt.encode("utf-8")).hexdigest()

    def to_json(self) -> dict[str, Any]:
        payload = {
            "format_version": "robotwin_semantic_plan_v2",
            "prompt_version": self.prompt_version,
            "annotation_mode": self.annotation_mode.value,
            "required_object_roles": list(self.annotation_spec.required_role_names),
            "episode": self.episode.to_json(),
            "model": self.model,
            "prompt_sha256": self.prompt_sha256,
            "input_frame_ids": list(self.input_frame_ids),
            "roles": {plan.role: plan.to_json() for plan in self.role_plans},
            "raw_response": self.raw_response,
        }
        if self.target_profile is not None:
            payload["target_profile"] = self.target_profile.value
        return payload
