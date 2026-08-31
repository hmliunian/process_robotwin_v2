"""Small, JSON-safe provenance helpers shared by dataset workflows.

The pipeline has a number of public metadata files (dynamic manifests,
process summaries and per-episode manifests).  Keeping the profile and prompt
identity calculation in one dependency-free module prevents those files from
slowly developing different interpretations of the same run.

The helpers deliberately accept duck-typed configuration objects.  A few
legacy callers and unit-test doubles only expose a subset of ``PipelineConfig``
and should continue to produce the old metadata envelope when no prompt files
are available.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from robotwin_annotation_v2.domain import TargetProfile

PROMPT_BUNDLE_FORMAT_VERSION = "robotwin_prompt_bundle_v1"
OBJECT_RESOLUTION_FORMAT_VERSION = "robotwin_object_resolution_v1"
_WORKFLOW_PROFILE_ALIASES = frozenset(
    {
        "pick_place",
        "pickplace",
        "target_only",
        "targetonly",
        "shared",
    }
)
_SEMANTIC_PROFILE_ALIASES = frozenset(
    {
        "origin",
        "grasp_manipulation",
        "graspmanipulation",
        "contact_press",
        "contactpress",
        "door_open",
        "dooropen",
    }
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _file_identity(path: Any) -> dict[str, Any] | None:
    """Return a stable prompt-file identity, or ``None`` for unavailable files."""

    if path is None:
        return None
    try:
        candidate = Path(path).expanduser()
        # Prompt loaders follow symlinks, so provenance must hash the resolved
        # target rather than silently omitting a prompt from the bundle.
        # ``resolve`` also makes the recorded diagnostic path unambiguous.
        resolved = candidate.resolve()
        if not resolved.is_file():
            return None
        digest = hashlib.sha256()
        with resolved.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return {
            "path": str(resolved),
            "sha256": digest.hexdigest(),
            "bytes": resolved.stat().st_size,
        }
    except (OSError, RuntimeError, TypeError, ValueError):
        return None


def _profile_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, TargetProfile):
        return value.value
    if not isinstance(value, str) or not value.strip():
        raise ValueError("target_profile must be a non-empty string")
    normalized = value.strip().lower().replace("-", "_")
    aliases = {
        "origin": TargetProfile.GRASP_MANIPULATION.value,
        "graspmanipulation": TargetProfile.GRASP_MANIPULATION.value,
        "contactpress": TargetProfile.CONTACT_PRESS.value,
        "dooropen": TargetProfile.DOOR_OPEN.value,
    }
    normalized = aliases.get(normalized, normalized)
    try:
        return TargetProfile(normalized).value
    except ValueError as exc:
        choices = ", ".join(profile.value for profile in TargetProfile)
        raise ValueError(f"target_profile must be one of: {choices}") from exc


def target_profile_from_config(config: Any) -> str | None:
    """Extract the canonical target profile from a config-like object."""

    annotation = getattr(config, "annotation", None)
    profile = _profile_value(getattr(annotation, "profile", None))
    if profile is not None:
        return profile
    return None


def target_profile_from_manifest(manifest: Mapping[str, Any] | None) -> str | None:
    """Extract a profile from either modern or historical manifest spellings."""

    if not isinstance(manifest, Mapping):
        return None
    explicit_profile = _profile_value(manifest.get("target_profile"))
    # ``profile`` historically overloaded annotation mode.  Only map known
    # semantic aliases here; ordinary workflow values must not be mistaken for
    # a target profile during source-run validation.
    raw = manifest.get("profile")
    legacy_profile: str | None = None
    if raw is not None and (not isinstance(raw, str) or not raw.strip()):
        raise ValueError("manifest profile must be a non-empty string")
    if isinstance(raw, str) and raw.strip():
        normalized = raw.strip().lower().replace("-", "_")
        if normalized in _SEMANTIC_PROFILE_ALIASES:
            legacy_profile = _profile_value(normalized)
        elif normalized not in _WORKFLOW_PROFILE_ALIASES:
            choices = ", ".join(
                sorted(_WORKFLOW_PROFILE_ALIASES | _SEMANTIC_PROFILE_ALIASES)
            )
            raise ValueError(f"manifest profile must be one of: {choices}")
    if (
        explicit_profile is not None
        and legacy_profile is not None
        and explicit_profile != legacy_profile
    ):
        raise ValueError("manifest target_profile conflicts with semantic profile alias")
    return explicit_profile or legacy_profile


def _validate_prompt_bundle(bundle: Mapping[str, Any], *, label: str) -> None:
    recorded = bundle.get("sha256")
    if (
        not isinstance(recorded, str)
        or len(recorded) != 64
        or any(character not in "0123456789abcdef" for character in recorded)
    ):
        raise ValueError(f"{label} has no valid sha256")
    unhashed = _prompt_bundle_content_payload(bundle)
    actual = hashlib.sha256(_canonical_json(unhashed)).hexdigest()
    if actual == recorded:
        return
    # Bundles written before the portable hash contract included absolute
    # diagnostic paths in their digest.  Accept those historical envelopes so
    # existing source runs remain readable; all newly generated bundles use
    # the path-independent digest above.
    legacy_payload = {key: value for key, value in bundle.items() if key != "sha256"}
    legacy_actual = hashlib.sha256(_canonical_json(legacy_payload)).hexdigest()
    if legacy_actual != recorded:
        raise ValueError(f"{label} content hash is invalid")


def _prompt_bundle_content_payload(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Return the portable, content-only payload covered by a bundle hash.

    Prompt paths are useful diagnostics, but they are machine-local and must
    not affect the identity of an otherwise identical prompt bundle.  Keep
    them in the recorded metadata while hashing only the template content
    identities and the semantic profile.
    """

    payload = {key: value for key, value in bundle.items() if key != "sha256"}
    templates = payload.get("templates")
    if isinstance(templates, Mapping):
        payload["templates"] = {
            name: (
                {key: value for key, value in identity.items() if key != "path"}
                if isinstance(identity, Mapping)
                else identity
            )
            for name, identity in templates.items()
        }
    return payload


def _prompt_bundle_profile(bundle: Mapping[str, Any], *, label: str) -> str | None:
    """Resolve a bundle's semantic profile, including the legacy ``profile`` key."""

    explicit = _profile_value(bundle.get("target_profile"))
    raw_legacy = bundle.get("profile")
    if raw_legacy is not None and (
        not isinstance(raw_legacy, str) or not raw_legacy.strip()
    ):
        raise ValueError(f"{label} profile must be a non-empty string")
    legacy = target_profile_from_manifest({"profile": raw_legacy})
    if raw_legacy is not None and legacy is None:
        normalized_legacy = raw_legacy.strip().lower().replace("-", "_")
        if normalized_legacy not in {
            "pick_place",
            "pickplace",
            "target_only",
            "targetonly",
        }:
            legacy = _profile_value(raw_legacy)
    if explicit is not None and legacy is not None and explicit != legacy:
        raise ValueError(f"{label} target_profile conflicts with profile")
    return explicit or legacy


def _prompt_bundles_equivalent(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> bool:
    """Compare bundles by content identity, ignoring diagnostic paths."""

    return _prompt_bundle_content_payload(left) == _prompt_bundle_content_payload(right)


def prompt_bundle_for_config(
    config: Any,
    *,
    target_profile: str | TargetProfile | None = None,
) -> dict[str, Any] | None:
    """Build a content-addressed prompt-template bundle for ``config``.

    Missing files are omitted rather than treated as a fatal error.  Config
    validation remains responsible for deciding whether a required template
    must exist; this helper is used at metadata boundaries where legacy test
    doubles may intentionally provide no paths.
    """

    qwen = getattr(config, "qwen", None)
    mask = getattr(config, "mask", None)
    templates = {
        "semantic": _file_identity(getattr(qwen, "prompt_template", None)),
        "mask_qc": _file_identity(getattr(mask, "qc_prompt_template", None)),
        "bbox_localization": _file_identity(
            getattr(mask, "qc_bbox_prompt_template", None)
        ),
    }
    # The gripper seed-QC stage resolves this same path in ``episode_pipeline``.
    # Newer config-like objects may expose it explicitly; old PipelineConfig
    # instances do not, so derive the checked-in sibling path as a compatibility
    # fallback.  Hashing the actual file here keeps the persisted bundle tied to
    # every prompt used by a full SAM run.
    gripper_path = getattr(config, "gripper_qc_prompt_template", None)
    if gripper_path is None:
        config_path = getattr(config, "config_path", None)
        if config_path is not None:
            try:
                gripper_path = (
                    Path(config_path).expanduser().resolve().parent
                    / "prompts"
                    / "gripper_seed_candidate_qc.txt"
                )
            except (OSError, TypeError, ValueError):
                gripper_path = None
    if gripper_path is not None:
        templates["gripper_qc"] = _file_identity(gripper_path)
    available = {key: value for key, value in templates.items() if value is not None}
    generated: dict[str, Any] | None = None
    if available:
        profile = _profile_value(target_profile) or target_profile_from_config(config)
        generated = {
            "format_version": PROMPT_BUNDLE_FORMAT_VERSION,
            "target_profile": profile,
            "templates": available,
        }
        generated["sha256"] = hashlib.sha256(
            _canonical_json(_prompt_bundle_content_payload(generated))
        ).hexdigest()

    # A bound dynamic manifest may already carry an immutable bundle (for
    # example, a relocated extract).  Reuse it only after comparing it with
    # the files configured for this process; silently replacing it would make
    # episode artifacts disagree with the run-level contract.
    dataset = getattr(config, "dataset", None)
    recorded = prompt_bundle_from_manifest(
        getattr(dataset, "manifest_data", None),
        strict=True,
    )
    if recorded is not None:
        _validate_prompt_bundle(recorded, label="bound prompt_bundle")
        expected_recorded_profile = (
            _profile_value(target_profile) or target_profile_from_config(config)
        )
        recorded_profile = _prompt_bundle_profile(
            recorded,
            label="bound prompt_bundle",
        )
        if (
            expected_recorded_profile is not None
            and recorded_profile is not None
            and expected_recorded_profile != recorded_profile
        ):
            raise ValueError(
                "bound prompt_bundle target_profile differs from configured profile"
            )
        if generated is not None:
            validate_profile_provenance_pair(
                {"target_profile": generated.get("target_profile"), "prompt_bundle": generated},
                {"target_profile": recorded.get("target_profile"), "prompt_bundle": recorded},
                label="configured and bound prompt provenance",
            )
        return recorded
    return generated


def prompt_bundle_from_manifest(
    manifest: Mapping[str, Any] | None,
    *,
    strict: bool = False,
) -> dict[str, Any] | None:
    """Return a defensive copy of a previously recorded prompt bundle.

    Historical callers may encounter manifests that predate prompt bundles,
    so absence remains ``None``.  At a provenance boundary, pass
    ``strict=True`` to distinguish a present but malformed value from an
    omitted legacy field and fail closed.
    """

    if not isinstance(manifest, Mapping):
        return None
    value = manifest.get("prompt_bundle")
    if value is not None and not isinstance(value, Mapping):
        if strict:
            raise TypeError("manifest prompt_bundle must be an object")
        return None
    if not isinstance(value, Mapping):
        return None
    cloned = json.loads(json.dumps(dict(value), ensure_ascii=False))
    if not isinstance(cloned, dict):
        return None
    if strict:
        _validate_prompt_bundle(cloned, label="manifest prompt_bundle")
    return cloned


def object_resolution_strategy(
    *,
    target_profile: str | TargetProfile | None = None,
    prompt_bundle: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe the canonical S1-S3 object-mask resolution contract.

    S2 is prompt policy active from semantic planning onward, not a separate
    runtime attempt between text queries and bbox fallback.  A prompt bundle
    may supply the profile when the caller has no explicit profile value.
    """

    profile = _profile_value(target_profile)
    if profile is None and isinstance(prompt_bundle, Mapping):
        profile = _profile_value(prompt_bundle.get("target_profile"))
    strategy: dict[str, Any] = {
        "format_version": OBJECT_RESOLUTION_FORMAT_VERSION,
        "scope": "object_masks_only",
        "order": ["S1", "S2", "S3"],
        "runtime_attempt_order": [
            "text_query_all_legal_seeds",
            "bbox_fallback_same_seed_order",
        ],
        "S1": "semantic_query_bank_curated_aliases_and_legal_seed_rescue",
        "S2": {
            "capability": "mode_specific_open_set_semantic_and_visual_qc",
            "profile": profile or "mode_default",
            "semantic_prompt": "mode_specific_open_set_semantic",
            "visual_qc_prompt": "mode_specific_mask_candidate_qc",
            "runtime_position": "active_from_semantic_planning",
        },
        "S3": "qwen_bbox_to_sam_box_after_all_text_attempts",
    }
    return strategy


def attach_profile_provenance(
    payload: dict[str, Any],
    *,
    target_profile: str | TargetProfile | None = None,
    prompt_bundle: Mapping[str, Any] | None = None,
    include_default_profile: bool = True,
) -> dict[str, Any]:
    """Attach profile metadata without mutating caller-owned nested mappings."""

    profile = _profile_value(target_profile)
    if profile is not None and (
        include_default_profile
        or profile != TargetProfile.GRASP_MANIPULATION.value
    ):
        payload["target_profile"] = profile
    if prompt_bundle is not None:
        payload["prompt_bundle"] = json.loads(
            json.dumps(dict(prompt_bundle), ensure_ascii=False)
        )
    return payload


def validate_profile_provenance(
    payload: Mapping[str, Any],
    *,
    expected_profile: str | TargetProfile | None,
    expected_prompt_bundle: Mapping[str, Any] | None = None,
    strict: bool = False,
) -> None:
    """Validate profile/bundle metadata at a resume or derivation boundary.

    Old artifacts predate these fields and remain readable by default.  New
    contracts can pass ``strict=True`` to require both fields.
    """

    profile = _profile_value(expected_profile)
    actual_profile = _profile_value(payload.get("target_profile"))
    if profile is not None:
        if actual_profile is None:
            if strict:
                raise ValueError("provenance has no target_profile")
        elif actual_profile != profile:
            raise ValueError(
                f"provenance target_profile differs: {actual_profile!r} != {profile!r}"
            )
    if expected_prompt_bundle is None:
        return
    _validate_prompt_bundle(expected_prompt_bundle, label="expected prompt_bundle")
    expected_bundle_profile = _prompt_bundle_profile(
        expected_prompt_bundle,
        label="expected prompt_bundle",
    )
    if (
        profile is not None
        and expected_bundle_profile is not None
        and expected_bundle_profile != profile
    ):
        raise ValueError(
            "expected prompt_bundle target_profile differs from expected_profile"
        )
    actual_bundle = payload.get("prompt_bundle")
    if actual_bundle is None:
        if strict:
            raise ValueError("provenance has no prompt_bundle")
        return
    if not isinstance(actual_bundle, Mapping):
        raise TypeError("provenance prompt_bundle must be an object")
    _validate_prompt_bundle(actual_bundle, label="provenance prompt_bundle")
    bundle_profile = _prompt_bundle_profile(
        actual_bundle,
        label="provenance prompt_bundle",
    )
    if (
        profile is not None
        and bundle_profile is not None
        and bundle_profile != profile
    ):
        raise ValueError(
            "provenance prompt_bundle target_profile differs from expected_profile"
        )
    if actual_profile is not None and bundle_profile != actual_profile:
        raise ValueError("provenance prompt_bundle target_profile differs")
    if not _prompt_bundles_equivalent(actual_bundle, expected_prompt_bundle):
        raise ValueError("provenance prompt_bundle differs from the immutable run contract")


def validate_profile_provenance_pair(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    label: str = "provenance",
) -> None:
    """Require two metadata envelopes to carry the same profile contract.

    The prompt bundle is validated independently on both sides and compared
    by content identity, so relocation of a project does not look like a
    semantic change merely because diagnostic paths differ.
    """

    left_bundle = left.get("prompt_bundle")
    right_bundle = right.get("prompt_bundle")
    if (left_bundle is None) != (right_bundle is None):
        raise ValueError(f"{label} prompt_bundle presence differs")
    left_profile = target_profile_from_manifest(left)
    right_profile = target_profile_from_manifest(right)
    if left_bundle is None:
        if left_profile != right_profile:
            raise ValueError(f"{label} target_profile differs")
        return
    if not isinstance(left_bundle, Mapping) or not isinstance(right_bundle, Mapping):
        raise TypeError(f"{label} prompt_bundle must be an object")
    _validate_prompt_bundle(left_bundle, label=f"{label} left prompt_bundle")
    _validate_prompt_bundle(right_bundle, label=f"{label} right prompt_bundle")
    left_bundle_profile = _prompt_bundle_profile(
        left_bundle,
        label=f"{label} left prompt_bundle",
    )
    right_bundle_profile = _prompt_bundle_profile(
        right_bundle,
        label=f"{label} right prompt_bundle",
    )
    if left_bundle_profile != right_bundle_profile:
        raise ValueError(f"{label} prompt_bundle target_profile differs")
    if left_profile is not None and left_bundle_profile != left_profile:
        raise ValueError(f"{label} left target_profile differs from its prompt_bundle")
    if right_profile is not None and right_bundle_profile != right_profile:
        raise ValueError(f"{label} right target_profile differs from its prompt_bundle")
    if left_profile is None:
        left_profile = left_bundle_profile
    if right_profile is None:
        right_profile = right_bundle_profile
    if left_profile != right_profile:
        raise ValueError(f"{label} target_profile differs")
    validate_profile_provenance(
        left,
        expected_profile=right_profile,
        expected_prompt_bundle=right_bundle,
        strict=True,
    )
    validate_profile_provenance(
        right,
        expected_profile=left_profile,
        expected_prompt_bundle=left_bundle,
        strict=True,
    )


__all__ = [
    "OBJECT_RESOLUTION_FORMAT_VERSION",
    "PROMPT_BUNDLE_FORMAT_VERSION",
    "attach_profile_provenance",
    "object_resolution_strategy",
    "prompt_bundle_for_config",
    "prompt_bundle_from_manifest",
    "target_profile_from_config",
    "target_profile_from_manifest",
    "validate_profile_provenance",
    "validate_profile_provenance_pair",
]
