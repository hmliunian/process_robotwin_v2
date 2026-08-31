from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from robotwin_annotation_v2.application.provenance import (
    object_resolution_strategy,
    prompt_bundle_for_config,
    target_profile_from_manifest,
    validate_profile_provenance,
)
from robotwin_annotation_v2.domain import TargetProfile


def test_object_resolution_strategy_audits_door_open_s2_policy() -> None:
    strategy = object_resolution_strategy(
        prompt_bundle={
            "target_profile": "door_open",
            "templates": {"gripper_qc": {"sha256": "d" * 64}},
        },
    )

    assert strategy["scope"] == "object_masks_only"
    assert isinstance(strategy["S1"], str)
    assert strategy["S2"]["profile"] == "door_open"
    assert strategy["S2"]["semantic_prompt"] == "mode_specific_open_set_semantic"
    assert strategy["S2"]["visual_qc_prompt"] == "mode_specific_mask_candidate_qc"
    assert isinstance(strategy["S3"], str)
    assert "gripper_qc" not in str(strategy)


def _prompt_config(tmp_path: Path, profile: TargetProfile) -> SimpleNamespace:
    tmp_path.mkdir(parents=True, exist_ok=True)
    semantic = tmp_path / "semantic.txt"
    qc = tmp_path / "qc.txt"
    bbox = tmp_path / "bbox.txt"
    semantic.write_text("semantic", encoding="utf-8")
    qc.write_text("qc", encoding="utf-8")
    bbox.write_text("bbox", encoding="utf-8")
    return SimpleNamespace(
        annotation=SimpleNamespace(profile=profile),
        qwen=SimpleNamespace(prompt_template=semantic),
        mask=SimpleNamespace(
            qc_prompt_template=qc,
            qc_bbox_prompt_template=bbox,
        ),
    )


def test_workflow_profile_is_not_treated_as_target_profile() -> None:
    assert target_profile_from_manifest({"profile": "target_only"}) is None


def test_semantic_profile_alias_is_canonicalized() -> None:
    assert (
        target_profile_from_manifest({"profile": "contact-press"})
        == TargetProfile.CONTACT_PRESS.value
    )


def test_manifest_rejects_conflicting_profile_fields() -> None:
    with pytest.raises(ValueError, match="conflicts"):
        target_profile_from_manifest(
            {"profile": "contact_press", "target_profile": "door_open"}
        )


def test_profile_provenance_rejects_tampered_bundle_hash(tmp_path: Path) -> None:
    bundle = prompt_bundle_for_config(
        _prompt_config(tmp_path, TargetProfile.CONTACT_PRESS)
    )
    assert bundle is not None
    tampered = dict(bundle)
    tampered["target_profile"] = TargetProfile.DOOR_OPEN.value

    with pytest.raises(ValueError, match="content hash is invalid"):
        validate_profile_provenance(
            {
                "target_profile": TargetProfile.CONTACT_PRESS.value,
                "prompt_bundle": tampered,
            },
            expected_profile=TargetProfile.CONTACT_PRESS,
            expected_prompt_bundle=bundle,
            strict=True,
        )


def test_profile_provenance_rejects_non_object_bundle(tmp_path: Path) -> None:
    bundle = prompt_bundle_for_config(
        _prompt_config(tmp_path, TargetProfile.CONTACT_PRESS)
    )
    assert bundle is not None

    with pytest.raises(TypeError, match="must be an object"):
        validate_profile_provenance(
            {
                "target_profile": TargetProfile.CONTACT_PRESS.value,
                "prompt_bundle": "not-an-object",
            },
            expected_profile=TargetProfile.CONTACT_PRESS,
            expected_prompt_bundle=bundle,
            strict=True,
        )


def test_profile_provenance_keeps_legacy_metadata_optional(tmp_path: Path) -> None:
    bundle = prompt_bundle_for_config(
        _prompt_config(tmp_path, TargetProfile.GRASP_MANIPULATION)
    )
    assert bundle is not None

    validate_profile_provenance(
        {},
        expected_profile=TargetProfile.GRASP_MANIPULATION,
        expected_prompt_bundle=bundle,
        strict=False,
    )


def test_prompt_bundle_hash_is_portable_across_project_paths(tmp_path: Path) -> None:
    first = _prompt_config(tmp_path / "first", TargetProfile.CONTACT_PRESS)
    second = _prompt_config(tmp_path / "second", TargetProfile.CONTACT_PRESS)
    first_bundle = prompt_bundle_for_config(first)
    second_bundle = prompt_bundle_for_config(second)
    assert first_bundle is not None
    assert second_bundle is not None
    assert first_bundle["sha256"] == second_bundle["sha256"]
    assert first_bundle["templates"] != second_bundle["templates"]

    validate_profile_provenance(
        {"target_profile": TargetProfile.CONTACT_PRESS.value, "prompt_bundle": second_bundle},
        expected_profile=TargetProfile.CONTACT_PRESS,
        expected_prompt_bundle=first_bundle,
        strict=True,
    )


def test_prompt_bundle_hashes_symlinked_prompt_target(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    link = tmp_path / "link.txt"
    target.write_text("prompt", encoding="utf-8")
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is unavailable")

    config = SimpleNamespace(
        annotation=SimpleNamespace(profile=TargetProfile.GRASP_MANIPULATION),
        qwen=SimpleNamespace(prompt_template=link),
        mask=SimpleNamespace(qc_prompt_template=None, qc_bbox_prompt_template=None),
    )
    bundle = prompt_bundle_for_config(config)

    assert bundle is not None
    assert "semantic" in bundle["templates"]
    assert bundle["templates"]["semantic"]["path"] == str(target.resolve())


def test_prompt_bundle_includes_gripper_seed_qc_prompt_for_real_config() -> None:
    from robotwin_annotation_v2.config import load_config

    config = load_config(Path("configs/pilot_move_pillbottle_pad.yaml"))
    bundle = prompt_bundle_for_config(config)

    assert bundle is not None
    assert "gripper_qc" in bundle["templates"]
    assert bundle["templates"]["gripper_qc"]["path"].endswith(
        "configs/prompts/gripper_seed_candidate_qc.txt"
    )


def test_profile_provenance_rejects_expected_bundle_profile_conflict(
    tmp_path: Path,
) -> None:
    bundle = prompt_bundle_for_config(
        _prompt_config(tmp_path, TargetProfile.CONTACT_PRESS)
    )
    assert bundle is not None
    conflicting = dict(bundle)
    conflicting["target_profile"] = TargetProfile.DOOR_OPEN.value
    # Recompute the hash so this exercises the semantic cross-field check,
    # rather than the tamper/hash check above.
    import hashlib
    import json

    payload = {key: value for key, value in conflicting.items() if key != "sha256"}
    conflicting["sha256"] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    with pytest.raises(ValueError, match="expected prompt_bundle target_profile"):
        validate_profile_provenance(
            {
                "target_profile": TargetProfile.CONTACT_PRESS.value,
                "prompt_bundle": bundle,
            },
            expected_profile=TargetProfile.CONTACT_PRESS,
            expected_prompt_bundle=conflicting,
            strict=True,
        )
