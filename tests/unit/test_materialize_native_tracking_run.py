from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from robotwin_annotation_v2.application.provenance import prompt_bundle_for_config
from robotwin_annotation_v2.domain import TargetProfile
from scripts.materialize_native_tracking_run import _validate_source_prompt_provenance


def _config(tmp_path: Path, *, semantic_text: str = "semantic") -> SimpleNamespace:
    tmp_path.mkdir(parents=True, exist_ok=True)
    semantic = tmp_path / "semantic.txt"
    qc = tmp_path / "qc.txt"
    bbox = tmp_path / "bbox.txt"
    semantic.write_text(semantic_text, encoding="utf-8")
    qc.write_text("qc", encoding="utf-8")
    bbox.write_text("bbox", encoding="utf-8")
    return SimpleNamespace(
        annotation=SimpleNamespace(profile=TargetProfile.DOOR_OPEN),
        qwen=SimpleNamespace(prompt_template=semantic),
        mask=SimpleNamespace(qc_prompt_template=qc, qc_bbox_prompt_template=bbox),
    )


def test_materialize_source_prompt_bundle_must_match_current_config(tmp_path: Path) -> None:
    source_bundle = prompt_bundle_for_config(_config(tmp_path / "source"))
    current_bundle = prompt_bundle_for_config(
        _config(tmp_path / "current", semantic_text="changed semantic")
    )
    assert source_bundle is not None
    assert current_bundle is not None

    with pytest.raises(ValueError, match="prompt_bundle differs"):
        _validate_source_prompt_provenance(
            {"target_profile": "door_open", "prompt_bundle": source_bundle},
            target_profile="door_open",
            prompt_bundle=current_bundle,
        )


def test_materialize_source_prompt_bundle_matching_config_is_accepted(tmp_path: Path) -> None:
    bundle = prompt_bundle_for_config(_config(tmp_path))
    assert bundle is not None

    _validate_source_prompt_provenance(
        {"target_profile": "door_open", "prompt_bundle": bundle},
        target_profile="door_open",
        prompt_bundle=bundle,
    )


def test_materialize_source_prompt_bundle_profile_must_match(tmp_path: Path) -> None:
    source_bundle = prompt_bundle_for_config(_config(tmp_path / "source"))
    current_bundle = dict(source_bundle or {})
    current_bundle["target_profile"] = "contact_press"
    # Keep this a valid bundle so the failure exercises the profile contract,
    # not the hash-integrity check.
    import hashlib
    import json

    payload = {key: value for key, value in current_bundle.items() if key != "sha256"}
    current_bundle["sha256"] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert source_bundle is not None

    with pytest.raises(ValueError, match="target_profile differs"):
        _validate_source_prompt_provenance(
            {"target_profile": "door_open", "prompt_bundle": source_bundle},
            target_profile="contact_press",
            prompt_bundle=current_bundle,
        )


def test_materialize_legacy_source_without_prompt_bundle_remains_readable(
    tmp_path: Path,
) -> None:
    current_bundle = prompt_bundle_for_config(_config(tmp_path))
    assert current_bundle is not None

    _validate_source_prompt_provenance(
        {"target_profile": "door_open"},
        target_profile="door_open",
        prompt_bundle=current_bundle,
    )
