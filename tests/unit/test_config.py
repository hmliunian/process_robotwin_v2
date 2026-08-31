from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from robotwin_annotation_v2.config import (
    AnnotationConfig,
    ConfigError,
    DatasetBinding,
    GripperRoiConfig,
    MaskConfig,
    ParallelConfig,
    Sam3Config,
    _integers,
    _resolve_profile_document,
    _strict_bool,
    _strict_float,
    _string,
    bind_dataset,
    load_config,
    load_profile,
    parse_gpu_list,
    validate_dataset_component,
)
from robotwin_annotation_v2.domain import (
    AnnotationMode,
    GripperBackend,
    ObjectRole,
    TargetProfile,
    annotation_spec,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _write_config_with_override(
    tmp_path: Path,
    path: tuple[str, ...],
    value: Any,
) -> Path:
    """Copy a known-good config and replace one nested scalar value."""

    raw = yaml.safe_load(
        (PROJECT_ROOT / "configs/pilot_adjust_bottle_target_only.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert isinstance(raw, dict)
    section: dict[str, Any] = raw
    for key in path[:-1]:
        nested = section[key]
        assert isinstance(nested, dict)
        section = nested
    section[path[-1]] = value
    config_path = tmp_path / ("invalid-" + "-".join(path) + ".yaml")
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return config_path


def test_pilot_config_loads_new_pipeline_contract() -> None:
    config = load_config(PROJECT_ROOT / "configs/pilot_move_pillbottle_pad.yaml")

    assert config.dataset.task == "move_pillbottle_pad"
    assert config.annotation == AnnotationConfig(AnnotationMode.PICK_PLACE)
    assert config.annotation.profile is TargetProfile.GRASP_MANIPULATION
    assert config.annotation.spec.required_object_roles == (
        ObjectRole.TARGET,
        ObjectRole.RECEIVER,
    )
    assert config.annotation.spec.default_gripper_backend is GripperBackend.URDF
    assert config.dataset.camera == "cam_high"
    assert config.dataset.smoke_episode_ids == (7152,)
    assert len(config.dataset.regression_episode_ids) == 20
    assert config.qwen.query_selection == "first_recommended"
    assert config.qwen.runtime == "local"
    assert config.qwen.probe == "health"
    assert config.qwen.api_key_env is None
    assert not config.qwen.allow_query_fallback
    assert config.qwen.prompt_template.name == "target_receiver_semantic_open_set.txt"
    assert config.qwen.timeout_seconds == 600
    assert config.qwen.prompt_template.is_file()
    assert config.dataset.manifest.is_file()
    assert config.sam3.checkpoint.name == "sam3.pt"
    assert config.sam3.gpus == (2,)
    assert config.mask.qc_enabled
    assert config.mask.qc_prompt_template is not None
    assert config.mask.qc_prompt_template.is_file()
    assert config.mask.qc_max_candidates == 8
    assert config.mask.qc_max_attempts == 2
    assert config.mask.qc_query_fallback_enabled
    assert config.mask.qc_seed_fallback_enabled
    assert config.mask.qc_bbox_fallback_enabled
    assert config.mask.qc_bbox_prompt_template is not None
    assert config.mask.qc_bbox_prompt_template.name == "open_set_bbox_localization.txt"
    assert config.mask.qc_bbox_prompt_template.is_file()
    assert config.mask.qc_bbox_max_tokens == 180
    assert config.gripper_roi == GripperRoiConfig(
        prompt_axial_back_m=0.120,
        prompt_axial_front_m=0.060,
        hard_axial_back_m=0.120,
        hard_axial_front_m=0.045,
        fixed_half_width_m=0.085,
    )


def test_default_api_config_explicitly_selects_qwen38_max() -> None:
    config = load_config(PROJECT_ROOT / "configs/process_qwen38_api.yaml")

    assert config.qwen.runtime == "api"
    assert config.qwen.model == "qwen3.8-max"
    assert config.qwen.api_key_env == "QWEN_API_KEY"
    assert config.qwen.probe == "models"
    assert config.qwen.temperature == 0
    assert not config.qwen.enable_thinking
    assert config.parallel == ParallelConfig()


def test_legacy_qwen_config_without_runtime_defaults_to_local() -> None:
    config = load_config(PROJECT_ROOT / "configs/open_set_mask_fallback_bbox.yaml")

    assert config.qwen.runtime == "local"
    assert config.qwen.probe == "health"
    assert config.qwen.api_key_env is None


def test_load_profile_rejects_task_bound_dataset_block() -> None:
    with pytest.raises(ConfigError, match="must not contain a dataset block"):
        load_profile(PROJECT_ROOT / "configs/pilot_move_pillbottle_pad.yaml")


def test_load_profile_rejects_nested_dataset_block(tmp_path: Path) -> None:
    source = (PROJECT_ROOT / "configs/process.yaml").read_text(encoding="utf-8")
    config_path = tmp_path / "nested-dataset-profile.yaml"
    config_path.write_text(
        source.replace("defaults:\n", "defaults:\n  dataset:\n    task: accidental\n"),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="must not contain a dataset block"):
        load_profile(config_path, mode=AnnotationMode.PICK_PLACE)


def test_load_profile_rejects_dataset_block_in_another_mode(tmp_path: Path) -> None:
    source = (PROJECT_ROOT / "configs/process.yaml").read_text(encoding="utf-8")
    config_path = tmp_path / "unselected-dataset-profile.yaml"
    config_path.write_text(
        source.replace(
            "  target_only:\n",
            "  target_only:\n    dataset:\n      task: accidental\n",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="must not contain a dataset block"):
        load_profile(config_path, mode=AnnotationMode.PICK_PLACE)


def test_shared_profile_loads_each_supported_mode_without_dataset_identity() -> None:
    profile_path = PROJECT_ROOT / "configs/process.yaml"

    pick_place = load_profile(profile_path, mode=AnnotationMode.PICK_PLACE)
    target_only = load_profile(profile_path, mode=AnnotationMode.TARGET_ONLY)
    contact_press = load_profile(profile_path, mode="contact_press")
    door_open = load_profile(profile_path, mode="door_open")

    assert not hasattr(pick_place, "dataset")
    assert pick_place.annotation.mode is AnnotationMode.PICK_PLACE
    assert target_only.annotation.mode is AnnotationMode.TARGET_ONLY
    assert contact_press.annotation.profile is TargetProfile.CONTACT_PRESS
    assert door_open.annotation.mode is AnnotationMode.TARGET_ONLY
    assert door_open.annotation.profile is TargetProfile.DOOR_OPEN
    assert door_open.qwen.prompt_template.name == "target_only_door_open_semantic_open_set.txt"
    assert door_open.mask.qc_prompt_template is not None
    assert door_open.mask.qc_prompt_template.name == (
        "target_only_door_open_mask_candidate_qc_open_set.txt"
    )
    assert door_open.mask.qc_bbox_prompt_template is not None
    assert door_open.mask.qc_bbox_prompt_template.name == (
        "target_only_door_open_bbox_localization.txt"
    )
    assert door_open.mask.qc_max_tokens == 400
    assert pick_place.qwen.runtime == target_only.qwen.runtime == contact_press.qwen.runtime == "api"


@pytest.mark.parametrize(
    "setting",
    (
        "qc_enabled",
        "qc_query_fallback_enabled",
        "qc_seed_fallback_enabled",
        "qc_bbox_fallback_enabled",
    ),
)
def test_door_open_profile_requires_complete_s1_s3_object_resolution(
    tmp_path: Path,
    setting: str,
) -> None:
    raw = yaml.safe_load((PROJECT_ROOT / "configs/process.yaml").read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    defaults = raw["defaults"]
    assert isinstance(defaults, dict)
    mask = defaults["mask"]
    assert isinstance(mask, dict)
    mask[setting] = False
    if setting == "qc_enabled":
        for fallback in (
            "qc_query_fallback_enabled",
            "qc_seed_fallback_enabled",
            "qc_bbox_fallback_enabled",
        ):
            mask[fallback] = False
    config_path = tmp_path / "door-open-disabled-stage.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ConfigError, match="complete S1-S3"):
        load_profile(config_path, mode="door_open")


@pytest.mark.parametrize(
    ("semantic_profile", "prompt_name", "qc_name", "bbox_name"),
    (
        (
            "contact_press",
            "target_only_contact_press_semantic_open_set.txt",
            "target_only_contact_press_mask_candidate_qc_open_set.txt",
            "target_only_contact_press_bbox_localization.txt",
        ),
        (
            "door_open",
            "target_only_door_open_semantic_open_set.txt",
            "target_only_door_open_mask_candidate_qc_open_set.txt",
            "target_only_door_open_bbox_localization.txt",
        ),
    ),
)
def test_target_only_semantic_profile_routes_to_dedicated_overlay(
    semantic_profile: str,
    prompt_name: str,
    qc_name: str,
    bbox_name: str,
) -> None:
    profile = load_profile(
        PROJECT_ROOT / "configs/process.yaml",
        mode="target_only",
        target_profile=semantic_profile,
    )

    assert profile.annotation.mode is AnnotationMode.TARGET_ONLY
    assert profile.annotation.profile is TargetProfile(semantic_profile)
    assert profile.qwen.prompt_template.name == prompt_name
    assert profile.mask.qc_prompt_template is not None
    assert profile.mask.qc_prompt_template.name == qc_name
    assert profile.mask.qc_bbox_prompt_template is not None
    assert profile.mask.qc_bbox_prompt_template.name == bbox_name


def test_semantic_mode_and_target_profile_conflict_is_rejected() -> None:
    with pytest.raises(ConfigError, match="conflicts with target profile"):
        load_profile(
            PROJECT_ROOT / "configs/process.yaml",
            mode="contact_press",
            target_profile="door_open",
        )


@pytest.mark.parametrize("selector", ("contact_press", "door_open"))
def test_mode_less_semantic_profile_cannot_use_generic_prompts(selector: str) -> None:
    with pytest.raises(ConfigError, match="explicit modes"):
        _resolve_profile_document(
            {
                "defaults": {
                    "annotation": {"mode": "target_only"},
                    "qwen": {"prompt_template": "generic.txt"},
                }
            },
            target_profile=selector,
        )


@pytest.mark.parametrize("selector", ("contact_press", "door_open"))
def test_mode_less_profile_rejects_semantic_mode_selector(selector: str) -> None:
    with pytest.raises(ConfigError, match="explicit modes"):
        _resolve_profile_document(
            {"defaults": {"annotation": {"mode": "target_only"}}},
            mode=selector,
        )


@pytest.mark.parametrize("selector", ("contact_press", "door_open"))
def test_single_generic_overlay_cannot_hide_semantic_profile(selector: str) -> None:
    raw = {
        "defaults": {
            "annotation": {"mode": "target_only", "profile": selector},
            "qwen": {"prompt_template": "generic.txt"},
            "sam3": {"checkpoint": "sam3.pt", "gpus": [0]},
            "gripper_roi": {
                "prompt": {"axial_back_m": 0.1, "axial_front_m": 0.05},
                "hard": {"axial_back_m": 0.1, "axial_front_m": 0.04},
                "fixed_half_width_m": 0.08,
            },
        },
        "modes": {"target_only": {"annotation": {"mode": "target_only"}}},
    }

    with pytest.raises(ConfigError, match="explicit modes"):
        _resolve_profile_document(raw)


def test_contact_press_requires_a_dedicated_mode_overlay() -> None:
    """A mode-less profile must not masquerade as the contact profile."""

    with pytest.raises(
        ConfigError,
        match="contact_press requires an explicit modes[.]contact_press profile overlay",
    ):
        _resolve_profile_document(
            {"defaults": {"annotation": {"mode": "target_only"}}},
            mode="contact_press",
        )


@pytest.mark.parametrize("selector", (None, "target_only"))
def test_target_only_overlay_cannot_relabel_generic_prompts_as_semantic(
    selector: str | None,
) -> None:
    """An overlay's semantic profile must agree with its dedicated mode key."""

    raw = {
        "defaults": {
            "qwen": {"prompt_template": "generic.txt"},
        },
        "modes": {
            "target_only": {
                "annotation": {
                    "mode": "target_only",
                    "profile": "contact_press",
                },
                "qwen": {"prompt_template": "generic.txt"},
            },
        },
    }

    with pytest.raises(ConfigError, match="dedicated prompt overlay"):
        _resolve_profile_document(raw, mode=selector)


def test_dataset_binding_rejects_task_or_camera_identity_mismatch() -> None:
    with pytest.raises(ConfigError, match="binding task differs"):
        DatasetBinding(
            root=Path("/dataset/task"),
            task="task",
            camera="cam_high",
            episode_ids=(1,),
            manifest_data={"task": "other-task", "camera": "cam_high"},
        )

    with pytest.raises(ConfigError, match="binding camera differs"):
        DatasetBinding(
            root=Path("/dataset/task"),
            task="task",
            camera="cam_high",
            episode_ids=(1,),
            manifest_data={"task": "task", "camera": "cam_left"},
        )


@pytest.mark.parametrize("value", ("../escape", "a/b", r"a\b", ".", "..", "C:task", r"C:\\task", "task\x00name"))
def test_dataset_component_rejects_path_like_names(value: str) -> None:
    with pytest.raises(ConfigError, match="single path component"):
        validate_dataset_component(value, field="dataset.task")


def test_dataset_component_strips_surrounding_whitespace() -> None:
    assert validate_dataset_component("  move_task  ", field="dataset.task") == "move_task"


@pytest.mark.parametrize("field", ("task", "camera"))
def test_dataset_binding_rejects_path_like_identity(field: str) -> None:
    kwargs = {
        "root": Path("/dataset/task"),
        "task": "task",
        "camera": "cam_high",
        "episode_ids": (1,),
    }
    kwargs[field] = "../escape"
    with pytest.raises(ConfigError, match="single path component"):
        DatasetBinding(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ("task", "camera"))
def test_legacy_config_rejects_path_like_dataset_identity(
    tmp_path: Path,
    field: str,
) -> None:
    source = (PROJECT_ROOT / "configs/pilot_move_pillbottle_pad.yaml").read_text(
        encoding="utf-8"
    )
    needle = (
        "  task: move_pillbottle_pad\n"
        if field == "task"
        else "  camera: cam_high\n"
    )
    replacement = f"  {field}: ../escape\n"
    config_path = tmp_path / f"invalid-{field}.yaml"
    config_path.write_text(source.replace(needle, replacement), encoding="utf-8")
    with pytest.raises(ConfigError, match="single path component"):
        load_config(config_path)


def test_bind_dataset_keeps_profile_reusable_and_copies_manifest(tmp_path: Path) -> None:
    profile = load_profile(PROJECT_ROOT / "configs/process.yaml", mode="pick_place")
    manifest = {
        "task": "task",
        "camera": "cam_high",
        "dataset_root": "/old/mount/task",
        "episode_indices": [4, 9],
        "smoke_episode_ids": [4],
    }
    binding = DatasetBinding(
        root=tmp_path / "task",
        task="task",
        camera="cam_high",
        episode_ids=(9,),
        manifest_path=Path("EXTRACT_MANIFEST.json"),
        manifest_data=manifest,
    )

    bound = bind_dataset(profile, binding)

    assert bound.dataset.root == (tmp_path / "task").resolve()
    assert bound.dataset.manifest == (tmp_path / "task/EXTRACT_MANIFEST.json").resolve()
    assert bound.dataset.regression_episode_ids == (9,)
    assert bound.dataset.smoke_episode_ids == (9,)
    assert bound.dataset.manifest_data is not manifest
    assert bound.dataset.manifest_data is not binding.manifest_data
    assert bound.dataset.manifest_data["dataset_root"] == str(bound.dataset.root)
    assert manifest["dataset_root"] == "/old/mount/task"


def test_bind_dataset_syncs_all_episode_selection_aliases(tmp_path: Path) -> None:
    profile = load_profile(PROJECT_ROOT / "configs/process.yaml", mode="pick_place")
    binding = DatasetBinding(
        root=tmp_path / "task",
        task="task",
        camera="cam_high",
        episode_ids=(9,),
        manifest_data={
            "task": "task",
            "camera": "cam_high",
            "episode_indices": [4, 9],
            "regression_episode_ids": [4, 9],
            "episode_ids": [4, 9],
            "smoke_episode_ids": [4],
        },
    )

    bound = bind_dataset(profile, binding)

    assert bound.dataset.manifest_data is not None
    for key in ("episode_indices", "regression_episode_ids", "episode_ids"):
        assert bound.dataset.manifest_data[key] == [9]


def test_bind_dataset_rejects_manifest_profile_mismatch(tmp_path: Path) -> None:
    profile = load_profile(PROJECT_ROOT / "configs/process.yaml", mode="pick_place")
    binding = DatasetBinding(
        root=tmp_path / "task",
        task="task",
        camera="cam_high",
        episode_ids=(1,),
        manifest_data={
            "profile": "target_only",
            "task": "task",
            "camera": "cam_high",
        },
    )

    with pytest.raises(ConfigError, match="binding profile differs"):
        bind_dataset(profile, binding)


def test_bind_dataset_accepts_contact_profile_alias_for_target_only(tmp_path: Path) -> None:
    profile = load_profile(PROJECT_ROOT / "configs/process.yaml", mode="contact_press")
    binding = DatasetBinding(
        root=tmp_path / "task",
        task="task",
        camera="cam_high",
        episode_ids=(1,),
        manifest_data={
            "profile": "contact_press",
            "target_profile": "contact_press",
            "task": "task",
            "camera": "cam_high",
        },
    )

    bound = bind_dataset(profile, binding)

    assert bound.annotation.profile is TargetProfile.CONTACT_PRESS
    assert bound.dataset.manifest_data is not None
    assert bound.dataset.manifest_data["target_profile"] == "contact_press"


def test_bind_dataset_rejects_manifest_target_profile_conflict(tmp_path: Path) -> None:
    profile = load_profile(PROJECT_ROOT / "configs/process.yaml", mode="origin")
    binding = DatasetBinding(
        root=tmp_path / "task",
        task="task",
        camera="cam_high",
        episode_ids=(1,),
        manifest_data={
            "profile": "target_only",
            "target_profile": "door_open",
            "task": "task",
            "camera": "cam_high",
        },
    )

    with pytest.raises(ConfigError, match="target_profile"):
        bind_dataset(profile, binding)


def test_dataset_binding_rejects_conflicting_manifest_profile_fields(
    tmp_path: Path,
) -> None:
    with pytest.raises(ConfigError, match="conflicts"):
        DatasetBinding(
            root=tmp_path / "task",
            task="task",
            camera="cam_high",
            episode_ids=(1,),
            manifest_data={
                "profile": "contact_press",
                "target_profile": "door_open",
                "task": "task",
                "camera": "cam_high",
            },
        )


@pytest.mark.parametrize(
    ("first_key", "second_key"),
    (
        ("episode_indices", "regression_episode_ids"),
        ("episode_indices", "episode_ids"),
        ("regression_episode_ids", "episode_ids"),
    ),
)
def test_dataset_binding_rejects_conflicting_episode_selection_aliases(
    tmp_path: Path,
    first_key: str,
    second_key: str,
) -> None:
    manifest = {
        "task": "task",
        "camera": "cam_high",
        first_key: [1, 2],
        second_key: [2, 1],
        "smoke_episode_ids": [1],
    }

    with pytest.raises(ConfigError, match="conflicts"):
        DatasetBinding(
            root=tmp_path / "task",
            task="task",
            camera="cam_high",
            episode_ids=(1,),
            manifest_data=manifest,
        )


@pytest.mark.parametrize(
    "smoke",
    (
        [],
        [3],
    ),
)
def test_dataset_binding_rejects_invalid_manifest_smoke_selection(
    tmp_path: Path,
    smoke: list[int],
) -> None:
    with pytest.raises(ConfigError, match="smoke_episode_ids"):
        DatasetBinding(
            root=tmp_path / "task",
            task="task",
            camera="cam_high",
            episode_ids=(1,),
            manifest_data={
                "task": "task",
                "camera": "cam_high",
                "episode_indices": [1, 2],
                "regression_episode_ids": [1, 2],
                "smoke_episode_ids": smoke,
            },
        )


def test_bind_dataset_validates_episode_aliases_loaded_from_manifest_file(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "task" / "EXTRACT_MANIFEST.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text(
        json.dumps(
            {
                "task": "task",
                "camera": "cam_high",
                "episode_indices": [1],
                "regression_episode_ids": [2],
                "smoke_episode_ids": [1],
            }
        ),
        encoding="utf-8",
    )
    profile = load_profile(PROJECT_ROOT / "configs/process.yaml", mode="pick_place")

    with pytest.raises(ConfigError, match="conflicts"):
        bind_dataset(
            profile,
            DatasetBinding(
                root=manifest_path.parent,
                task="task",
                camera="cam_high",
                manifest_path=manifest_path,
            ),
        )


def test_legacy_config_accepts_episode_selection_aliases(tmp_path: Path) -> None:
    raw = yaml.safe_load(
        (PROJECT_ROOT / "configs/pilot_move_pillbottle_pad.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert isinstance(raw, dict)
    dataset = raw["dataset"]
    assert isinstance(dataset, dict)
    regression = dataset.pop("regression_episode_ids")
    dataset["episode_indices"] = regression
    config_path = tmp_path / "alias-config.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    config = load_config(config_path)

    assert config.dataset.regression_episode_ids == tuple(regression)


def test_legacy_config_rejects_conflicting_episode_selection_aliases(
    tmp_path: Path,
) -> None:
    raw = yaml.safe_load(
        (PROJECT_ROOT / "configs/pilot_move_pillbottle_pad.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert isinstance(raw, dict)
    dataset = raw["dataset"]
    assert isinstance(dataset, dict)
    dataset["episode_indices"] = [999]
    config_path = tmp_path / "conflicting-alias-config.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ConfigError, match="conflicts"):
        load_config(config_path)


def test_bind_dataset_builds_manifest_in_memory_when_native_manifest_is_absent(
    tmp_path: Path,
) -> None:
    profile = load_profile(PROJECT_ROOT / "configs/process.yaml", mode="pick_place")
    binding = DatasetBinding(
        root=tmp_path / "native-task",
        task="native-task",
        camera="cam_high",
        regression_episode_ids=(12, 15),
        smoke_episode_ids=(15,),
    )

    bound = bind_dataset(profile, binding)

    assert bound.dataset.manifest_data == {
        "format_version": "robotwin_dataset_manifest_bound_v1",
        "profile": "pick_place",
        "task": "native-task",
        "camera": "cam_high",
        "dataset_root": str((tmp_path / "native-task").resolve()),
        "episode_indices": [12, 15],
        "smoke_episode_ids": [15],
        "regression_episode_ids": [12, 15],
    }
    # The synthetic record is runtime-only; no manifest is materialized.
    assert not bound.dataset.manifest.is_file()


def test_api_runtime_requires_environment_credential_name(tmp_path: Path) -> None:
    source = (PROJECT_ROOT / "configs/process_qwen38_api.yaml").read_text(encoding="utf-8")
    config_path = tmp_path / "missing-api-key-env.yaml"
    config_path.write_text(
        source.replace("  api_key_env: QWEN_API_KEY\n", ""),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="api_key_env"):
        load_config(config_path)


def test_parallel_config_is_opt_in_and_requires_api_runtime(tmp_path: Path) -> None:
    source = (PROJECT_ROOT / "configs/process_qwen38_api.yaml").read_text(encoding="utf-8")
    config_path = tmp_path / "parallel.yaml"
    config_path.write_text(
        source + "\nparallel:\n  sam_worker_gpus: [1, 4]\n  qwen_max_in_flight: 3\n",
        encoding="utf-8",
    )

    config = load_config(config_path)
    assert config.parallel.sam_worker_gpus == (1, 4)
    assert config.parallel.qwen_max_in_flight == 3
    assert config.parallel.enabled


def test_parallel_config_rejects_duplicate_gpu_and_invalid_limits() -> None:
    with pytest.raises(ConfigError, match="duplicate GPUs"):
        ParallelConfig(sam_worker_gpus=(1, 1))
    with pytest.raises(ConfigError, match="positive integer"):
        ParallelConfig(qwen_max_in_flight=0)


def test_parse_gpu_list_accepts_whitespace_and_rejects_malformed_values() -> None:
    assert parse_gpu_list(" 1, 4,7 ") == (1, 4, 7)
    assert parse_gpu_list("") == ()
    with pytest.raises(ConfigError, match="comma-separated"):
        parse_gpu_list("1,,2")
    with pytest.raises(ConfigError, match="duplicate GPUs"):
        parse_gpu_list("1,1")
    with pytest.raises(ConfigError, match="non-negative"):
        parse_gpu_list("-1")


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("dataset", "smoke_episode_ids"), ["0"]),
        (("sam3", "gpus"), ["2"]),
        (("qwen", "runtime"), False),
        (("qwen", "endpoint"), 123),
        (("qwen", "model"), False),
        (("qwen", "probe"), 1),
        (("qwen", "temperature"), "0.0"),
        (("qwen", "timeout_seconds"), "600"),
        (("qwen", "max_tokens"), 400.0),
        (("qwen", "query_selection"), False),
        (("qwen", "allow_query_fallback"), "false"),
        (("mask", "qc_enabled"), "false"),
        (("mask", "target_envelope_padding_px"), 4.0),
        (("mask", "temporal_qc_min_adjacent_iou_p05"), "0.5"),
        (("mask", "qc_max_attempts"), 2.0),
        (("mask", "qc_query_fallback_enabled"), 0),
        (("gripper_roi", "prompt", "axial_back_m"), "0.120"),
    ),
)
def test_load_config_rejects_implicitly_coerced_values(
    tmp_path: Path,
    path: tuple[str, ...],
    value: Any,
) -> None:
    config_path = _write_config_with_override(tmp_path, path, value)

    with pytest.raises(ConfigError, match=path[-1]):
        load_config(config_path)


def test_strict_scalar_helpers_reject_implicit_coercion() -> None:
    assert _integers([0, 2], field="episodes") == (0, 2)
    assert _strict_float(2, field="threshold") == 2.0
    assert _strict_bool(False, field="enabled") is False
    assert _string("  value  ", field="name") == "value"

    with pytest.raises(ConfigError, match="episodes"):
        _integers(["2"], field="episodes")
    with pytest.raises(ConfigError, match="episodes"):
        _integers([1.5], field="episodes")
    with pytest.raises(ConfigError, match="threshold"):
        _strict_float("2.0", field="threshold")
    with pytest.raises(ConfigError, match="threshold"):
        _strict_float(True, field="threshold")
    with pytest.raises(ConfigError, match="enabled"):
        _strict_bool("false", field="enabled")
    with pytest.raises(ConfigError, match="name"):
        _string(123, field="name")


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="positive-inf"),
        pytest.param(float("-inf"), id="negative-inf"),
        pytest.param(10**10000, id="huge-int"),
    ],
)
def test_strict_float_rejects_non_finite_or_unrepresentable_values(value: Any) -> None:
    with pytest.raises(ConfigError, match="finite number"):
        _strict_float(value, field="threshold")


def test_parallel_sam_workers_require_api_runtime(tmp_path: Path) -> None:
    source = (PROJECT_ROOT / "configs/pilot_move_pillbottle_pad.yaml").read_text(encoding="utf-8")
    config_path = tmp_path / "parallel-local.yaml"
    config_path.write_text(
        source + "\nparallel:\n  sam_worker_gpus: [1]\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="qwen.runtime=api"):
        load_config(config_path)


def test_place_container_plate_config_pins_depth_complete_subset() -> None:
    config = load_config(PROJECT_ROOT / "configs/pilot_place_container_plate.yaml")
    manifest = json.loads(config.dataset.manifest.read_text(encoding="utf-8"))

    assert config.dataset.task == "place_container_plate"
    assert config.dataset.smoke_episode_ids == (14850,)
    assert len(config.dataset.regression_episode_ids) == 547
    assert config.dataset.regression_episode_ids == tuple(manifest["regression_episode_ids"])
    assert {int(value) for value in manifest["excluded_source_episodes"]} == {
        14941,
        15022,
        15360,
    }


def test_target_only_pilot_config_pins_close_and_hold_dataset() -> None:
    config = load_config(PROJECT_ROOT / "configs/pilot_adjust_bottle_target_only.yaml")
    manifest = json.loads(config.dataset.manifest.read_text(encoding="utf-8"))

    assert config.annotation == AnnotationConfig(AnnotationMode.TARGET_ONLY)
    assert config.annotation.profile is TargetProfile.GRASP_MANIPULATION
    assert config.annotation.spec.required_object_roles == (ObjectRole.TARGET,)
    assert config.annotation.spec.default_gripper_backend is GripperBackend.URDF
    assert config.dataset.task == "adjust_bottle"
    assert config.dataset.smoke_episode_ids == (0,)
    assert len(config.dataset.regression_episode_ids) == 20
    assert config.dataset.regression_episode_ids == tuple(manifest["regression_episode_ids"])
    assert config.qwen.prompt_template.name == "target_only_semantic_open_set.txt"
    assert config.qwen.timeout_seconds == 600
    assert config.qwen.max_tokens == 400
    assert config.mask.qc_prompt_template is not None
    assert config.mask.qc_prompt_template.name == (
        "target_only_mask_candidate_qc_open_set.txt"
    )
    assert config.mask.qc_max_candidates == 8
    assert config.mask.qc_query_fallback_enabled
    assert config.mask.qc_seed_fallback_enabled
    assert config.mask.qc_bbox_fallback_enabled
    assert config.mask.qc_bbox_prompt_template is not None
    assert config.mask.qc_bbox_prompt_template.name == "open_set_bbox_localization.txt"


def test_open_set_bbox_experiment_explicitly_enables_bbox_fallback() -> None:
    config = load_config(PROJECT_ROOT / "configs/open_set_mask_fallback_bbox.yaml")

    assert config.mask.qc_bbox_fallback_enabled
    assert config.mask.qc_bbox_prompt_template is not None
    assert config.mask.qc_bbox_prompt_template.name == "open_set_bbox_localization.txt"
    assert config.mask.qc_bbox_prompt_template.is_file()
    assert config.mask.qc_bbox_max_tokens == 180


def test_bbox_fallback_requires_qc_and_an_explicit_prompt() -> None:
    with pytest.raises(ConfigError, match="requires mask QC"):
        MaskConfig(qc_bbox_fallback_enabled=True)

    with pytest.raises(ConfigError, match="qc_bbox_prompt_template"):
        MaskConfig(
            qc_enabled=True,
            qc_prompt_template=Path("mask-qc.txt"),
            qc_bbox_fallback_enabled=True,
        )


@pytest.mark.parametrize(
    "field",
    [
        "temporal_envelope_guard_retry_enabled",
        "qc_bbox_directional_expand_enabled",
        "qc_border_retry_enabled",
        "qc_border_retry_prompt_template",
        "qc_border_retry_max_tokens",
    ],
)
def test_config_rejects_removed_s4_fields(tmp_path: Path, field: str) -> None:
    source = (PROJECT_ROOT / "configs/open_set_mask_fallback_bbox.yaml").read_text(encoding="utf-8")
    config_path = tmp_path / "removed-s4.yaml"
    config_path.write_text(source.replace("mask:\n", f"mask:\n  {field}: true\n"), encoding="utf-8")

    with pytest.raises(ConfigError, match="removed S4"):
        load_config(config_path)


@pytest.mark.parametrize(
    "field",
    ["qc_query_fallback_enabled", "qc_seed_fallback_enabled"],
)
def test_text_fallbacks_require_qc(field: str) -> None:
    with pytest.raises(ConfigError, match="requires mask QC"):
        MaskConfig(**{field: True})


def test_config_rejects_automatic_query_fallback(tmp_path: Path) -> None:
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        """
dataset:
  root: /tmp/data
  manifest: manifest.json
  task: task
  camera: cam_high
  smoke_episode_ids: [1]
  regression_episode_ids: [1]
qwen:
  endpoint: http://127.0.0.1:1/v1/chat/completions
  model: qwen
  prompt_template: prompt.txt
  allow_query_fallback: true
sam3:
  checkpoint: sam3.pt
gripper_roi:
  prompt:
    axial_back_m: 0.120
    axial_front_m: 0.060
  hard:
    axial_back_m: 0.120
    axial_front_m: 0.045
  fixed_half_width_m: 0.085
output:
  root: output
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="fallback"):
        load_config(config_path)


def test_annotation_specs_only_declare_roles_and_backend() -> None:
    pick_place = annotation_spec(AnnotationMode.PICK_PLACE)
    target_only = annotation_spec(AnnotationMode.TARGET_ONLY)

    assert pick_place.required_object_roles == (ObjectRole.TARGET, ObjectRole.RECEIVER)
    assert target_only.required_object_roles == (ObjectRole.TARGET,)
    assert target_only.canonical_object_roles == pick_place.canonical_object_roles
    assert not target_only.requires(ObjectRole.RECEIVER)
    assert target_only.default_gripper_backend is GripperBackend.URDF


def test_config_rejects_unknown_annotation_mode(tmp_path: Path) -> None:
    source = (PROJECT_ROOT / "configs/pilot_move_pillbottle_pad.yaml").read_text(encoding="utf-8")
    config_path = tmp_path / "bad-mode.yaml"
    config_path.write_text(source.replace("mode: pick_place", "mode: mystery"), encoding="utf-8")

    with pytest.raises(ConfigError, match="annotation.mode"):
        load_config(config_path)


def test_target_only_config_accepts_explicit_contact_press_profile(tmp_path: Path) -> None:
    source = (PROJECT_ROOT / "configs/pilot_adjust_bottle_target_only.yaml").read_text(
        encoding="utf-8"
    )
    config_path = tmp_path / "contact-press.yaml"
    config_path.write_text(
        source.replace("mode: target_only", "mode: target_only\n  profile: contact_press"),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.annotation.profile is TargetProfile.CONTACT_PRESS


def test_target_only_config_accepts_origin_profile_alias(tmp_path: Path) -> None:
    source = (PROJECT_ROOT / "configs/pilot_adjust_bottle_target_only.yaml").read_text(
        encoding="utf-8"
    )
    config_path = tmp_path / "origin-profile.yaml"
    config_path.write_text(
        source.replace("mode: target_only", "mode: target_only\n  profile: origin"),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.annotation.profile is TargetProfile.GRASP_MANIPULATION


def test_contact_press_profile_requires_target_only_mode(tmp_path: Path) -> None:
    source = (PROJECT_ROOT / "configs/pilot_move_pillbottle_pad.yaml").read_text(
        encoding="utf-8"
    )
    config_path = tmp_path / "invalid-contact-press.yaml"
    config_path.write_text(
        source.replace("mode: pick_place", "mode: pick_place\n  profile: contact_press"),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="contact_press requires target_only"):
        load_config(config_path)


def test_config_rejects_unknown_annotation_profile(tmp_path: Path) -> None:
    source = (PROJECT_ROOT / "configs/pilot_adjust_bottle_target_only.yaml").read_text(
        encoding="utf-8"
    )
    config_path = tmp_path / "bad-profile.yaml"
    config_path.write_text(
        source.replace("mode: target_only", "mode: target_only\n  profile: mystery"),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="annotation.profile"):
        load_config(config_path)


def test_sam_config_requires_one_gpu() -> None:
    with pytest.raises(ConfigError, match="exactly one"):
        Sam3Config(Path("sam3.pt"), gpus=(0, 1))


def test_temporal_qc_thresholds_are_validated() -> None:
    with pytest.raises(ConfigError, match="IoU"):
        MaskConfig(temporal_qc_min_adjacent_iou_p05=1.1)
    with pytest.raises(ConfigError, match="signal count"):
        MaskConfig(temporal_qc_quarantine_signal_count=4)


@pytest.mark.parametrize("value", [0.0, -0.1, float("nan"), float("inf")])
def test_gripper_roi_requires_positive_finite_values(value: float) -> None:
    with pytest.raises(ConfigError, match="greater than zero"):
        GripperRoiConfig(
            prompt_axial_back_m=value,
            prompt_axial_front_m=0.060,
            hard_axial_back_m=0.120,
            hard_axial_front_m=0.045,
            fixed_half_width_m=0.085,
        )
