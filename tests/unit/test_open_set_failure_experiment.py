from __future__ import annotations

from pathlib import Path

import pytest

import scripts.run_open_set_failure_experiment as experiment_runner

CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "open_set_mask_fallback_failures.yaml"
)


def test_failure_config_declares_exactly_52_unique_episodes() -> None:
    experiment = experiment_runner._load_experiment(CONFIG_PATH)

    episode_ids = [
        episode_id
        for _task, task_episode_ids in experiment.task_episode_ids
        for episode_id in task_episode_ids
    ]

    assert experiment.expected_failure_count == 52
    assert len(episode_ids) == 52
    assert len(set(episode_ids)) == 52
    assert experiment.pipeline.qwen.timeout_seconds == 600
    assert experiment.pipeline.mask.qc_max_candidates == 8
    assert experiment.pipeline.mask.qc_query_fallback_enabled is True
    assert experiment.pipeline.mask.qc_seed_fallback_enabled is True


def test_s1_s2_s3_configs_use_distinct_outputs_and_only_s3_enables_bbox() -> None:
    names = (
        "open_set_mask_fallback_failures.yaml",
        "open_set_mask_fallback_appearance.yaml",
        "open_set_mask_fallback_bbox.yaml",
    )
    experiments = tuple(
        experiment_runner._load_experiment(CONFIG_PATH.with_name(name)) for name in names
    )

    assert len({item.run_id_prefix for item in experiments}) == 3
    assert len({item.pipeline.output_root for item in experiments}) == 3
    assert [item.pipeline.mask.qc_bbox_fallback_enabled for item in experiments] == [
        False,
        False,
        True,
    ]


def test_smoke_selection_uses_only_the_six_representative_failures() -> None:
    experiment = experiment_runner._load_experiment(CONFIG_PATH)

    selected = experiment_runner._select_episodes(
        experiment,
        tasks=None,
        episode_ids=None,
        smoke=True,
    )

    assert {episode_id for _task, values in selected for episode_id in values} == {
        14899,
        16534,
        17087,
        18209,
        18710,
        19273,
    }
    assert sum(len(values) for _task, values in selected) == 6


def test_selection_rejects_episode_outside_failure_inventory() -> None:
    experiment = experiment_runner._load_experiment(CONFIG_PATH)

    with pytest.raises(ValueError, match="outside the declared failures"):
        experiment_runner._select_episodes(
            experiment,
            tasks=None,
            episode_ids=(19300,),
            smoke=False,
        )
