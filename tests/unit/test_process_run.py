from __future__ import annotations

from pathlib import Path

from robotwin_annotation_v2.models.process_run import (
    EpisodeRecord,
    ProcessRequest,
    ProcessSummary,
)


def test_episode_record_round_trip_preserves_legacy_field_order() -> None:
    record = EpisodeRecord.from_payload(
        {
            "episode": 7,
            "status": "failed",
            "reason": "bad_input",
            "error": "details",
        }
    )

    assert record.to_json() == {
        "episode": 7,
        "status": "failed",
        "reason": "bad_input",
        "error": "details",
    }


def test_episode_record_without_episode_omits_episode_key() -> None:
    assert EpisodeRecord.from_payload(
        {"status": "render_failed", "error": "renderer"}
    ).to_json() == {"status": "render_failed", "error": "renderer"}


def test_process_summary_omits_variant_fields_when_unused() -> None:
    summary = ProcessSummary(
        format_version="robotwin_process_dataset_summary_v1",
        annotation_mode="pick_place",
        required_object_roles=("target", "receiver"),
        gripper_backend="sam",
        run_id="run-1",
        dataset_root="/dataset",
        task="task",
        camera="cam_high",
        discovered_episode_ids=(7,),
        requested_episode_ids=(7,),
        dynamic_manifest={"format_version": "manifest"},
        qwen_health={"status": "ok"},
        records=(EpisodeRecord(7, "completed"),),
        render=None,
        fatal_error=None,
        backend={"object_masks": "sam", "gripper": "sam"},
        passed=True,
    )

    payload = summary.to_json()

    assert "stage_mode" not in payload
    assert "plan" not in payload
    assert "artifact" not in payload
    assert list(payload) == [
        "format_version",
        "annotation_mode",
        "required_object_roles",
        "gripper_backend",
        "run_id",
        "dataset_root",
        "task",
        "camera",
        "discovered_episode_ids",
        "requested_episode_ids",
        "dynamic_manifest",
        "qwen_health",
        "records",
        "render",
        "fatal_error",
        "backend",
        "passed",
    ]


def test_process_summary_variant_and_artifact_fields_follow_legacy_order() -> None:
    summary = ProcessSummary(
        format_version="robotwin_process_dataset_summary_v1",
        annotation_mode="pick_place",
        required_object_roles=("target",),
        gripper_backend="sam",
        run_id="run-1",
        dataset_root="/dataset",
        task="task",
        camera="cam_high",
        discovered_episode_ids=(7,),
        requested_episode_ids=(7,),
        dynamic_manifest={},
        qwen_health=None,
        records=(),
        render=None,
        fatal_error=None,
        backend={},
        passed=True,
        stage_mode="object_source_only",
        plan={"status": "planned"},
        artifact="/output/process_summary.json",
    )

    payload = summary.to_json()

    assert payload["stage_mode"] == "object_source_only"
    assert payload["plan"] == {"status": "planned"}
    assert payload["artifact"] == "/output/process_summary.json"
    assert list(payload)[-4:] == ["stage_mode", "passed", "plan", "artifact"]


def test_process_summary_round_trip_preserves_payload() -> None:
    payload = {
        "format_version": "robotwin_process_dataset_summary_v1",
        "annotation_mode": "pick_place",
        "required_object_roles": ["target", "receiver"],
        "gripper_backend": "sam",
        "run_id": "run-1",
        "dataset_root": "/dataset",
        "task": "task",
        "camera": "cam_high",
        "discovered_episode_ids": [7],
        "requested_episode_ids": [7],
        "dynamic_manifest": {"format_version": "manifest"},
        "qwen_health": {"status": "ok"},
        "records": [{"episode": 7, "status": "completed"}],
        "render": None,
        "fatal_error": None,
        "backend": {"object_masks": "sam", "gripper": "sam"},
        "stage_mode": "full_sam",
        "passed": True,
        "artifact": "/output/process_summary.json",
    }

    assert ProcessSummary.from_payload(payload).to_json() == payload


def test_process_request_keeps_shared_identity_and_selection() -> None:
    request = ProcessRequest(
        dataset_root=Path("/dataset"),
        output_root=Path("/output"),
        task="task",
        camera="cam_high",
        run_id="run-1",
        episode_ids=(7, 8),
        skip_render=True,
    )

    assert request.episode_ids == (7, 8)
    assert request.skip_render is True
