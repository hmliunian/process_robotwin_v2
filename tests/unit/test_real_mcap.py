from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from robotwin_annotation_v2.adapters.real_mcap import (
    McapEpisode,
    RealMcapError,
    SourceMetadata,
    align_episode,
    load_text_records,
    materialize_real_mcap_dataset,
    normalize_gripper_loop,
)

REPOSITORY_ROOT = Path(__file__).parents[2]
TEXT_MANIFEST = REPOSITORY_ROOT / "configs" / "datasets" / "pick_and_place_real_texts.json"


def test_reviewed_text_manifest_selects_only_complete_pick_place() -> None:
    records = load_text_records(TEXT_MANIFEST)

    assert len(records) == 33
    assert sum(record.quality_status == "complete_pick_place" for record in records) == 29
    assert sum(record.quality_status == "incomplete_no_release" for record in records) == 2
    assert sum(record.quality_status == "non_pick_place" for record in records) == 2
    assert all(
        record.receiver is not None
        for record in records
        if record.quality_status == "complete_pick_place"
    )


def test_reviewed_text_manifest_preserves_corrected_visual_roles() -> None:
    records = {record.source_mcap: record for record in load_text_records(TEXT_MANIFEST)}

    failed_grasp = records["28ec209220db49b245509e0e516a6600.mcap"]
    assert failed_grasp.quality_status == "incomplete_no_release"
    assert "never secures or transports" in failed_grasp.quality_reason
    assert (
        records["a2a81be2bd09e9b01068627644c09241.mcap"].target,
        records["a2a81be2bd09e9b01068627644c09241.mcap"].receiver,
    ) == (
        "black rectangular pump bottle",
        "top surface of white three-drawer cabinet",
    )
    assert (
        records["f880addbeaf365882d3aa434115b373a.mcap"].target,
        records["f880addbeaf365882d3aa434115b373a.mcap"].receiver,
    ) == (
        "small white rectangular pump bottle",
        "top surface of right-hand dryer",
    )
    assert records["e8823d88445fd1d5606407de704f65f7.mcap"].target == (
        "brown fabric oven mitt"
    )
    assert records["996b5e8bf7b2a00fface0e9402c79c9c.mcap"].receiver == (
        "white sheet covering wooden seat"
    )
    assert records["39f6766a215e189986d74ee7dd731b90.mcap"].target == (
        "small pink pump bottle"
    )


def test_text_manifest_rejects_complete_record_without_receiver(tmp_path: Path) -> None:
    path = tmp_path / "texts.json"
    path.write_text(
        json.dumps(
            {
                "format_version": "real_pnp_texts_v1",
                "records": [
                    {
                        "source_mcap": "source.mcap",
                        "task_text": "Pick up the item and place it.",
                        "task_text_zh": "拿起物体并放置。",
                        "target": "item",
                        "receiver": None,
                        "quality_status": "complete_pick_place",
                        "quality_reason": "test",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RealMcapError, match="has no receiver"):
        load_text_records(path)


def test_normalize_gripper_loop_encodes_one_binary_close_release() -> None:
    raw = np.asarray([0.0, 0.08, 0.08, 0.08, 0.02, 0.02, 0.02, 0.09, 0.09, 0.09, 0.0])

    normalized, close_frame, release_frame = normalize_gripper_loop(raw)

    assert close_frame == 4
    assert release_frame == 7
    np.testing.assert_array_equal(
        normalized,
        np.asarray([1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1], dtype=np.float32),
    )


def test_normalize_gripper_loop_rejects_recording_without_release() -> None:
    raw = np.asarray([0.0, 0.08, 0.08, 0.08, 0.02, 0.02, 0.02, 0.01, 0.01])

    with pytest.raises(RealMcapError, match="no stable release"):
        normalize_gripper_loop(raw)


def test_normalize_gripper_loop_allows_target_only_hold_without_release() -> None:
    raw = np.asarray([0.0, 0.04, 0.05, 0.05, 0.05, 0.02, 0.01, 0.01, 0.01])

    normalized, close_frame, release_frame = normalize_gripper_loop(
        raw,
        require_release=False,
    )

    assert close_frame == 5
    assert release_frame == len(raw)
    assert np.array_equal(normalized, np.asarray([1.0] * 5 + [0.0] * 4))


def test_align_episode_uses_video_timestamps_as_frame_authority(tmp_path: Path) -> None:
    count = 11
    timestamps = np.arange(count, dtype=np.int64) * 100_000_000
    joints = np.zeros((count, 8), dtype=np.float64)
    joints[:, 0] = np.linspace(0.0, 1.0, count)
    joints[:, 7] = [0.0, 0.08, 0.08, 0.08, 0.02, 0.02, 0.02, 0.09, 0.09, 0.09, 0.0]
    positions = np.column_stack(
        (
            np.linspace(0.1, 0.2, count),
            np.linspace(-0.1, 0.1, count),
            np.linspace(0.5, 0.6, count),
        )
    )
    quaternions = np.tile(np.asarray([0.0, 0.0, 0.0, 1.0]), (count, 1))
    episode = McapEpisode(
        source=SourceMetadata(tmp_path / "source.mcap", {"start_time_us": "1"}),
        video_times_ns=timestamps,
        video_payloads=tuple(b"frame" for _ in range(count)),
        joint_times_ns=timestamps,
        joint_values=joints,
        eef_times_ns=timestamps,
        eef_positions=positions,
        eef_quaternions=quaternions,
    )

    aligned = align_episode(episode)

    assert aligned.states.shape == (count, 14)
    assert aligned.actions.shape == (count, 14)
    assert aligned.joint_absolute.shape == (count, 14)
    np.testing.assert_allclose(aligned.timestamps_s, np.arange(count) / 10)
    np.testing.assert_allclose(aligned.states[:, 7:10], positions)
    np.testing.assert_array_equal(aligned.states[:, 6], np.ones(count))
    np.testing.assert_array_equal(aligned.states[:, 13], [1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1])
    assert np.isfinite(aligned.actions).all()


def test_materializer_refuses_to_overwrite_existing_directory(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()

    with pytest.raises(RealMcapError, match="refusing to overwrite"):
        materialize_real_mcap_dataset(source, output, text_manifest=TEXT_MANIFEST)
