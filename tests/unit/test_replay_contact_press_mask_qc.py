from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from robotwin_annotation_v2.adapters.qwen_client import QwenCompletion
from robotwin_annotation_v2.config import MaskConfig
from scripts import replay_contact_press_mask_qc as replay


class _FakeClient:
    model_id = "test-qwen"

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[list[dict[str, Any]]] = []

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "model": self.model_id}

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
    ) -> QwenCompletion:
        assert max_tokens == 80
        self.calls.append(messages)
        return QwenCompletion(self.responses[len(self.calls) - 1], self.model_id)


def _candidate(
    candidate_id: str,
    *,
    query_field: str,
    query: str,
    seed_frame_id: int,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "query_field": query_field,
        "query": query,
        "nonempty": True,
        "area_fraction": 0.25,
        "component_count": 1,
        "basic_valid": True,
        "basic_reason": None,
        "duplicate_of": None,
        "seed_frame_id": seed_frame_id,
    }


def _attempt(
    candidate: dict[str, Any],
    *,
    method: str,
    seed_frame_id: int,
) -> dict[str, Any]:
    return {
        "seed_frame_id": seed_frame_id,
        "status": "rejected",
        "selected_candidate": None,
        "selected_query_field": None,
        "selected_query": None,
        "confidence": 0.9,
        "reason": "old prompt rejected candidate",
        "method": method,
        "candidates": [candidate],
        "model": "test-qwen",
        "raw_response": "old response",
        "rendered_prompt": "old prompt",
        "provenance": {},
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    artifact_root = tmp_path / "artifact-root"
    episode_dir = artifact_root / "archive" / "click_bell" / "episode_000007" / "cam_high"
    episode_dir.mkdir(parents=True)
    loop = {
        "format_version": "robotwin_loop_context_v4",
        "annotation_mode": "target_only",
        "timeline_kind": "close_hold",
        "required_object_roles": ["target"],
        "episode": {
            "task": "click_bell",
            "episode_index": 7,
            "episode_id": "000007",
            "camera": "cam_high",
        },
        "task_text": "Press the bell button.",
        "frame_count": 6,
        "events": {
            "active_arm": "left",
            "t_remove_start": 2,
            "t_close_start": 3,
            "t_close_end": 4,
        },
        "windows": {
            "operation": [2, 5],
            "target_0": [2, 5],
            "receiver_0": None,
            "gripper": [2, 5],
        },
        "semantic_frames": [
            {
                "frame_id": 0,
                "purpose": "pre_grasp_seed_candidate",
                "eligible_roles": ["target"],
            },
            {
                "frame_id": 1,
                "purpose": "pre_grasp_seed_candidate",
                "eligible_roles": ["target"],
            },
            {
                "frame_id": 4,
                "purpose": "post_grasp_context",
                "eligible_roles": ["target"],
            },
        ],
        "sources": {"state": "/unused/state.parquet", "video": "/unused/video.mp4"},
    }
    _write_json(episode_dir / "loop.json", loop)

    candidate_a = _candidate(
        "A",
        query_field="category_query",
        query="bell button",
        seed_frame_id=0,
    )
    candidate_bbox = _candidate(
        "BBOX",
        query_field="bbox_fallback",
        query="Qwen-localized target bounding box",
        seed_frame_id=1,
    )
    attempts = [
        _attempt(candidate_a, method="text_query", seed_frame_id=0),
        _attempt(candidate_bbox, method="bbox_fallback", seed_frame_id=1),
    ]
    mask_qc = {
        "format_version": "robotwin_mask_qc_v2",
        "roles": {
            "target": {
                "role": "target",
                "status": "rejected",
                "selected_candidate": None,
                "selected_query_field": None,
                "selected_query": None,
                "selected_seed_frame_id": None,
                "confidence": 0.9,
                "reason": "old final rejection",
                "model": "test-qwen",
                "raw_response": "old final response",
                "rendered_prompt": "old final prompt",
                "attempts": attempts,
            }
        },
        "artifacts": {
            "attempts": {
                "target": {
                    "frame_000000": {
                        "seed_frame_id": 0,
                        "candidate_masks": {
                            "A": "target/qc_candidates/frame_000000/candidate_A.mask.png"
                        },
                    },
                    "frame_000001": {
                        "seed_frame_id": 1,
                        "candidate_masks": {
                            "BBOX": (
                                "target/qc_candidates/frame_000001/"
                                "candidate_BBOX.mask.png"
                            )
                        },
                    },
                }
            }
        },
    }
    _write_json(episode_dir / "mask_qc.json", mask_qc)
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[:2, :2] = 255
    for relative in (
        "target/qc_candidates/frame_000000/candidate_A.mask.png",
        "target/qc_candidates/frame_000001/candidate_BBOX.mask.png",
    ):
        path = episode_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask, mode="L").save(path)

    audit_path = tmp_path / "audit.json"
    _write_json(
        audit_path,
        {
            "episodes": [
                {
                    "task": "click_bell",
                    "episode": 7,
                    "stage": "mask_qc_rejected",
                    "artifacts": {
                        "source_episode": (
                            "archive/click_bell/episode_000007/cam_high"
                        )
                    },
                },
                {
                    "task": "ignored_grasp_task",
                    "episode": 8,
                    "stage": "completed",
                    "artifacts": {},
                },
            ]
        },
    )
    prompt = tmp_path / "new-qc-prompt.txt"
    prompt.write_text(
        "task={task_text} role={role} seed={seed_frame_id} ids={candidate_ids}\n"
        "arm={active_arm} remove={remove_start} close={close_start}-{close_end}\n"
        "{candidate_panels}\n{context_frames}\n",
        encoding="utf-8",
    )
    return audit_path, artifact_root, episode_dir, prompt


def test_replay_reuses_every_archived_visual_attempt_and_resolves_first_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit, artifact_root, episode_dir, prompt = _fixture(tmp_path)
    frames = {frame_id: Image.new("RGB", (4, 4), "white") for frame_id in (0, 1, 4)}
    monkeypatch.setattr(replay, "_decode_frames", lambda *_args, **_kwargs: frames)
    client = _FakeClient(
        [
            json.dumps(
                {
                    "decision": "accept",
                    "selected_candidate": "A",
                    "confidence": 0.95,
                    "reason": "correct upcoming contact site",
                }
            ),
            json.dumps(
                {
                    "decision": "accept",
                    "selected_candidate": "BBOX",
                    "confidence": 0.92,
                    "reason": "also covers contact site",
                }
            ),
        ]
    )
    mask_config = MaskConfig(
        qc_enabled=True,
        qc_prompt_template=prompt,
        qc_max_tokens=80,
        qc_max_attempts=1,
    )

    report = replay.replay_archive(
        audit_path=audit,
        artifact_root=artifact_root,
        config_path=tmp_path / "config.yaml",
        qc_prompt_path=prompt,
        mask_config=mask_config,
        client=client,
    )

    assert len(client.calls) == 2
    assert report["summary"] == {
        "baseline_episode_count": 2,
        "baseline_success_count": 1,
        "baseline_success_rate": 0.5,
        "projected_success_count": 2,
        "projected_success_rate": 1.0,
        "projected_delta_percentage_points": 50.0,
        "episode_count": 1,
        "archived_attempt_count": 2,
        "replayed_visual_attempt_count": 2,
        "transition_counts": {"rejected->passed": 1},
        "old_rejected_new_passed_count": 1,
        "old_rejected_new_passed": [{"task": "click_bell", "episode": 7}],
    }
    episode = report["episodes"][0]
    assert episode["old"]["status"] == "rejected"
    assert episode["new"]["status"] == "passed"
    assert episode["new"]["selected_candidate"] == "A"
    assert episode["selected_new_attempt_index"] == 0
    assert [attempt["new"]["status"] for attempt in episode["attempts"]] == [
        "passed",
        "passed",
    ]
    mask_path = episode_dir / "target/qc_candidates/frame_000000/candidate_A.mask.png"
    assert episode["attempts"][0]["candidate_artifacts"][0]["sha256"] == hashlib.sha256(
        mask_path.read_bytes()
    ).hexdigest()
    assert report["replacement_qc_prompt"]["sha256"] == hashlib.sha256(
        prompt.read_bytes()
    ).hexdigest()


def test_replay_fails_closed_when_an_archived_candidate_mask_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit, artifact_root, episode_dir, prompt = _fixture(tmp_path)
    missing = episode_dir / "target/qc_candidates/frame_000000/candidate_A.mask.png"
    missing.unlink()
    frames = {frame_id: Image.new("RGB", (4, 4), "white") for frame_id in (0, 1, 4)}
    monkeypatch.setattr(replay, "_decode_frames", lambda *_args, **_kwargs: frames)
    client = _FakeClient([])

    with pytest.raises(replay.ReplayError, match="archived candidate mask is missing"):
        replay.replay_archive(
            audit_path=audit,
            artifact_root=artifact_root,
            config_path=tmp_path / "config.yaml",
            qc_prompt_path=prompt,
            mask_config=MaskConfig(
                qc_enabled=True,
                qc_prompt_template=prompt,
                qc_max_tokens=80,
                qc_max_attempts=1,
            ),
            client=client,
        )

    assert not client.calls
