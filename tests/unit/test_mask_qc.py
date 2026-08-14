from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from robotwin_annotation_v2.adapters import ArtifactStore, QwenCompletion
from robotwin_annotation_v2.config import MaskConfig
from robotwin_annotation_v2.domain import AnnotationMode
from robotwin_annotation_v2.models import (
    EpisodeRef,
    FramePurpose,
    LoopContext,
    LoopEvents,
    MaskQCStatus,
    QueryBank,
    RoleSemanticPlan,
    SemanticFrame,
    SemanticPlan,
    SemanticStatus,
    TargetOnlyEvents,
)
from robotwin_annotation_v2.pipeline import (
    MaskQCError,
    parse_mask_qc_response,
    run_mask_qc_stage,
    save_mask_qc_artifacts,
)
from robotwin_annotation_v2.pipeline.mask_qc import _context_items

FRAME_SHAPE = (12, 16)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _context() -> LoopContext:
    return LoopContext(
        episode=EpisodeRef("move_pillbottle_pad", 7274, "cam_high"),
        task_text="Place the brown bottle with wide white label onto the pad.",
        frame_count=20,
        events=LoopEvents("right", 2, 6, 8, 14, 17),
        semantic_frames=(
            SemanticFrame(
                0,
                FramePurpose.PRE_GRASP_SEED_CANDIDATE,
                ("target", "receiver"),
            ),
            SemanticFrame(7, FramePurpose.POST_GRASP_CONTEXT, ("target",)),
            SemanticFrame(15, FramePurpose.PLACE_CONTEXT, ("receiver",)),
        ),
        state_source="state.parquet",
        video_source="video.mp4",
    )


def _role(role: str) -> RoleSemanticPlan:
    if role == "target":
        bank = QueryBank(
            category_query="bottle",
            color_category_query="brown bottle",
            shape_category_query="cylindrical bottle",
            general_fallback_query="container",
            recommended_order=(
                "category_query",
                "color_category_query",
                "shape_category_query",
                "general_fallback_query",
            ),
        )
    else:
        bank = QueryBank(
            category_query="pad",
            color_category_query="blue square pad",
            shape_category_query="square pad",
            general_fallback_query="mat",
            recommended_order=(
                "color_category_query",
                "category_query",
                "shape_category_query",
                "general_fallback_query",
            ),
        )
    return RoleSemanticPlan(
        role=role,  # type: ignore[arg-type]
        status=SemanticStatus.OK,
        seed_frame_id=0,
        query_bank=bank,
        exclude=(),
        reason=f"{role} reason",
    )


def _plan() -> SemanticPlan:
    return SemanticPlan(
        episode=_context().episode,
        role_plans=(_role("target"), _role("receiver")),
        model="fake-qwen",
        prompt_sha256=hashlib.sha256(b"prompt").hexdigest(),
        input_frame_ids=(0, 7, 15),
        raw_response="{}",
    )


def _target_only_context() -> LoopContext:
    base = _context()
    return LoopContext(
        episode=base.episode,
        task_text=base.task_text,
        frame_count=base.frame_count,
        events=TargetOnlyEvents("right", 2, 6, 8),
        semantic_frames=(
            SemanticFrame(
                0,
                FramePurpose.PRE_GRASP_SEED_CANDIDATE,
                ("target",),
            ),
            SemanticFrame(7, FramePurpose.POST_GRASP_CONTEXT, ("target",)),
            SemanticFrame(15, FramePurpose.POST_GRASP_CONTEXT, ("target",)),
        ),
        state_source=base.state_source,
        video_source=base.video_source,
        annotation_mode=AnnotationMode.TARGET_ONLY,
    )


def _target_only_plan() -> SemanticPlan:
    context = _target_only_context()
    return SemanticPlan(
        episode=context.episode,
        role_plans=(_role("target"),),
        model="fake-qwen",
        prompt_sha256=hashlib.sha256(b"prompt").hexdigest(),
        input_frame_ids=(0, 7, 15),
        raw_response="{}",
        annotation_mode=AnnotationMode.TARGET_ONLY,
    )


def _images() -> dict[int, Image.Image]:
    return {
        frame_id: Image.fromarray(np.full((*FRAME_SHAPE, 3), 100 + frame_id, dtype=np.uint8))
        for frame_id in (0, 7, 15)
    }


class FakeCandidateBackend:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def text_query_masks(
        self,
        _resource_path: Path,
        texts: tuple[str, ...],
        **_kwargs: Any,
    ) -> dict[str, np.ndarray]:
        positions = {
            "bottle": (1, 1),
            "brown bottle": (4, 9),
            "cylindrical bottle": (4, 9),
            "blue square pad": (8, 10),
            "pad": (8, 10),
            "square pad": (8, 10),
        }
        result: dict[str, np.ndarray] = {}
        for text in texts:
            self.calls.append(text)
            mask = np.zeros(FRAME_SHAPE, dtype=bool)
            row, column = positions[text]
            mask[row : row + 2, column : column + 3] = True
            result[text] = mask
        return result


class EmptyReceiverBackend(FakeCandidateBackend):
    def text_query_masks(
        self,
        resource_path: Path,
        texts: tuple[str, ...],
        **kwargs: Any,
    ) -> dict[str, np.ndarray]:
        result = super().text_query_masks(resource_path, texts, **kwargs)
        for text in texts:
            if text in {"blue square pad", "pad", "square pad"}:
                result[text] = np.zeros(FRAME_SHAPE, dtype=bool)
        return result


class FakeQCClient:
    model_id = "fake-qwen"

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.messages: list[list[dict[str, Any]]] = []

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "model": self.model_id}

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
    ) -> QwenCompletion:
        assert max_tokens == 123
        self.messages.append(messages)
        return QwenCompletion(self.responses.pop(0), self.model_id)


class FlakyQCClient(FakeQCClient):
    def __init__(self, responses: list[str]) -> None:
        super().__init__(responses)
        self.attempts = 0

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
    ) -> QwenCompletion:
        self.attempts += 1
        if self.attempts == 1:
            raise TimeoutError("temporary timeout")
        return super().complete(messages, max_tokens=max_tokens)


def _response(candidate: str, confidence: float = 0.95) -> str:
    return json.dumps(
        {
            "decision": "accept",
            "selected_candidate": candidate,
            "confidence": confidence,
            "reason": "The contour covers the requested complete instance.",
        }
    )


def _config(prompt: Path, **overrides: Any) -> MaskConfig:
    values = {
        "qc_enabled": True,
        "qc_prompt_template": prompt,
        "qc_max_candidates": 3,
        "qc_max_tokens": 123,
        "qc_min_confidence": 0.7,
        "qc_min_area_fraction": 0.001,
        "qc_max_area_fraction": 0.5,
    }
    values.update(overrides)
    return MaskConfig(0, 0, **values)


def _prompt(tmp_path: Path) -> Path:
    path = tmp_path / "mask-qc.txt"
    path.write_text(
        "task={task_text}; role={role}; seed={seed_frame_id}\n"
        "candidates:\n{candidate_panels}\ncontext:\n{context_frames}\n"
        "return strict json",
        encoding="utf-8",
    )
    return path


def test_context_sampling_keeps_first_and_last_evidence_frames() -> None:
    context = LoopContext(
        episode=_context().episode,
        task_text=_context().task_text,
        frame_count=20,
        events=_context().events,
        semantic_frames=(
            SemanticFrame(0, FramePurpose.PRE_GRASP_SEED_CANDIDATE, ("receiver",)),
            SemanticFrame(3, FramePurpose.POST_GRASP_CONTEXT, ("receiver",)),
            SemanticFrame(7, FramePurpose.POST_GRASP_CONTEXT, ("receiver",)),
            SemanticFrame(11, FramePurpose.POST_GRASP_CONTEXT, ("receiver",)),
            SemanticFrame(15, FramePurpose.PLACE_CONTEXT, ("receiver",)),
        ),
        state_source="state.parquet",
        video_source="video.mp4",
    )
    images = {
        frame_id: Image.fromarray(np.zeros((*FRAME_SHAPE, 3), dtype=np.uint8))
        for frame_id in (0, 3, 7, 11, 15)
    }

    sampled = _context_items(context, "receiver", 0, images)

    assert [frame_id for frame_id, _image in sampled] == [3, 15]


def test_context_sampling_prefers_action_evidence_over_extra_seed_frames() -> None:
    context = LoopContext(
        episode=_target_only_context().episode,
        task_text=_target_only_context().task_text,
        frame_count=20,
        events=TargetOnlyEvents("right", 2, 6, 8),
        semantic_frames=(
            SemanticFrame(0, FramePurpose.PRE_GRASP_SEED_CANDIDATE, ("target",)),
            SemanticFrame(3, FramePurpose.PRE_GRASP_SEED_CANDIDATE, ("target",)),
            SemanticFrame(7, FramePurpose.POST_GRASP_CONTEXT, ("target",)),
            SemanticFrame(15, FramePurpose.POST_GRASP_CONTEXT, ("target",)),
        ),
        state_source="state.parquet",
        video_source="video.mp4",
        annotation_mode=AnnotationMode.TARGET_ONLY,
    )
    images = {
        frame_id: Image.fromarray(np.zeros((*FRAME_SHAPE, 3), dtype=np.uint8))
        for frame_id in (0, 3, 7, 15)
    }

    sampled = _context_items(context, "target", 0, images)

    assert [frame_id for frame_id, _image in sampled] == [7, 15]


def test_parse_mask_qc_response_validates_candidate_contract() -> None:
    decision, selected, confidence, reason = parse_mask_qc_response(
        _response("b"),
        candidate_ids=("A", "B"),
    )

    assert (decision, selected, confidence) == ("accept", "B", 0.95)
    assert reason.startswith("The contour")

    with pytest.raises(MaskQCError, match="not one of"):
        parse_mask_qc_response(_response("C"), candidate_ids=("A", "B"))


def test_mask_qc_selects_actual_candidate_masks_and_saves_provenance(
    tmp_path: Path,
) -> None:
    prompt = _prompt(tmp_path)
    backend = FakeCandidateBackend()
    client = FakeQCClient([_response("B"), _response("A")])
    images = _images()

    result = run_mask_qc_stage(
        _context(),
        _plan(),
        backend,
        Path("/tmp/resource"),
        seed_images={0: images[0]},
        context_images=images,
        frame_shape=FRAME_SHAPE,
        mask_config=_config(prompt),
        client=client,
    )

    assert result.target.status is MaskQCStatus.PASSED
    assert result.target.selected_query == "brown bottle"
    assert result.receiver.status is MaskQCStatus.PASSED
    assert result.receiver.selected_query == "blue square pad"
    assert result.selected_masks["target"][4, 9]
    assert backend.calls == [
        "bottle",
        "brown bottle",
        "cylindrical bottle",
        "blue square pad",
        "pad",
        "square pad",
    ]
    assert len(client.messages) == 2

    path = save_mask_qc_artifacts(
        ArtifactStore(tmp_path / "artifacts"),
        "qc-test",
        _context(),
        result,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["roles"]["target"]["status"] == "passed"
    assert payload["roles"]["target"]["selected_query"] == "brown bottle"
    assert payload["roles"]["target"]["candidates"][2]["duplicate_of"] == "B"
    candidate_dir = path.parent / "target/qc_candidates"
    assert sorted(item.name for item in candidate_dir.glob("*.mask.png")) == [
        "candidate_A.mask.png",
        "candidate_B.mask.png",
        "candidate_C.mask.png",
    ]
    assert sorted(item.name for item in candidate_dir.glob("*.overlay.png")) == [
        "candidate_A.overlay.png",
        "candidate_B.overlay.png",
        "candidate_C.overlay.png",
    ]


def test_mask_qc_fails_closed_when_qwen_confidence_is_low(tmp_path: Path) -> None:
    images = _images()
    result = run_mask_qc_stage(
        _context(),
        _plan(),
        FakeCandidateBackend(),
        Path("/tmp/resource"),
        seed_images={0: images[0]},
        context_images=images,
        frame_shape=FRAME_SHAPE,
        mask_config=_config(_prompt(tmp_path), qc_min_confidence=0.8),
        client=FakeQCClient([_response("B", 0.4), _response("A", 0.4)]),
    )

    assert result.target.status is MaskQCStatus.AMBIGUOUS
    assert result.target.selected_query is None
    assert result.receiver.status is MaskQCStatus.AMBIGUOUS
    assert result.selected_masks == {}


def test_mask_qc_retries_transient_qwen_request_without_regenerating_masks(
    tmp_path: Path,
) -> None:
    images = _images()
    backend = FakeCandidateBackend()
    client = FlakyQCClient([_response("B"), _response("A")])

    result = run_mask_qc_stage(
        _context(),
        _plan(),
        backend,
        Path("/tmp/resource"),
        seed_images={0: images[0]},
        context_images=images,
        frame_shape=FRAME_SHAPE,
        mask_config=_config(_prompt(tmp_path)),
        client=client,
    )

    assert result.target.status is MaskQCStatus.PASSED
    assert result.receiver.status is MaskQCStatus.PASSED
    assert client.attempts == 3
    assert len(backend.calls) == 6


def test_target_only_qc_runs_no_receiver_candidates(tmp_path: Path) -> None:
    context = _target_only_context()
    images = _images()
    backend = FakeCandidateBackend()

    result = run_mask_qc_stage(
        context,
        _target_only_plan(),
        backend,
        Path("/tmp/resource"),
        seed_images={0: images[0]},
        context_images={0: images[0], 7: images[7], 15: images[15]},
        frame_shape=FRAME_SHAPE,
        mask_config=_config(_prompt(tmp_path)),
        client=FakeQCClient([_response("B")]),
    )

    assert tuple(report.role for report in result.role_reports) == ("target",)
    assert set(result.candidate_masks) == {"target"}
    assert all("pad" not in query for query in backend.calls)
    with pytest.raises(KeyError, match="non-applicable"):
        _ = result.receiver


def test_mask_qc_rejects_a_semantic_plan_from_another_mode(
    tmp_path: Path,
) -> None:
    context = _target_only_context()

    with pytest.raises(MaskQCError, match="different annotation modes"):
        run_mask_qc_stage(
            context,
            _plan(),
            FakeCandidateBackend(),
            Path("/tmp/resource"),
            seed_images={0: _images()[0]},
            context_images=_images(),
            frame_shape=FRAME_SHAPE,
            mask_config=_config(_prompt(tmp_path)),
            client=FakeQCClient([_response("A")]),
        )


def test_target_only_qc_prompt_contains_only_target_rules() -> None:
    context = _target_only_context()
    images = _images()
    client = FakeQCClient([_response("B")])

    run_mask_qc_stage(
        context,
        _target_only_plan(),
        FakeCandidateBackend(),
        Path("/tmp/resource"),
        seed_images={0: images[0]},
        context_images={0: images[0], 7: images[7], 15: images[15]},
        frame_shape=FRAME_SHAPE,
        mask_config=_config(
            PROJECT_ROOT / "configs/prompts/target_only_mask_candidate_qc.txt"
        ),
        client=client,
    )

    content = client.messages[0][0]["content"]
    prompt_text = "\n".join(
        part["text"] for part in content if part["type"] == "text"
    )
    assert "本次只检查 target" in prompt_text
    assert "随后真正被夹爪抓取并移动的实例" in prompt_text
    assert "receiver" not in prompt_text
    assert "active_arm: right" in prompt_text
    assert "remove_start: 2" in prompt_text
    assert "close_start: 6" in prompt_text
    assert "close_end: 8" in prompt_text
    assert "episode_end: 19" in prompt_text
    assert "open_start" not in prompt_text
    assert "open_done" not in prompt_text


def test_receiver_blue_region_prior_recovers_empty_text_candidates(
    tmp_path: Path,
) -> None:
    images = _images()
    seed = np.asarray(images[0]).copy()
    seed[7:11, 9:15] = np.asarray([10, 20, 240], dtype=np.uint8)
    images[0] = Image.fromarray(seed)
    backend = EmptyReceiverBackend()

    result = run_mask_qc_stage(
        _context(),
        _plan(),
        backend,
        Path("/tmp/resource"),
        seed_images={0: images[0]},
        context_images=images,
        frame_shape=FRAME_SHAPE,
        mask_config=_config(_prompt(tmp_path)),
        client=FakeQCClient([_response("B"), _response("C")]),
    )

    assert result.receiver.status is MaskQCStatus.PASSED
    assert result.receiver.selected_query_field == "blue_region_prior"
    assert result.receiver.selected_query == "blue planar region"
    assert result.selected_masks["receiver"][8, 10]
    assert [candidate.query_field for candidate in result.receiver.candidates] == [
        "color_category_query",
        "category_query",
        "blue_region_prior",
    ]


def test_receiver_qc_prompt_defines_contact_and_place_context() -> None:
    images = _images()
    client = FakeQCClient([_response("B"), _response("A")])

    run_mask_qc_stage(
        _context(),
        _plan(),
        FakeCandidateBackend(),
        Path("/tmp/resource"),
        seed_images={0: images[0]},
        context_images=images,
        frame_shape=FRAME_SHAPE,
        mask_config=_config(
            PROJECT_ROOT / "configs/prompts/mask_candidate_qc.txt"
        ),
        client=client,
    )

    receiver_content = client.messages[1][0]["content"]
    receiver_text = "\n".join(
        part["text"] for part in receiver_content if part["type"] == "text"
    )
    assert "任务完成时应与 target 直接接触" in receiver_text
    assert "核心判断依据是二者的直接接触关系" in receiver_text
    assert "不能只凭“位于 target 下方或附近”判断" in receiver_text
    assert "不能选择 target 本身" in receiver_text
    assert "不得因为候选“不是 target”而拒绝它" in receiver_text
    assert "完整的薄平面目标区域是合法 receiver" in receiver_text
    assert "purpose=place_context" in receiver_text
    assert "open_done: 17" in receiver_text
