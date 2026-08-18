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
    MaskQCAttemptMethod,
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


def _multi_seed_context() -> LoopContext:
    base = _context()
    return LoopContext(
        episode=base.episode,
        task_text=base.task_text,
        frame_count=base.frame_count,
        events=base.events,
        semantic_frames=(
            SemanticFrame(
                0,
                FramePurpose.PRE_GRASP_SEED_CANDIDATE,
                ("target", "receiver"),
            ),
            SemanticFrame(
                3,
                FramePurpose.PRE_GRASP_SEED_CANDIDATE,
                ("target",),
            ),
            SemanticFrame(
                5,
                FramePurpose.PRE_GRASP_SEED_CANDIDATE,
                ("target",),
            ),
            SemanticFrame(7, FramePurpose.POST_GRASP_CONTEXT, ("target",)),
            SemanticFrame(15, FramePurpose.PLACE_CONTEXT, ("receiver",)),
        ),
        state_source=base.state_source,
        video_source=base.video_source,
    )


def _multi_seed_plan() -> SemanticPlan:
    context = _multi_seed_context()
    return SemanticPlan(
        episode=context.episode,
        role_plans=(_role("target"), _role("receiver")),
        model="fake-qwen",
        prompt_sha256=hashlib.sha256(b"prompt").hexdigest(),
        input_frame_ids=(0, 3, 5, 7, 15),
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


class SeedAwareCandidateBackend(FakeCandidateBackend):
    def __init__(self, *, empty_target_frames: tuple[int, ...] = ()) -> None:
        super().__init__()
        self.empty_target_frames = frozenset(empty_target_frames)
        self.frame_calls: list[tuple[int, tuple[str, ...]]] = []

    def text_query_masks(
        self,
        resource_path: Path,
        texts: tuple[str, ...],
        *,
        frame_id: int,
        **kwargs: Any,
    ) -> dict[str, np.ndarray]:
        self.frame_calls.append((frame_id, texts))
        result = super().text_query_masks(resource_path, texts, **kwargs)
        if frame_id in self.empty_target_frames and "bottle" in texts:
            return {text: np.zeros(FRAME_SHAPE, dtype=bool) for text in texts}
        return result


class QueryFallbackBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def text_query_masks(
        self,
        _resource_path: Path,
        texts: tuple[str, ...],
        **_kwargs: Any,
    ) -> dict[str, np.ndarray]:
        self.calls.append(texts)
        result: dict[str, np.ndarray] = {}
        for text in texts:
            mask = np.zeros(FRAME_SHAPE, dtype=bool)
            if text in {"bottle", "brown bottle", "cylindrical bottle", "container"}:
                mask[2:4, 2:5] = True
            elif text == "blue table mat":
                mask[7:11, 8:14] = True
            result[text] = mask
        return result


class BboxFallbackBackend(FakeCandidateBackend):
    def __init__(self, *, empty_target_frames: tuple[int, ...] = (0,)) -> None:
        super().__init__()
        self.empty_target_frames = frozenset(empty_target_frames)
        self.frame_calls: list[tuple[str, int]] = []
        self.box_calls: list[tuple[int, tuple[float, ...]]] = []

    def text_query_masks(
        self,
        resource_path: Path,
        texts: tuple[str, ...],
        *,
        frame_id: int,
        **kwargs: Any,
    ) -> dict[str, np.ndarray]:
        self.frame_calls.append(("text", frame_id))
        result = super().text_query_masks(resource_path, texts, **kwargs)
        if frame_id in self.empty_target_frames and "bottle" in texts:
            return {text: np.zeros(FRAME_SHAPE, dtype=bool) for text in texts}
        return result

    def box_mask(
        self,
        _resource_path: Path,
        box_xyxy: tuple[float, ...],
        *,
        frame_id: int,
        **_kwargs: Any,
    ) -> np.ndarray:
        self.frame_calls.append(("bbox", frame_id))
        self.box_calls.append((frame_id, tuple(box_xyxy)))
        mask = np.zeros(FRAME_SHAPE, dtype=bool)
        mask[3:7, 4:9] = True
        return mask


class BrokenTextBackend(BboxFallbackBackend):
    def __init__(self, failure: str) -> None:
        super().__init__(empty_target_frames=())
        self.failure = failure
        self.text_calls: list[tuple[int, tuple[str, ...]]] = []

    def text_query_masks(
        self,
        resource_path: Path,
        texts: tuple[str, ...],
        *,
        frame_id: int,
        **kwargs: Any,
    ) -> dict[str, np.ndarray]:
        self.text_calls.append((frame_id, texts))
        if "bottle" in texts and self.failure == "exception":
            raise RuntimeError("text backend unavailable")
        result = super().text_query_masks(
            resource_path,
            texts,
            frame_id=frame_id,
            **kwargs,
        )
        if "bottle" not in texts:
            return result
        if self.failure == "missing_query":
            result.pop(texts[0])
        elif self.failure == "invalid_shape":
            result[texts[0]] = np.zeros((2, 3), dtype=bool)
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


def _decision_response(decision: str, confidence: float = 0.95) -> str:
    return json.dumps(
        {
            "decision": decision,
            "selected_candidate": None,
            "confidence": confidence,
            "reason": "The current seed does not provide a usable candidate.",
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


def _bbox_prompt(tmp_path: Path) -> Path:
    path = tmp_path / "bbox-localization.txt"
    path.write_text(
        "task={task}; instruction={task_text}; episode={episode_id}; "
        "role={role}; seed={seed_frame_id}",
        encoding="utf-8",
    )
    return path


def _bbox_response(
    bbox: tuple[float, float, float, float] = (0.2, 0.25, 0.6, 0.75),
) -> str:
    return json.dumps(
        {
            "status": "ok",
            "bbox_xyxy": list(bbox),
            "confidence": 0.86,
            "reason": "the manipulated bottle is visible",
        }
    )


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


def test_mask_qc_seed_fallback_defaults_to_the_semantic_seed_only(
    tmp_path: Path,
) -> None:
    context = _multi_seed_context()
    images = {
        frame_id: Image.fromarray(np.full((*FRAME_SHAPE, 3), 100 + frame_id, dtype=np.uint8))
        for frame_id in (0, 3, 5, 7, 15)
    }
    backend = SeedAwareCandidateBackend(empty_target_frames=(0,))

    result = run_mask_qc_stage(
        context,
        _multi_seed_plan(),
        backend,
        Path("/tmp/resource"),
        seed_images={frame_id: images[frame_id] for frame_id in (0, 3, 5)},
        context_images=images,
        frame_shape=FRAME_SHAPE,
        mask_config=_config(_prompt(tmp_path)),
        client=FakeQCClient([_response("A")]),
    )

    assert result.target.status is MaskQCStatus.REJECTED
    assert result.receiver.status is MaskQCStatus.PASSED
    assert [frame_id for frame_id, _texts in backend.frame_calls] == [0, 0]


def test_mask_qc_seed_fallback_uses_the_first_later_passing_seed(
    tmp_path: Path,
) -> None:
    context = _multi_seed_context()
    images = {
        frame_id: Image.fromarray(np.full((*FRAME_SHAPE, 3), 100 + frame_id, dtype=np.uint8))
        for frame_id in (0, 3, 5, 7, 15)
    }
    backend = SeedAwareCandidateBackend(empty_target_frames=(0,))

    result = run_mask_qc_stage(
        context,
        _multi_seed_plan(),
        backend,
        Path("/tmp/resource"),
        seed_images={frame_id: images[frame_id] for frame_id in (0, 3, 5)},
        context_images=images,
        frame_shape=FRAME_SHAPE,
        mask_config=_config(
            _prompt(tmp_path),
            qc_seed_fallback_enabled=True,
        ),
        client=FakeQCClient([_response("B"), _response("A")]),
    )

    assert result.target.status is MaskQCStatus.PASSED
    assert result.target.selected_seed_frame_id == 3
    assert result.target.selected_query == "brown bottle"
    assert all(candidate.seed_frame_id == 3 for candidate in result.target.candidates)
    assert result.target.rendered_prompt is not None
    assert "seed=3" in result.target.rendered_prompt
    assert result.selected_masks["target"][4, 9]
    assert [(attempt.seed_frame_id, attempt.status) for attempt in result.target.attempts] == [
        (0, MaskQCStatus.REJECTED),
        (3, MaskQCStatus.PASSED),
    ]
    assert set(result.attempt_candidate_masks["target"]) == {0, 3}
    assert {
        seed_frame_id: set(candidates)
        for seed_frame_id, candidates in result.attempt_candidate_masks["target"].items()
    } == {
        0: {"A", "B", "C"},
        3: {"A", "B", "C"},
    }
    assert result.receiver.selected_seed_frame_id == 0
    assert [frame_id for frame_id, _texts in backend.frame_calls] == [0, 3, 0]


@pytest.mark.parametrize("decision", ("reject_all", "ambiguous"))
def test_mask_qc_seed_fallback_continues_after_qwen_non_acceptance_and_stops_on_pass(
    tmp_path: Path,
    decision: str,
) -> None:
    context = _multi_seed_context()
    images = {
        frame_id: Image.fromarray(np.full((*FRAME_SHAPE, 3), 100 + frame_id, dtype=np.uint8))
        for frame_id in (0, 3, 5, 7, 15)
    }
    backend = SeedAwareCandidateBackend()

    result = run_mask_qc_stage(
        context,
        _multi_seed_plan(),
        backend,
        Path("/tmp/resource"),
        seed_images={frame_id: images[frame_id] for frame_id in (0, 3, 5)},
        context_images=images,
        frame_shape=FRAME_SHAPE,
        mask_config=_config(
            _prompt(tmp_path),
            qc_seed_fallback_enabled=True,
        ),
        client=FakeQCClient([_decision_response(decision), _response("B"), _response("A")]),
    )

    assert result.target.status is MaskQCStatus.PASSED
    assert result.target.selected_seed_frame_id == 3
    assert [frame_id for frame_id, _texts in backend.frame_calls] == [0, 3, 0]


def test_mask_qc_seed_fallback_stops_after_qwen_error(tmp_path: Path) -> None:
    context = _multi_seed_context()
    images = {
        frame_id: Image.fromarray(np.full((*FRAME_SHAPE, 3), 100 + frame_id, dtype=np.uint8))
        for frame_id in (0, 3, 5, 7, 15)
    }
    backend = SeedAwareCandidateBackend()

    result = run_mask_qc_stage(
        context,
        _multi_seed_plan(),
        backend,
        Path("/tmp/resource"),
        seed_images={frame_id: images[frame_id] for frame_id in (0, 3, 5)},
        context_images=images,
        frame_shape=FRAME_SHAPE,
        mask_config=_config(
            _prompt(tmp_path),
            qc_seed_fallback_enabled=True,
        ),
        client=FakeQCClient(["not-json", _response("A")]),
    )

    assert result.target.status is MaskQCStatus.ERROR
    assert result.receiver.status is MaskQCStatus.PASSED
    assert [frame_id for frame_id, _texts in backend.frame_calls] == [0, 0]


@pytest.mark.parametrize("failure", ("exception", "missing_query", "invalid_shape"))
def test_text_candidate_generation_errors_are_structured_and_stop_fallback(
    tmp_path: Path,
    failure: str,
) -> None:
    context = _multi_seed_context()
    images = {
        frame_id: Image.fromarray(np.full((*FRAME_SHAPE, 3), 100 + frame_id, dtype=np.uint8))
        for frame_id in (0, 3, 5, 7, 15)
    }
    backend = BrokenTextBackend(failure)

    result = run_mask_qc_stage(
        context,
        _multi_seed_plan(),
        backend,
        Path("/tmp/resource"),
        seed_images={frame_id: images[frame_id] for frame_id in (0, 3, 5)},
        context_images=images,
        frame_shape=FRAME_SHAPE,
        mask_config=_config(
            _prompt(tmp_path),
            qc_seed_fallback_enabled=True,
            qc_bbox_fallback_enabled=True,
            qc_bbox_prompt_template=_bbox_prompt(tmp_path),
        ),
        client=FakeQCClient([_response("A")]),
    )

    assert result.target.status is MaskQCStatus.ERROR
    assert len(result.target.attempts) == 1
    assert result.target.attempts[0].status is MaskQCStatus.ERROR
    assert result.target.attempts[0].method is MaskQCAttemptMethod.TEXT_QUERY
    assert result.target.reason.startswith("text candidate generation failed:")
    assert [frame_id for frame_id, texts in backend.text_calls if "bottle" in texts] == [0]
    assert backend.box_calls == []
    assert result.receiver.status is MaskQCStatus.PASSED


def test_invalid_visual_qc_prompt_is_structured_and_stops_fallback(tmp_path: Path) -> None:
    context = _multi_seed_context()
    images = {
        frame_id: Image.fromarray(np.full((*FRAME_SHAPE, 3), 100 + frame_id, dtype=np.uint8))
        for frame_id in (0, 3, 5, 7, 15)
    }
    prompt = tmp_path / "invalid-mask-qc.txt"
    prompt.write_text("missing image markers", encoding="utf-8")
    backend = BboxFallbackBackend(empty_target_frames=())
    client = FakeQCClient([])

    result = run_mask_qc_stage(
        context,
        _multi_seed_plan(),
        backend,
        Path("/tmp/resource"),
        seed_images={frame_id: images[frame_id] for frame_id in (0, 3, 5)},
        context_images=images,
        frame_shape=FRAME_SHAPE,
        mask_config=_config(
            prompt,
            qc_seed_fallback_enabled=True,
            qc_bbox_fallback_enabled=True,
            qc_bbox_prompt_template=_bbox_prompt(tmp_path),
        ),
        client=client,
    )

    assert result.target.status is MaskQCStatus.ERROR
    assert len(result.target.attempts) == 1
    assert result.target.reason.startswith("mask QC request could not be rendered:")
    assert backend.box_calls == []
    assert client.messages == []


def test_bbox_fallback_is_default_off_and_never_runs_for_a_text_failure(
    tmp_path: Path,
) -> None:
    images = _images()
    backend = BboxFallbackBackend()

    result = run_mask_qc_stage(
        _context(),
        _plan(),
        backend,
        Path("/tmp/resource"),
        seed_images={0: images[0]},
        context_images=images,
        frame_shape=FRAME_SHAPE,
        mask_config=_config(_prompt(tmp_path)),
        client=FakeQCClient([_response("A")]),
    )

    assert result.target.status is MaskQCStatus.REJECTED
    assert result.receiver.status is MaskQCStatus.PASSED
    assert backend.box_calls == []
    assert [attempt.method for attempt in result.target.attempts] == [
        MaskQCAttemptMethod.TEXT_QUERY
    ]


def test_bbox_fallback_runs_only_after_all_text_seeds_and_uses_normal_visual_qc(
    tmp_path: Path,
) -> None:
    context = _multi_seed_context()
    images = {
        frame_id: Image.fromarray(np.full((*FRAME_SHAPE, 3), 100 + frame_id, dtype=np.uint8))
        for frame_id in (0, 3, 5, 7, 15)
    }
    backend = BboxFallbackBackend(empty_target_frames=(0, 3, 5))
    requested_box = (0.2, 0.25, 0.6, 0.75)
    client = FakeQCClient(
        [
            _bbox_response(requested_box),
            _response("BBOX"),
            _response("A"),
        ]
    )

    result = run_mask_qc_stage(
        context,
        _multi_seed_plan(),
        backend,
        Path("/tmp/resource"),
        seed_images={frame_id: images[frame_id] for frame_id in (0, 3, 5)},
        context_images=images,
        frame_shape=FRAME_SHAPE,
        mask_config=_config(
            _prompt(tmp_path),
            qc_seed_fallback_enabled=True,
            qc_bbox_fallback_enabled=True,
            qc_bbox_prompt_template=_bbox_prompt(tmp_path),
            qc_bbox_max_tokens=123,
        ),
        client=client,
    )

    assert result.target.status is MaskQCStatus.PASSED
    assert result.target.selected_candidate == "BBOX"
    assert result.target.selected_query_field == "bbox_fallback"
    assert result.target.selected_seed_frame_id == 0
    assert backend.frame_calls[:4] == [
        ("text", 0),
        ("text", 3),
        ("text", 5),
        ("bbox", 0),
    ]
    assert backend.box_calls == [(0, requested_box)]
    assert [attempt.method for attempt in result.target.attempts] == [
        MaskQCAttemptMethod.TEXT_QUERY,
        MaskQCAttemptMethod.TEXT_QUERY,
        MaskQCAttemptMethod.TEXT_QUERY,
        MaskQCAttemptMethod.BBOX_FALLBACK,
    ]
    bbox_attempt = result.target.attempts[-1]
    assert bbox_attempt.provenance["candidate_generation"] == "qwen_bbox_to_sam_box"
    assert bbox_attempt.provenance["localization"]["bbox_xyxy"] == list(requested_box)
    assert bbox_attempt.provenance["sam_prompt"]["coordinates_clamped"] is False
    assert set(result.attempt_candidate_masks["target"][0]) == {
        "A",
        "B",
        "C",
        "BBOX",
    }
    assert result.selected_masks["target"][3, 4]
    # localization + bbox candidate QC + normal receiver QC
    assert len(client.messages) == 3

    artifact = save_mask_qc_artifacts(
        ArtifactStore(tmp_path / "artifacts"),
        "bbox-fallback-audit",
        context,
        result,
    )
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    serialized_attempts = payload["roles"]["target"]["attempts"]
    assert [attempt["method"] for attempt in serialized_attempts] == [
        "text_query",
        "text_query",
        "text_query",
        "bbox_fallback",
    ]
    assert serialized_attempts[-1]["provenance"]["sam_prompt"]["bbox_xyxy"] == list(requested_box)
    frame_zero = payload["artifacts"]["attempts"]["target"]["frame_000000"]
    assert set(frame_zero["candidate_masks"]) == {"A", "B", "C", "BBOX"}
    assert (artifact.parent / frame_zero["candidate_masks"]["BBOX"]).is_file()


def test_bbox_candidate_is_not_accepted_when_normal_visual_qc_rejects_it(
    tmp_path: Path,
) -> None:
    images = _images()
    backend = BboxFallbackBackend()

    result = run_mask_qc_stage(
        _context(),
        _plan(),
        backend,
        Path("/tmp/resource"),
        seed_images={0: images[0]},
        context_images=images,
        frame_shape=FRAME_SHAPE,
        mask_config=_config(
            _prompt(tmp_path),
            qc_bbox_fallback_enabled=True,
            qc_bbox_prompt_template=_bbox_prompt(tmp_path),
            qc_bbox_max_tokens=123,
        ),
        client=FakeQCClient(
            [
                _bbox_response(),
                _decision_response("reject_all"),
                _response("A"),
            ]
        ),
    )

    assert result.target.status is MaskQCStatus.REJECTED
    assert result.target.selected_candidate is None
    assert "target" not in result.selected_masks
    assert backend.box_calls == [(0, (0.2, 0.25, 0.6, 0.75))]
    assert result.target.attempts[-1].method is MaskQCAttemptMethod.BBOX_FALLBACK
    assert result.target.attempts[-1].status is MaskQCStatus.REJECTED


def test_bbox_fallback_never_runs_after_a_text_candidate_passes(tmp_path: Path) -> None:
    images = _images()
    backend = BboxFallbackBackend(empty_target_frames=())

    result = run_mask_qc_stage(
        _context(),
        _plan(),
        backend,
        Path("/tmp/resource"),
        seed_images={0: images[0]},
        context_images=images,
        frame_shape=FRAME_SHAPE,
        mask_config=_config(
            _prompt(tmp_path),
            qc_bbox_fallback_enabled=True,
            qc_bbox_prompt_template=_bbox_prompt(tmp_path),
            qc_bbox_max_tokens=123,
        ),
        client=FakeQCClient([_response("B"), _response("A")]),
    )

    assert result.target.status is MaskQCStatus.PASSED
    assert result.receiver.status is MaskQCStatus.PASSED
    assert backend.box_calls == []
    assert all(
        attempt.method is MaskQCAttemptMethod.TEXT_QUERY
        for report in result.role_reports
        for attempt in report.attempts
    )


def test_mask_qc_query_fallback_is_opt_in_and_can_select_a_curated_alias(
    tmp_path: Path,
) -> None:
    images = _images()
    baseline_backend = QueryFallbackBackend()

    baseline = run_mask_qc_stage(
        _context(),
        _plan(),
        baseline_backend,
        Path("/tmp/resource"),
        seed_images={0: images[0]},
        context_images=images,
        frame_shape=FRAME_SHAPE,
        mask_config=_config(_prompt(tmp_path)),
        client=FakeQCClient([_response("A")]),
    )

    assert baseline.target.status is MaskQCStatus.PASSED
    assert baseline.receiver.status is MaskQCStatus.REJECTED
    assert baseline_backend.calls == [
        ("bottle", "brown bottle", "cylindrical bottle"),
        ("blue square pad", "pad", "square pad"),
    ]

    fallback_backend = QueryFallbackBackend()
    fallback = run_mask_qc_stage(
        _context(),
        _plan(),
        fallback_backend,
        Path("/tmp/resource"),
        seed_images={0: images[0]},
        context_images=images,
        frame_shape=FRAME_SHAPE,
        mask_config=_config(
            _prompt(tmp_path),
            qc_max_candidates=8,
            qc_query_fallback_enabled=True,
        ),
        client=FakeQCClient([_response("A"), _response("F")]),
    )

    assert fallback.receiver.status is MaskQCStatus.PASSED
    assert fallback.receiver.selected_query == "blue table mat"
    assert fallback.receiver.selected_query_field is not None
    assert fallback.receiver.selected_query_field.startswith("curated_alias_")
    assert fallback_backend.calls == [
        ("bottle", "brown bottle", "cylindrical bottle", "container"),
        (
            "blue square pad",
            "pad",
            "square pad",
            "mat",
            "blue rectangle",
            "blue table mat",
        ),
    ]
    assert {candidate.query_field for candidate in fallback.receiver.candidates} >= {
        "general_fallback_query",
        "curated_alias_1",
        "curated_alias_2",
    }


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
    assert "只指当前 seed 图像内可见的主体" in receiver_text
    assert "不得要求覆盖画外不可见部分" in receiver_text
    assert "仅因轮廓接触图像边缘而拒绝" in receiver_text
    assert "phone 是 target" in receiver_text
    assert "stand/holder 是 receiver" in receiver_text
    assert "不得把它误判" in receiver_text
    assert "为 smartphone" in receiver_text
    assert "purpose=place_context" in receiver_text
    assert "open_done: 17" in receiver_text


def test_open_set_qc_prompt_requires_visible_and_grasped_target_parts() -> None:
    template = (PROJECT_ROOT / "configs/prompts/mask_candidate_qc_open_set.txt").read_text(
        encoding="utf-8"
    )

    assert "任务描述明确点名且画面中可见的部件" in template
    assert "刚性连接、同步运动的可见结构" in template
    assert "被夹爪直接接触、夹住或带走的可见结构" in template
    assert "透明或低对比主体" in template
    assert "不能抵消轮廓外" in template
    assert "完全看不见、只能靠类别常识猜出的结构" in template
