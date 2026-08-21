from __future__ import annotations

import json

import pytest

from robotwin_annotation_v2.pipeline.bbox_localization import (
    BboxLocalizationError,
    parse_bbox_localization,
)


def _response(**overrides: object) -> str:
    payload: dict[str, object] = {
        "status": "ok",
        "bbox_xyxy": [0.1, 0.2, 0.8, 0.9],
        "confidence": 0.75,
        "reason": "  full   object visible  ",
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_parser_preserves_raw_normalized_coordinates_without_clamping() -> None:
    result = parse_bbox_localization(_response(bbox_xyxy=[0.0, 0.125, 1.0, 0.875]))

    assert result.status == "ok"
    assert result.bbox_xyxy == (0.0, 0.125, 1.0, 0.875)
    assert result.confidence == 0.75
    assert result.reason == "full object visible"


@pytest.mark.parametrize("status", ("ambiguous", "not_visible"))
def test_non_ok_status_requires_and_preserves_null_bbox(status: str) -> None:
    result = parse_bbox_localization(_response(status=status, bbox_xyxy=None))

    assert result.status == status
    assert result.bbox_xyxy is None


def test_parser_rejects_markdown_fence() -> None:
    with pytest.raises(BboxLocalizationError, match="Markdown fence"):
        parse_bbox_localization(f"```json\n{_response()}\n```")


def test_parser_rejects_duplicate_fields() -> None:
    raw = (
        '{"status":"ok","status":"ambiguous",'
        '"bbox_xyxy":[0.1,0.2,0.8,0.9],"confidence":0.7,"reason":"visible"}'
    )

    with pytest.raises(BboxLocalizationError, match="duplicate field 'status'"):
        parse_bbox_localization(raw)


@pytest.mark.parametrize("constant", ("NaN", "Infinity", "-Infinity"))
def test_parser_rejects_non_finite_json_constants(constant: str) -> None:
    raw = _response().replace("0.75", constant)

    with pytest.raises(BboxLocalizationError, match="non-finite JSON number"):
        parse_bbox_localization(raw)


@pytest.mark.parametrize(
    "raw",
    (
        '{"status":"ok","bbox_xyxy":[0.1,0.2,0.8,0.9],"confidence":0.7}',
        (
            '{"status":"ok","bbox_xyxy":[0.1,0.2,0.8,0.9],'
            '"confidence":0.7,"reason":"visible","extra":1}'
        ),
    ),
)
def test_parser_rejects_missing_or_extra_fields(raw: str) -> None:
    with pytest.raises(BboxLocalizationError, match="fields do not match schema"):
        parse_bbox_localization(raw)


@pytest.mark.parametrize("status", ("passed", "error", ""))
def test_parser_rejects_unknown_status(status: str) -> None:
    with pytest.raises(BboxLocalizationError, match="status must be"):
        parse_bbox_localization(_response(status=status))


@pytest.mark.parametrize("status", ("ambiguous", "not_visible"))
def test_parser_rejects_bbox_for_non_ok_status(status: str) -> None:
    with pytest.raises(BboxLocalizationError, match="requires bbox_xyxy=null"):
        parse_bbox_localization(_response(status=status))


@pytest.mark.parametrize(
    "bbox",
    (
        [-0.1, 0.2, 0.8, 0.9],
        [0.1, 0.2, 1.1, 0.9],
        [0.8, 0.2, 0.1, 0.9],
        [0.1, 0.9, 0.8, 0.2],
        [0.1, 0.2, 0.1, 0.9],
        [0.1, 0.2, 0.8, 0.2],
    ),
)
def test_parser_rejects_unordered_or_out_of_range_bbox_without_repair(
    bbox: list[float],
) -> None:
    with pytest.raises(BboxLocalizationError, match="rejected rather than clamped"):
        parse_bbox_localization(_response(bbox_xyxy=bbox))


@pytest.mark.parametrize(
    "bbox",
    (
        None,
        [0.1, 0.2, 0.8],
        [0.1, 0.2, 0.8, True],
        [0.1, 0.2, 0.8, "0.9"],
    ),
)
def test_ok_status_requires_exactly_four_json_numbers(bbox: object) -> None:
    with pytest.raises(BboxLocalizationError, match="bbox_xyxy"):
        parse_bbox_localization(_response(bbox_xyxy=bbox))


@pytest.mark.parametrize("confidence", (-0.01, 1.01, True, "0.7"))
def test_parser_rejects_invalid_confidence(confidence: object) -> None:
    with pytest.raises(BboxLocalizationError, match="confidence"):
        parse_bbox_localization(_response(confidence=confidence))
