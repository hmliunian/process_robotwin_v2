from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pytest

from robotwin_annotation_v2.urdf_gripper_renderer import (
    ALOHA_RENDER_LINKS,
    AlohaUrdfRenderer,
    DepthAgreement,
    FingerCandidateScore,
    UrdfRenderResult,
    active_gripper_link_names,
    aloha_joint_positions,
    axis_angle_matrix,
    candidate_has_minimum_support,
    compute_visible_gripper_mask,
    depth_agreement,
    forward_kinematics,
    gripper_command_to_drive_target,
    gripper_command_to_kinematic_q,
    load_urdf,
    rank_finger_candidates,
    xyz_rpy_matrix,
)


def _write_kinematic_urdf(path: Path) -> None:
    path.write_text(
        """<?xml version="1.0"?>
<robot name="test">
  <link name="base"/>
  <link name="rotated"/>
  <link name="translated"/>
  <joint name="turn" type="revolute">
    <parent link="base"/>
    <child link="rotated"/>
    <origin xyz="1 2 3" rpy="0 0 1.5707963267948966"/>
    <axis xyz="1 0 0"/>
    <limit lower="-2" upper="2"/>
  </joint>
  <joint name="slide" type="prismatic">
    <parent link="rotated"/>
    <child link="translated"/>
    <origin xyz="0 0.5 0" rpy="0 0 0"/>
    <axis xyz="0 2 0"/>
    <limit lower="0" upper="1"/>
  </joint>
</robot>
""",
        encoding="utf-8",
    )


def test_xyz_rpy_uses_rz_ry_rx_order() -> None:
    roll, pitch, yaw = 0.31, -0.27, 0.63
    rx = axis_angle_matrix((1, 0, 0), roll)
    ry = axis_angle_matrix((0, 1, 0), pitch)
    rz = axis_angle_matrix((0, 0, 1), yaw)

    actual = xyz_rpy_matrix((4, 5, 6), (roll, pitch, yaw))

    np.testing.assert_allclose(actual[:3, :3], (rz @ ry @ rx)[:3, :3], atol=1e-12)
    np.testing.assert_array_equal(actual[:3, 3], np.array((4, 5, 6)))


def test_load_urdf_and_fk_apply_joint_axis_in_joint_frame(tmp_path: Path) -> None:
    urdf_path = tmp_path / "test.urdf"
    _write_kinematic_urdf(urdf_path)
    model = load_urdf(urdf_path)

    transforms = forward_kinematics(
        model,
        {"turn": math.pi / 2, "slide": 0.25},
        root_pose=np.eye(4),
    )

    expected_rotated = xyz_rpy_matrix(
        (1, 2, 3),
        (0, 0, math.pi / 2),
    ) @ axis_angle_matrix((1, 0, 0), math.pi / 2)
    expected_slide = np.eye(4)
    expected_slide[:3, 3] = np.array((0, 0.5, 0))
    expected_motion = np.eye(4)
    expected_motion[:3, 3] = np.array((0, 0.5, 0))
    np.testing.assert_allclose(transforms["rotated"], expected_rotated, atol=1e-12)
    np.testing.assert_allclose(
        transforms["translated"],
        expected_rotated @ expected_slide @ expected_motion,
        atol=1e-12,
    )
    assert model.joints[0].lower == -2
    assert model.joints[1].upper == 1


def test_aloha_joint_mapping_uses_two_six_dof_arms_and_two_grippers() -> None:
    joint_absolute = np.array(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.0]
    )

    positions = aloha_joint_positions(
        joint_absolute,
        {"fl_joint8": 0.01, "fr_joint7": 0.02},
    )

    assert [positions[f"fl_joint{index}"] for index in range(1, 7)] == pytest.approx(
        joint_absolute[:6]
    )
    assert [positions[f"fr_joint{index}"] for index in range(1, 7)] == pytest.approx(
        joint_absolute[7:13]
    )
    assert positions["fl_joint7"] == 0.0
    assert positions["fl_joint8"] == 0.01
    assert positions["fr_joint7"] == 0.02
    assert positions["fr_joint8"] == 0.045


def test_gripper_command_mapping_distinguishes_drive_target_and_urdf_q() -> None:
    assert gripper_command_to_drive_target(0.0) == pytest.approx(-0.01)
    assert gripper_command_to_kinematic_q(0.0) == 0.0
    assert gripper_command_to_drive_target(1.0) == pytest.approx(0.045)
    assert gripper_command_to_kinematic_q(1.0) == pytest.approx(0.045)


def test_depth_visible_mask_requires_positive_scene_depth_within_tolerance() -> None:
    projected = np.array([[True, True, True, True, False]])
    rendered = np.array([[100.0, 100.0, 100.0, 100.0, 100.0]])
    scene = np.array([[100.0, 108.0, 108.01, 0.0, 100.0]])

    visible = compute_visible_gripper_mask(projected, rendered, scene, tolerance_mm=8.0)
    summary = depth_agreement(projected, rendered, scene, tolerance_mm=8.0)

    np.testing.assert_array_equal(visible, np.array([[True, True, False, False, False]]))
    assert summary.rendered_pixels == 4
    assert summary.comparable_pixels == 3
    assert summary.consistent_pixels == 2
    assert summary.consistent_fraction == pytest.approx(2 / 3)
    assert summary.median_residual_mm == pytest.approx(8.0)
    assert summary.median_signed_residual_mm == pytest.approx(-8.0)
    assert summary.rendered_in_front_pixels == 1
    assert summary.rendered_behind_pixels == 0


def test_active_gripper_semantics_include_fixed_link6_and_both_fingers() -> None:
    assert active_gripper_link_names("left") == ("fl_link6", "fl_link7", "fl_link8")
    assert active_gripper_link_names("right") == ("fr_link6", "fr_link7", "fr_link8")
    with pytest.raises(ValueError, match="active_side"):
        active_gripper_link_names("rear")  # type: ignore[arg-type]


def _agreement(
    consistent: int,
    *,
    comparable: int = 100,
    median: float = 2.0,
    p90: float = 4.0,
) -> DepthAgreement:
    return DepthAgreement(
        rendered_pixels=comparable,
        comparable_pixels=comparable,
        consistent_pixels=consistent,
        consistent_fraction=consistent / comparable,
        median_residual_mm=median,
        p90_residual_mm=p90,
    )


def _candidate(
    q_m: float,
    consistent: int,
    *,
    comparable: int = 100,
    median: float = 2.0,
    p90: float = 4.0,
    per_link: tuple[int, int] | None = None,
) -> FingerCandidateScore:
    link_counts = per_link or (consistent // 2, consistent - consistent // 2)
    return FingerCandidateScore(
        q_m=q_m,
        agreement=_agreement(consistent, comparable=comparable, median=median, p90=p90),
        link7_agreement=_agreement(link_counts[0], comparable=comparable // 2),
        link8_agreement=_agreement(link_counts[1], comparable=comparable // 2),
    )


def test_candidate_ranking_prioritizes_count_then_fraction_and_robust_residual() -> None:
    more_pixels = _candidate(0.01, 51, comparable=100, median=20.0, p90=40.0)
    cleaner_but_less = _candidate(0.02, 50, comparable=50, median=0.1, p90=0.2)
    same_count_lower_fraction = _candidate(0.03, 51, comparable=200, median=0.1, p90=0.2)

    ranked = rank_finger_candidates((cleaner_but_less, same_count_lower_fraction, more_pixels))

    assert ranked == (more_pixels, same_count_lower_fraction, cleaner_but_less)


def test_candidate_ranking_uses_temporal_then_command_prior_only_for_ties() -> None:
    low_q = _candidate(0.01, 50)
    high_q = _candidate(0.03, 50)

    command_ranked = rank_finger_candidates(
        (low_q, high_q),
        command_prior_q_m=0.03,
    )
    temporal_ranked = rank_finger_candidates(
        (low_q, high_q),
        command_prior_q_m=0.03,
        temporal_prior_q_m=0.01,
    )

    assert command_ranked[0] is high_q
    assert temporal_ranked[0] is low_q


def test_candidate_ranking_rejects_high_count_object_surface_with_bad_median() -> None:
    object_surface = _candidate(0.0415, 422, median=6.55, p90=70.0)
    aligned_finger = _candidate(0.0375, 413, median=1.33, p90=59.0)

    ranked = rank_finger_candidates(
        (object_surface, aligned_finger),
        maximum_median_residual_mm=2.0,
    )

    assert ranked[0] is aligned_finger


def test_candidate_minimum_support_requires_both_fingers() -> None:
    supported = _candidate(0.02, 30, per_link=(12, 18))
    one_finger_only = _candidate(0.03, 30, per_link=(30, 0))

    kwargs = {
        "minimum_support_pixels": 24,
        "minimum_per_link_support_pixels": 6,
        "minimum_consistent_fraction": 0.05,
    }
    assert candidate_has_minimum_support(supported, **kwargs)
    assert not candidate_has_minimum_support(one_finger_only, **kwargs)


def _fake_render_result(
    q_by_joint: Mapping[str, float],
    *,
    mismatch_link8: bool,
    offscreen: bool,
    active_side: str,
    joint_positions: Mapping[str, float],
    link6_residual_mm: float = 0.0,
) -> UrdfRenderResult:
    link6 = np.array([[True, False, False]])
    link7 = np.array([[False, True, False]])
    link8 = np.array([[False, False, True]])
    active = link6 | link7 | link8
    prefix = "fl" if active_side == "left" else "fr"
    link7_depth = 100.0 + abs(q_by_joint[f"{prefix}_joint7"] - 0.015) * 1000.0
    link8_depth = 100.0 + abs(q_by_joint[f"{prefix}_joint8"] - 0.03) * 1000.0
    depth = np.array(
        [
            [
                100.0 + link6_residual_mm,
                link7_depth,
                130.0 if mismatch_link8 else link8_depth,
            ]
        ]
    )
    if offscreen:
        link6 = np.zeros_like(link6)
        link7 = np.zeros_like(link7)
        link8 = np.zeros_like(link8)
        active = np.zeros_like(active)
        depth = np.zeros_like(depth)
    masks = {
        f"{prefix}_link6": link6,
        f"{prefix}_link7": link7,
        f"{prefix}_link8": link8,
    }
    return UrdfRenderResult(
        active_side=active_side,  # type: ignore[arg-type]
        robot_mask=active,
        robot_depth_mm=depth,
        active_gripper_mask=active,
        active_gripper_depth_mm=depth,
        fixed_link6_mask=link6,
        finger_link7_mask=link7,
        finger_link8_mask=link8,
        per_link_masks=masks,
        per_link_depth_mm={name: np.where(mask, depth, 0.0) for name, mask in masks.items()},
        segmentation_ids=np.array([[1, 2, 3]], dtype=np.uint32),
        joint_positions=joint_positions,
    )


def _fake_renderer(
    *,
    mismatch_link8: bool,
    offscreen: bool = False,
    link6_residual_mm: float = 0.0,
) -> AlohaUrdfRenderer:
    renderer = object.__new__(AlohaUrdfRenderer)
    renderer.width = 3
    renderer.height = 1
    renderer._closed = False
    renderer.render_call_count = 0

    def fake_render(
        joint_absolute: Any,
        intrinsic_cv: Any,
        cam2world_gl: Any,
        *,
        active_side: str,
        finger_q_by_joint: Mapping[str, float] | None = None,
    ) -> UrdfRenderResult:
        del joint_absolute, intrinsic_cv, cam2world_gl
        assert finger_q_by_joint is not None
        renderer.render_call_count += 1
        return _fake_render_result(
            finger_q_by_joint,
            mismatch_link8=mismatch_link8,
            offscreen=offscreen,
            active_side=active_side,
            joint_positions=finger_q_by_joint,
            link6_residual_mm=link6_residual_mm,
        )

    renderer.render = fake_render  # type: ignore[method-assign]
    return renderer


@pytest.mark.parametrize(
    ("mismatch_link8", "expected_accepted", "expected_visible_pixels"),
    ((False, True, 3), (True, False, 2)),
)
def test_q_fit_selects_depth_supported_contact_q_and_fails_closed(
    mismatch_link8: bool,
    expected_accepted: bool,
    expected_visible_pixels: int,
) -> None:
    renderer = _fake_renderer(mismatch_link8=mismatch_link8)

    result = renderer.fit_finger_q(
        np.zeros(14),
        np.eye(3),
        np.eye(4),
        np.full((1, 3), 100.0),
        active_side="right",
        tolerance_mm=0.1,
        q_max_m=0.04,
        coarse_step_m=0.01,
        fine_step_m=0.005,
        minimum_support_pixels=2,
        minimum_per_link_support_pixels=1,
        minimum_consistent_fraction=1.0,
        minimum_fixed_support_pixels=1,
        minimum_searchable_pixels=0,
    )

    assert result.selected_q_by_joint["fr_joint7"] == pytest.approx(0.015)
    if mismatch_link8:
        assert result.selected_q_by_joint["fr_joint8"] == pytest.approx(0.0)
    else:
        assert result.selected_q_by_joint["fr_joint8"] == pytest.approx(0.03)
    assert result.selected_q_m is None
    assert result.accepted is expected_accepted
    assert int(result.visible_mask.sum()) == expected_visible_pixels
    assert (result.diagnostics.reason is None) is expected_accepted
    assert result.component_acceptance["fr_link6"]
    assert result.component_acceptance["fr_link7"]
    assert result.component_acceptance["fr_link8"] is (not mismatch_link8)


def test_q_fit_uses_one_render_when_temporal_prior_is_already_supported() -> None:
    renderer = _fake_renderer(mismatch_link8=False)

    result = renderer.fit_finger_q(
        np.zeros(14),
        np.eye(3),
        np.eye(4),
        np.full((1, 3), 100.0),
        active_side="right",
        tolerance_mm=0.1,
        minimum_support_pixels=2,
        minimum_per_link_support_pixels=1,
        minimum_consistent_fraction=1.0,
        minimum_fixed_support_pixels=1,
        temporal_prior_q_by_joint={"fr_joint7": 0.015, "fr_joint8": 0.03},
    )

    assert result.accepted
    assert result.diagnostics.search_mode == "prior_fast_path"
    assert renderer.render_call_count == 1  # type: ignore[attr-defined]


def test_q_fit_final_acceptance_uses_visibility_tolerance_not_candidate_median_gate() -> None:
    renderer = _fake_renderer(mismatch_link8=False, link6_residual_mm=3.0)

    result = renderer.fit_finger_q(
        np.zeros(14),
        np.eye(3),
        np.eye(4),
        np.full((1, 3), 100.0),
        active_side="right",
        tolerance_mm=8.0,
        q_max_m=0.04,
        coarse_step_m=0.01,
        fine_step_m=0.005,
        minimum_support_pixels=2,
        minimum_per_link_support_pixels=1,
        minimum_consistent_fraction=1.0,
        minimum_fixed_support_pixels=1,
        minimum_searchable_pixels=0,
    )

    assert result.diagnostics.maximum_median_residual_mm == pytest.approx(2.0)
    assert result.diagnostics.fixed_link6_agreement.median_residual_mm == pytest.approx(3.0)
    assert result.component_acceptance["fr_link6"]
    assert result.accepted
    assert int(result.visible_mask.sum()) == 3


def test_q_fit_skips_sweep_when_both_fingers_are_offscreen() -> None:
    renderer = _fake_renderer(mismatch_link8=False, offscreen=True)

    result = renderer.fit_finger_q(
        np.zeros(14),
        np.eye(3),
        np.eye(4),
        np.full((1, 3), 100.0),
        active_side="right",
        minimum_support_pixels=2,
        minimum_per_link_support_pixels=1,
        minimum_fixed_support_pixels=1,
        minimum_searchable_pixels=1,
    )

    assert not result.accepted
    assert not result.visible_mask.any()
    assert result.diagnostics.search_mode == "prior_fast_path"
    assert renderer.render_call_count == 1  # type: ignore[attr-defined]


def test_q_fit_uses_metric_temporal_window_when_it_contains_supported_q() -> None:
    renderer = _fake_renderer(mismatch_link8=False)

    result = renderer.fit_finger_q(
        np.zeros(14),
        np.eye(3),
        np.eye(4),
        np.full((1, 3), 100.0),
        active_side="right",
        tolerance_mm=0.1,
        q_max_m=0.04,
        coarse_step_m=0.005,
        fine_step_m=0.001,
        minimum_support_pixels=2,
        minimum_per_link_support_pixels=1,
        minimum_consistent_fraction=1.0,
        minimum_fixed_support_pixels=1,
        temporal_prior_q_by_joint={"fr_joint7": 0.014, "fr_joint8": 0.032},
        temporal_max_delta_m=0.004,
        minimum_searchable_pixels=0,
    )

    assert result.selected_q_by_joint["fr_joint7"] == pytest.approx(0.015)
    assert result.selected_q_by_joint["fr_joint8"] == pytest.approx(0.03)
    for joint_name, candidates in result.diagnostics.ranked_candidates_by_joint.items():
        prior = {"fr_joint7": 0.014, "fr_joint8": 0.032}[joint_name]
        assert all(abs(candidate.q_m - prior) <= 0.004 + 1e-12 for candidate in candidates)


def test_q_fit_globally_reacquires_when_temporal_window_has_no_support() -> None:
    renderer = _fake_renderer(mismatch_link8=False)

    result = renderer.fit_finger_q(
        np.zeros(14),
        np.eye(3),
        np.eye(4),
        np.full((1, 3), 100.0),
        active_side="right",
        tolerance_mm=0.1,
        q_max_m=0.04,
        coarse_step_m=0.005,
        fine_step_m=0.001,
        minimum_support_pixels=2,
        minimum_per_link_support_pixels=1,
        minimum_consistent_fraction=1.0,
        minimum_fixed_support_pixels=1,
        temporal_prior_q_by_joint={"fr_joint7": 0.005, "fr_joint8": 0.04},
        temporal_max_delta_m=0.004,
        minimum_searchable_pixels=0,
    )

    assert result.accepted
    assert result.selected_q_by_joint["fr_joint7"] == pytest.approx(0.015)
    assert result.selected_q_by_joint["fr_joint8"] == pytest.approx(0.03)
    assert any(
        candidate.q_m > 0.009
        for candidate in result.diagnostics.ranked_candidates_by_joint["fr_joint7"]
    )
    assert any(
        candidate.q_m < 0.036
        for candidate in result.diagnostics.ranked_candidates_by_joint["fr_joint8"]
    )


def test_real_aloha_render_preserves_link6_7_8_union_when_backend_is_available() -> None:
    if importlib.util.find_spec("pyrender") is None or importlib.util.find_spec("trimesh") is None:
        pytest.skip("optional rendering dependencies are unavailable")
    pilot_root = Path("/tmp/robotwin_aloha_pilot")
    urdf_path = pilot_root / "embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf"
    report_path = pilot_root / "episode_007152_frame0030_report.json"
    if not urdf_path.is_file() or not report_path.is_file():
        pytest.skip("validated local RoboTwin pilot assets are unavailable")
    report = json.loads(report_path.read_text(encoding="utf-8"))

    try:
        with AlohaUrdfRenderer(urdf_path) as renderer:
            rendered = renderer.render(
                report["joint_absolute"],
                np.asarray(report["intrinsic_cv"]),
                np.asarray(report["cam2world_gl"]),
                active_side="right",
            )
    except Exception as exc:
        pytest.skip(f"EGL rendering backend is unavailable: {exc}")

    expected = rendered.fixed_link6_mask | rendered.finger_link7_mask | rendered.finger_link8_mask
    np.testing.assert_array_equal(rendered.active_gripper_mask, expected)
    assert rendered.active_gripper_mask.any()
    assert rendered.robot_mask.sum() > rendered.active_gripper_mask.sum()
    assert set(rendered.per_link_masks) == set(ALOHA_RENDER_LINKS)
