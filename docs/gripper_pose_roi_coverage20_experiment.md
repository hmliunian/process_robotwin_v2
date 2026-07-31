# Gripper Pose ROI Coverage20 Extra Experiment

## 1. Status and scope

This is an extra feasibility experiment, not a change to the production V2
three-stage target/receiver pipeline.

```text
repository: /DATA/disk8/xuran/add_mask_robotwin/process_data_v2
branch: experiment/gripper-pose-roi-coverage20
base commit: 51807ee
dataset: /DATA/disk8/xuran/add_mask_robotwin/dataset/move_pillbottle_pad_coverage20_original
camera: cam_high
date: 2026-07-30 to 2026-07-31
```

The experiment asks whether robot state can provide a stable per-frame spatial
constraint for active-gripper masks, and whether target/receiver contamination
inside that constraint can be removed with already-known object masks.

It does not use simulator instance masks, scene object poses, or asset IDs.

## 2. State projection

RoboTwin stores absolute EEF `xyz + roll/pitch/yaw` and gripper opening for both
arms. The TCP is 0.12 m along the EEF local `+x` axis:

```text
R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
tcp_world = eef_xyz + R @ [0.12, 0, 0]
tcp_pixel = project(K, world_to_cam, tcp_world)
```

The implementation projects an oriented 3-D box centered around the TCP. Its
default dimensions are:

```text
axial_back = 0.025 m
axial_front = 0.060 m
closed_half_width = 0.045 m
open_half_width = 0.085 m
half_thickness = 0.050 m
pixel_margin = 3 px
```

The gripper opening linearly controls the lateral half-width. A historical
ep7150/frame51 check exactly reproduces the previously verified projection:

```text
EEF pixel = (121.02764577, 15.26118804)
TCP pixel = (139.94583300, 56.22266571)
```

## 3. Coverage20 geometry audit

All 20 regression episodes were audited: 10 left arm, 10 right arm, 10 clean,
and 10 randomized.

| Metric | Result |
|---|---:|
| Episodes completed | 20/20 |
| Episodes whose projected ROI intersects the image | 20/20 |
| First TCP-in-frame range | frame 14..36 |
| Episode-level median ROI area range | 4190..6444 px |

Four compact review sheets show that the ROI follows the correct active
gripper and excludes long wrist/forearm segments. During grasp, transport, and
release it also contains part or all of the target bottle. Therefore the pose
projection is a spatial upper bound, not a final pixel mask.

## 4. Same-frame SAM3 candidate tests

Two representative episodes were tested at seven keyframes each:

- ep7152: clean, right arm;
- ep7317: randomized, left arm.

### 4.1 Pose box only

SAM3 received only the projected 2-D bounding box. All 14 masks were nonempty,
and the projected polygon removed 6692 pixels outside the 3-D ROI. Cropped mask
areas were 355..4132 px.

The visual result was not sufficient by itself. SAM3 frequently selected the
bottle rather than the gripper, especially during transport and release.

### 4.2 `black robot gripper` plus pose box

Joint text and visual-box prompting also returned 14/14 nonempty masks. Cropped
areas were 755..2537 px, and the pose polygon removed 5218 outside-ROI pixels.
Seven of 14 keyframes had at least 80% dark pixels.

This improved gripper recall but did not make the candidate object-pure. In
ep7317 frame120/121, the unmodified candidate still contained large target
regions and had dark fractions of only 15.1% and 19.0%.

## 5. Exact known-object exclusion

The tested operation is deliberately simple:

```text
gripper_residual = candidate AND NOT target_track AND NOT receiver_track
```

No dilation, erosion, temporal fill, interpolation, or amodal completion is
used. Target has priority only for attribution when target and receiver masks
overlap; it does not change the final Boolean result.

### 5.1 Failed inputs: object identity was not actually known

The first exclusion run reused target/receiver seeds from
`coverage20-e2e-v1`. It removed only 365 of 22757 candidate pixels across 14
keyframes. Review found that ep7317's generic `bottle` target seed had selected
the red rectangular distractor in the upper-right of the image.

This was not caught by the old run's `status=ok`, which only established a
nonempty technical result. Qwen's semantic plan had correctly described the
target as the orange bottle with a blue stripe, but configuration selected the
first generic query.

A second attempt used Qwen's unused `orange bottle` query. SAM3 selected a
different thin orange bottle in the middle of the scene. It again removed zero
target pixels from all seven ep7317 candidates.

These are wrong-instance failures, not evidence against Boolean subtraction.

### 5.2 Controlled correct-target run

To isolate the subtraction hypothesis, ep7317 frame0 used the known tight
target box `[30,75,100,170]` in the 320x240 image. SAM3 box segmentation created
a 2571-pixel target seed, which was natively propagated through frame134. This
manual/known box is recorded as a controlled input and is not claimed as an
automatic target-localization solution.

Using the text+pose-box gripper candidate produced:

| Frame | Phase | Candidate | Target removed | Receiver removed | Residual | Residual dark |
|---:|---|---:|---:|---:|---:|---:|
| 20 | approach | 1863 | 0 | 0 | 1863 | 87.7% |
| 50 | approach | 1381 | 88 | 0 | 1293 | 96.4% |
| 63 | close | 1887 | 888 | 0 | 999 | 97.1% |
| 92 | transport | 1584 | 432 | 0 | 1152 | 97.9% |
| 120 | transport | 2537 | 2122 | 2 | 413 | 90.3% |
| 121 | release | 2526 | 2018 | 0 | 508 | 92.3% |
| 134 | release | 2066 | 1 | 0 | 2065 | 97.1% |

The frame120/121 result directly supports the proposed idea: 80% or more of a
mixed candidate was identified as the target, while the remaining mask was a
compact dark gripper region. Frame134 needed almost no subtraction because the
joint prompt already selected the gripper rather than the bottle.

Receiver exclusion was nearly inactive in these seven frames (0..2 pixels), so
this run does not independently validate difficult gripper/receiver overlap.

## 6. Why box-only is still insufficient

The same correct target/receiver tracks were applied offline to the earlier
pose-box-only candidates:

| Frame | Candidate | Target removed | Residual | Residual dark |
|---:|---:|---:|---:|---:|
| 63 | 1665 | 1473 | 192 | 89.6% |
| 92 | 3134 | 2063 | 1071 | 98.2% |
| 120 | 2202 | 2123 | 78 | 16.7% |
| 121 | 2274 | 2124 | 150 | 54.0% |
| 134 | 4132 | 2075 | 2057 | 97.5% |

Object subtraction can remove contamination, but it cannot recover gripper
pixels that the candidate never contained. Frame120/121 are the counterexample:
box-only SAM3 largely selected the target, so subtraction left a weak or wrong
residual. The joint gripper text prompt was necessary on these frames.

## 7. Current conclusion

The state-projected gripper ROI idea is useful and the user's object-subtraction
observation is correct under explicit conditions:

```text
pose ROI
  + a candidate with actual gripper recall
  + identity-correct visible target/receiver masks
  + exact object exclusion
= promising compact gripper mask
```

The current evidence supports continuing with `gripper text + pose box + known
object exclusion`. It rejects `pose box-only SAM3` as a complete producer.

This is not yet a production pass. The controlled positive result covers seven
keyframes in one randomized left-arm episode. The geometry-only audit covers all
20 episodes, but full per-frame segmentation and exclusion do not.

## 8. Failures and limitations

1. The pose ROI is deliberately wider than the visible fingers and includes the
   bottle during contact. It cannot be exported as the final mask.
2. Generic or color-only target text can select distractors. Nonempty SAM output
   and `status=ok` are not identity validation.
3. Exact subtraction depends on visible, identity-correct object masks. A wrong
   target mask silently leaves contamination in the gripper channel.
4. Target tracking across contact may absorb or lose boundary pixels. There is
   no simulator ground truth here to quantify false removal of dark gripper
   pixels at occlusion boundaries.
5. Box-only SAM3 can omit the gripper entirely. No subtraction rule can repair
   that failure.
6. Receiver overlap was too small in the controlled keyframes to validate the
   hardest release-on-pad cases.
7. No full-trajectory temporal continuity, inactive-channel, or wrist-leak
   acceptance test has been run for this new producer.
8. The target box in the positive ep7317 run was manually supplied to isolate
   the hypothesis; it is not an automatic solution.
9. One attempted rerun failed before inference because GPU7 had only about
   5 MiB free and PyTorch could not allocate another 20 MiB. The empty failed
   output directory was removed, and the same experiment completed on GPU4.
10. `ruff` is not installed in this environment. Python compilation and the
    focused tests passed, but lint was not run.

## 9. Implementation and verification

Branch-local files:

```text
src/robotwin_annotation_v2/experiments/gripper_pose_roi.py
src/robotwin_annotation_v2/experiments/__init__.py
scripts/experiment_gripper_pose_roi_coverage20.py
tests/unit/test_gripper_pose_roi.py
```

Verification:

```text
9 passed
python -m py_compile: passed
git diff --check: passed
```

Generated artifacts are ignored by Git and remain local under:

```text
artifacts/gripper_pose_roi_coverage20/geometry_v1/
artifacts/gripper_pose_roi_coverage20/sam3_seeds_ep7152_ep7317_v1/
artifacts/gripper_pose_roi_coverage20/sam3_text_box_black_gripper_ep7152_ep7317_v1/
artifacts/gripper_pose_roi_coverage20/object_exclusion_ep7152_ep7317_v1/
artifacts/gripper_pose_roi_coverage20/object_exclusion_ep7317_orange_target_v1/
artifacts/gripper_pose_roi_coverage20/object_exclusion_ep7317_known_target_box_v1/
```

Manifest SHA-256 values for the main evidence runs:

```text
geometry_v1: 065ff363d26bda474303361d23ad5926a1130ecc5321187c32866c837ccf3c35
box_only: 1d51a7fb66097149b00f4f8102f557117b5a24276901014fd3b6b653be663053
text_box: b6821c04449b21ddc2fcce62e633e894e8d526834d2801f35d472d31e19f4efe
known_target_exclusion: a46ffe7f52c01f5bad59eace1da28223d4b16ab10ad0a8123d4ace93770d0ba2
```

To avoid oversized responses and the earlier HTTP 413 failure mode, model logs
were redirected to `/tmp`, review JPEGs were kept around 52..100 KB, and only
small summaries were inspected in the conversation.
