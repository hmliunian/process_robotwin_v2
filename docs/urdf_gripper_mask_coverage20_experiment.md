# URDF Gripper Mask Coverage20 Experiment

## 1. Goal

Replace only the gripper SAM3 producer with deterministic RoboTwin Aloha
geometry on the 20-episode `move_pillbottle_pad` coverage set. Target and
receiver masks remain unchanged.

The experiment must produce a **visible-only active-gripper mask**. A plain
2-D projection of the URDF silhouette is not sufficient because it includes
gripper surfaces hidden by the target, receiver, table, or other scene
geometry.

## 2. Dataset

```text
/DATA/disk8/xuran/add_mask_robotwin/dataset/
  move_pillbottle_pad_coverage20_original
```

The experiment uses, for every usable Parquet frame:

- `observation.state.joint_absolute`: left arm 6 joints + normalized gripper,
  followed by right arm 6 joints + normalized gripper;
- `sidecars/episode_<id>.hdf5`: per-frame `cam_high` OpenCV intrinsics and
  world-to-camera extrinsics;
- `sidecars/videos/.../observation.depths.cam_high/episode_<id>.mkv`: aligned
  16-bit millimetre scene depth;
- `videos/.../observation.images.cam_high/episode_<id>.mp4`: source RGB;
- Stage-1 loop events: active arm and inclusive active-gripper time window.

Parquet length is authoritative. RGB and depth videos contain one trailing
raw frame in this extract; that frame must not be published.

All 20 coverage episodes are marked `geometry_valid=true` and have `cam_high`
depth.

## 3. Required RoboTwin geometry

Use the exact `aloha-agilex` asset family used to create RoboTwin 2.0 data:

- the original URDF;
- visual meshes, not collision approximations;
- original articulation root pose;
- original six-joint order for each arm;
- original gripper scale and mimic-joint mapping;
- an explicit list of fixed and actuated gripper links.

The asset identity and hashes must be recorded in the output manifest. A
different Aloha model is not an acceptable silent fallback.

The implementation reuses the already-local RoboTwin 2 asset archive instead
of downloading assets again:

```text
archive: /tmp/robotwin2_embodiments_c15cc97.zip
URDF:   /tmp/robotwin_aloha_pilot/embodiments/aloha-agilex/urdf/
          arx5_description_isaac.urdf
```

The run contract hashes the URDF and all 20 referenced visual meshes. It also
hashes, with byte sizes, every episode's Parquet, HDF5 sidecar, RGB video,
depth video, and source four-channel mask.

## 4. Mask algorithm

For frame `t`:

1. Set both six-DOF arms from
   `observation.state.joint_absolute[t]` and convert the normalized gripper
   drive command to an initial prismatic-joint hypothesis.
2. Fit active-side `joint7` and `joint8` independently against recorded depth.
   A previous accepted q limits the first search to `previous q +/- 10 mm`;
   if that bounded search has no supported solution, retry once over the full
   physical range. A high-confidence previous pose uses a one-render fast
   path.
3. Render `cam_high` at 320x240 using the recorded calibration, all links of
   both arms, integer visual IDs, and robot camera-Z depth.
4. Define the active gripper as exactly active-side `link6 | link7 | link8`,
   then compare its rendered depth with recorded scene depth:

   ```text
   depth_consistent = abs(robot_depth_mm - scene_depth_mm) <= tolerance_mm
   visible_gripper = accepted_component AND depth_consistent
   ```

5. Emit the mask only inside the Stage-1 inclusive active window. The inactive
   gripper channel remains empty.

The dataset gripper value is a **drive target**, not the realized contact qpos.
For example, episode 7152 frame 69 renders poorly at the closed command but
fits both fingers at approximately `0.0375 m`. This is why per-frame depth
fitting is required rather than simply forwarding the recorded command.

The renderer's Z-buffer handles gripper self-occlusion. Recorded scene depth
handles occlusion by the bottle, pad, table, other robot links, and clutter.
No RGB segmentation model, SAM seed, Qwen gripper QC, temporal propagation,
target/receiver subtraction, or colour threshold is used.

The publication tolerance is fixed at `8 mm` for every episode. A stricter
default `2 mm` median-residual gate is used only to rank finger-q hypotheses
and accept the fast path. It must not remove a final component whose pixels
satisfy the `8 mm` publication support and fraction gates.

## 5. Outputs

Use a versioned experiment directory outside the source dataset:

```text
artifacts/urdf_gripper_mask_coverage20/<run_id>/
  manifest.json
  episode_<id>/
    gripper_masks.npz
    masks.npz
    diagnostics.json
    overlay.mp4
  review/
    contact_sheets/
```

`gripper_masks.npz` contains at least:

- `gripper_track`: `[T,H,W] bool`, active arm only;
- `rendered_amodal_track`: `[T,H,W] bool`;
- `depth_evaluable_track`: valid positive rendered and scene depth inside the
  amodal projection;
- `depth_consistent_track`: the raw, pre-component-gate `8 mm` consistency
  mask, with `gripper_track <= depth_consistent_track <=
  depth_evaluable_track <= rendered_amodal_track`;
- `active_arm`, active-window bounds, frame count, and format version.

`masks.npz` preserves target and receiver byte-for-byte, clears both old SAM
gripper channels, and fills only the active left/right gripper channel with the
URDF result.

The manifest records dataset paths, episode IDs, Git revision plus
implementation-file hashes, input and asset identities, camera/depth
conventions, thresholds, per-episode output hashes, and failure reasons.
Resume validates the immutable run contract and already-published artifact
hashes before it may skip an episode.

## 6. Validation gates

Before the 20-episode batch, inspect at least one left-arm and one right-arm
episode at approach, close, transport, release, and post-release frames.

The pilot passes only if:

- the URDF gripper outline follows the RGB gripper through motion and opening;
- the left/right identity and active window are correct;
- grasped-object pixels inside the amodal projection are removed by depth;
- finger pixels are not systematically erased at contact;
- no forearm or inactive-gripper links enter the published mask;
- no unexplained frame offset is visible;
- masks are nonempty on expected active frames and empty outside the window.

For the automatic gate, a frame is eligible when the selected amodal gripper
has at least one pixel with valid rendered and scene depth. At least 90% of
eligible active frames must publish a nonempty visible mask. Fully offscreen
frames are therefore allowed without weakening the gate.

The 20-episode batch passes only if all episodes render, every artifact has the
Parquet frame count, and the generated videos and contact sheets receive a
visual review. Failures remain fail-closed and are listed in the manifest.

## 7. Known risks

1. The local RoboTwin checkout has a broken `assets` symlink. The experiment
   therefore uses the already-local official archive listed above and records
   exact hashes; it must not silently fall back to another Aloha model.
2. Gripper normalized opening is only a drive target. Treating it as realized
   qpos shifts the fingers during contact; depth fitting is mandatory.
3. A collision-only URDF render is too coarse for pixel masks.
4. Depth is quantized to millimetres. Too-small tolerance removes visible
   edges; too-large tolerance preserves surfaces hidden at close contact.
5. Rendering conventions must be checked explicitly: OpenCV camera-Z versus
   OpenGL renderer depth, image origin, principal point, and half-pixel rules.
6. RGB/depth contain one raw trailing frame; using video length instead of
   Parquet length introduces a temporal contract violation.

## 8. Branch and status

```text
branch: experiment/urdf-gripper-mask-coverage20
base: bf6a8604b22241404b8b1446501998d1a19c27db
```

Implementation status before the full-20 run:

- dataset/source artifacts remain read-only and nothing was downloaded;
- 20/20 episode dry-run passed, including five input identities per episode;
- unit tests: `149 passed, 1 skipped`;
- real EGL renderer tests: `18 passed`;
- Ruff, PyCompile, and the independent runner review passed;
- right-arm episode 7152 and left-arm episode 7157 passed approach, contact,
  transport, release, post-window, channel-preservation, link-membership, and
  saved-vs-rerender checks in the accepted renderer pilot
  `qa2-7152-7157-20260810T1854`.

The runner contract was strengthened after that pilot, so a final pilot with
the frozen implementation is required before starting the independent full-20
run. Final run paths and aggregate metrics are appended after completion.
