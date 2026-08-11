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

## 8. Final experiment status

```text
branch: experiment/urdf-gripper-mask-coverage20
base: bf6a8604b22241404b8b1446501998d1a19c27db
generation commit: 16d8bc87af76ac16167cba80e06dd10e4915d1cc
```

The source dataset and source SAM artifacts remained read-only throughout the
experiment. No dataset, model, or RoboTwin asset was downloaded. Validation of
the frozen implementation included:

- 20/20 episode dry-run passed, including five input identities per episode;
- unit tests: `149 passed, 1 skipped`;
- real EGL renderer tests: `18 passed`;
- focused runner tests: `21 passed`;
- combined runner and real-renderer tests: `39 passed`;
- Ruff, PyCompile, `git diff --check`, and an independent runner review passed.

The final frozen pilot is:

```text
artifacts/urdf_gripper_mask_coverage20/
  final-pilot-7152-7157-16d8bc8/
```

It completed 2/2 episodes with no failures. Right-arm episode 7152 published a
nonempty mask on `116/117` eligible frames (99.15%), while left-arm episode
7157 published on `149/149` (100%). Both passed approach, contact, transport,
release, post-window, channel-preservation, link-membership, and
saved-vs-rerender review. Its masks and overlays are identical to the visually
accepted pre-freeze renderer pilot `qa2-7152-7157-20260810T1854`.

## 9. Coverage20 results

The independent 20-episode run is:

```text
artifacts/urdf_gripper_mask_coverage20/
  coverage20-urdf-gripper-v1-16d8bc8/
    manifest.json
    episode_<id>/overlay.mp4
```

The frozen commit completed all 20 episodes with 20 successes, zero failed
episodes, and zero failure attempts. The first render took approximately
704.3 seconds wall-clock. A subsequent full `--resume` validation took
approximately 13.9 seconds; all 20 episodes were anchored to their published
hashes and recorded as `validated_skip`.

There are 20 overlay videos containing 2,940 frames in total. Every video is
320x240 at 50 FPS, and every video frame count matches its authoritative
Parquet frame count.

Aggregate mask quality from the final manifest is:

| Metric | Result |
| --- | ---: |
| Eligible frames with a nonempty mask | `2561/2570` (99.65%) |
| Lowest per-episode eligible fraction | `125/128` (97.66%), episode 7571 |
| Mean of the 20 per-episode eligible fractions | 99.66% |
| Visible gripper pixels | 6,929,725 |
| `link6` component acceptance | `2544/2759` (92.21%) |
| `link7` component acceptance | `2371/2759` (85.94%) |
| `link8` component acceptance | `2353/2759` (85.28%) |
| Maximum fitted-q jump | 10.5 mm, episode 7274 `fr_joint8` |
| Active-arm distribution | 10 left / 10 right |

The 10.5 mm maximum jump came from the permitted full-range reacquisition
path after the bounded temporal search had no reliable solution. It did not
cause a quality-gate or visual-review failure.

## 10. Visual review and conclusion

The formal-run review material is stored under:

```text
artifacts/urdf_gripper_mask_coverage20/
  coverage20-urdf-gripper-v1-16d8bc8/review/
    review_manifest.json
    contact_sheets/
      target_early.jpg
      target_late.jpg
      receiver_early.jpg
      receiver_late.jpg
      gripper_early.jpg
      gripper_late.jpg
```

All 20 episodes passed review across the six contact sheets. No wrong-arm
selection, long-forearm leakage, bottle-body inclusion, or release-opening
error was found. Occlusion clipping and image-edge truncation were consistent
with the RGB evidence. The sheets sample the last active frame rather than a
post-window frame; the frozen pilot visually checked post-window behavior,
and the full-run artifact validator enforces an empty gripper mask outside the
inclusive active window.

The experiment therefore demonstrates that replacing the **gripper** SAM
producer with RoboTwin URDF geometry is feasible on this coverage20 dataset.
The published gripper channel is SAM-free and deterministic with respect to
the recorded joints, calibration, depth, assets, and thresholds. Target and
receiver channels intentionally remain the byte-preserved masks from the
existing SAM run; this experiment does not claim to remove SAM from those two
object channels.

## 11. `just process` integration and `place_empty_cup_full550`

本节的集成边界和 public artifact contract 已正式整理到
`process_data_v3_1_architecture_design.md`。v3 设计继续作为 live visual pipeline 基线；
URDF 被定义为复用冻结 target/receiver source run 的 derived-run backend。

On 2026-08-10 the experiment was generalized on the uncommitted branch and
worktree below. The frozen coverage20 results in the preceding sections keep
their original historical meaning.

```text
branch:   experiment/urdf-gripper-mask-coverage20
worktree: /DATA/disk8/xuran/add_mask_robotwin/process_data_v2/
            .worktrees/process_data_v2-urdf-gripper-mask-coverage20
```

`just process` now accepts trailing CLI arguments. With no extra arguments it
still runs the original Qwen/SAM pipeline. With `--gripper-backend urdf`, it
does not start Qwen, SAM gripper segmentation, or the legacy gripper renderer.
It reuses only the source run's QC-passed target and receiver masks, clears the
two old gripper channels, and writes the visible URDF mask into the active-arm
channel.

The integration audit used only data already present on disk:

```text
dataset: /DATA/disk8/xuran/add_mask_robotwin/dataset/
           place_empty_cup_full550_original
source:  /DATA/disk8/xuran/add_mask_robotwin/process_data_v2/artifacts/runs/
           20260807T105004Z-47ee3def
```

The dataset contains 550 discovered episodes. The frozen source run has 456
completed episodes whose target and receiver artifacts satisfy the manifest,
annotation, and QC contracts. It excludes 94 episodes: 91 have an incomplete
target or receiver SAM result, and episodes 16027, 16336, and 16345 have no
publishable source mask. The old SAM gripper result is deliberately not part
of source eligibility.

Automatic discovery is fail-closed across both contracts: every URDF episode
must have Parquet, RGB, sidecar, depth video, and frame-aligned source masks,
and its target/receiver review windows must be valid. Because this source has
excluded episodes, the 456-episode subset requires an explicit
`--allow-partial-source`. Explicit `--episode-ids` remain fail-closed and never
silently drop a requested episode. The initial 456-episode dry-run passed in
about 63 seconds. After the generalized contract checks were tightened, the
final 456-episode dry-run also passed in about 115 seconds. Dry-run created no
output run. The formal 456-episode render has not been started; its estimated
runtime is 5--7 hours.

The generalized integration finished with `170 passed, 1 skipped` unit tests,
plus PyCompile, Ruff `E/F/I`, and `git diff --check` passing.

The real left/right pilot is stored at:

```text
artifacts/urdf_gripper_place_empty_cup/
  place-empty-cup-urdf-pilot-15950-15955/
```

Both episodes are currently complete and `process_summary.json` reports
`passed=true`. Episode 15950 uses the right gripper and publishes a nonempty
mask on `111/112` eligible frames (99.11%); episode 15955 uses the left gripper
and publishes on `144/144` (100%). Their overlays contain 179 and 166 frames,
respectively, at 320x240 and 50 FPS. Target and receiver masks are pixelwise
identical to the source artifacts. Six contact sheets were generated. Visual
review found correct alignment and occlusion handling; the initial empty
right-arm frames occur before that gripper enters the image. The pilot
manifest retains early failed attempts caused by missing renderer packages,
but the current episode states and final process summary are successful.

Review output is paginated at 32 episodes per page. Its top-level manifest
records `page_size`, `page_count`, page manifests, and all generated sheets.
For a failed or interrupted immutable run, `--resume` requires the same
explicit `--run-id`. The URDF backend does not support `--force`; use a new run
ID for an intentional clean rerun.

Install the renderer dependencies declared by the `urdf` project extra before
running outside the temporary pilot environment. In a new worktree-local
environment, use:

```bash
uv sync --extra urdf
```

This installs Python dependencies only; it does not download or replace a
dataset. Do not run `uv sync` against an existing environment that must retain
unselected SAM extras, because synchronization can prune them. To add URDF
packages to such an environment without pruning its other packages, run from
this worktree:

```bash
uv pip install --python /absolute/path/to/python -e '.[urdf]'
```

The currently verified RoboTwin asset is already local at the path in the
command below. Because it is under `/tmp`, verify that the complete URDF and
its referenced meshes are still present before starting a long run.

After dependencies are available, the formal subset command is:

```bash
just process \
  /DATA/disk8/xuran/add_mask_robotwin/dataset/place_empty_cup_full550_original \
  /DATA/disk8/xuran/add_mask_robotwin/process_data_v2/artifacts/urdf_gripper_place_empty_cup \
  --gripper-backend urdf \
  --source-run-dir /DATA/disk8/xuran/add_mask_robotwin/process_data_v2/artifacts/runs/20260807T105004Z-47ee3def \
  --urdf-path /tmp/robotwin_aloha_pilot/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf \
  --allow-partial-source \
  --run-id place-empty-cup-urdf-456-v1
```

The optional second positional argument is the output root; the actual run is
written to `<output-root>/<run-id>`. If it is omitted and the next argument
starts with `-`, the recipe uses `artifacts/runs`. Variadic arguments are passed
without shell word-splitting, including paths containing spaces. In the
experiment worktree, either create its local `.venv` with the command above or
override the recipe variable before the recipe name. Merely pointing at
another interpreter is not sufficient unless that environment already
contains the `urdf` extra.

```bash
just --set python /absolute/path/to/urdf-enabled/python process \
  DATASET OUTPUT_ROOT --gripper-backend urdf \
  --source-run-dir SOURCE_RUN --urdf-path URDF --run-id RUN_ID
```
