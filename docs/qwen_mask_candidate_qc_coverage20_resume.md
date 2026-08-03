# Qwen mask candidate QC coverage20 completion status

Last updated: 2026-08-03 (Asia/Shanghai)

## Repository state

- Repository: `/DATA/disk8/xuran/add_mask_robotwin/process_data_v2`
- Branch: `fix/qwen-mask-candidate-qc`
- Coverage run ID: `coverage20-qc-contact-v5-native`
- Config: `configs/pilot_move_pillbottle_pad.yaml`
- Dataset: `/DATA/disk8/xuran/add_mask_robotwin/dataset/move_pillbottle_pad_coverage20_original`
- Commit boundary: source, config, tests, and documentation only; local runtime artifacts stay
  excluded by `.gitignore`

All implementation and generated artifacts for this task belong to the nested
`process_data_v2` repository on `fix/qwen-mask-candidate-qc`. The outer repository `main`
branch is not the implementation branch.

## Final status

The coverage20 run is complete:

```text
masks.npz:        20/20
mask_qc.json:     20/20
run_manifest.json 20/20
target QC passed: 20/20
receiver QC passed: 20/20
target status ok: 20/20
receiver status ok: 20/20
overlay videos:   20/20
```

Completed episode IDs:

```text
7152 7156 7157 7163 7168 7179 7181 7185 7187 7188
7274 7317 7335 7367 7424 7464 7571 7621 7673 7674
```

The final five incomplete episodes (`7187 7317 7464 7571 7674`) were rerun together with
the resident SAM3 worker. The batch exited successfully and every record in
`sam_batch_summary.json` is `completed` with `fatal_error: null`.

## Implemented behavior

- Qwen generates a short ordered query bank for each target and receiver.
- SAM3 creates real seed-mask candidates; mechanical checks reject empty, abnormal, and
  duplicate candidates before Qwen mask QC.
- Qwen sees contour-only candidate panels plus action context and must accept one candidate
  or fail closed with `reject_all`/`ambiguous`.
- Semantic parsing repairs omitted query-order entries and preserves discriminating
  two-color target descriptions such as `teal white bottle`.
- Receiver rules use final direct contact rather than requiring a support object, and the QC
  prompt prevents rejecting a receiver merely because it is not the target.
- Saturated-blue planar-region candidates recover blue pads that text-only SAM3 misses; Qwen
  still performs the final identity decision.
- Transient Qwen mask-QC requests are attempted at most twice without regenerating candidates.
- `sam-batch` keeps one `Sam3Adapter` resident across sequential episodes, reuses one video
  session for an episode's text queries, skips already complete episodes, and fails fast on
  CUDA-level faults.

SAM3 residence is scoped to one batch worker process. This avoids reloading the checkpoint
for every episode while keeping episode sessions isolated; it is not a separate long-running
network daemon that must be deployed and monitored.

## Final artifacts

Per-episode masks, QC reports, selected seeds, native tracks, and manifests:

```text
artifacts/runs/coverage20-qc-contact-v5-native/
```

Final 20 overlay videos and exact-run render manifest:

```text
artifacts/rendered_videos/coverage20_qc_contact_v5_native/
artifacts/rendered_videos/coverage20_qc_contact_v5_native/manifest.json
```

The render manifest reports `episode_count: 20`; every target/receiver annotation status is
`valid`, and every target/receiver QC status is `passed`.

Full-run visual review sheets:

```text
artifacts/rendered_videos/coverage20_qc_contact_v5_native/review_sheets/target_early.jpg
artifacts/rendered_videos/coverage20_qc_contact_v5_native/review_sheets/target_late.jpg
artifacts/rendered_videos/coverage20_qc_contact_v5_native/review_sheets/receiver_early.jpg
artifacts/rendered_videos/coverage20_qc_contact_v5_native/review_sheets/receiver_late.jpg
```

The four sheets contain all 20 episodes. Visual inspection confirmed that target masks follow
the requested bottle and receiver masks follow the blue pad. Missing mask pixels under the
gripper or placed bottle are expected visible-only occlusion, not identity drift.

## Reproduction commands

Run or resume a coverage batch while keeping SAM3 resident:

```bash
.venv/bin/python scripts/run_target_receiver.py sam-batch \
  --config configs/pilot_move_pillbottle_pad.yaml \
  --run-id coverage20-qc-contact-v5-native \
  --episode-ids 7152 7156 7157 7163 7168 7179 7181 7185 7187 7188 \
                7274 7317 7335 7367 7424 7464 7571 7621 7673 7674
```

The config addresses physical GPU index 2 directly. Do not also set
`CUDA_VISIBLE_DEVICES=2`, because that exposes the physical device as logical device 0 while
the config still requests device 2.

Regenerate the exact-run videos and review sheets:

```bash
.venv/bin/python scripts/render_coverage20_videos.py \
  --config configs/pilot_move_pillbottle_pad.yaml \
  --run-id coverage20-qc-contact-v5-native \
  --output-dir artifacts/rendered_videos/coverage20_qc_contact_v5_native \
  --overwrite

.venv/bin/python scripts/build_tracking_review_sheets.py \
  --render-manifest artifacts/rendered_videos/coverage20_qc_contact_v5_native/manifest.json \
  --output-dir artifacts/rendered_videos/coverage20_qc_contact_v5_native/review_sheets \
  --columns 4
```

## Verification

Final CPU verification after the GPU run:

```text
69 passed in 3.94s
```

Generated files under `artifacts/` are local run outputs and are not part of the source
commit.
