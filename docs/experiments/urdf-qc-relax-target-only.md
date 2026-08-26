# Target-only URDF QC probe

The target-only full-220 audit had 21 `urdf_gripper` failures. The URDF
publisher's only episode-level visibility gate was `eligible_nonempty_fraction
>= 0.90`; this is the count of depth-evaluable active-window frames with at
least one visible URDF pixel. The renderer's structural checks and
per-component depth checks remain independent of this threshold.

## Offline threshold-zero replay

On branch `experiment/relax-urdf-qc`, the 21 failed episodes were replayed
with `--minimum-eligible-nonempty-fraction 0 --skip-overlay`.  Every episode
completed and produced diagnostics.  No episode reaches the old 0.90 gate:

| task | episodes | eligible fraction range | active nonempty range |
| --- | ---: | ---: | ---: |
| click_alarmclock | 1 | 0.167 | 0.026 |
| move_playingcard_away | 2 | 0.784–0.786 | 0.686–0.784 |
| shake_bottle | 11 | 0.581–0.886 | 0.336–0.742 |
| shake_bottle_horizontally | 1 | 0.836 | 0.777 |
| turn_switch | 6 | 0.559–0.884 | 0.358–0.708 |
| **total** | **21** | **0.167–0.886** | **0.026–0.784** |

The failure is therefore not a one-frame numerical boundary issue.  Sixteen
of 21 episodes have at least one active frame with zero rendered amodal
support (66/78 such frames in click_alarmclock/2249; 36/95 in
turn_switch/27254; 108/256 in shake_bottle/23119).  Across the other episodes,
the amodal/depth support is present but the visible mask is often empty.  The
per-frame diagnostic reason is predominantly `insufficient_depth_support`.

The failure pattern is strongly post-close: e.g. shake_bottle/23117 has
100% visible support in the 64-frame approach/close interval but only 19% of
177 post-close frames; turn_switch/27124 has 100% pre-close and 87% post-close.
This is consistent with the gripper leaving the camera/depth view or becoming
occluded after the first close, rather than a malformed URDF mask. Manual
review of the complete `click_alarmclock/2249` video confirmed the operated
gripper never enters the camera view. Its nearly empty visible track is
therefore a valid geometric result, not a silent rendering failure.

## Code change in this branch

The branch also carries the minimal target-only loop-context v4 compatibility
needed to replay the current source artifacts: an optional `t_reopen_start`
ends only the held-target encoding; the first close remains the operation
boundary.  Hold-aware source validation accepts v3 and v4.  Tests were updated
for the new version and reopen semantics.

Decision: remove URDF gripper visibility/nonempty QC, including the global 0.90
gate and the requirement for any active-window pixel. Keep structural,
depth/component, artifact and lineage validation. Coverage remains in
diagnostics only. This decision is URDF-specific; SAM gripper visual and
temporal QC remains unchanged.
