# Bundled Aloha gripper render asset

This directory contains the minimum RoboTwin 2.0 `aloha-agilex` geometry
needed by the current gripper renderer.

Use it from the repository root with:

```bash
--urdf-path configs/assets/aloha-agilex/arx5_description_isaac_gripper.urdf
```

No separate mesh-root argument is required. The URDF resolves every visual
mesh relative to this directory.

The asset is derived from
`embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf` in the local
RoboTwin 2 asset archive `robotwin2_embodiments_c15cc97.zip`. The source URDF
SHA-256 is
`097c59fb19a7b482249c6097df8319586ea7cfd268c015516f103b289a7e761a`.

The derived URDF preserves the exact follower-arm kinematic tree, joint
origins, axes, limits, visual origins, and the original DAE geometry for
`fl/fr_base_link` and `fl/fr_link1..8`. It removes unrelated robot branches,
collision geometry, and non-rendered visuals. It is therefore a render-only
asset and must not be used as a physics or collision model.

`asset_manifest.json` records the expected size and SHA-256 of every bundled
file. The two tiny PNG files are material references embedded in the original
DAE files; segmentation does not use their colours, but retaining them keeps
the mesh directory self-contained.
