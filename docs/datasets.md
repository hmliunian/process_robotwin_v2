# Dataset contract

Runtime 接受单 task dataset 或 task collection，不从 pipeline YAML 读取机器相关 dataset path。

## 单 task 目录

```text
DATASET_ROOT/
  EXTRACT_MANIFEST.json
  data/chunk-<NNN>/episode_<id>.parquet
  videos/chunk-<NNN>/observation.images.<camera>/episode_<id-padded-6>.mp4
  sidecars/episode_<id-padded-6>.hdf5
  sidecars/videos/chunk-<NNN>/observation.depths.<camera>/episode_<id-padded-6>.mkv
  meta/
```

Chunk number 是 `episode_id // 1000`。当前 discovery 要求 Parquet、RGB video 和 HDF5
sidecar；URDF 额外要求 depth video。Parquet `frame_index` 必须从零连续，并作为 usable frame
count。

最小 manifest：

```json
{
  "profile": "target_only",
  "task": "open_door",
  "camera": "cam_high",
  "episode_indices": [9350],
  "task_kind": "door_open_action_site"
}
```

`profile` 选择 annotation mode，应为 `pick_place` 或 `target_only`；兼容的旧 aliases 会先被
规整。Target-only 数据通过 `task_kind` 选择 target specialization：

| Task kind | Target profile |
| --- | --- |
| `single_movable_target` | `grasp_manipulation` |
| `single_movable_target_conditional` | `grasp_manipulation` |
| `articulated_action_site` | `grasp_manipulation` |
| `contact_action_site` | `contact_press` |
| `door_open_action_site` | `door_open` |

Door dataset 必须显式使用 `door_open_action_site`；generic articulated dataset 保留原来的
target semantics。不支持或与 mode 冲突的值会在启动模型前失败。

## Collection 目录

```text
COLLECTION_ROOT/
  EXTRACT_MANIFEST.json
  task_a/EXTRACT_MANIFEST.json
  task_b/EXTRACT_MANIFEST.json
```

Collection manifest 声明共同 profile 和 task names：

```json
{
  "profile": "target_only",
  "datasets": [{"task": "task_a"}, {"task": "task_b"}]
}
```

Child path 根据本地 task name 解析，不信任复制 manifest 中过期的 absolute path。使用
`--task NAME` 只处理一个 child。显式 `--episode-ids` 是 strict selection；自动 discovery
则把不完整 episode 记录为 skipped。

## 可复现 selection

仓库只在 `configs/datasets/` 提交精简 manifest。目前示例覆盖 pick-place coverage set、
target-only selection 和审核后的 real-MCAP text labels。外部 video、Parquet/HDF5、depth、
checkpoint 和 generated artifacts 都不能提交。

审核后的 real pick-place MCAP 可这样转换：

```bash
uv sync --extra real-mcap
just convert-real INPUT_ROOT OUTPUT_ROOT --limit 1
```

Converter 只选择标记为 `complete_pick_place` 的记录，拒绝覆盖已有 output。转换结果没有 depth，
因此应使用 SAM gripper backend。
