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

Chunk number 是 `episode_id // 1000`。Discovery 要求 Parquet 和 RGB video；完整夹爪流程
还要求 HDF5 sidecar，URDF 额外要求 depth video。`--object-source-only` 不要求 HDF5。
Parquet `frame_index` 必须从零连续，并作为 usable frame
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

Remote-operation `task_manifest` 是跨来源索引。可以生成按来源、任务分组的只读输入视图：

```bash
.venv/bin/python scripts/prepare_task_manifest.py TASK_MANIFEST_JSON BATCH_ROOT \
  --task-kind door_open_action_site
```

视图软链接原始 Parquet/RGB，保留源 episode 编号和 split，并从索引写入纠正后的任务文本。
单右臂 7D 状态使用已有 real-MCAP 夹爪归一化规则，再重新推导 close/hold 时间线；
末尾没有 reopen 时保持 terminal hold。无法确认 close 的 episode 单独绑定为完整视频窗口，
在元数据中记录原因，不生成 held 编码。所有输入视图只用于 `--object-source-only`。
`batch_plan.json` 记录原始索引 hash、分组和完整 episode selection。

准备后可以逐组试跑，再以新的 runs 目录完整计算：

```bash
.venv/bin/python -m scripts.process_task_manifest_batch BATCH_ROOT --gpu 2 --smoke
.venv/bin/python -m scripts.process_task_manifest_batch BATCH_ROOT --gpu 2
```

批次入口拒绝覆盖已有 run，写入 `batch_status.json` 和 `episode_results.json`，完成后生成
`review_sheets/{overall_index,early,late}.jpg`。失败条目保留原始视频帧；同任务中来自不同
数据集的相同 episode 编号通过 `episode_uid` 区分。

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

UMI 的 target-only 任务只提取 head-left RGB，并用左右夹爪各一个 aperture 信号派生真实的
close/hold 边界；输出 Parquet 不包含机械臂 joint、EEF、state 或 action：

```bash
just convert-umi INPUT_ROOT OUTPUT_ROOT
just process OUTPUT_ROOT --object-source-only --ui plain
```

转换后的 `timeline_source=episode_metadata`，target window 保守覆盖完整 episode；
`--object-source-only` 仅生成 target/receiver 对象源，其中 target-only 的 receiver 为 N/A，
不运行 gripper mask 或四通道 canonical publication。
