# 架构与产物合同

本文只描述当前合同。修改 stage 边界或四通道 `masks.npz` 前，应先更新本文。

## Pipeline

```text
EXTRACT_MANIFEST.json + episode files
  -> 把共享 pipeline profile 绑定到 dataset
  -> Stage 1：推导 timeline 与合法 semantic frames
  -> Stage 2：生成并校验 Qwen semantic plan
  -> Stage 3：用 SAM3 解析并传播必要的 object masks
  -> 用 SAM 或 URDF 生成 active gripper mask
  -> 原子发布 canonical artifacts
  -> 渲染 overlays 与 review sheets
```

`configs/process.yaml` 由一个 defaults block 和四个 overlay 组成，不保存 dataset identity。
运行时由 `DatasetBinding` 提供 root、task、camera、episode selection 和 extract manifest；
`bind_dataset()` 生成本次运行使用的 immutable `PipelineConfig`。

| Overlay | Annotation mode | Target profile | Required roles |
| --- | --- | --- | --- |
| `pick_place` | `pick_place` | `grasp_manipulation` | target, receiver |
| `target_only` | `target_only` | `grasp_manipulation` | target |
| `contact_press` | `target_only` | `contact_press` | target |
| `door_open` | `target_only` | `door_open` | target |

Annotation mode 决定角色数和 timeline；target profile 只细化 target identity 与 prompts。
Specialized profile 不能用于 `target_only` 以外的 mode。

## Stage contracts

### State timeline

Parquet state 是 frame-count authority。RGB 可以包含 manifest 声明的尾帧 surplus，该部分会被
忽略。Stage 1 推导一个 active arm 和有序事件边界。Pick-place 要求
close/transport/reopen；target-only 要求 close-and-hold，并允许记录稍后的 reopen。只有标记为
`seed_candidate=yes` 的帧能成为 semantic seed。

输出 `loop.json`，包含 annotation mode、required roles、事件、输出窗口和 sparse semantic
frames。无效或含糊的 state 会 fail closed。

### Semantic plan

Qwen 接收交错的 frame label 与 image。严格 JSON 响应必须只包含 annotation mode 要求的角色。
每个可用角色包含：

- 合法的 `seed_frame_id`；
- 互不重复的 category、color、shape 和可选 general fallback query；
- 覆盖所有非空 query 的顺序，且 general fallback 位于最后；
- exclusions 与简短 reason。

Contact-press profile 识别可见的被按压控件。Door-open profile 只接受以 `handle` 为中心的
query，并把 `microwave door handle` 之类 appliance context 规整到物理 handle。
`door panel` 或 `moving door panel` 仅在没有可分离 handle 时作为唯一 category query。
因截断或 content filtering 结束的 Qwen completion 会在解析 JSON 前被拒绝。

`semantic_plan.json` 记录非默认 target profile 和 rendered prompt 的 SHA-256。

### Object masks

每个 required role 都会在合法 seeds 上尝试 semantic query bank，执行基础 mask 检查，并让
Qwen 比较实际 candidate panels。配置可以增加 curated aliases 和 seed fallback。Qwen bbox
localization 是最后一级 fallback，其 SAM box mask 仍要通过相同的 candidate 与 temporal QC。
Identity QC 未通过的 candidate 不得进入 propagation。

SAM3 native video tracking 从选定 seed 双向传播，并裁剪到 role window。Temporal QC 记录
coverage、gaps、adjacent IoU、centroid motion 和 area jumps；多个严重信号会 quarantine 该
role，而不是发布已知错误的 pixels。

### Gripper masks

只标注 active arm，inactive gripper channel 保持全空并标记 `not_annotated`。Pick-place 可以
使用 SAM pose-ROI backend。默认 URDF backend 按 scene depth 渲染 geometry，也是 target-only
完整 pipeline 的 gripper backend。Object pixels 的优先级高于 gripper pixels。

URDF 可以消费 fresh object-source run 或显式固定的 frozen source。发布和 resume 前都会校验
source inputs、artifacts、assets 和 implementation identity 的 hashes。

## Canonical artifacts

```text
<output>/<run-id>/
  process_summary.json
  <task>/episode_<id>/<camera>/
    loop.json
    semantic_plan.json
    mask_qc.json
    masks.npz
    run_manifest.json
    frame_provenance.json
    target_0/...
    receiver_0/...        # 仅 pick-place
    gripper_<active>/...
  rendered_videos/
```

新 `masks.npz` 使用 `robotwin_visible_masks_v3`，严格包含：

```text
format_version, frame_count, masks, instance_names, roles,
annotation_status, qc_status, frame_encoding
```

`masks` 是固定顺序的 bool `[4,T,H,W]`：

| Index | Instance | 含义 |
| ---: | --- | --- |
| 0 | `target_0` | required target |
| 1 | `receiver_0` | receiver，或全零 `not_applicable` |
| 2 | `gripper_left` | left gripper |
| 3 | `gripper_right` | right gripper |

`frame_encoding` 是 `uint8 [4,T]`：`0` 为空，`1` 为普通 visible mask，`2` 为 close 后、reopen
前的 held target。Schema format version 与 package Semantic Versioning 相互独立。

`run_manifest.json` 列出 role artifacts 和 algorithms；`frame_provenance.json` 记录每个 channel
的 producer、seed、window、backend、annotation mode 和 target profile；
`process_summary.json` 是 dataset-level 状态和 render 索引。Writer 原子发布，reader 严格校验
keys、shape、dtype、status 和 lineage。

## 模块边界

- `models/`、`domain/`：稳定数据与词汇；
- `application/`：dataset 与 episode orchestration；
- `pipeline/`：stage algorithms 与 validation；
- `adapters/`：datasets、Qwen、SAM3、rendering 与 storage；
- `scripts/`：薄 executable entry points；
- `configs/`：共享 profiles、prompts、manifests 与 bundled render assets。

Domain code 不依赖 adapters。生成的视频、dataset、checkpoint、run 和 secret 不得进入 Git。
