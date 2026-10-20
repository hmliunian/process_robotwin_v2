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

`configs/process.yaml` 由共享 defaults 和各工作流 overlay 组成，不保存 dataset identity。
运行时由 `DatasetBinding` 提供 root、task、camera、episode selection 和 extract manifest；
`bind_dataset()` 生成本次运行使用的 immutable `PipelineConfig`。

| Overlay | Annotation mode | Target profile | Required roles |
| --- | --- | --- | --- |
| `pick_place` | `pick_place` | `grasp_manipulation` | target, receiver |
| `target_only` | `target_only` | `grasp_manipulation` | target |
| `contact_press` | `target_only` | `contact_press` | target |
| `door_open` | `target_only` | `door_open` | target |
| `video_object` | `target_only` | `video_object` | target |
| `tool_use` | `tool_use` | `grasp_manipulation` | target, receiver |

Annotation mode 决定角色数和 timeline；target profile 只细化 target identity 与 prompts。
Specialized profile 不能用于 `target_only` 以外的 mode。

## Stage contracts

### State timeline

Parquet 行数是 frame-count authority。RGB 可以包含 manifest 声明的尾帧 surplus，该部分会被
忽略。Stage 1 推导一个 active arm 和有序事件边界。Pick-place 要求
close/transport/reopen；target-only 要求 close-and-hold，并允许记录稍后的 reopen。只有标记为
`seed_candidate=yes` 的帧能成为 semantic seed。

UMI 单臂物体标注可显式声明 `timeline_source=video_window`。这种输入不要求
夹爪开合或机械臂状态，而由逐段审核的 metadata 声明 active arm 和有效视频窗口。
`video_window` 只允许 object-source-only，均匀抽取窗口内的可见候选，不伪造抓取事件。
它使用 `robotwin_loop_context_v5`，target 和适用的 receiver 都在完整窗口内跟踪；
没有确认的 held 区间，因此非空 mask 使用普通 visible 编码 1。已有 v1–v4 合同不变。
`tool_use` 的 target 是完整工具，receiver 是被工具作用的独立容器或有明确边界的表面，
不要求发生放置或松开。具体对象定义保存在任务配置中。

嵌套 UMI 批次可用 `scripts/process_umi_batch.py` 先按头部及左右腕部多时刻预览筛选。
模型筛选只生成单臂候选，来源记录为 `qwen_multiframe_screen`，不等同于人工验收。
双臂、低置信度、证据不足和接口失败分别留在批次索引中，不进入 masks 处理。
转换兼容 head-left 的 `left_h264/video` 和旧 `left/video` topic；同一片段混用两者会拒绝。
转换保留解码帧对应的源序号、时间戳、实际 topic、原始片段 ID 和逐段任务文本，
按动作及实际解码分辨率分组。`scripts/run_umi_batch.py` 复用现有 SAM GPU worker，
为每组生成 overlays 和 review sheets；video-window 产物始终标为需要视觉复核。

UMI 批处理可显式选择 `--inference-mode resident` 或 `async`。每个 GPU 进程跨组持有一个
SAM backend，任务携带各自的 dataset、mode 和 profile，每个角色使用独立 session。
`async` 在同一个 SAM owner 线程周围重叠有界 Qwen 规划和磁盘文件可视化；候选 QC、
bbox 救援、时序传播及 canonical 保存仍保持原有依赖。多 GPU 共享文件 token 限制 Qwen
总请求数，各线程分别持有 token 句柄；不提供全局 QC 请求优先级。

快速模式对文本、bbox、文本加 bbox 候选均可只加载对应 JPEG。多候选帧的搜索、query
顺序和 QC 规则不变。原生传播可选择只计算跟踪所需的视觉特征，编码 batch 为 1 或 4，
tracker 仍逐帧更新记忆，特征缓存只保留一个 chunk。`sam3.seed_image_only` 和
`sam3.tracking_batch_size` 通过配置传入所有 backend 构造入口。

快速批次在输出根的 `.pipeline/selection.json` 保存完整选择与输入、配置、权重和实现身份，
在每个原有 episode 目录的 `.pipeline/` 中保存任务身份、结果和可视化。分组目录和 canonical
mask 路径不变，结果索引中的 `video` 指向实际 overlay；分组 review sheets 仍统一生成。
恢复先校验整批及逐任务身份，通过校验的有效 masks 可只补渲染。渲染就绪和推理完成分别
记账；失败、中断和未执行任务均保留。模式和运行参数见 [UMI 批处理说明](umi_batch.md)。

输出 `loop.json`，包含 annotation mode、required roles、事件、输出窗口和 sparse semantic
frames。无效或含糊的 state 会 fail closed。

### Semantic plan

跨 close/hold 与 video-window 的语义模板可使用 `timeline_kind` 和 `timeline_events`。
后者只序列化实际声明的事件；video-window 不补造 close 或 hold 时间。

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

共享 profile 默认启用完整 resolution chain：S1 在所有合法 seeds 上尝试 semantic query bank
与 curated aliases；S2 从语义规划开始应用 profile 专用 identity、完整性与 action-context QC；
S3 仅在全部文本尝试失败后执行 Qwen bbox localization，并通过相同的 candidate 与 temporal QC。
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
    receiver_0/...        # pick-place 或 tool-use
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
