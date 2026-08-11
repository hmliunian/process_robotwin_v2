# 文档总览

> 更新时间：2026-08-11。本文是 `docs/` 的入口；当前实现契约以
> [architecture.md](architecture.md) 为准，实验数字以
> [experiments.md](experiments.md) 为准。

本项目为 RoboTwin 单次 pick-and-place episode 生成 visible-only mask：

```text
state loop
  -> Qwen target/receiver semantic plan
  -> SAM3 seed candidate + Qwen identity QC
  -> SAM3 native video propagation
  -> gripper backend (SAM pose-ROI 或 URDF geometry/depth)
  -> canonical four-channel masks + overlay/review sheets
```

固定输出通道为：

```text
target_0, receiver_0, gripper_left, gripper_right
```

当前正式范围是 `cam_high`、单 active arm、一次闭合—张开操作循环。mask 只描述 RGB
中实际可见的像素；遮挡区域不会做 amodal 补全。

## 当前能力

| 模式 | target/receiver | gripper | 状态 |
| --- | --- | --- | --- |
| 默认 `sam` | 当前 run 内由 Qwen + SAM3 生成 | pose ROI + SAM3 + Qwen QC | 已实施；默认入口 |
| `urdf` live | 先生成并冻结内部 Qwen/SAM source run | joints + calibration + depth + URDF | 入口已实施；fresh-only |
| `urdf` frozen-source | 逐像素复用已有 QC-passed source run | joints + calibration + depth + URDF | 已实施；coverage20 20/20 验收 |
| `active_wrist` | close/open 阶段分别选 seed | 不生成 | 仅完成可行性实验，尚未接入 pipeline |

`urdf` 只替换 gripper producer；target/receiver 始终来自 Qwen/SAM。两种 backend
发布相同的四通道 `masks.npz`、manifest/provenance schema 和 overlay/review 结构。

## 推荐入口

默认 SAM gripper：

```bash
# 终端 A
just serve-qwen

# 终端 B
just process DATASET_ROOT [OUTPUT_ROOT]
```

从原始数据一体化生成 URDF gripper：

```bash
# 需要 Qwen 服务，因为仍要生成 target/receiver
just process DATASET_ROOT [OUTPUT_ROOT] --gripper-backend urdf
```

该模式默认使用仓库内置的
`configs/assets/aloha-agilex/arx5_description_isaac_gripper.urdf`，先将对象结果冻结到：

```text
OUTPUT_ROOT/_sources/<run-id>-target-receiver/
```

随后发布最终 run。内部 source 是最终 lineage 的组成部分，不能删除、移动或修改。

复用已有 frozen source，可跳过 Qwen/SAM：

```bash
just process DATASET_ROOT OUTPUT_ROOT \
  --gripper-backend urdf \
  --source-run-dir SOURCE_RUN \
  --run-id RUN_ID
```

恢复已开始的 immutable URDF run：

```bash
just process DATASET_ROOT OUTPUT_ROOT \
  --gripper-backend urdf \
  --source-run-dir SOURCE_RUN \
  --run-id RUN_ID \
  --resume
```

live URDF 模式不支持 `--dry-run` 或 `--resume`；二者要求显式
`--source-run-dir`。URDF backend 不支持 `--force`，主动重跑必须使用新 run ID。

分阶段调试、依赖安装和完整参数见 [architecture.md](architecture.md#7-运行入口)。

## 主要产物

```text
<output-root>/<run-id>/
  process_summary.json
  <task>/episode_<id>/<camera>/
    loop.json
    semantic_plan.json
    mask_qc.json
    masks.npz
    run_manifest.json
    frame_provenance.json
    target_0/...
    receiver_0/...
    gripper_<active-arm>/...
  rendered_videos/
    manifest.json
    episode_*_overlay.mp4
    review_sheets/
```

URDF run 还包含私有审计目录 `_backend/urdf/`。它不是 downstream API；下游只应读取
canonical episode 目录和 `process_summary.json`。

## 关键语义

- `target`：被 active gripper 抓取并移动的物体。
- `receiver`：任务完成时与 target 直接接触的完整物体或目标区域，不要求位于 target
  下方。
- `visible-only`：被 gripper、target、桌面或其他物体遮挡的像素不补全。
- `fail closed`：语义、身份、输入合同或 immutable lineage 无法验证时，不静默发布猜测结果。
- `temporal QC`：检查传播连续性；它不能识别“稳定跟错实例”，身份 QC 必须在 seed
  阶段单独完成。

## 文档导航

| 文档 | 内容 | 适合什么时候看 |
| --- | --- | --- |
| [architecture.md](architecture.md) | 当前 pipeline、CLI、数据与 artifact 契约 | 实现、运行或排障 |
| [experiments.md](experiments.md) | Qwen/SAM、tracking、SAM gripper、URDF 和 active-wrist 实验结论 | 查参数依据和证据边界 |
| [datasets.md](datasets.md) | RoboTwin pick-and-place 兼容任务、深度完整性和迁移顺序 | 选择新数据集 |

推荐阅读顺序：本页 → `architecture.md` 的“运行入口”和“公共产物契约”；只有需要理解
参数来源、历史失败或实验数字时再看 `experiments.md`。

## 旧文档合并映射

| 原内容 | 新位置 |
| --- | --- |
| v2/v3/v3.1 architecture、v3 overview/progress | 本页 + `architecture.md` |
| Qwen candidate QC、video tracking | `experiments.md` 第 2 节 |
| gripper pose ROI、初始欠分割分析、fixed bbox | `experiments.md` 第 3 节 |
| URDF coverage20 设计、实验和验收 | `architecture.md` 第 6/9 节 + `experiments.md` 第 4 节 |
| active-wrist phase-seed 草案 | `experiments.md` 第 5 节 |
| RoboTwin compatible datasets | `datasets.md` |

## 版本演变

| 版本 | 已固化的变化 |
| --- | --- |
| v2 | State Loop → Qwen Semantic Plan → SAM target/receiver；加入 seed candidate QC 和 native tracking |
| v3 | SAM gripper stage 接入正式 pipeline；统一四通道 NPZ；render 自动生成 review sheets；新增 `just process` |
| v3.1 | 增加 URDF derived-run backend、canonical publisher、source lineage、immutable resume 和共享 renderer |
| 当前增量 | URDF 可在未提供 source run 时先生成内部 target/receiver source；提供 bundled render-only URDF |

旧架构设计、实施进度和逐轮实验流水账已经合并到上述三份文档。需要逐字查看历史内容时使用
Git history；不要再把旧版本设计中的计划项当作当前接口。
