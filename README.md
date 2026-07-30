# RoboTwin Target / Receiver Masks V2

这个项目只实现一个小而明确的三阶段 pipeline：

```text
State Loop → Qwen Semantic Plan → SAM Mask + Propagation
```

当前范围固定为 `move_pillbottle_pad / cam_high / target_0 + receiver_0`：

- Stage 1 从 state 提取一个机械臂 loop 和五个事件边界；
- Stage 2 通过可配置 Qwen client/server 联合确定 target/receiver、seed frame 和短 query bank；
- Stage 3 使用 SAM3 text-only seed、native mask propagation 和同帧 text evidence 输出
  visible masks；
- 不做人工 mask 选择、QC、Qwen bbox、gripper mask 或 amodal 补全。

## 测试数据

外部测试集：

```text
/DATA/disk8/xuran/add_mask_robotwin/dataset/move_pillbottle_pad_coverage20_original
```

项目只提交 [dataset manifest](configs/datasets/move_pillbottle_pad_coverage20.json)，不提交视频、
Parquet 或 HDF5。

## 目录

```text
configs/                         pipeline、prompt 和 dataset manifest
src/robotwin_annotation_v2/
  models/                        LoopContext / SemanticPlan / MaskRun
  pipeline/                      state_loop / qwen_stage / sam_stage
  adapters/                      dataset / Qwen HTTP / SAM3 / artifacts
scripts/                         server 和运行入口
tests/                           unit + integration
docs/process_data_v2_architecture_design.md
```

## 快速验证

```bash
just test
just preflight
just loop 7152
```

Qwen server、SAM smoke 和完整运行命令将在对应阶段实现后加入 `justfile`。

完整设计见 [docs/process_data_v2_architecture_design.md](docs/process_data_v2_architecture_design.md)。
