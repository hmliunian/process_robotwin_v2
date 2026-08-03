# RoboTwin Target / Receiver Masks V2

这个项目只实现一个小而明确的三阶段 pipeline：

```text
State Loop → Qwen Semantic Plan → SAM Mask + Propagation
```

当前范围固定为 `move_pillbottle_pad / cam_high / target_0 + receiver_0`：

- Stage 1 从 state 提取一个机械臂 loop 和五个事件边界；
- Stage 2 通过可配置 Qwen client/server 联合确定 target/receiver、seed frame 和短 query bank；
- Stage 3 只在 seed frame 运行一次 SAM3 text prompt，随后用 SAM3 native video tracker
  传播，并按角色时间窗输出 visible masks；
- 时序 QC 会记录覆盖率、断帧、相邻 IoU、质心和面积突变，并隔离多信号严重异常；
- 不做逐帧 text SAM、自动身份纠错、Qwen bbox、gripper mask 或 amodal 补全。

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
just qwen 7152
just sam <run_id> 7152
just run 7152
```

运行 Qwen 或完整 pipeline 前先在另一个终端执行 `just serve-qwen`。`sam` 只消费指定
`run_id` 中已经保存的 Stage 2 产物，不会再次调用 Qwen；`run` 依次执行三个阶段。

完整设计见 [docs/process_data_v2_architecture_design.md](docs/process_data_v2_architecture_design.md)。
视频跟踪方法对比、coverage20 指标和产物位置见
[docs/video_mask_tracking_experiment.md](docs/video_mask_tracking_experiment.md)。
