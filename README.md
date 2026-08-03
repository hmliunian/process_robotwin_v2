# RoboTwin Target / Receiver Masks V2

这个项目实现一个小而明确的 target/receiver pipeline：

```text
State Loop → Qwen Semantic Plan → SAM Seed Candidates → Qwen Mask QC → Propagation
```

当前范围固定为 `move_pillbottle_pad / cam_high / target_0 + receiver_0`：

- Stage 1 从 state 提取一个机械臂 loop 和五个事件边界；
- Stage 2 通过可配置 Qwen client/server 联合确定 target/receiver、seed frame 和短 query bank；
- Stage 3 对 query bank 生成多个 SAM3 seed mask，用 Qwen 比较实际候选，并只传播
  `qc_status=passed` 的候选；
- 通过候选 QC 的 seed 使用 SAM3 native video tracker 传播，并按角色时间窗输出 visible masks；
- 时序 QC 会记录覆盖率、断帧、相邻 IoU、质心和面积突变，并隔离多信号严重异常；
- 不做逐帧 text SAM、人工 mask 选择、自动身份纠错、Qwen bbox、gripper mask 或 amodal 补全。

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
  models/                        LoopContext / SemanticPlan / MaskQCResult / MaskRun
  pipeline/                      state_loop / qwen_stage / mask_qc / sam_stage
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

# 多 episode 顺序执行；一个 SAM3 adapter 在整个 batch 内常驻
.venv/bin/python scripts/run_target_receiver.py sam-batch \
  --config configs/pilot_move_pillbottle_pad.yaml \
  --run-id <run_id> \
  --episode-ids 7152 7156 7157
```

运行 `qwen`、启用 mask QC 的 `sam` 或完整 pipeline 前，先在另一个终端执行
`just serve-qwen`。`sam` 会消费指定 `run_id` 的 Stage 2 产物，并调用 Qwen 比较实际
SAM seed 候选；可用 `mask.qc_enabled: false` 兼容旧行为。

`sam-batch` 每次只在配置指定的一张 GPU 上处理一个 episode，但会跨 episode 复用已经加载的
SAM3 adapter；已完整通过的 episode 默认跳过，CUDA 级故障会立即终止该 worker。它解决的是
批量任务中的重复模型加载，不是需要长期维护的独立 SAM3 网络服务。

完整设计见 [docs/process_data_v2_architecture_design.md](docs/process_data_v2_architecture_design.md)。
视频跟踪方法对比、coverage20 指标和产物位置见
[docs/video_mask_tracking_experiment.md](docs/video_mask_tracking_experiment.md)。
