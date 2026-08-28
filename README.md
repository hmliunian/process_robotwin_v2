# RoboTwin Target / Receiver / Gripper Masks V3

这个项目实现一套可追溯的 RoboTwin visible-mask pipeline：

```text
State Loop → Qwen Semantic Plan → SAM target/receiver → Gripper Stage → 4-channel masks
```

当前验证范围是 `move_pillbottle_pad / cam_high`，输出 target、receiver 和 active gripper：

- Stage 1 从 state 提取一个机械臂 loop 和五个事件边界；
- Stage 2 通过可配置 Qwen client/server 联合确定 target/receiver、seed frame 和短 query bank；
- Stage 3 对 query bank 生成多个 SAM3 seed mask，用 Qwen 比较实际候选，并只传播
  `qc_status=passed` 的候选；
- 通过候选 QC 的 seed 使用 SAM3 native video tracker 传播，并按角色时间窗输出 visible masks；
- gripper stage 读取已保存的 target/receiver native track，使用 pose ROI + Qwen QC
  选择 gripper seed，并做已知物体排除；不会重跑 target/receiver；
- 同一份 `masks.npz` 固定包含 target_0、receiver_0、gripper_left、gripper_right 四通道；
- 时序 QC 会记录覆盖率、断帧、相邻 IoU、质心和面积突变，并隔离多信号严重异常；
- 不做逐帧 text SAM、人工 mask 选择、自动身份纠错、Qwen bbox 或 amodal 补全。

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
  pipeline/                      state_loop / qwen_stage / mask_qc / sam_stage / gripper_stage
  adapters/                      dataset / Qwen HTTP / SAM3 / artifacts
scripts/                         server 和运行入口
tests/                           unit + integration
docs/README.md                   文档入口与当前状态
docs/architecture.md             当前架构、CLI 和产物契约
docs/experiments.md              实验结论与参数依据
docs/datasets.md                 兼容任务与数据完整性
```

## 快速验证

```bash
just test
just preflight
just loop 7152
just qwen 7152
just sam <run_id> 7152
just run 7152
just process ../dataset/move_pillbottle_pad_coverage20_original

# Qwen API + multi-GPU SAM；SAM_WORKER_GPUS 必填，API 并发默认 4
SAM_WORKER_GPUS=0,1,2,3 just run-parallel \
  ../dataset/move_pillbottle_pad_coverage20_original

# 将审核后的真实 pick-and-place MCAP 转成 RoboTwin 目录
just convert-real <input_root> <new_output_root> --limit 1

# 多 episode 顺序执行；一个 SAM3 adapter 在整个 batch 内常驻
.venv/bin/python scripts/run_target_receiver.py sam-batch \
  --config configs/pilot_move_pillbottle_pad.yaml \
  --run-id <run_id> \
  --episode-ids 7152 7156 7157

.venv/bin/python scripts/run_target_receiver.py gripper-batch \
  --config configs/pilot_move_pillbottle_pad.yaml \
  --run-id <run_id> \
  --episode-ids 7152 7156 7157
```

运行分阶段的 `qwen`、启用 mask QC 的 `sam` 或 `run` 前，先在另一个终端执行
`just serve-qwen`。该命令会自动选择空闲显存最多的合格 GPU。`sam` 会消费指定 `run_id`
的 Stage 2 产物，并调用 Qwen 比较实际 SAM seed 候选；可用 `mask.qc_enabled: false`
兼容旧行为。

`sam-batch` 与 `gripper-batch` 都会跨 episode 复用一个 SAM3 adapter；已完整通过的
episode 默认跳过，CUDA 级故障会立即终止 worker。`run` 会按 qwen → sam → gripper
运行一个 episode。

`just process <dataset_root>` 是推荐的一键入口：自动扫描
`data/chunk-*/episode_*.parquet`，核对 video/sidecar，处理全部完整 episode，最后生成
四通道 overlay 视频以及 target/receiver/gripper 的 early/late review sheets。配置为
`qwen.runtime=api` 时，它只探测远程 endpoint，失败即退出，绝不启动本地模型；只有
`qwen.runtime=local` 才会复用已有服务，或排除 SAM/显式 EGL GPU 后选择合格 GPU 启动服务。
由本次命令启动的本地 Qwen 会在 process 成功、失败或被中断后自动关闭，加载日志保存在
`artifacts/qwen-services/`。

默认 API 配置优先读取环境变量 `QWEN_API_KEY`；未设置时，`just process` 会读取被 Git
忽略的 `secrets/qwen_api_key.txt`。可复制 `secrets/qwen_api_key.txt.example` 后填入单行
key，并设置权限 `chmod 600 secrets/qwen_api_key.txt`。也可通过 `QWEN_API_KEY_FILE` 指定
其他本地文件；凭据不会写入 YAML、命令行或运行产物。

交互终端默认显示 episode 总进度、当前阶段、跳过/失败状态、耗时和最终 artifact；stderr
不是交互终端时改为稳定的逐行日志，stdout 被重定向时仍会写出最终 JSON。可显式选择输出方式：

```bash
# 强制动态终端 UI
just process <dataset_root> --ui rich

# 适合日志采集的无 ANSI 逐行输出
just process <dataset_root> --ui plain

# stdout 只输出一个可直接解析的最终 summary
just process <dataset_root> --ui json > process-summary.json

# 回放嵌套 Qwen/SAM/URDF 阶段的详细 payload
just process <dataset_root> --verbose
```

`--output-format` 是 `--ui` 的等价别名；设置 `NO_COLOR` 会关闭 Rich 颜色，`CI` 或
`TERM=dumb` 环境会自动使用纯文本模式。完整机器可读结果仍以 run 目录中的
`process_summary.json` 为准。

`just run-parallel` 是 Qwen API + 多 GPU SAM 的快捷入口。`SAM_WORKER_GPUS` 必须显式设置为
逗号分隔的 physical GPU ID；每张卡常驻一个 SAM worker。`QWEN_MAX_IN_FLIGHT` 可选，默认
为 `4`，限制所有 worker 共享的远程 Qwen HTTP 并发数。其余参数与 `just process` 完全相同：

```bash
SAM_WORKER_GPUS=0,1,2,3 QWEN_MAX_IN_FLIGHT=4 \
just run-parallel <dataset_root> \
  --urdf-egl-device-id 4
```

该快捷入口要求所选配置使用 `qwen.runtime=api`，且不会查询或等待 GPU 空闲显存；它只校验
重复 SAM GPU，以及 live URDF 中 SAM pool 与 EGL GPU 是否重叠。需要直接控制 CLI 覆盖时，
仍可使用 `just process ... --sam-worker-gpus ... --qwen-max-in-flight ...`；不传 worker pool 时
沿用 YAML，YAML 也未配置则保持单进程串行。

真实 pick-and-place MCAP 可通过 `just convert-real INPUT_ROOT OUTPUT_ROOT [OPTIONS...]` 转换。
首次使用先安装 `uv sync --extra real-mcap`；`--texts` 默认是已审核的
`configs/datasets/pick_and_place_real_texts.json`，`--limit N` 可用于 smoke conversion。转换器
只选择 `complete_pick_place` 记录并拒绝覆盖已有输出目录。转换结果没有 depth，后续必须使用
`--gripper-backend sam`。完整流程见 [docs/datasets.md](docs/datasets.md#0-真实-pp-mcap-转换)。

完整文档从 [docs/README.md](docs/README.md) 开始；当前实现契约见
[docs/architecture.md](docs/architecture.md)，coverage20 实验、参数依据和证据边界见
[docs/experiments.md](docs/experiments.md)。
