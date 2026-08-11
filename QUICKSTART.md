# 快速开始

```bash
cd /DATA/disk8/xuran/add_mask_robotwin/process_data_v2

# 当前开发环境复用 process_data 的依赖
ln -s ../process_data/.venv .venv  # 已存在时跳过

# CPU 单元测试
just test

# 检查 20-episode 外部数据契约
just preflight

# 只运行 Stage 1
just loop 7152

# 终端 A：启动独立 Qwen server（只加载模型，不包含 prompt 逻辑）
just serve-qwen

# 终端 B：运行 Stage 1 + Stage 2
just qwen 7152

# 对上一步输出的 run_id 运行 target/receiver SAM；Qwen server 需保持运行
just sam <run_id> 7152

# 复用上一步保存的 native track 运行 gripper，不重传播 target/receiver
.venv/bin/python scripts/run_target_receiver.py gripper \
  --config configs/pilot_move_pillbottle_pad.yaml \
  --episode 7152 \
  --run-id <run_id>

# 多条 episode 使用一个进程内常驻的 SAM3 adapter，完整条目会自动跳过
.venv/bin/python scripts/run_target_receiver.py sam-batch \
  --config configs/pilot_move_pillbottle_pad.yaml \
  --run-id <run_id> \
  --episode-ids 7152 7156 7157

.venv/bin/python scripts/run_target_receiver.py gripper-batch \
  --config configs/pilot_move_pillbottle_pad.yaml \
  --run-id <run_id> \
  --episode-ids 7152 7156 7157

# 或一次运行 qwen → sam → gripper
just run 7152

# 推荐：自动发现目录中的全部 episode，一键处理并渲染
just process ../dataset/move_pillbottle_pad_coverage20_original

# 终端显示可切换为 rich、plain 或单一 JSON；默认 auto
just process ../dataset/move_pillbottle_pad_coverage20_original --ui plain
just process ../dataset/move_pillbottle_pad_coverage20_original --ui json \
  > process-summary.json

# 使用已有四通道 masks.npz 生成全长 overlay 视频和 review sheets
.venv/bin/python scripts/render_coverage20_videos.py --overwrite
```

主配置是 `configs/pilot_move_pillbottle_pad.yaml`。Qwen 的 prompt 模板位于
`configs/prompts/target_receiver_semantic.txt`；mask 候选验证 prompt 位于
`configs/prompts/mask_candidate_qc.txt`。物体名称和视觉属性不能写死在 Python 或 YAML 中。
Stage 2 会写出 `loop.json`、`semantic_plan.json`、rendered prompt 和 Qwen raw response；
SAM/gripper 会额外写出 `mask_qc.json`、gripper seed QC、候选 seed masks、选中
seed、native track、四通道 `masks.npz` 和 provenance。

`sam-batch`、`gripper-batch` 和 `just process` 在批次中复用一个 SAM3 adapter，
并为每个 episode 单独创建、清理视频 session。配置中的 `sam3.gpus` 使用物理 GPU
索引时，不要再用 `CUDA_VISIBLE_DEVICES` 把同一张卡重映射成逻辑索引 0。

`just process` 在交互终端显示 episode/阶段进度和最终汇总；非交互环境输出稳定日志，
并在 stdout 保留最终 JSON。`--verbose` 可显示默认收起的阶段 payload，`NO_COLOR` 可关闭
颜色。完整结果始终写入 run 目录中的 `process_summary.json`。

overlay 视频默认写到 `artifacts/rendered_videos/coverage20_best_current/`。默认样式是
`alpha=0.32` 的内部填充、mask 外侧 `3 px` 高亮角色轮廓，以及扩张到总计 `5 px` 的黑色
衬边。可通过 `--alpha`、`--outline-radius` 和 `--halo-radius` 调整；渲染只读取并显示已有
mask，不会在渲染时修复断帧、漂移或目标身份。review sheets 默认随 render 自动生成；
需要跳过二次解码时传 `--skip-review-sheets`。

运行产物写到 `artifacts/runs/`，不会进入 Git。
