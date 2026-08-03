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

# 对上一步输出的 run_id 运行 Stage 3，不重复调用 Qwen
just sam <run_id> 7152

# 或一次运行完整三阶段
just run 7152

# 使用已有 masks.npz 覆盖生成 20 条全长 overlay 视频
.venv/bin/python scripts/render_coverage20_videos.py --overwrite
```

主配置是 `configs/pilot_move_pillbottle_pad.yaml`。Qwen 的 prompt 模板位于
`configs/prompts/target_receiver_semantic.txt`，物体名称和视觉属性不能写死在 Python 或 YAML
中。Stage 2 会写出 `loop.json`、`semantic_plan.json`、rendered prompt 和 Qwen raw response；
Stage 3 会写出 seed、native track、same-frame text observation、`masks.npz` 和 provenance。

overlay 视频默认写到 `artifacts/rendered_videos/coverage20_best_current/`。默认样式是
`alpha=0.32` 的内部填充、mask 外侧 `3 px` 高亮角色轮廓，以及扩张到总计 `5 px` 的黑色
衬边。可通过 `--alpha`、`--outline-radius` 和 `--halo-radius` 调整；渲染只读取并显示已有
mask，不会修复 mask 的断帧、漂移或目标身份。

运行产物写到 `artifacts/runs/`，不会进入 Git。
