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
```

主配置是 `configs/pilot_move_pillbottle_pad.yaml`。Qwen 的 prompt 模板位于
`configs/prompts/target_receiver_semantic.txt`，物体名称和视觉属性不能写死在 Python 或 YAML
中。Stage 2 会写出 `loop.json`、`semantic_plan.json`、rendered prompt 和 Qwen raw response。

运行产物写到 `artifacts/runs/`，不会进入 Git。
