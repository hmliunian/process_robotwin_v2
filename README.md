# RoboTwin Annotation 3

本项目把 RoboTwin episode 转换为可追溯的 visible masks：

```text
state timeline -> Qwen semantic plan -> SAM object masks -> gripper mask -> masks.npz
```

当前 package 版本为 `3.0.0`，要求 Python 3.13，并默认使用项目根目录下的 `.venv`。

## 支持范围

| Dataset mode | Target profile | Object channels |
| --- | --- | --- |
| `pick_place` | `grasp_manipulation` | target + receiver |
| `target_only` | `grasp_manipulation` | target |
| `target_only` | `contact_press` | 被按压的 action site |
| `target_only` | `door_open` | 被操作的完整 handle，或显式 door-panel proxy |

Profile 只保存算法参数。Dataset root、task、camera、episode IDs 和 task kind 都在运行时从
`EXTRACT_MANIFEST.json` 绑定。共享配置是
[`configs/process.yaml`](configs/process.yaml)。

## 快速开始

```bash
uv sync --extra sam3 --extra urdf
cp secrets/qwen_api_key.txt.example secrets/qwen_api_key.txt

just test
just process DATASET_ROOT --episode-ids EPISODE_ID --ui plain
```

`just process` 会从 manifest 推断 mode 和 target profile。默认 Qwen runtime 使用配置中的 API；
环境变量 `QWEN_API_KEY` 优先于 Git 忽略的本地 secret 文件。默认 gripper backend 是 URDF，
因此需要 depth。没有 depth 的 pick-place 数据可使用 `--gripper-backend sam`；只需要对象
mask 时使用 `--object-source-only`，它会跳过 gripper 与 canonical publication。

结果写入 `artifacts/runs/<run-id>/`。机器可读入口是 `process_summary.json`，overlay 和 review
sheets 位于 `rendered_videos/`。

常用检查：

```bash
just test-all
just lint
.venv/bin/python -m mypy src
just version-check
```

## 文档

- [架构与产物合同](docs/architecture.md)
- [数据集与 manifest 合同](docs/datasets.md)
- [运行、测试与发布](docs/operations.md)
- [版本变更](CHANGELOG.md)

实验报告和迁移过程不再作为“当前文档”维护；已完成的调查仍可从 Git 历史查看。
