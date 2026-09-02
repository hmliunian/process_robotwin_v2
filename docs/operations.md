# 运行、测试与发布

## 环境

使用 Python 3.13，并把环境放在 `.venv`：

```bash
uv sync --extra sam3 --extra urdf
```

共享 profile 默认使用 Qwen API。可以导出 `QWEN_API_KEY`，或复制 Git 忽略的本地文件：

```bash
cp secrets/qwen_api_key.txt.example secrets/qwen_api_key.txt
chmod 600 secrets/qwen_api_key.txt
```

Credential 不会写入 YAML、命令行或 run artifacts。Legacy local-runtime config 仍可配合
`just serve-qwen` 使用本地 Qwen。

## 运行 dataset

推荐 smoke command：

```bash
just process DATASET_ROOT OUTPUT_ROOT \
  --episode-ids EPISODE_ID \
  --ui plain
```

Output root 可省略，默认是 `artifacts/runs`。Dataset manifest 决定 mode 和 target profile。
`--pick-place` 或 `--target-only` 可以作为显式一致性检查，发生冲突时直接拒绝。常用参数：

```text
--task NAME
--camera NAME
--run-id ID
--skip-render
--force                         # 只用于 SAM rerun
--ui auto|rich|plain|json
--verbose
```

默认 URDF gripper 需要 HDF5 calibration 和 depth video。它可以 fresh 生成 objects，也可以消费
pinned source：

```bash
just process DATASET_ROOT OUTPUT_ROOT --gripper-backend urdf

just process DATASET_ROOT OUTPUT_ROOT \
  --gripper-backend urdf \
  --source-run-dir SOURCE_RUN \
  --run-id NEW_RUN_ID
```

Frozen-source `--dry-run` 只验证 plan；`--resume` 必须提供显式 run ID，且只接受 identity-compatible
artifacts。Source run 被固定后不要再修改。

Pick-place 也可以使用 SAM gripper：

```bash
just process DATASET_ROOT --gripper-backend sam
```

API-backed multi-GPU object processing 会为每张 GPU 启动一个 persistent SAM worker：

```bash
SAM_WORKER_GPUS=0,1,2,3 QWEN_MAX_IN_FLIGHT=4 \
just run-parallel DATASET_ROOT --urdf-egl-device-id 4
```

SAM GPUs 不能与 live URDF EGL GPU 重叠。Scheduler 不等待外部 GPU jobs，也不按 utilization 自动
换卡；device allocation 由 operator 负责。

## 检查 run

按下面顺序查看：

```text
process_summary.json
rendered_videos/manifest.json
rendered_videos/review_sheets/
<task>/episode_<id>/<camera>/run_manifest.json
<task>/episode_<id>/<camera>/frame_provenance.json
```

Process success 不能替代 visual review。应检查 target 的 early/late identity、pick-place receiver、
active gripper coverage，以及 target/gripper overlap。Quarantined role 或异常 Qwen
`finish_reason` 都属于失败，而不是 partial success。

## Validation

日常 gates：

```bash
just test
just lint
.venv/bin/python -m mypy src
git diff --check
```

合并 profile 或 artifact 变化前：

```bash
just test-all
just preflight
```

然后为每个受影响 profile 运行并目检一个 representative episode。V3 merge matrix 是
pick-place、generic target-only、contact-press 和 door-open。PR 中记录 task、camera、episode
ID、command、result 与 review artifact，但不要把 generated run 提交到 Git。

## Release version

Package 从 `3.0.0` 开始遵循 Semantic Versioning。`_version.py` 是 package 与 build metadata
共用的单一版本源；schema `format_version` 单独演进。

```bash
just version
just version-check
```

Release 前先把变化写在 `[Unreleased]`，然后执行：

```bash
just bump-version patch
just bump-version minor
just bump-version major
just bump-version 3.2.0
```

命令会推进版本，并把 pending changelog 放到带日期的 release heading 下。检查 diff、运行测试和
`just version-check` 后，再手动 commit/tag；该命令不会自动 commit、tag 或 push。
