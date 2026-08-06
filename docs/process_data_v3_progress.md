# v3 实施进度

> 跟踪 `process_data_v3_architecture_design.md` 第 7 节 P1–P6 的完成情况。每个阶段完成
> 并测试通过后在这里更新状态和 commit hash。

当前分支：`experiment/gripper-pose-roi-coverage20`

## 状态总览

| 阶段 | 内容 | 状态 | commit |
|---|---|---|---|
| P1 | 删除孤立实验脚本 + 空测试目录 | ✅ 完成 | `eb2b7dc`, `5494e47` |
| P2 | gripper 算法迁移到 `pipeline/gripper_stage.py` | ✅ 完成 | `05f5893` |
| P3 | pipeline CLI 接入 gripper + 四通道 `masks.npz` | ✅ 完成 | 待提交 |
| P4 | render 简化（删事后 merge、并入 review sheet） | ✅ 完成 | 待提交 |
| P5 | 一键处理任意 RoboTwin 目录 | ✅ 完成 | 待提交 |
| P6 | 文档同步（v2 文档过时声明、README/QUICKSTART） | ✅ 完成 | 待提交 |

---

## P1：删除孤立实验脚本（完成）

删除了 4 个只在 ROI 参数选型阶段用过一次的脚本及其测试，加上从未被使用过的空目录：

```text
scripts/analyze_gripper_roi_variants.py
scripts/render_fixed_gripper_bbox_sweep.py
scripts/render_gripper_roi_comparison.py
scripts/experiment_gripper_pose_roi_coverage20.py
tests/unit/test_analyze_gripper_roi_variants.py
tests/unit/test_render_fixed_gripper_bbox_sweep.py
tests/unit/test_render_gripper_roi_comparison.py
tests/contract/            # 只有空的 __init__.py，从未有过 contract 测试
```

选型结论保留在 `docs/gripper_pose_roi_coverage20_experiment.md` 等实验文档中，未受影响。

验证：`pytest -q` 101 passed；grep 确认仓库内（`.worktrees/` 之外）无遗留引用。

## P2：gripper 算法迁移（完成）

`src/robotwin_annotation_v2/experiments/{gripper_pose_roi.py, gripper_seed_qc.py}`
（合计 1114 行）原样合并迁移为 `src/robotwin_annotation_v2/pipeline/gripper_stage.py`
（1109 行）。纯搬迁：函数体逐字节比对确认未改动，只合并了 import 头部并把两个源文件之间
原有的跨模块 import 变成同文件内引用。

更新了保留调用方和测试的 import 路径；旧的两个 gripper batch/preview
入口随后在 P4 删除，算法只由 pipeline stage 对外提供。

删除了整个 `src/robotwin_annotation_v2/experiments/` 目录。`pipeline/__init__.py` 导出
新增 25 个 gripper 相关符号。

验证：迁移阶段 `pytest -q` 101 passed；无遗留
`robotwin_annotation_v2.experiments` 代码引用。

## P3：pipeline CLI 接入（完成）

- 新增 `GripperStageResult`、`GripperSeedQualityGateConfig` 和
  `run_gripper_stage()`，封装 pose ROI、候选生成、Qwen QC、一次 native propagation
  与已知物体排除。
- `run_target_receiver.py` 新增 `gripper` / `gripper-batch`；从同一 run 的
  `run_manifest.json` 与 `native_track.npz` 重建 SAM 结果，不重传播 target/receiver。
- `run` 已改为 `qwen → sam → gripper`。
- `save_sam_artifacts(gripper_result=...)` 会原子重写统一四通道 `masks.npz`，并在
  manifest/provenance 中记录 active arm、seed、QC、ROI policy 和诊断文件。

新增 `tests/unit/test_gripper_stage.py`，并扩展 SAM/CLI 测试覆盖四通道和 batch 前置条件。

## P4：render 简化（完成）

- 删除 `--gripper-mask-root`、`_merge_gripper_track()` 和
  `_load_and_merge_gripper()`；render 直接读取统一四通道 `masks.npz`。
- 将 review sheet 逻辑并入 `render_coverage20_videos.py`，默认生成
  target/receiver/gripper 的 early/late 六张图；新增 `--skip-review-sheets`。
- 删除 `build_tracking_review_sheets.py` 和两个旧 gripper batch/preview 脚本。

## P5：一键处理任意目录（完成）

- 新增 `scripts/process_dataset.py`：扫描 parquet，核对 video/sidecar，记录不完整
  episode，实测首条 episode 的 frame shape/raw surplus，并构造内存 manifest。
- DatasetConfig/RoboTwinDataset 支持可选内存 manifest，原有 YAML manifest 行为不变。
- 一键流程复用一个常驻 SAM3 adapter，顺序执行 qwen → sam → gripper，默认再生成 overlay
  和 review sheets。
- `justfile` 新增 `just process <dataset_root> [output_dir]`。

新增 `tests/unit/test_process_dataset.py` 覆盖发现、缺文件、非法 chunk 和动态 manifest。

## P6：文档同步（完成）

`docs/process_data_v2_architecture_design.md` 已标注为 v2 基线并指向 v3；
`README.md`、`QUICKSTART.md` 和 v3 overview 已更新四通道、gripper CLI、一键命令和
自动 review sheets 的用法。

---

## 已知的无关事项

`src/robotwin_annotation_v2/pipeline/qwen_stage.py` 和 `tests/unit/test_qwen_stage.py`
有一份未提交的小改动（`_string_list` 增加 `deduplicate` 参数，用于 `exclude` 字段去重），
这份改动在本次 v3 工作开始前就已经存在于工作区，与 gripper 集成无关，未被本次任何 commit
触碰，仍处于未提交状态。

## 运行验收

真实 smoke episode `7152` 已使用 GPU 7 完成 `qwen → sam → gripper`：

- run id：`v3-smoke-7152-20260806`；
- `masks.npz`：`(4, 138, 240, 320)`，四通道固定为
  `target_0/receiver_0/gripper_left/gripper_right`；
- `gripper_right` 为 active arm，Qwen QC 选择候选 `C`，117 帧非空，target/receiver
  均保持 `status=ok`、QC `passed`；
- render 已生成 overlay MP4，以及 `review_sheets/` 下 target、receiver、gripper
  的 early/late 六张 JPG。

产物目录：

```text
artifacts/runs/v3-smoke-7152-20260806/
artifacts/rendered_videos/v3-smoke-7152-20260806/
```

## 验证

- 全量单元/集成测试：`pytest -q`，109 passed。
- Python 语法检查通过。
- 真实 GPU smoke 和 render 验收已通过；完整 coverage20 批量运行仍需按资源情况另行执行。
