# `process_data_v2` v3 变更设计（面向实现）

> **状态：v3 baseline 已实施；URDF 增量见 v3.1（2026-08-11）。本文是 v2 架构
> （`process_data_v2_architecture_design.md`）之上的
> 增量设计，不重写 v2 已验证的 State Loop / Qwen Semantic Plan / SAM target-receiver
> 部分。v2 文档第 2/13 节中"本版本不做 gripper mask"的边界声明，从本文件生效起视为已被
> 取代。**

本文是给实现者（人或 AI）看的详细契约：改哪些文件、删哪些文件、新格式长什么样、按什么顺序
提交。简短的背景和动机见 `process_data_v3_overview.md`。

> v3 完成后新增的 URDF gripper derived-run backend 不回写本设计的历史决策；其增量架构见
> `process_data_v3_1_architecture_design.md`。

---

## 1. v3 要解决的问题

1. gripper mask 的算法已经写好并在 coverage20 上跑通，但完全游离在标准三阶段 pipeline 之
   外：核心逻辑在 `src/robotwin_annotation_v2/experiments/`，入口是
   `scripts/generate_gripper_mask_video_qwen_qc.py`，自己管理 manifest、自己渲染视频，产
   物是独立目录下的 `gripper_masks.npz`，靠 `render_coverage20_videos.py` 的
   `--gripper-mask-root` 事后合并进最终 `masks.npz`。
2. `scripts/` 和 `tests/` 中积累了多个一次性实验对比脚本（ROI 参数选型），已经没有复用
   价值，只被自己的单元测试引用。
3. 没有一个命令能指向任意 RoboTwin 格式目录、自动发现全部 episode 并一次跑完；当前所有
   命令都要求 episode 落在写死的 `regression_episode_ids` 名单里。

---

## 2. 明确边界（相对 v2 的变化）

v2 第 2 节写的"本版本不做：gripper mask"在 v3 撤销。v3 增加：

- gripper mask 成为 Stage 3 的第三种角色，产物写入同一份四通道 `masks.npz`；
- 一键处理任意 RoboTwin 格式目录（自动发现 episode，不要求预注册 manifest）。

v3 仍然不做：

- 人工选择或确认 mask；
- gripper 的 hidden/amodal 补全（依旧只做 visible-only、pose ROI 裁剪、已知物体排除）；
- 多任务、多相机、动态相机的通用化处理；
- 一键命令里自动拉起/管理 Qwen server 的生命周期（假定用户已经在另一个终端跑
  `just serve-qwen`，一键命令只做 health check，失败即报错退出）。

---

## 3. gripper mask 集成设计

### 3.1 集成深度：浅集成，独立 stage

gripper 作为 `run_target_receiver.py` 的第四个子命令（`gripper` / `gripper-batch`），在
`sam` 阶段之后运行，复用已保存的 target/receiver native track 做已知物体排除。**不**把
`_run_role()` 改造成通用的可插拔 seeder 去同时处理 target/receiver/gripper——三者的 seed
策略差异太大（text query bank vs. pose-ROI + box + text），硬统一只会让 `sam_stage.py` 更
难读，且需要重跑并重新验证已经稳定的 20-episode target/receiver 回归。

```text
LoopContext → SemanticPlan → MaskQCResult → MaskRun(target, receiver)
                                                  │
                                                  ▼ (native tracks 已保存)
                                          GripperStage(pose ROI + Qwen QC)
                                                  │
                                                  ▼
                                   MaskRun(target, receiver, gripper_left, gripper_right)
```

### 3.2 文件搬迁

新建 `src/robotwin_annotation_v2/pipeline/gripper_stage.py`，从以下两个文件原样迁入算法
（不重写核心逻辑，只调整 import 路径和对外接口）：

```text
src/robotwin_annotation_v2/experiments/gripper_pose_roi.py  → pipeline/gripper_stage.py
src/robotwin_annotation_v2/experiments/gripper_seed_qc.py   → pipeline/gripper_stage.py
```

迁入内容包括：

- `CameraCalibration` / `GripperRoiGeometry` / `ProjectedGripperRoi` / `rotation_from_rpy`
  / `project_gripper_roi`（3-D ROI 投影）；
- `ObjectExclusionResult` / `exclude_known_objects` / `GripperTrackResult` /
  `compose_gripper_track`（已知物体排除）；
- `GripperSeedCandidate` / `build_gripper_seed_candidate` /
  `apply_gripper_seed_quality_gate` / `mark_same_frame_duplicates`（候选生成与机械质量
  门）；
- `GripperSeedQCResult` / `build_gripper_qwen_request` / `run_gripper_seed_qc`（Qwen 候选
  选择，复用 `pipeline/mask_qc.py` 的 `parse_mask_qc_response`）；
- `render_gripper_candidate_panel` / `render_gripper_candidate_sheet`（候选可视化，QC 阶段
  产物，非最终渲染）。

迁移后删除整个 `src/robotwin_annotation_v2/experiments/` 目录（包括
`experiments/__init__.py`）。

对外新增一个编排函数，签名对齐 v2 `run_sam_stage` 的风格：

```python
def run_gripper_stage(
    context: LoopContext,
    *,
    backend: Sam3Adapter,
    resource_path: Path,
    frame_shape: tuple[int, int],
    gripper_roi_config: GripperRoiConfig,
    target_native_track: np.ndarray,   # 复用已保存的 target track，不重新传播
    receiver_native_track: np.ndarray,
    qc_client: GripperQwenClient,
    qc_prompt_template: Path,
    qc_max_tokens: int,
    qc_max_attempts: int,
    qc_min_confidence: float,
    seed_quality_gate: GripperSeedQualityGateConfig,
) -> GripperStageResult: ...
```

`GripperStageResult` 携带 `active_arm`、`active_window`、`gripper_track`（bool
`[T,H,W]`）、seed 候选与 QC 记录（用于写 `gripper_seed_qc.json`），风格对齐现有
`SamStageResult`/`RoleMaskData`。

### 3.3 pipeline CLI 改动（`scripts/run_target_receiver.py`）

新增两个子命令：

```text
gripper --config <path> --episode <id> --run-id <id>
gripper-batch --config <path> --run-id <id> [--episode-ids ...] [--force]
```

`gripper`/`gripper-batch` 的前置条件：同一 `run_id` 下该 episode 的 `sam`（target/
receiver）必须已经跑完且 `status=ok`——直接读取已保存的 `native_track.npz`，不重新调用
Qwen semantic 或重新传播 target/receiver。这与现有 `sam-batch` 常驻一个 `Sam3Adapter`、
CUDA 级故障 fail fast、单 episode 失败记录后继续的模式完全一致，`gripper-batch` 复用同一
套 `SAM_EXECUTION_ERRORS`/`_fatal_cuda_error` 逻辑。

`run_pipeline()`（`run` 子命令）扩展为四步：

```python
def run_pipeline(config, episode_index, run_id):
    run_qwen(config, episode_index, run_id)
    run_sam(config, episode_index, run_id)
    run_gripper(config, episode_index, run_id)
```

### 3.4 `masks.npz` 写入方式（关键改动：不再事后 merge）

扩展 `save_sam_artifacts()`（`pipeline/sam_stage.py`）签名，新增可选参数：

```python
def save_sam_artifacts(
    store, run_id, context, semantic_plan, result,
    *, seed_images,
    gripper_result: GripperStageResult | None = None,
) -> MaskRun: ...
```

当 `gripper_result` 为 `None` 时行为不变（`gripper_left`/`gripper_right` 通道仍是
`not_annotated`，兼容只跑到 Stage 3 的旧调用）。当传入时：

- `masks.npz` 的四个通道在同一次写入中全部落盘，不再有独立的 `gripper_masks.npz`；
- `annotation_status[gripper_active_arm] = "valid"`，非活动臂保持 `not_annotated`；
- `frame_provenance.json` 的 `gripper_left`/`gripper_right` 字段从固定的
  `{"status": "not_annotated"}` 改为按实际结果填充（`seed_frame`、`selected_candidate`、
  `qc_status`、`active_window` 等），非活动臂仍为 `not_annotated`；
- `run_manifest.json` 新增 `roi_policy`（prompt/hard ROI 几何）和 gripper QC 摘要字段。

`run_gripper`/`run_gripper_batch` 调用方式：先 `run_sam`（或确认已跑过），再
`run_gripper` 时重新打开同一 episode 的 `run_manifest.json` 读 native track 路径，跑
gripper stage，最后用扩展后的 `save_sam_artifacts` **重写** `masks.npz`（读出已有 target/
receiver 数据 + 新 gripper 数据，一次性写完整四通道），而不是在旧文件基础上打补丁。

### 3.5 render 简化：删除事后 merge

`scripts/render_coverage20_videos.py` 删除：

- `--gripper-mask-root` 参数；
- `_merge_gripper_track()` 与 `_load_and_merge_gripper()` 两个函数（约 150 行）；
- `MaskArtifact` 不再需要区分"原始四通道"和"合并后四通道"两种状态。

render 直接从 `masks.npz` 读 4 个通道，`overlay_frame()` 的角色着色逻辑不变（target 绿、
receiver 蓝、gripper 红，同一套 0.32 alpha + 3px 描边 + 5px 黑边）。

### 3.6 旧 gripper 脚本处置

删除：

```text
scripts/generate_gripper_mask_video_qwen_qc.py
scripts/generate_gripper_mask_video_preview.py
```

两者的算法部分已迁入 `pipeline/gripper_stage.py`；批处理循环、临时目录管理、`sha256`/
`manifest.json` 落盘等脚手架被 `run_target_receiver.py` 的 `gripper-batch` 取代。如果
`episode_gripper_review.mp4`（单角色 review 视频，非最终三色 overlay）仍有诊断价值，其渲
染逻辑可在后续按需迁到一个独立的诊断脚本，本次不强制保留。

### 3.7 测试迁移

```text
tests/unit/test_gripper_pose_roi.py        → 保留，import 路径改为 pipeline.gripper_stage
tests/unit/test_gripper_seed_qc.py         → 保留，import 路径改为 pipeline.gripper_stage；
                                              删除对 scripts.generate_gripper_mask_video_
                                              qwen_qc 的 importlib 依赖
tests/unit/test_gripper_qwen_roi_policy.py → 保留，同上
```

新增：

- `tests/unit/test_gripper_stage.py`：覆盖 `run_gripper_stage()` 编排（fake backend + fake
  Qwen client），对齐 `test_sam_stage.py` 的写法；
- `tests/unit/test_sam_stage.py` 追加：`save_sam_artifacts(gripper_result=...)` 写四通道
  的用例；
- `tests/unit/test_pipeline_cli.py` 追加：`gripper`/`gripper-batch` 子命令解析和前置条件
  校验（缺少 target/receiver 完成态时应报错，不静默跳过）。

---

## 4. 一次性实验脚本清理

删除（结论已固化进文档，可从 Git 历史找回代码）：

```text
scripts/analyze_gripper_roi_variants.py
scripts/render_fixed_gripper_bbox_sweep.py
scripts/render_gripper_roi_comparison.py
tests/unit/test_analyze_gripper_roi_variants.py
tests/unit/test_render_fixed_gripper_bbox_sweep.py
tests/unit/test_render_gripper_roi_comparison.py
```

删除空目录：

```text
tests/contract/    # 只有 __init__.py，从未有过 contract 测试
```

`docs/gripper_pose_roi_coverage20_experiment.md`、`docs/qwen_mask_gripper_fixed_bbox_
experiment.md`、`docs/qwen_mask_gripper.md`、`docs/video_mask_tracking_experiment.md` 四份
实验文档保留不动——它们记录的是选型过程和已验证结论，是 3.1/3.2 节参数取值（如
`front45` ROI profile）的来源依据。

---

## 5. review sheet 并入 render

`scripts/build_tracking_review_sheets.py` 删除，其 `build_sheets()` 逻辑并入
`scripts/render_coverage20_videos.py`：

- render 主流程写完全部 episode 视频和 `manifest.json` 后，自动调用同一份 early/late
  contact sheet 生成逻辑，产物路径不变：
  `<output_dir>/review_sheets/{target,receiver,gripper}_{early,late}.jpg`；
- 新增 `--skip-review-sheets` 开关（默认不跳过），用于不想承担二次解码开销的场景；
- gripper 角色加入后，review sheet 从两组（target/receiver）扩展到三组。

`tests/unit/test_render_coverage20_videos.py` 吸收原 `build_sheets` 的断言；删除单独的
`tests/unit/test_build_tracking_review_sheets.py`（如存在）。

---

## 6. 一键处理任意 RoboTwin 目录

### 6.1 目录发现

新建 `scripts/process_dataset.py`。核心差异于现有 `RoboTwinDataset`：不要求预先存在一份
登记了 `regression_episode_ids` 的 manifest JSON，而是扫描目录结构自动发现 episode：

```text
<dataset_root>/data/chunk-*/episode_*.parquet
```

对每个匹配文件解析出 `episode_index`，与
`<dataset_root>/videos/chunk-*/observation.images.<camera>/episode_*.mp4`、
`<dataset_root>/sidecars/episode_*.hdf5` 交叉核对是否三者都存在；缺失任一文件的 episode
记录为 `discovery_skipped`，不中断整个批次。

发现结果动态构造一份内存态的 manifest 字典（复用 `RoboTwinDataset.__init__` 期望的字段：
`task`、`camera`、`dataset_root`、`frame_shape_hw`、`raw_video_frame_surplus`,
`regression_episode_ids`），不写入磁盘、不要求 `configs/datasets/*.json` 预先存在。
`frame_shape_hw` 和 `raw_video_frame_surplus` 从第一个成功发现的 episode 实测得到，而不是
写死。

### 6.2 命令行为

```text
scripts/process_dataset.py
  --dataset-root <path>       # 必填；RoboTwin 格式目录
  --config <path>             # 默认 configs/pilot_move_pillbottle_pad.yaml，仅取
                                 qwen/sam3/mask/gripper_roi 配置段，忽略其中的 dataset 段
  --task <str>                # 默认取 config.dataset.task
  --camera <str>               # 默认取 config.dataset.camera
  --output-dir <path>          # 默认 artifacts/runs
  --run-id <str>                # 默认自动生成
  --episode-ids <int...>        # 可选，缺省时处理全部扫描发现的 episode
  --skip-render                 # 可选，跳过最终 overlay 视频渲染
```

执行顺序（每个 episode 内部）：

```text
loop → qwen → sam(target/receiver) → gripper
```

跨 episode：复用现有 `sam-batch`/`gripper-batch` 的常驻 `Sam3Adapter` 模式，一个 GPU worker
顺序处理，已完整跑过的 episode 默认跳过（`--force` 强制重跑）。全部 episode 处理完后，除非
传了 `--skip-render`，自动调用 render 逻辑（含 3.5/5 节的四通道读取和 review sheet 生成）。

Qwen server 生命周期不由本脚本管理：开始前做一次 health check，失败直接报错退出并提示
`just serve-qwen`。

### 6.3 `justfile` 新增入口

```text
process dataset_root="../dataset/move_pillbottle_pad_coverage20_original"
        output_dir="artifacts/runs" *process_args:
    @dataset_root="$1"; output_dir="$2"; shift 2; if [ "${output_dir#-}" != "$output_dir" ]; then set -- "$output_dir" "$@"; output_dir="artifacts/runs"; fi; exec env PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}" {{quote(python)}} scripts/process_dataset.py --config {{quote(config)}} --dataset-root "$dataset_root" --output-dir "$output_dir" "$@"
```

`OUTPUT_ROOT` is optional when invoking `just process`; if the token after the dataset root starts
with `-`, it is treated as the first process argument and the output defaults to
`artifacts/runs`. The default backend remains SAM. The URDF derived-run options
(`--gripper-backend urdf`, `--source-run-dir`, `--urdf-path`, explicit `--run-id`, and related
immutable-run flags) are specified in `process_data_v3_1_architecture_design.md`.

原有 `preflight`/`loop`/`qwen`/`sam`/`run` 等分阶段命令保留，用于单 episode debug；`process`
是新增的顶层入口，不替代它们。

### 6.4 测试

- `tests/unit/test_process_dataset.py`：目录扫描发现逻辑（正常目录、缺文件 episode、空
  目录、非法 chunk 命名）；动态 manifest 构造字段校验。
- `tests/integration/test_coverage20_dataset.py` 保留现状，作为"固定 20 条 episode"的回归
  锚点；`process_dataset.py` 在 coverage20 数据集上跑通视为等价验收标准之一（发现的 20 个
  episode id 应与现有 manifest 完全一致）。

---

## 7. 提交顺序

对齐 v2 文档"每完成一个可独立验证的阶段，测试通过再提交"的节奏：

```text
P1  删除 3 个孤立实验脚本 + 对应测试 + tests/contract/         → 测试通过 → commit
P2  gripper 算法迁移：experiments/* → pipeline/gripper_stage.py
    （只搬迁 + 改 import，不改行为）                            → 测试通过 → commit
P3  pipeline 接入：run_target_receiver.py 加 gripper/
    gripper-batch；save_sam_artifacts 扩展写四通道               → 测试通过 → commit
P4  render 简化：删除事后 merge；build_tracking_review_sheets
    并入 render_coverage20_videos.py                             → 测试通过 → commit
P5  一键命令：process_dataset.py + justfile process target        → 测试通过 → commit
P6  文档同步：process_data_v2_architecture_design.md 第 2/13 节
    去掉过时的"不做 gripper mask"声明；README/QUICKSTART/
    DOCS_INDEX 更新                                               → commit
```

P1/P4/P6 风险低。P2 是纯搬迁，不改变已验证行为。P3 是本次核心改动，涉及 `masks.npz` 写入
契约变化，需要在真实 GPU 上重跑至少 smoke episode `7152` 验证四通道产物正确。P5 相对独立，
可以在 P3 完成后随时插入。

删除 `src/robotwin_annotation_v2/experiments/` 前，确认 `.worktrees/` 下的两个实验分支
worktree（`process_data_v2-gripper-text-only-roi`、`process_data_v2-qwen-mask-gripper-roi`）
不依赖主仓库当前的 `experiments/` 模块路径——它们是独立 worktree，各自有自己的文件系统副
本，删除主仓库文件不影响其已存在的副本，但后续如果要把这些实验分支合并回主线，需要同步改
它们的 import 路径。
