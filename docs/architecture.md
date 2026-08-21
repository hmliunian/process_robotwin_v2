# 当前架构与运行契约

> 本文合并原 v2、v3、v3.1 架构设计和实施进度，只描述当前实现中的生效行为。
> 历史实验和数字见 [experiments.md](experiments.md)。
> 2026-08-20 完成的结构重构结果与兼容层清单见
> [refactoring_architecture.md](refactoring_architecture.md) 和
> [refactoring_ai_guide.md](refactoring_ai_guide.md#13-实施结果与兼容层清单)。

## 1. 范围与总流程

输入是 RoboTwin 格式的单次 episode。当前 pipeline 假设：

- 一个 active arm；
- 一个由 annotation mode 声明的 loop：pick-place 为
  approach → close → transport → open，target-only 为 remove/approach → close → hold；
- pick-place 需要一个 `target_0` 和一个 `receiver_0`；target-only 只需要 `target_0`，
  `receiver_0` 保留为全零的 canonical `not_applicable` channel；
- 固定 `cam_high` 视角；
- 输出 visible-only mask，不补全遮挡区域。

默认流程：

```text
RoboTwin episode
  │
  ├─ Stage 1: State Loop
  │    state -> active arm, mode-specific events, role windows, semantic frames
  │
  ├─ Stage 2: Qwen Semantic Plan
  │    sparse RGB + action context -> mode-required object identity, seed, query bank
  │
  ├─ Stage 3: Object SAM
  │    required-role queries -> candidate masks -> Qwen identity QC
  │    -> native propagation -> role-window composition -> temporal QC
  │
  ├─ Gripper backend
  │    ├─ urdf（默认）: joints + calibration + scene depth + visual geometry
  │    └─ sam（显式）: pose ROI + text/box seed + Qwen QC + native propagation
  │
  └─ Canonical publication
       four-channel masks.npz + manifests + overlay + review sheets
```

Stage 之间的核心对象是：

```text
LoopContext -> SemanticPlan -> MaskQCResult -> MaskRun
```

Qwen HTTP、SAM3 session、URDF renderer 和文件写入分别封装在 adapter/stage/publisher 中；
顶层脚本只启动 package API，application 层负责发现、调度和汇总。

## 2. 输入与数据发现

### 2.1 基础目录合同

SAM backend（以及 live URDF 的 object-source 阶段）发现：

```text
<dataset-root>/
  data/chunk-*/episode_<id>.parquet
  videos/chunk-*/observation.images.<camera>/episode_<id>.mp4
  sidecars/episode_<id>.hdf5
```

默认 URDF backend 还要求同一 camera 的 depth：

```text
sidecars/videos/chunk-*/observation.depths.<camera>/episode_<id>.mkv
```

`scripts/process_dataset.py` 从 Parquet 自动发现 episode，并与所需 RGB、sidecar（以及
URDF 模式下的 depth）交叉核对。不完整条目进入 summary 的 discovery/excluded 记录，不能
伪装成已处理。

### 2.2 帧数 authority

Parquet 的连续 `frame_index` 是所有 mask 和 URDF 几何计算的有效帧数 authority。已验证数据
中的 RGB/depth 常比 Parquet 多一个尾帧：

- Qwen、SAM、URDF 和 `masks.npz` 只消费 Parquet 帧；
- depth 尾帧不进入几何计算；
- shared renderer 可以保留 RGB 尾帧，但必须保持无 overlay，并在 manifest 中记录
  `unmasked_trailing_frames`。

不能用视频解码长度扩张 mask 时间轴。

### 2.3 动态 manifest 与固定回归集

一键入口运行时构造内存 manifest，不要求 episode 预先写入
`configs/datasets/*.json`。固定 manifest 仍用于 coverage20 regression、分阶段命令和可重复
验收。动态发现不会修改原数据集。

## 3. Stage 1：State Loop

Stage 1 只读取 metadata、state 和帧数，不判断视觉实例，也不调用 Qwen/SAM。

### 3.1 事件和窗口

pick-place 事件顺序必须满足：

```text
t_move_start <= t_close_start < t_close_done < t_open_start < t_open_done
```

target-only 使用独立三边界合同：

```text
t_remove_start <= t_close_start < t_close_end < frame_count
```

下表语义阶段适用于 pick-place：

| 阶段 | 时间范围 | 含义 |
| --- | --- | --- |
| approach/move | `[move_start, close_start)` | 接近 target |
| close/grasp | `[close_start, close_done]` | 闭合并完成抓取 |
| hold/transport | `(close_done, open_start)` | 移动 target |
| open/release | `[open_start, open_done]` | 释放 target |

pick-place 输出窗口：

```text
loop      = [move_start, open_done]
target    = [move_start, open_start - 1]
receiver  = [close_done, open_done]
gripper   = [move_start, open_done]
```

target 内部再分成两种逐帧编码：普通段 `[move_start, close_done]` 使用 `1`，持有段
`[close_done + 1, open_start - 1]` 使用 `2`；从 `open_start` 起 target 归零。Target-only
没有 release 事件，普通段为 `[remove_start, close_end]`，持有段为
`[close_end + 1, T - 1]`。边界均按 inclusive 处理。事件帧由 detector 从 state 计算，不能为
特定 episode 写死。

### 3.2 语义帧

Stage 1 只抽取少量带原始 frame ID 和用途标签的 RGB：

| purpose | 用途 | seed 资格 |
| --- | --- | --- |
| `pre_grasp_seed_candidate` | target/receiver 在动作前的清晰视图 | target、receiver |
| `post_grasp_context` | 判断哪个物体随 gripper 移动 | 仅上下文 |
| `place_context` | 判断最终直接接触对象 | 仅上下文 |

receiver 可以用动作前帧做 seed，但只在 receiver 输出窗口发布 mask。seed 候选窗口和最终
输出窗口不是同一个概念。

### 3.3 输出与失败

新写 `loop.json` 使用 `robotwin_loop_context_v3`，至少保存 episode/camera、frame count、active
arm、事件、窗口、语义帧、annotation mode 和 state/video source。统一 codec 只读兼容 v1/v2/v3；
兼容读取不会授权新 writer 降级。state 缺失、多个/零合法 loop、事件顺序错误或窗口越界时保存
失败原因，后续阶段不得运行。

## 4. Stage 2：Qwen Semantic Plan

### 4.1 client/server 边界

Qwen server 是独立基础设施，只加载模型并提供 OpenAI-compatible endpoint。
`adapters/qwen_client.py` 只负责 health/completion HTTP transport 和图像编码；prompt 渲染与
response parser 属于各 pipeline stage，artifact persistence 属于 application 层。当前有四个
彼此独立的 Qwen 边界：

| 边界 | owner | 合同 |
| --- | --- | --- |
| semantic plan | `pipeline/qwen_stage.py` | sparse labeled RGB → mode-required role、seed、query bank |
| object visual QC | `pipeline/mask_qc.py` | 实际 SAM contour panel → select/reject/ambiguous |
| bbox localization fallback | `pipeline/bbox_localization.py` + `pipeline/mask_qc.py` | strict bbox prompt/parser + Qwen/SAM execution；全局进入顺序由 object resolver 控制 |
| SAM gripper keyframe QC | `pipeline/gripper/sam/` | 只供显式 SAM gripper backend 使用；服务/合同失败时按已记录的 availability policy 处理 |

pick-place 的 target 和 receiver 在一次 semantic request 中联合判断，避免角色交换或指向同一
实例；target-only request/response 只包含 target。pipeline 内部只检查 endpoint health；
`just process` 的外层 launcher 在 endpoint 不可用时自动选卡并启动本地 server，且只在退出时
回收自己启动的进程。已有健康服务保持外部所有权。

### 4.2 角色语义

- target：随后被 gripper 抓取并移动的物体。
- receiver：任务完成时与 target 直接接触的完整物体或目标区域；不要求承托 target，也不
  要求位于其下方。
- receiver 身份先由 `place_context` 确认，再回到合法 seed 帧中选择同一对象的清晰视图。

### 4.3 query bank 合同

每个角色输出 `status`、`seed_frame_id`、有序 query bank、`exclude` 和语义理由。核心字段：

```json
{
  "status": "ok",
  "seed_frame_id": 0,
  "category_query": "bottle",
  "color_category_query": "orange bottle",
  "shape_category_query": null,
  "general_fallback_query": "container",
  "recommended_order": [
    "category_query",
    "color_category_query",
    "general_fallback_query"
  ]
}
```

约束压缩如下：

- `status=ok` 时 `category_query` 必填；无清晰 seed 时返回 `no_clear_seed` 和空 bank；
- 每条 query 是 1–4 个小写英文词、单数完整物体名词短语；
- 可使用可靠的颜色/形状/材质修饰，但不能只有颜色、形状或 cap/logo 等子部件；
- 禁止位置关系、比较级、动作、OCR/品牌和 `object/thing/item` 等空泛词；
- canonical query bank 中的非空候选必须互异；完全相同的输入候选只做窄 canonicalization；
  general fallback 永远最后；
- `recommended_order` 由 Qwen 按预期分割可靠性排序，而不是按描述长度排序；
- Python/YAML 不写死 `bottle`、`pad` 等任务特定文本；
- schema 不要求 Qwen bbox。历史实验表明 bbox 可能过紧或只覆盖子部件。

semantic parser 要求 exact role/field schema，但允许单层 JSON fence，并对两个已冻结的窄情况
做 canonicalization：合并完全相同的候选；从 `recommended_order` 删除 null field、按字段稳定
顺序补上遗漏的非空候选，再把 general fallback 移到末尾。未知 field、重复 order entry、非法
seed/query 或其他合同错误仍拒绝该角色；parser 不生成新的对象语义。

## 5. Stage 3：mode-required object SAM

### 5.1 seed candidate 和身份 QC

对每个 mode-required role：

1. `no_clear_seed` 直接 rejected；target-only 不创建 receiver candidates；
2. 先按 semantic `recommended_order` 读取最多四个 query；启用 query fallback 时，再追加配置
   驱动、规范化并去重的 curated aliases；
3. 先尝试 semantic plan 选择的 seed；启用 seed fallback 时，再按
   `LoopContext.seed_candidates(role)` 的稳定顺序尝试其他合法 seed；
4. 每个 seed 的实际 proposal 总数不超过 `mask.qc_max_candidates`；必要时加入的 saturated-blue
   planar receiver proposal 也占该上限；
5. 拒绝空 mask、异常面积和机械合同失败项，按 IoU 去除近重复候选，再生成不遮挡纹理的 contour
   panel 交给 Qwen visual QC；
6. `passed` 立即停止；`rejected/ambiguous` 才进入下一合法 seed；request、parser、prompt、候选
   生成或 shape error 立即停止当前角色；
7. 只有所有 text seed attempts 都是 rejected/ambiguous，才按相同 seed 顺序进入可选 bbox
   fallback；Qwen bbox 必须通过 strict parser，原 bbox 生成的 SAM mask 仍经过同一 mechanical
   gate 和 visual QC。

object seed QC 是 fail-closed：全部 attempts 未通过、置信度不足、身份歧义或服务/合同错误都
停止该角色传播。curated alias、seed retry 和 bbox 是显式配置的恢复链路；不得按像素面积自动
选择、合并候选或静默生成任务特定 query。

### 5.2 native video propagation

通过 QC 的 mask 作为一次 native-mask prompt，SAM3 从 seed 向前、向后传播：

```text
final_role_mask[t] = native_track[t], t in role output window
final_role_mask[t] = empty,           otherwise
```

当前组合明确不再使用：

```text
native_track & per-frame text mask & fixed envelope
```

逐帧 text mask 会造成闪烁，固定 envelope 会裁掉移动或遮挡后的合法像素。envelope 只保留为
seed 诊断。native tracker 维持实例身份但不生成 amodal mask；被遮挡部分保持不可见。

`sam-batch` 在一个 worker 进程内复用一个 `Sam3Adapter`，每个 episode 的 video session 和
临时帧独立清理。普通 episode 错误记入 summary 后继续，CUDA 初始化/launch 等致命错误
立即终止 worker。

### 5.3 Temporal QC

每个角色的 `temporal_qc.json` 记录实际执行严格 QC 的 inclusive `window`，以及：

- 输出窗口覆盖率、存在性切换、内部断帧；
- adjacent IoU mean/p05；
- centroid jump p95；
- area-ratio jump p95；
- 相对 seed 的最大质心距离。

默认严重阈值：

```text
adjacent IoU p05 < 0.5
centroid jump p95 > 5 px
area-ratio jump p95 > 0.4
```

IoU、质心、面积三类信号至少两类越界才 `quarantined`；单一信号进入 review，避免将真实
遮挡直接判失败。Temporal QC 只判断连续性：错误 seed 也可能被稳定传播，因此不能取代
candidate identity QC。

target 的严格 temporal QC 只覆盖普通编码段（截至 close 完成）；close 后目标随夹爪搬运会
发生合法的大幅位移，因此 hold 段不参与 quarantine 判定。hold 像素仍照常发布，并在
`frame_provenance.json` 的 target channel 中单独记录窗口覆盖率。

## 6. Gripper backend

### 6.1 公共语义

两种 backend 都只写 active arm 的 visible gripper，inactive channel 全空且为
`not_annotated/not_run`。pick-place active window 为 `[move_start, open_done]`；target-only 为
`[remove_start, T - 1]`。object channel 的像素归属优先于 gripper；任何 producer 都不能在发布
或 render 时偷偷把对象像素并入 gripper。

### 6.2 显式 SAM pose-ROI backend

完整 dataset 入口只允许 pick-place 使用该 backend；target-only 显式请求 SAM gripper 会
fail closed，必须改用默认 URDF 或只运行 object-source 阶段。

状态提供两臂 EEF `xyz + roll/pitch/yaw` 与开合量。TCP 位于 EEF local `+x` 方向
`0.120 m`：

```text
R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
tcp_world = eef_xyz + R @ [0.120, 0, 0]
```

当前固定 ROI profile：

```yaml
gripper_roi:
  prompt:
    axial_back_m: 0.120
    axial_front_m: 0.060
  hard:
    axial_back_m: 0.120
    axial_front_m: 0.045
  fixed_half_width_m: 0.085
```

其他固定值为 `half_thickness_m=0.050`、`margin_px=3`。prompt ROI 比最终 hard ROI 在
指尖方向多 `1.5 cm`，便于 SAM 找到完整 gripper；hard ROI 从 TCP 向后覆盖到 EEF 平面，
避免旧 ROI 只保留指尖，同时不主动延伸到长 forearm。横向宽度不再随开合动态收缩。

正式步骤：

```text
black robot gripper + projected prompt bbox
  -> prompt ROI intersection
  -> exact target/receiver exclusion
  -> mechanical seed gate
  -> Qwen keyframe selection
  -> SAM3 native bidirectional propagation
  -> per-frame hard ROI
  -> exact target/receiver exclusion
```

默认 seed gate：至少 24 px、dark fraction ≥ 0.80、连通域 ≤ 3、最大连通域比例 ≥ 0.80、
距 TCP ≤ 20 px，duplicate IoU 为 0.98。Qwen 正常情况下只在通过 gate 的 text+box keyframe
间选择；若模型/合同失败但已经生成候选，当前 availability policy 会选择 deterministic
fallback，并在 `forced_fallback` 中显式记录。该策略提高批量可用性，不等于像素级 GT。

gripper stage 复用已保存的 target/receiver native tracks，不重跑对象传播。结果与对象通道
一起原子重写同一份 canonical `masks.npz`，不存在独立 `gripper_masks.npz` 的下游合并步骤。

### 6.3 默认 URDF geometry/depth backend

URDF backend 不运行 gripper SAM 或 gripper Qwen QC。每帧算法：

1. 从 Parquet 读取左右六关节和 normalized gripper drive；
2. 根据 HDF5 标定和 depth 独立拟合 active finger `joint7/joint8`；
3. 用 visual meshes、记录的 `cam_high` 标定和 robot camera-Z depth 渲染两臂；
4. active gripper 严格定义为 active-side `link6 | link7 | link8`；
5. 保留满足场景深度一致性的像素：

   ```text
   visible = accepted_component
             & (abs(robot_depth_mm - scene_depth_mm) <= 8 mm)
   ```

6. 只在 active window 发布；inactive arm 始终为空。

normalized opening 是 drive target，不是接触后的真实 finger qpos，因此 depth fitting 是必要
步骤。Z-buffer 处理 self-occlusion，scene depth 去除被 target、receiver、桌面、其他 link
或 clutter 遮挡的几何。不会使用 RGB 分割、颜色阈值、对象 mask subtraction 或时序填充。

自动质量门只统计 rendered/scene depth 都有效的 eligible active frame；默认要求至少 90%
eligible frames 发布非空 mask。完全出画帧不会错误降低该比例。

仓库默认资产：

```text
configs/assets/aloha-agilex/arx5_description_isaac_gripper.urdf
```

它是 render-only 等价资产，保留 follower-arm kinematics 和 renderer 使用的九个 visual
meshes，省略无关 robot branch/collision geometry。可用 `--urdf-path` 和
`--urdf-mesh-root` 显式覆盖，但资产及引用 mesh 都会进入 immutable identity。

### 6.4 live source 与 frozen source

live URDF 总是先冻结 mode-required object source，内部 run ID 后缀固定为 `-object-source`：

```text
depth-complete episode discovery
  -> freeze OUTPUT/_sources/<final-id>-object-source
  -> object Qwen/SAM source
       ├─ independent EGL GPU available: per-episode completion receipt
       │    -> bounded ready queue -> persistent URDF worker（与 source 重叠）
       └─ no independent EGL GPU / --no-urdf-pipeline: source 完成并释放后串行 URDF
  -> canonical publisher + shared renderer
```

默认会尝试 streaming；streaming 时 EGL 必须使用与 SAM 不同的 physical GPU，显式选择同一
GPU 会拒绝，自动选择不到独立设备则退回串行。串行路径在 EGL 启动前释放 SAM，因此可共享
physical GPU。`--urdf-pipeline-buffer-size` 限制 source-ready episode queue，
`--no-urdf-pipeline` 强制串行。streaming source 在推理前写 immutable run contract，并在每个
完整 episode 后原子写 completion receipt；URDF worker 只消费 receipt 校验通过的 episode。

内部 source 只生成 annotation mode 要求的对象，不运行 SAM gripper、不渲染：pick-place 生成
target/receiver；target-only 只生成 target，receiver channel 必须全零且为 `not_applicable`，也
不得存在 receiver role artifacts。live 模式 fresh-only，source 或 final run 已存在即拒绝，不能
用 `--resume`/`--dry-run` 接着跑。

显式 `--source-run-dir` 是 frozen-source 快速路径：不启动 Qwen/SAM，可用于 A/B、dry-run
和 immutable resume。所有 mode-required object role 必须 `annotation_status=valid`、
`qc_status=passed`；非 required receiver 必须是全零 `not_applicable`。loop、帧数、annotation
mode、identity 和引用 artifacts 也必须全部通过校验。

## 7. 运行入口

### 7.1 一键入口

未传 `--gripper-backend` 时，CLI 与 `AnnotationSpec` 都选择 `urdf`：

```bash
just process DATASET_ROOT [OUTPUT_ROOT] [PROCESS_ARGS...]
```

默认 live URDF 仍需先用 Qwen/SAM 生成 mode-required object source，并要求发现同 camera
depth。该入口自动复用健康 Qwen endpoint；若 endpoint 不可用，会排除 SAM 和显式 EGL GPU 后按
空闲显存选卡，等待服务就绪，并在 process 成功、失败或中断后回收服务。分阶段入口仍可用
`just serve-qwen` 手动维持服务。

常用参数：

```text
--config PATH
--task NAME
--camera NAME
--run-id ID
--episode-ids ID...
--force                    # 仅 SAM
--skip-render
--ui {auto,rich,plain,json}
--verbose
```

若第二个 positional token 以 `-` 开头，它会被当作 process 参数，输出根仍为
`artifacts/runs`。

也可以只提供带兼容 `EXTRACT_MANIFEST.json` 的单任务目录或 collection，而不显式传 pipeline
配置：

```bash
just process --data-path DATASET_OR_COLLECTION --pick-place
just process --data-path DATASET_OR_COLLECTION --target-only
```

path 模式分别加载 `configs/pilot_move_pillbottle_pad.yaml` 和
`configs/pilot_adjust_bottle_target_only.yaml` 作为默认推理 profile；数据目录中的 manifest 只
替换 dataset root、task、camera 和 episode ids。两个默认 profile 都启用完整的 S1–S3
open-set object-mask 路径：最多 8 个候选、curated query fallback、多合法 seed fallback、
mode-specific appearance prompt，以及所有文本尝试失败后的 Qwen bbox → SAM box fallback。
因此 collection 中的每个 task 使用同一套 mode profile，不需要逐 task 配置这些开关。

`EXTRACT_MANIFEST.json` 必须显式声明与 mode 匹配的 `profile`。缺少该字段的旧 extract 会
fail closed；此时使用 `just process DATASET_ROOT [OUTPUT_ROOT] --config PROFILE`，不要同时传
`--pick-place/--target-only`。

live URDF：

```bash
just process DATASET_ROOT [OUTPUT_ROOT] \
  --gripper-backend urdf \
  [--run-id NEW_ID] \
  [--episode-ids ID...]
```

frozen-source URDF：

```bash
just process DATASET_ROOT OUTPUT_ROOT \
  --gripper-backend urdf \
  --source-run-dir SOURCE_RUN \
  [--urdf-path ROBOT.urdf] \
  --run-id RUN_ID
```

URDF-only 参数：

```text
--source-run-dir
--urdf-path
--urdf-mesh-root
--urdf-depth-tolerance-mm
--urdf-minimum-eligible-nonempty-fraction
--urdf-fit-config-json
--urdf-egl-device-id GPU       # physical EGL GPU；streaming 时必须与 SAM GPU 不同
--urdf-pipeline-buffer-size N  # live streaming source-ready queue，默认 2
--no-urdf-pipeline             # live 模式强制串行 Source -> URDF
--allow-partial-source
--dry-run                  # frozen source only
--resume                   # frozen source + explicit run ID
```

显式 `--episode-ids` 永远 fail-closed：请求项有一个不合格就拒绝。自动发现若存在 dataset
或 source 排除项，默认同样拒绝；只有显式 `--allow-partial-source` 才处理合格子集，summary
必须记录 selection 是否完整。

### 7.2 分阶段调试

```bash
just preflight
just loop EPISODE_ID
just qwen EPISODE_ID
just sam RUN_ID EPISODE_ID

.venv/bin/python scripts/run_target_receiver.py gripper \
  --config CONFIG --episode EPISODE_ID --run-id RUN_ID

.venv/bin/python scripts/run_target_receiver.py sam-batch \
  --config CONFIG --run-id RUN_ID --episode-ids ID...

.venv/bin/python scripts/run_target_receiver.py gripper-batch \
  --config CONFIG --run-id RUN_ID --episode-ids ID...
```

`run` 子命令按 `qwen -> sam -> gripper` 运行单 episode。`gripper` 前置要求同一 run 的
mode-required object SAM 已完成且 QC passed。

配置里的 `sam3.gpus` 使用物理 GPU index 时，不要同时用 `CUDA_VISIBLE_DEVICES` 把同一设备
重新映射为 logical 0。

### 7.3 依赖

核心依赖只覆盖数据读取和通用 CLI；SAM3 与 URDF 分为 extras。URDF 环境：

```bash
uv sync --extra urdf
```

对必须保留 SAM packages 的既有环境，避免同步时裁剪未选择 extras，可改用：

```bash
uv pip install --python /absolute/path/to/python -e '.[urdf]'
```

安装 Python 依赖不会下载或替换数据集/模型/URDF 资产。URDF frozen-source 模式通过 lazy
import 不要求 torch/SAM/OpenCV；live 模式仍需要对象 Qwen/SAM runtime。

## 8. 公共产物契约

### 8.1 run 布局

```text
<output-root>/<run-id>/
  process_summary.json
  <task>/episode_<id>/<camera>/
    loop.json
    semantic_plan.json
    mask_qc.json
    masks.npz
    run_manifest.json
    frame_provenance.json
    target_0/
      seed.rgb.png
      seed.mask.png
      native_track.npz
      temporal_qc.json
      ...
    receiver_0/...                 # pick-place only；target-only 仅保留 canonical N/A channel
    gripper_<active-arm>/...
  rendered_videos/
    manifest.json
    episode_*_overlay.mp4
    review_sheets/
      target_early.jpg
      target_late.jpg
      receiver_early.jpg           # applicable roles only
      receiver_late.jpg
      gripper_early.jpg
      gripper_late.jpg
```

URDF run 可增加 `<run>/_backend/urdf/`，它只用于中间产品与审计，不能成为下游读取 mask
的前置条件。

### 8.2 `masks.npz`

新生成的 canonical NPZ 使用 `robotwin_visible_masks_v3`，严格包含八个 key：

```text
format_version
frame_count
masks
instance_names
roles
annotation_status
qc_status
frame_encoding
```

`masks` 是 bool `[4,T,H,W]`；顺序固定：

| index | instance | role |
| ---: | --- | --- |
| 0 | `target_0` | target |
| 1 | `receiver_0` | receiver |
| 2 | `gripper_left` | gripper |
| 3 | `gripper_right` | gripper |

`frame_encoding` 是 `uint8 [4,T]`，只描述当前 instance/frame 的渲染语义，不替代 bool
像素 mask：

| code | 含义 |
| ---: | --- |
| 0 | 当前帧 mask 为空 |
| 1 | 普通可见 mask |
| 2 | close 完成后、open 开始前的 held target；只允许出现在 `target_0` |

旧 `robotwin_visible_masks_v2` 七键产物仍可读取，loader 会按非空帧合成 `0/1` 编码；新写入
和 URDF canonical publication 一律输出 v3。

未运行的 gripper channel 是全零并标记 `not_annotated/not_run`，不能被下游当作负样本。被
temporal QC 隔离的对象标记 `quarantined`，不得为了覆盖率发布已知坏像素。

SAM 和 URDF 的 public contract 相同。URDF publisher 固定执行：source object 两通道逐像素
复制、source 旧 gripper 两通道丢弃、只写 active URDF channel、重新构造 public
manifest/provenance。render-time merge 已被删除。

### 8.3 manifest、provenance 和 summary

`run_manifest.json` 记录 episode、role artifacts、算法和 QC；
`frame_provenance.json` 记录每个 channel 的 producer、seed/window/backend；
`process_summary.json` 的公共顶层包括：

```text
format_version, gripper_backend, run_id, dataset_root, task, camera,
discovered_episode_ids, requested_episode_ids, dynamic_manifest,
qwen_health, records, render, fatal_error, backend, passed
```

两种 backend 的 `gripper_qc` 使用同一字段集合：

```text
backend, status, qc_status, active_arm, selected_candidate,
confidence, reason, forced_fallback, nonempty_frames, quality
```

SAM 填 candidate/confidence/fallback；URDF 将这些视觉字段设为 `null/false`，几何质量放在
`quality`。下游不应根据 backend 猜测字段是否存在。

### 8.4 shared renderer

当前 canonical 路径先用 strict codec 只读 `masks.npz`，不修复 mask；strict 读取拒绝时，仅为
已登记的历史 archive 启用受控 read-only compatibility fallback，该 fallback 不是新 schema 或
writer authority。默认：内部填充 alpha 0.32、mask 外侧 3 px 角色色轮廓、扩张到总计 5 px 的
黑色 halo。处理完为 applicable 的 target/receiver/gripper 自动生成 early/late review sheet；
pick-place 最多六张，target-only 不生成 receiver sheet。`--skip-review-sheets` 可关闭二次解码。

普通 target 使用 `(36, 180, 92)`，`frame_encoding=2` 的 held target 使用黄色
`(255, 255, 0)`；shared renderer 与 URDF standalone renderer 共用这一约定。

实验或发布审查应固定 exact run ID，避免“最佳 run”选择器回退到旧结果。

## 9. URDF lineage、原子发布与 resume

### 9.1 source lineage

`SourceLineageValidator` 同时接受两种 derived-episode lineage；版本由 source 是否存在 immutable
`source_run_contract.json` 决定：

- `robotwin_derivation_source_lineage_v1`：source 不含 `source_run_contract.json` 时使用，包括
  串行 live source 和历史 frozen source；锚定 source `process_summary.json`、四个 control
  artifacts 及 manifest 实际引用的 role artifacts；
- `robotwin_derivation_source_lineage_v2`：source 含 immutable run contract 时使用，当前由
  streaming source 生成；锚定 source run contract、完整 episode artifact identity 和原子
  completion receipt。receipt 是 episode 可交给 URDF worker 的 readiness gate，不由目录存在性
  替代。

两版都记录 source run/dataset/task/camera/episode/frame identity，以及每个引用 regular file 的
run-relative path、SHA-256 和 byte size。publisher implementation identity 由 private URDF run
contract 和最终 canonical derivation 另行锚定，不混入 source lineage。

引用路径必须位于 source run 内且为 regular file；symlink、目录逃逸、缺失或 hash/size 不符
都 fail closed。source `loop.json` 是 URDF event authority；URDF 不从 Parquet 重新猜事件。

### 9.2 immutable run contract

URDF private run contract 固定 dataset inputs、source lineage、episode plan、URDF/mesh、fit
config、阈值以及 runner/data/renderer/publisher 实现文件 identity。任一 source、输入、资产、
配置或实现文件变化，旧 run 不可 resume。

验证边界：

```text
source preflight + lineage snapshot
  -> private URDF render
  -> publish 前重验 source/backend
  -> staging tree 全量验证 + atomic rename
  -> shared render 前重验 canonical tree
  -> overlay/review
```

canonical episode 先写同父目录 staging tree，文件集合、JSON、hash 和八键 masks 全部通过后
原子 rename。非 resume 不覆盖既有 destination；resume 也不“修复”或重写被篡改的 complete
episode。

只有明确表示“episode loop 已结束、部分条目失败且 checkpoint 完整”的
`UrdfBatchIncompleteError` 可以带着成功子集继续发布；contract/tamper、配置、I/O 和编程
错误原样向上抛出，不能回读旧 manifest 掩盖失败。

## 10. 配置与模块边界

主配置段：

```yaml
dataset: {root, manifest, task, camera, smoke_episode_ids, regression_episode_ids}
qwen: {endpoint, model, prompt_template, timeout_seconds, max_tokens}
sam3: {checkpoint, gpus}
mask:
  temporal_qc_min_adjacent_iou_p05: 0.5
  temporal_qc_max_centroid_jump_p95_px: 5.0
  temporal_qc_max_area_ratio_jump_p95: 0.4
  temporal_qc_quarantine_signal_count: 2
  qc_enabled: true
  qc_max_candidates: 8
  qc_query_fallback_enabled: true
  qc_seed_fallback_enabled: true
  qc_bbox_fallback_enabled: true
  qc_min_confidence: 0.70
gripper_roi:
  prompt: {axial_back_m: 0.120, axial_front_m: 0.060}
  hard: {axial_back_m: 0.120, axial_front_m: 0.045}
  fixed_half_width_m: 0.085
output: {root}
```

关键代码职责：

```text
scripts/process_dataset.py                    薄 CLI/兼容启动入口
scripts/run_target_receiver.py                薄分阶段 CLI/兼容启动入口
scripts/render_coverage20_videos.py            薄 canonical renderer 启动入口
scripts/render_urdf_gripper_masks.py           薄 URDF batch 启动入口

src/robotwin_annotation_v2/application/
  dataset_pipeline.py                         typed public facade、SAM convenience API 与 backend dispatcher
  episode_pipeline.py                         stage 实现/分阶段 CLI seam 与单 episode executors
  sam_workflow.py                             正式 SAM dataset/per-episode 顺序、resume 与失败策略 owner
  urdf_workflow.py                            frozen/live URDF 用例编排
  urdf_runtime.py                             URDF runner、GPU/EGL handoff、spawn/streaming IPC owner
  urdf_batch.py                               package-owned private URDF batch engine
  mask_qc_artifacts.py                        object-mask QC diagnostics publication
  dataset_runtime.py                          CLI/path-mode 参数、hook wiring 与旧调用兼容层
  sam_artifacts.py                            SAM 结果到 canonical publication

src/robotwin_annotation_v2/pipeline/
  timeline_detector.py                        唯一当前 timeline detector
  state_loop.py                               LoopContext 与 semantic frame 构造
  qwen_stage.py                               semantic prompt/render/parser owner
  bbox_localization.py                        strict bbox prompt/parser contract
  mask_qc.py                                  object visual-QC 与 bbox-localization execution boundary
  object_mask/resolver.py                     text -> legal seed -> bbox 唯一顺序 owner
  object_mask/{planner,proposals,qc}.py        query/proposal/mechanical QC
  sam_stage.py                                object propagation 纯 stage 结果
  gripper/sam/                                pose-ROI SAM gripper 分层实现

src/robotwin_annotation_v2/adapters/
  canonical_masks.py                          canonical v2 reader/v3 DTO/validator
  canonical_publication.py                    SAM/URDF 共用的 v3 原子 NPZ publisher
  loop_context_codec.py                       loop v1/v2/v3 读取与 v3 当前语义
  rendering.py                                package-owned public renderer

src/robotwin_annotation_v2/urdf_gripper_publisher.py
                                                SourceLineageValidator 与
                                                UrdfCanonicalEpisodePublisher owner
```

正式 `just process` 的当前调用图是：

```text
manage_qwen_process.py -> scripts/process_dataset.py -> dataset_runtime._run_from_args
  ├─ SAM -> DatasetPipeline.run dispatch -> process_dataset compatibility adapter
  │          -> DatasetPipeline.run dispatch -> SamWorkflow.run
  │          -> episode_pipeline module-level executors
  ├─ live URDF -> DatasetPipeline dispatch -> UrdfWorkflow.run_live
  │          -> urdf_runtime streaming/serial workers
  │          -> process_urdf_source_run compatibility adapter -> UrdfWorkflow.run
  └─ frozen URDF -> DatasetPipeline dispatch -> UrdfWorkflow.run
             -> urdf_batch -> UrdfCanonicalEpisodePublisher -> shared renderer
```

`EpisodePipeline` class 仍是分阶段 CLI 使用的单 episode facade；正式 dataset run 的
Qwen → object SAM → optional SAM gripper 顺序由 `SamWorkflow` 持有。`DatasetPipeline.run()` 只按
typed `ProcessRequest` 选择注入的 backend runner，不重复实现 SAM/URDF 生命周期。

`dataset_runtime.py`、旧私有 renderer 名称、timeline type alias 和 publisher 函数入口仍为受控
兼容面，不是第二套事实来源；剩余调用方与删除条件记录在
[refactoring_ai_guide.md](refactoring_ai_guide.md#132-compatibility-inventory)。

## 11. 验证与明确非目标

最低验证顺序：

```bash
just test
just test-all
just lint
git diff --check
```

2026-08-20 的结构重构 CPU/static 验收记录：`just test` 为 662 passed，`just test-all` 为
669 passed，`just lint` 通过，
`.venv/bin/python -m mypy src` 为 0 issues，`git diff --check` 通过。该结果覆盖 unit、filesystem/
dataset contract、fake backend、resume/tamper、import boundary 和 schema 测试；它不等于重新运行
Qwen、SAM3、CUDA/EGL、URDF 实际数据或人工 overlay QC。

涉及真实 backend 时再做：单 episode smoke → 左右臂各一条 → coverage subset → full batch →
exact-run overlay/review。URDF 还需核对 source object channels 逐像素相同、inactive arm 全空、
八键 NPZ、lineage 和 resume tamper tests。

2026-08-20 已新增重构后真实默认 URDF 单 episode 证据：pick-place episode 7152（right arm）和
target-only episode 0（left arm）分别通过 direct CLI 与正式 `just process`，四个 run 均
`passed=true`。正式入口结果的严格八键四通道 NPZ、source object 逐像素一致、inactive arm
全零、target-only receiver `not_applicable`、publisher validator、lineage/provenance、默认 URDF
资产 identity 与 overlay/review 均通过。exact run ID 见
[refactoring_ai_guide.md](refactoring_ai_guide.md#21-实施完成快照)。该结果只完成单 episode/左右臂
smoke；文档中既有 coverage20 20/20 仍是重构前证据，coverage subset、full batch 与像素级人工
签字仍须按上一段单独执行。

当前不保证：

- 多 target、多 receiver、多次抓取循环或双臂协同任务；
- 动态 receiver、articulated drawer 等额外状态；
- wrist/dynamic camera 正式支持；
- hidden/amodal object 或 gripper mask；
- 像素级 ground-truth accuracy；
- 自动修正稳定跟错实例；
- 自动下载数据集、模型或外部 RoboTwin assets。

active-wrist 的 close/open phase-seed 方案仍是未实施实验，不得将其 seed window、full-video
QC 或 no-gripper profile当作当前 `cam_high` pipeline 的行为。

## 12. 全量数据集兼容性与 target 基数

### 12.1 统计口径

本节扫描日期为 2026-08-11，统计对象是完整 RoboTwin 2.0 数据集：50 个 coarse task、
27,500 个 episode，每个 task 固定包含 50 条 clean 和 500 条 randomized。coarse task 以
`meta/episodes.jsonl` 的 `full_structured_tasks[0]` 为 authority；事件结构另外审计了全部
27,500 个 Parquet 的 `observation.state`。

“直接兼容当前 pick-and-place pipeline”同时要求：

1. 一个独立、可移动的刚体 target；
2. 一个稳定且可识别、最终与 target 直接接触的 receiver 或目标区域；
3. 单 active arm、一次完整 grasp → transport → release；
4. 不需要额外受控实体、动态/articulated receiver 或任务专用 outcome。

“单 target”采用更宽但明确的实体口径：统计被机器人直接控制并主动改变 pose 的独立
root-level movable entity。静态 reference/receiver 和 fixed-root articulated link 不计；同一
物体被 handover 或双臂共同夹持仍计一个；后续又被抓起移动的 receiver 计另一个 target；
容器内随容器被动运动的 payload 不逐个计数。因此“单 target”只是数据模型复杂度，不等于
满足当前 pick-and-place 时间线或 receiver 语义。

### 12.2 当前 pick-and-place 覆盖

保守的 full-task 兼容集是 **10/50 个 task、5,500/27,500 个 episode（20.00%）**：

```text
move_pillbottle_pad
move_stapler_pad
place_container_plate
place_empty_cup
place_fan
place_mouse_pad
place_object_scale
place_object_stand
place_phone_stand
place_shoe
```

其中 `move_pillbottle_pad` 是现有 pilot；[datasets.md](datasets.md) 中列出的 9 类是“pilot
之外可直接扩展”的任务，不能把 9 误写成全数据集兼容总数。这 5,500 条全部通过一次完整
loop 且双手最终 open 的 state 检查；其中 5,401 条 `geometry_valid=true`。geometry flag 只
表示 corrected replay geometry 可用，不替代 backend 自己的 depth/discovery 检查。

`place_bread_basket` 是唯一的 episode-level 混合边界：144 条为单 bread、406 条为双 bread。
144 条单 bread episode 均通过完整 loop 和 geometry 检查，可以在显式预筛选后条件纳入。
因此 episode 级的条件上限是 **5,644 条（20.52%）**，其中 5,545 条 geometry valid；不能把
整个 `place_bread_basket` task 标为直接兼容。

### 12.3 target 基数

| 实际 target 数 | task 组成 | episode | 全集占比 |
| --- | --- | ---: | ---: |
| 1 | 27 个固定单 target task + `place_bread_basket` 的 144 条 | **14,994** | **54.52%** |
| 2 | 11 个固定双 target task + `place_bread_basket` 的 406 条 | **6,456** | **23.48%** |
| 3 | 5 个固定三 target task | **2,750** | **10.00%** |
| 0 | 6 个 fixed-root contact/articulation task | **3,300** | **12.00%** |
| 合计 | 50 个 task | **27,500** | **100.00%** |

task-level 可概括为 27 个 always-single、17 个 multi-capable（含一个 1–2 可变 task）和
6 个没有独立 movable target 的 task。多 target 合计 9,206 条（33.48%）。

27 个 always-single task：

```text
adjust_bottle              beat_block_hammer          dump_bin_bigbin
grab_roller                handover_block              handover_mic
hanging_mug                lift_pot                    move_can_pot
move_pillbottle_pad        move_playingcard_away       move_stapler_pad
place_a2b_left             place_a2b_right             place_container_plate
place_empty_cup            place_fan                   place_mouse_pad
place_object_scale         place_object_stand          place_phone_stand
place_shoe                 put_object_cabinet          rotate_qrcode
shake_bottle               shake_bottle_horizontally   stamp_seal
```

17 个 multi-capable task：

- 固定 2 target：`pick_diverse_bottles`、`pick_dual_bottles`、`place_bread_skillet`、
  `place_burger_fries`、`place_can_basket`、`place_cans_plasticbox`、`place_dual_shoes`、
  `place_object_basket`、`scan_object`、`stack_blocks_two`、`stack_bowls_two`；
- 固定 3 target：`blocks_ranking_rgb`、`blocks_ranking_size`、`put_bottles_dustbin`、
  `stack_blocks_three`、`stack_bowls_three`；
- 1–2 target 可变：`place_bread_basket`。

6 个没有独立 movable target 的 task：`click_alarmclock`、`click_bell`、`open_laptop`、
`open_microwave`、`press_stapler`、`turn_switch`。它们仍有 task entity，但需要
`parent + part/link/action_site`，不能强塞进当前 rigid `target_0`。

### 12.4 为什么 state loop 不能单独判兼容

全量 state 审计得到：

| gate | episode | 说明 |
| --- | ---: | --- |
| 当前 detector 返回 exactly-one complete loop | 11,969 | 只累计完整 loop |
| 再要求 episode 结束时双手都 open | 8,944 | 去掉 3,025 条未结束动作 |
| full-task 语义也满足当前 pick-and-place 合同 | 5,500 | 保守直接兼容集 |
| 再加入预筛后的单 bread 子集 | 5,644 | episode-level 条件集 |

严格的双手终态 gate 仍包含 3,300 条语义假阳性：`move_can_pot`、
`move_playingcard_away`、`place_a2b_left/right`、`rotate_qrcode` 和 `stamp_seal`。它们的 state
形状像一次 pick-and-place，但终点是相对区域、重新定向或接触动作，不满足当前“最终直接
接触 receiver”合同。

另外，`detect_arm_loops()` 遇到后续不完整 close 会停止扫描并保留此前完整 loop，
`detect_episode_loop()` 只检查已收集的完整 candidate 数量。这会让另一只手或同一只手在
episode 尾部仍 closed 的 3,025 条样本通过 exactly-one 检查。state detector 应继续作为事件
候选 gate，但 task profile、实体基数和终态检查必须是独立的 compatibility gate。

### 12.5 对可扩展架构的直接要求

全量任务不能只用 `receiver: optional` 区分。最小可扩展 profile 至少应覆盖：

| profile | 典型任务 | 需要新增的合同 |
| --- | --- | --- |
| `pick_place` | 当前 10 类 + 单 bread 子集 | 当前 target/receiver + 单 loop |
| `relative_place` | `move_can_pot`、`place_a2b_*` | reference、relation、goal region |
| `grasp_hold/reorient` | `adjust_bottle`、`shake_*`、`rotate_qrcode` | open-ended window、pose/trajectory outcome |
| `multi_entity` | `pick_*`、stack/ranking、双物体 place | 动态 entity/channel、并行或连续事件 |
| `multi_stage_place` | handover、动态 basket/skillet、cabinet | 多 effector、多 segment、动态/articulated receiver |
| `tool_contact` | hammer、stamp、scan、click/press | tool、patient/action-site、contact outcome |
| `articulate` | `open_*`、`turn_switch` | parent/link/handle、joint/visual outcome |

因此 14,994 条单-target episode 中，只有 5,500 条可按 full task 直接进入当前 pipeline；
其余单-target 数据仍需要新的 timeline、entity role 或 outcome contract。
