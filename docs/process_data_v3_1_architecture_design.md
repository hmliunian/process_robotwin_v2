# `process_data_v2` v3.1 架构设计：URDF gripper backend

> **状态：实验分支实施中。** 本文是
> `process_data_v3_architecture_design.md` 的增量设计。v3 的视觉 pipeline、四通道
> `masks.npz` 和一键入口仍是基线；v3.1 只增加一个从冻结 source run 派生 URDF gripper
> channel 的 backend。

## 1. 目标与非目标

v3.1 的目标是让同一条 `just process` 命令支持两种 gripper producer，并让 downstream
只面对一种 public artifact contract：

| 契约 | `sam` backend（默认） | `urdf` backend |
|---|---|---|
| target/receiver | 当前 run 内由 Qwen + SAM 生成 | 从冻结 source run 逐像素复用 |
| gripper | v3 视觉 pose-ROI + SAM + Qwen QC | 关节状态 + 标定 + depth + URDF visual geometry |
| loop authority | 当前 run 的 `loop.json` | 冻结 source episode 的同一 `loop.json` |
| Qwen/SAM runtime | 需要 | 不加载 |
| public `masks.npz` | 七键、四通道 | 同一七键、同一通道顺序 |
| `run_manifest.json` / provenance | canonical v3 schema | 同一 schema，增加可审计 derivation |
| `gripper_qc` | shared schema，`backend=sam` | 同一字段集合，`backend=urdf` |
| overlay/review | shared renderer | 同一 shared renderer |
| backend 差异 | provenance 中为 `sam` | provenance 中为 `urdf` |

v3.1 不做：

- 不用 URDF 生成 target/receiver mask；
- 不重写 v3 已验证的视觉 gripper stage；
- 不把视觉 seeder 和几何 renderer 硬抽象成同一套算法内部接口；
- 不做 hidden/amodal gripper 补全，只发布 RGB 中可见、通过 depth occlusion 的像素；
- 不自动下载数据集、URDF 或 mesh；输入必须来自用户指定的本地路径；
- 不在 URDF-only 命令里启动 Qwen server 或加载 SAM/OpenCV extra。

## 2. 核心决策：derived-run，而不是第二套公开 pipeline

用户已经有一份 target/receiver 通过 QC 的 canonical run。URDF 模式只替换其 gripper
producer，因此定义为显式的 **derived-run/import mode**：

```text
                                   ┌─ sam  → v3 live visual gripper stage
dataset discovery → object masks ──┤
                                   └─ urdf → frozen source target/receiver
                                              + joints/calibration/depth/URDF
                                                        │
                                                        ▼
                               canonical four-channel run + shared renderer
```

这保留 v3 3.1 节“浅集成、独立 stage”的原则。统一的是 stage 的输入/输出语义、public
artifact contract、summary 和 renderer；两种算法的内部实现不强行统一。

`urdf` 模式不声称执行了 `loop → qwen → sam(target/receiver)`。它把 source run 已发布的
`loop.json` 作为 episode 的 authoritative loop，把 QC-passed target/receiver 作为派生输入，
并在 public manifest/provenance 中记录完整、可内容寻址的 source lineage。继承数据不得被
表述为当前 run 重新计算的结果。

## 3. 组件边界

```text
justfile
  └─ scripts/process_dataset.py       # discovery、backend dispatch、summary、shared render
       ├─ sam runtime (lazy import)    # 原 v3 live pipeline
       └─ URDF derived-run
            ├─ URDF data loader        # parquet/HDF5/depth/calibration
            ├─ URDF geometry renderer  # visual mesh + depth visibility
            ├─ canonical publisher     # source object channels + new gripper channel
            └─ shared renderer         # final MP4 + six review sheets
```

文件职责：

- `scripts/process_dataset.py`：唯一顶层 batch 入口；选择 backend、构造统一 summary，并只调用
  一次最终 renderer；URDF 模式在调用 shared renderer 前再次验证每个已发布 episode；
- `src/robotwin_annotation_v2/urdf_gripper_data.py`：读取 RoboTwin episode 的状态、标定和
  depth，并严格加载 source `robotwin_loop_context_v1`，不写 public artifact；
- `src/robotwin_annotation_v2/urdf_gripper_renderer.py`：URDF visual geometry 的纯计算与
  visibility，不管理 run 目录；
- `scripts/render_urdf_gripper_masks.py`：现阶段的内部 URDF batch engine。其 overlay/review
  必须关闭，输出只能位于 private backend tree；它把 source lineage、输入、资产和实现 identity
  固定在 immutable run contract 中；
- `src/robotwin_annotation_v2/urdf_gripper_publisher.py`：source contract 和 derivation lineage
  的唯一 owner；验证 source/backend identity，构造 public manifest/provenance，原子发布
  canonical episode，并在 resume 与 render 前做严格内容校验；
- `scripts/render_coverage20_videos.py`：两种 backend 唯一的 public overlay/review renderer。

后续若继续产品化，可把 URDF batch engine 的编排从 `scripts/` 下沉到 `src/`；这不改变本文
定义的 public contract，也不是本轮 20-episode 验收的前置条件。

## 4. CLI 与兼容性

入口保持：

```bash
just process DATASET_ROOT [OUTPUT_ROOT] [PROCESS_ARGS...]
```

不带额外参数时必须保持 v3 原行为。新增参数：

```text
--gripper-backend {sam,urdf}       # 默认 sam
--source-run-dir <path>            # urdf 必填
--urdf-path <path>                 # urdf 必填
--urdf-mesh-root <path>            # 可选；默认按 URDF 相对路径解析
--urdf-depth-tolerance-mm <float>
--urdf-minimum-eligible-nonempty-fraction <float>
--urdf-fit-config-json <path>
--allow-partial-source
--dry-run                           # 仅 urdf
--resume                            # 仅 urdf，且要求显式 --run-id
```

约束：

- `sam` 模式传入 URDF-only 参数必须报错；
- `urdf` 模式缺少 source run 或 URDF 必须报错；
- `--force` 不适用于 immutable derived-run；主动重跑使用新 run id；
- `--dry-run` 与 `--resume` 互斥；resume 必须显式提供原 run id；
- URDF run id 必须是 simple directory name，禁止空值、路径分隔符、`.`/`..` 和 parent
  traversal；非 resume 的新 run 不得复用已经存在的 canonical run 目录；
- 显式 `--episode-ids` 必须 fail closed，任何一个不合格都整体拒绝；
- 自动发现模式默认也 fail closed。只有显式 `--allow-partial-source` 才可处理合格子集；
- optional positional `OUTPUT_ROOT` 省略且下一个 token 以 `-` 开头时，`just` recipe 必须仍把
  该 token 当作 process 参数，默认输出到 `artifacts/runs`。

默认 SAM 行为不增加参数：

```bash
just process /absolute/path/to/DATASET
just process /absolute/path/to/DATASET /absolute/path/to/OUTPUT_ROOT \
  --episode-ids 7 8
```

URDF derived-run 使用同一 recipe，只增加 backend 参数：

```bash
just process /absolute/path/to/DATASET /absolute/path/to/OUTPUT_ROOT \
  --gripper-backend urdf \
  --source-run-dir /absolute/path/to/SOURCE_RUN \
  --urdf-path /absolute/path/to/ROBOT.urdf \
  --run-id RUN_ID \
  --episode-ids 7 8
```

恢复必须重复同一 dataset/source/assets/config/episode 集合，并显式给出原 run id：

```bash
just process /absolute/path/to/DATASET /absolute/path/to/OUTPUT_ROOT \
  --gripper-backend urdf \
  --source-run-dir /absolute/path/to/SOURCE_RUN \
  --urdf-path /absolute/path/to/ROBOT.urdf \
  --run-id RUN_ID --resume \
  --episode-ids 7 8
```

这些命令只读取用户给出的本地 dataset/source/URDF/mesh；recipe 和 Python 入口都不得隐式下载
或复制一份新数据集。

## 5. Source contract、authoritative loop 与 lineage

### 5.1 Source `loop.json` 是唯一事件 authority

URDF backend 不得根据 Parquet action/state 再推断抓取事件。它必须读取 source episode 的
`loop.json`，并严格验证：

- `format_version=robotwin_loop_context_v1`；
- task、episode index/id、camera 与请求完全一致；
- `frame_count` 与 Parquet/source masks 一致；
- `active_arm` 只能是 `left` 或 `right`；
- `t_move_start <= t_close_start < t_close_done < t_open_start < t_open_done`，且窗口不越界；
- `windows.loop/target_0/receiver_0` 必须分别与 events 导出的 active、target、receiver 窗口
  完全一致；
- `sources.state/video` 必须指向本次 dataset 中同一 episode/camera。

这个 loop 同时决定 URDF active arm/window、target/receiver source role window 和 published
`loop.json`。Parquet 仍是 joint state 与 usable frame count 的 authority，但不是事件边界的
authority。缺少、格式错误或与其他 source artifact 不一致时必须 fail closed，不能退回启发式
推断。

### 5.2 Publisher-centric source lineage v1

source 验证与 lineage 构造只在 canonical publisher 中实现一份。每个 episode 生成
`robotwin_derivation_source_lineage_v1`，至少固定：

```text
source_run
  run_id, path, dataset_root
  process_summary {path, sha256, bytes}
episode
  task, episode_index, episode_id, camera
frame_count, frame_shape_hw
control_artifacts
  loop, run_manifest, frame_provenance, masks
    {path, sha256, bytes}
role_artifacts
  target, receiver
    seed_rgb_path / seed_mask_path / canonical_envelope_path /
    native_track_path / temporal_qc_path
      {path, sha256, bytes}           # 仅记录 source manifest 实际引用的项
lineage_sha256                        # 上述对象 canonical JSON 的摘要
```

所有 path 必须位于 source run 内、指向 regular file，禁止 symlink 和目录逃逸；`sha256` 与
`bytes` 缺一不可。source process summary、loop、episode manifest、frame provenance、
`masks.npz` 以及 target/receiver manifest 引用的 role artifacts 因而都被内容寻址，而不是只
记录一个可变的绝对路径。

publisher 还记录自身的 dirty-worktree-safe implementation identity（格式版本、实现文件相对
路径、`sha256`、`bytes`）。public `run_manifest.json.derivation` 必须同时嵌入完整 source
lineage 和 publisher identity，使任何继承像素和生成规则都能追溯到具体内容。

### 5.3 Episode eligibility

一个 episode 只有同时满足下列条件才能进入 URDF engine：

1. dataset 中存在 Parquet、RGB MP4、HDF5 sidecar 和对应 camera 的 depth video；
2. Parquet `frame_index` 连续且 frame count 与 authoritative loop/source `masks.npz` 一致；
3. source summary、episode path、loop、manifest、provenance 的 run/task/camera/episode identity
   一致；
4. source `masks.npz` 严格包含 canonical 七键，shape 为 bool `[4,T,H,W]`，通道固定为
   `target_0, receiver_0, gripper_left, gripper_right`；
5. target 与 receiver 均为 `annotation_status=valid`、`qc_status=passed`，role window 与 loop
   一致；
6. source manifest/provenance 格式受支持，所有引用文件通过 lineage identity 校验；
7. URDF、全部引用 visual mesh、fit config、backend 实现与 publisher 实现均可读取并记录
   identity。

source 中旧的两个 gripper channel 不参与 eligibility。publisher 必须清空它们，只把新
active-arm track 写入对应 channel；inactive arm 全空且保持 `not_annotated/not_run`。

## 6. Canonical public artifact contract

两种 backend 的 public run 使用相同布局：

```text
<output-root>/<run-id>/
  process_summary.json
  <task>/episode_<id>/<camera>/
    masks.npz
    run_manifest.json
    frame_provenance.json
    loop.json
    target_0/...
    receiver_0/...
    gripper_<active-arm>/...
  rendered_videos/
    manifest.json
    episode_*_overlay.mp4
    review_sheets/
      target_early.jpg
      target_late.jpg
      receiver_early.jpg
      receiver_late.jpg
      gripper_early.jpg
      gripper_late.jpg
```

URDF backend 可额外保存：

```text
<run>/_backend/urdf/...
```

它是 private、可审计的几何中间层，不属于 downstream API。URDF engine 自己的 overlay 与
review 必须关闭。对外有用的 `native_track.npz`、URDF product、depth/geometry diagnostics
同时发布到 canonical episode 的 active gripper 目录。

public `masks.npz` 必须严格保持七键：

```text
format_version, frame_count, masks, instance_names,
roles, annotation_status, qc_status
```

target/receiver 必须与 source 对应 channel 逐像素相同；不允许 render-time merge，也不允许
downstream 读取 private `gripper_masks.npz`。canonical publisher 的派生规则固定为：

- source `masks[0:2]` 原样成为 public target/receiver；
- source 两个旧 gripper channel 一律丢弃；
- URDF visible track 只写入 active-arm channel，inactive channel 全空；
- source `loop.json` 及 target/receiver materialized artifacts 保持内容一致；
- public `run_manifest.json`、`frame_provenance.json` 和 `masks.npz` 由 publisher 重新构造，不能
  从 source 或 private backend 盲拷贝；
- public file tree 必须与预期集合完全相等，多文件、少文件、symlink 或任一 hash/内容不符都
  视为失败。

两种 backend 的 downstream contract 是相同的七键 masks、manifest/provenance format、role
语义、overlay manifest 和六张 review sheet；backend-specific diagnostics 只能作为 active
gripper role 的附加审计 artifact，不得成为读取 canonical masks 的前置条件。

## 7. Manifest、provenance 与 summary

`run_manifest.json` 和 `frame_provenance.json` 沿用 v3 format version。backend 判别位置固定为：

```text
run_manifest.gripper_backend = "sam" | "urdf"
frame_provenance.gripper_backend = "sam" | "urdf"
frame_provenance.channels.gripper_<active-arm>.backend = "sam" | "urdf"
```

SAM/URDF 的 `run_manifest.json.gripper_qc` 必须是精确相同的字段集合，不能让 downstream
根据 backend 猜 key 是否存在：

```text
backend              "sam" | "urdf"
status               episode gripper stage status
qc_status            passed/rejected/... 的 canonical QC status
active_arm           "left" | "right"
selected_candidate   string | null
confidence           number | null
reason               string | null
forced_fallback      bool
nonempty_frames      int
quality              object | null
```

视觉 backend 填候选、confidence 和 fallback；没有统一几何质量对象时 `quality=null`。URDF
backend 没有候选或 confidence，因此对应字段为 `null`、`forced_fallback=false`，并把 depth/
geometry quality 放入 `quality`。共同字段的类型和缺省值必须固定，backend-specific 细节放在
`algorithm.gripper_stage`、provenance 或 role artifact，不能扩张 `gripper_qc` key set。

URDF public manifest 由 publisher 派生：只继承已验证的 source target/receiver role record 和
object algorithm metadata，删除旧 gripper role/provenance，再加入一个 active URDF gripper
role、`algorithm.gripper_stage.backend=urdf` 以及：

```text
derivation
  format_version = robotwin_urdf_gripper_derivation_v1
  source          = 完整 robotwin_derivation_source_lineage_v1
  publisher       = publisher implementation identity
```

public `frame_provenance.json` 保留 source `target_0/receiver_0` channel 的原语义，并重新生成
active gripper channel；inactive gripper 只能是 `{"status":"not_annotated"}`。URDF gripper
provenance 至少记录 producer、active arm/window、source lineage digest、backend run/product
identity、URDF/mesh、depth tolerance、fit config、质量和 canonical artifact path。继承的
target/receiver 不能伪装成本 run 重新计算。

`process_summary.json` 使用统一的 `robotwin_process_dataset_summary_v1` 顶层字段：

```text
format_version, gripper_backend, run_id, dataset_root, task, camera,
discovered_episode_ids, requested_episode_ids, dynamic_manifest,
qwen_health, records, render, fatal_error, backend, passed
```

URDF 特有配置只放入 `backend`。`qwen_health` 在 URDF 模式为 `null`。自动发现时所有被
dataset/source contract 排除的 episode 必须同时出现在顶层 `records` 和
`backend.dataset_excluded/source_excluded`，避免 summary 看似完整却静默少 episode。

`passed=true` 表示没有 backend batch error，且所有实际 selected episode 已成功发布、render
成功；若使用
`--allow-partial-source`，是否覆盖全部请求由 `backend.source_selection_complete` 明确表达。

## 8. Immutable contract、复验、原子发布与 resume

### 8.1 Backend immutable run contract

private URDF engine 在 render 前建立 immutable `run_contract`。除了 dataset/source/output、
task/camera、threshold、fit config、episode plan 和 URDF/mesh identity，还必须固定：

- 每个 episode 的 Parquet、sidecar、RGB、depth、source masks、authoritative loop identity；
- publisher 生成的完整 source lineage v1（包括 lineage digest）；
- URDF runner/data/renderer implementation identity（实现文件 `sha256/bytes`，可附 Git
  revision）；
- canonical publisher implementation identity。

具体落盘时，`run_contract.episode_plans[*].source_lineage` 保存完整 lineage，不只保存 digest；
同一 plan 的 `inputs` 保存 dataset/source input identities。`run_contract.implementation.files`
必须包含 runner、data loader、geometry renderer 和 canonical publisher 实现文件的 hash/bytes。
public derivation 中仍保留带独立 format version 的 publisher identity；两者分别锚定“backend
计划由什么代码执行”和“canonical JSON/tree 由什么代码生成”。

因此只要 source summary、loop、manifest、provenance、masks、任一被引用 role artifact、
dataset input、asset、配置或实现文件发生改变，resume contract 就不再相等。`--resume` 必须在
任何 episode 写入前拒绝 changed contract；不能用新代码或新 source 接着旧 backend manifest
跑。

### 8.2 三个验证边界

```text
source preflight + lineage snapshot
        ↓
private URDF render under immutable contract
        ↓
publish 前：重验 source lineage + backend record/artifact identities
        ↓
staging tree 全量校验 → atomic rename
        ↓
shared render 前：从 source/backend 重建 contract，再验 canonical tree
        ↓
shared overlay/review renderer
```

不能把 preflight 结果当作数小时任务后的永久授权。publisher 在每个 episode 发布前重新读取
source control/role artifacts，并核对 backend product、combined masks、diagnostics、quality 和
artifact identities。episode 发布成功后，顶层 batch 在调用 shared renderer 前必须再调用
canonical validator；只有通过第二次 public tree/content 校验的 episode 才进入 render id
集合。

### 8.3 原子性与 fail-closed resume

- canonical episode 先写同父目录临时 staging tree；七键 masks、JSON、role artifacts、文件
  集合和内容全部验证通过后才原子 rename；失败清理 staging，不留下看似完整的 episode；
- 同一 destination 已存在时，非 resume 必须拒绝覆盖；
- resume 不信任 `manifest.status`。它从当前 source、backend 和 publisher implementation 重建
  预期 contract，再验证 public masks payload、JSON 全对象相等、role product 和完整文件树；
- source/backend/public 任一文件被篡改、缺失、多出、变成 symlink，或 hash/bytes 不一致时
  fail closed；不得删除、覆盖或“修复”已有 canonical episode；
- backend 已标记 complete 的 episode 也必须以 manifest 中原 artifact identity 为 anchor；
  complete artifact 缺失或内容改变属于 immutable resume failure，不能重渲染覆盖；
- 单 episode publisher 校验失败时，该 episode 不得加入 shared render；其他 episode 是否继续
  由 batch 的逐 episode failure policy 决定，但 summary 必须保留明确失败记录；
- `render_failed` 属于顶层失败状态；已验证并发布的 masks 保留用于诊断。

### 8.4 唯一可恢复的 runner 异常

URDF engine 只把“逐 episode 尝试已经结束、部分 episode failed、当次 manifest 已完整
checkpoint”表示为 `UrdfBatchIncompleteError`。该异常携带本次内存 `result`，而不是要求调用者
去磁盘猜一个可能过期的 manifest。

`process_urdf_source_run()` 只能捕获这一种异常：它可以继续发布、复验和渲染 result 中 complete
的 episode，同时逐条记录 incomplete episode；summary 的 `fatal_error` 与 `backend.error`
必须写入带异常类型的原因，且 `passed=false`。

其他 runner 异常——包括 resume contract/anchor/tamper error、配置错误、I/O contract error、
`TypeError` 等编程错误——必须原样向上传播。顶层不得回读旧 manifest，不得调用 publisher，
也不得启动 shared renderer。这一边界防止“旧 manifest 看起来有 complete record”掩盖本次
运行在进入 episode loop 前已经失败。

## 9. 依赖边界

依赖拆分为：

```text
core: av/numpy/pandas/pyarrow/Pillow/PyYAML
sam3 extra: torch/torchvision/OpenCV/SAM3
urdf extra: h5py/pyrender/trimesh/PyOpenGL/pyglet/pycollada
```

`process_dataset.py` 顶层只能加载 core 与 backend-neutral 类型。Qwen client、Sam3Adapter、
`run_target_receiver` 和其间接 OpenCV import 必须在选择 `sam` 后延迟加载。以下命令必须在只
安装 `.[urdf]` 的环境中成功：

```bash
PYTHONPATH=src python scripts/process_dataset.py --help
PYTHONPATH=src python -c 'import scripts.render_coverage20_videos'
```

依赖安装不得下载或替换数据集。`pyrender==0.1.45` 在 Python 3.13 下通过项目级 uv override
使用兼容的 `PyOpenGL==3.1.10`。

## 10. 验收顺序

1. 单元测试覆盖 authoritative loop identity/window/frame count，且证明 URDF 不重新推断 loop；
2. lineage 测试逐类覆盖 source summary、loop、manifest、provenance、masks 和 target/receiver
   referenced artifact 的 `sha256/bytes`，并覆盖 publisher/backend implementation drift；
3. artifact parity 测试断言 SAM/URDF masks 七键、manifest/provenance format、十键
   `gripper_qc` 和 shared renderer 输入一致；
4. failure-boundary 测试证明只有 `UrdfBatchIncompleteError(result)` 会发布成功子集；resume
   changed-contract/tamper、配置和编程异常不会读旧 manifest，也不会调用 publisher/render；
5. publisher 测试覆盖 staging 原子性、publish 前 source/backend mutation、resume tree tamper，
   顶层测试覆盖 shared render 前再次 mutation；
6. CLI 测试覆盖默认 SAM、URDF-only 参数约束、optional output positional、显式 episode
   fail-closed、`--allow-partial-source` 和带显式 run id 的 resume；
7. 全量 pytest、Ruff `E/F/I`、PyCompile、`git diff --check` 通过；
8. 对一个右臂和一个左臂 episode 做真实 smoke，核对 active/inactive channel；
9. 验证 target/receiver 与 source 像素完全相同、public `masks.npz` 严格七键；
10. 验证两个 overlay MP4 和六张 shared review sheet；
11. 最后在 `move_pillbottle_pad_coverage20_original` 的 20 个显式 episode 上运行，避免
    `--allow-partial-source` 静默改变验收集合；
12. 20 个 episode 的 canonical artifact、视频、review sheet 与 summary 全部通过后，才把
    实验结论写为“可行”。

coverage20 验收应复用本地已处理 source run 和本地 URDF/mesh，不重新下载数据集。
