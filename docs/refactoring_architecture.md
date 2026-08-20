# 精简重构架构结果

> 状态：结构、CPU/static 与重构后真实默认 URDF 单 episode/正式入口验收完成
>
> 日期：2026-08-20
>
> 详细实施约束见 [refactoring_ai_guide.md](refactoring_ai_guide.md)。

本文保留了迁移时使用的目标、原则与阶段设计，便于解释当前边界为什么形成。正文中的“目标”、
“建议”和“应”主要是设计记录；当前生效合同仍以 [architecture.md](architecture.md) 和代码/
测试为准。

## 0. 实施结果

从 `feat/default-open-set-s1-s3@fb31b6c` 开始的 R0–R8 结构迁移已经落到本分支。结果不是
简单搬文件，而是形成了以下 canonical owner：

| 责任 | 当前 owner | 实施结果 |
| --- | --- | --- |
| dataset dispatch / workflow | `application/dataset_pipeline.py` + `sam_workflow.py` / `urdf_workflow.py` | `DatasetPipeline` 提供 typed backend dispatch 与 public SAM entry；两个 workflow 持有各自生命周期，旧 runtime 负责 CLI/hook 组装并反向委托 |
| URDF runtime 依赖 | `application/urdf_runtime.py` | URDF runner、CUDA/EGL handoff、spawn workers、source selection 和 ownership validation 只有一份实现；`dataset_runtime.py` 仅保留兼容 alias 与 hook/CLI 组装 |
| SAM episode 编排 | `application/sam_workflow.py` + `application/episode_pipeline.py` module-level executors | `SamWorkflow` 持有正式 dataset 路径的 per-episode 顺序、resident backend 生命周期和失败策略；`EpisodePipeline` 仅为 standalone/分阶段 CLI façade |
| object resolution | `pipeline/object_mask/resolver.py` | 一个 resolver 持有 text → legal seed → bbox 的顺序与 error-stop；planner/proposal/QC 已按职责拆分 |
| object-mask QC artifact publication | `application/mask_qc_artifacts.py` | 文件写入位于 application；`pipeline/object_mask/artifacts.py` 仅保留旧 import 委托，不再依赖 `ArtifactStore` |
| canonical NPZ | `adapters/canonical_masks.py` + `adapters/canonical_publication.py` | 一个 DTO/validator/reader 和一个 `CanonicalMaskPublisher`；SAM 与 URDF 都调用该 publisher，v2 只读兼容 |
| URDF derived tree | `UrdfCanonicalEpisodePublisher` | 在独立 lineage 信任边界组装、验证并原子 rename 整棵 episode tree；内部 NPZ 仍委托中立 publisher |
| source lineage | `SourceLineageValidator` | 无 source contract 的 lineage v1 与带 contract/receipt 的 lineage v2 共用一个验证 owner；frozen-source 模式可消费任一版本，identity/hash 仍在各信任边界重复校验 |
| Qwen 边界 | `pipeline/qwen_stage.py` + `adapters/qwen_client.py` + application | pipeline 持有 prompt/request/领域响应校验，adapter 持有 HTTP transport，application 持有 artifact 与 endpoint 生命周期；本地 server/model 仍是 standalone utility |
| timeline | `pipeline/timeline_detector.py` + `models/timeline.py` | 一个当前 detector/type 体系；`loop_context_codec.py` 隔离 v1/v2/v3 读取兼容 |
| renderer/URDF engine | `adapters/rendering.py` public renderer API；`application/urdf_batch.py` 编排 `urdf_gripper_renderer.AlohaUrdfRenderer` | renderer 负责 canonical mask selection/load 与视频资源；URDF geometry/FK owner 仍是 `urdf_gripper_renderer.py`，对应 script 只导入 package `main` 并启动 |

`src/` 已无对 `scripts/` 的反向导入，两个 renderer launcher 的 `sys.modules` proxy 也已删除。
`dataset_runtime`、`run_target_receiver`、旧类型/函数名等仍有真实调用方；它们只委托 canonical
owner，且逐项记录了删除条件。完整清单见
[refactoring_ai_guide.md](refactoring_ai_guide.md#132-compatibility-inventory)。

### 0.1 验收边界

本分支 2026-08-20 验收记录：`just test` 662 passed，`just test-all` 669 passed，`just lint` 通过，
`.venv/bin/python -m mypy src` 为 0 issues，`git diff --check` 通过。测试覆盖 contract、filesystem、
fake backend、schema、lineage、resume/tamper 与 import boundary。

本轮已用真实数据重跑默认 URDF：pick-place episode 7152（right arm）与 target-only episode 0
（left arm）分别通过 direct CLI 和正式 `just process`，四个 exact run 均 `passed=true`，并生成
overlay/review sheet。严格八键四通道 NPZ、source object 逐像素一致、inactive arm 全零、
target-only receiver `not_applicable`、publisher validator、lineage/provenance 和默认 URDF 资产
identity 均复验通过。exact run ID 与 LOC 口径见
[refactoring_ai_guide.md](refactoring_ai_guide.md#21-实施完成快照)。该证据仍只覆盖左右臂各一个
episode；历史 coverage20、coverage subset/full batch 与像素级人工签字不能由此替代。

## 1. 目标

这次重构的目标不是重写算法，而是在保持当前行为和产物合同不变的前提下，让代码的真实
职责与目录结构一致：

- 顶层流程容易阅读，能够直接看出一次 episode 和一次 dataset run 如何执行；
- Qwen、SAM3、URDF、视频和文件系统等外部依赖具有明确的职责边界；
- pipeline 计算与 artifact 发布分离；
- S1–S3 open-set fallback 作为正式的 object-mask resolution 合同保留；
- 同一份 timeline、mask schema 和 publication 规则只有一个事实来源；
- 只为有状态、有生命周期或需要替换依赖的职责使用类；
- 删除迁移遗留的空转 facade、反向依赖和真正重复的实现，但保留安全校验。

本次重构不追求：

- 改变 mask 算法、阈值、prompt 或任务覆盖范围；
- 修改四通道 `masks.npz`、manifest、provenance 或 lineage 格式；
- 为 50 个 task 建立类继承树；
- 引入 DI 容器、事件总线、通用 Stage 基类或插件框架；
- 为了表面上的 DRY，删除 publication、lineage、resume 和 tamper 的边界复验。

## 2. 重构前判断（历史）

本节记录 `fb31b6c` 附近的审查结论，用于解释迁移优先级，不再描述当前模块 owner。实施后的
对应关系见第 0 节。

仓库已经具备合理的基础结构，不需要推倒重写。以下设计应继续保留：

- `FrameWindow`、timeline events、`LoopContext` 等不可变值对象及其不变量；
- `AnnotationSpec` 对 annotation mode 和 object role 的数据化描述；
- Qwen、SAM3、RoboTwin dataset 和 artifact storage 的 adapter 方向；
- text-first、多 query、多合法 seed、最终 bbox fallback 的 S1–S3 恢复链路；
- fail-closed、visible-only、Parquet frame authority 等 pipeline 合同；
- canonical mask、source lineage、atomic publication、resume/tamper 校验。

当时的主要问题集中在编排和发布层：

1. `application/dataset_runtime.py` 同时承担 CLI、发现、进程通信、GPU、SAM/URDF 调度、
   发布、渲染、UI 和 summary。
2. application 代码会反向导入 `scripts/render_*.py`；脚本已经成为生产业务模块。
3. `DatasetPipeline` 和 `EpisodePipeline` 目前主要是转发壳，真正职责仍在巨型过程函数中。
4. SAM stage、URDF publisher 和 renderer 各自包含一部分 canonical mask 读写与校验。
5. State Loop 与 URDF 路径存在两套相近的 timeline/event 处理。
6. `mask_qc.py` 同时承担 query/seed/bbox 调度、候选生成、视觉判定和 artifact 写入；拆分时
   容易破坏 S1–S3 的固定顺序和 attempt provenance。
7. 若干大模块把纯计算、外部调用、artifact 写入和流程控制混在一起。

因此，优先级应是修正依赖方向和职责边界，而不是先按行数机械拆文件。

### 2.1 精简与去冗余路线

代码精简按“减少职责重复”衡量，不按删除行数衡量。建议固定四个唯一事实来源：

1. timeline/event/window 只有一个 detector 和 codec；
2. object mask 只有一个 resolver 负责 text → legal seed → bbox 的顺序；
3. canonical mask 只有一个已校验 DTO、codec 和 publisher；
4. 每个完整用例只有一个 workflow coordinator；standalone façade 不与 dataset workflow 争夺
   编排所有权。

执行时先用 `rg`、调用图和测试定位重复实现，再让旧入口委托到新实现；确认仓内调用、测试和
monkeypatch seam 已迁移后，才删除 compatibility shim。纯函数可以合并，跨信任边界的复验、
lineage 检查和失败分类不能为了 DRY 删除。每个阶段都应能说明：删除了哪一份重复逻辑、保留了
哪一个 owner、旧新输出如何逐字段对拍，以及新增了多少类/接口（目标是尽量不增加）。

## 3. 架构原则

### 3.1 依赖方向

目标依赖方向如下：

```text
scripts / cli
      │
      v
application  ----->  pipeline
      │                 │
      v                 v
   ports            models/domain
      ^                 ^
      │                 │
adapters ---------------+
```

具体规则：

- `scripts/` 只能解析入口参数并调用包内 API；`src/` 不得导入 `scripts/`。
- `application/` 负责流程顺序、资源生命周期、失败策略和批处理策略。
- `pipeline/` 负责一个 stage 内的计算，不负责 CLI、终端输出或公共 artifact 发布。
- `domain/` 和 `models/` 保存稳定语义、不变量和跨阶段 typed contracts。
- `adapters/` 实现 dataset、Qwen、SAM3、URDF、文件、视频等外部边界。
- 内部模块从具体模块导入；包级 `__init__.py` 只导出轻量且稳定的公共类型。

当前已消除 `src/ -> scripts/` 反向依赖，但尚未完成严格 ports 反转：若干 pipeline stage 仍直接
复用 `adapters.qwen_client` 的 completion DTO 和 image encoding helper。Qwen 的现状是
`pipeline/qwen_stage.py` 持有 prompt/request/领域响应校验，`adapters/qwen_client.py` 持有 HTTP
transport，application 持有 artifact 与 endpoint 生命周期；本地 server/model 仍是已登记的
standalone utility。

### 3.2 精简不等于减少校验

可以合并的是校验实现，不能删除的是校验时机。例如 source preflight、publish 前复验、
staging tree 验证和 render 前 canonical 验证处在不同信任边界，应继续执行，但调用同一个
validator，避免规则漂移。

### 3.3 面向对象的使用边界

优先使用：

- frozen dataclass 表达事件、窗口、plan、result 和 run request；
- 小型 application service 表达一个完整用例；
- `Protocol` 表达真正需要替换的外部端口；
- 组合表达 backend 差异。

继续使用纯函数处理：矩阵计算、窗口推导、mask composition、QC metrics、候选评分和各边界
已有的 JSON 解析。不要在结构迁移中统一不同边界的 parser 宽严程度，也不要把每个 helper
包装成类。

## 4. 目标模块结构

目标结构保持层次少而清楚；第一阶段不要求立即移动所有现有文件：

```text
src/robotwin_annotation_v2/
  domain/
    annotation_spec.py       # mode、role、backend 语义
    timeline.py              # 唯一 timeline/event/window 规则（最终目标）

  models/
    loop_context.py          # 跨阶段不可变数据
    semantic_plan.py
    mask_qc.py
    mask_run.py
    process_run.py           # 少量 batch request/record/summary 类型

  application/
    episode_pipeline.py      # 单 episode executors + standalone/分阶段 façade
    dataset_pipeline.py      # discovery、resume、batch 和资源生命周期
    sam_workflow.py          # dataset 级 SAM workflow
    urdf_workflow.py         # source -> URDF -> publish -> validate -> render
    urdf_runtime.py          # worker、Pipe/Queue、并发协调和 GPU handoff
    discovery.py             # dataset discovery 与动态 manifest
    streaming.py             # multiprocessing wire message DTO/decoder
    ports.py                 # 少量稳定外部端口；不做接口垃圾桶

  pipeline/
    state_loop.py            # Stage 1 计算
    qwen_stage.py            # Stage 2 请求构造与响应解析
    object_mask/
      planner.py             # query、alias 与合法 seed 的有序计划
      resolver.py            # S1–S3 attempt 顺序和停止条件
      proposals.py           # text mask 与 bbox -> SAM mask
      qc.py                  # 共享机械检查与 visual QC
      open_set_queries.py    # 配置驱动的 curated aliases
    object_tracking.py       # native propagation 与 temporal QC
    gripper/sam/             # geometry、candidate、QC、composition

  adapters/
    robotwin_dataset.py
    qwen_client.py
    sam3_adapter.py
    canonical_masks.py       # 唯一 NPZ DTO/reader/validator
    canonical_publication.py # SAM/URDF 共用的原子 NPZ publisher
    urdf/
      data.py
      renderer.py
      finger_fit.py
      runner.py
    rendering.py

  urdf_gripper_publisher.py  # source lineage + derived-tree publisher；兼容路径暂留

  cli/
    process_dataset.py
    run_episode.py

scripts/
  process_dataset.py         # 薄启动入口
  run_target_receiver.py     # 薄启动入口
  render_*.py                # 薄启动入口或兼容 shim
```

`models/` 与 `domain/` 暂时不做大规模重命名：先解决真实职责，最后再根据稳定语义决定是否
移动文件，避免产生大量无收益的 import churn。

## 5. 两级编排

### 5.1 单 episode

迁移目标曾计划由 `EpisodePipeline` 统一单 episode 用例：

```text
LoopContext
  -> SemanticPlan
  -> ObjectMaskResolver: text query -> legal seed -> bbox
  -> ObjectMaskResolution (S1–S3)
  -> MaskQCResult + selected seed masks
  -> ObjectMaskResult
  -> GripperMaskResult
  -> CanonicalMaskBundle
  -> publication
```

最终实现没有让这个 façade 成为正式 dataset 路径的第二个 coordinator。`SamWorkflow` 持有
per-episode 阶段顺序、resident backend 生命周期和失败策略；
`application/episode_pipeline.py` 的 module-level executors 实现具体操作，`EpisodePipeline`
只为 standalone/分阶段 CLI 组合这些 executors。上述 application 边界共同保持 fail-closed
策略，并且不应：

- 解析 argparse；
- `print()` JSON；
- 直接管理 dataset batch；
- 手工拼接公共 NPZ、manifest 和 provenance；
- 知道 HTTP、CUDA session 或具体文件布局的实现细节。

### 5.2 Dataset run

迁移目标中的完整 dataset run 包含：

```text
discover
  -> select / resume scan
  -> acquire backend resources
  -> execute episodes
  -> classify ordinary/fatal failures
  -> publish summary
  -> optional render
```

最终所有权按 backend 拆分：`DatasetPipeline` 提供 typed backend dispatch、discovery/manifest
helper 和 public SAM convenience API；resume scan、资源生命周期、episode 执行、失败分类、
summary/render 分别由 `SamWorkflow` 与 `UrdfWorkflow` 持有。不应为了统一表面接口，把两者
强塞入一个包含大量布尔参数的方法。共同点放入 typed request、record、summary 和 publisher，
流程差异保留在 workflow 中。

## 6. 建议保留的少量类

| 类或端口 | 责任 | 不应承担 |
| --- | --- | --- |
| `EpisodePipeline` | standalone/分阶段 CLI 的单 episode 顺序 façade | dataset batch、resident backend 生命周期 |
| `SamWorkflow` | 正式 SAM dataset 路径的 per-episode 顺序、backend 生命周期与失败策略 | 单 episode 算法、CLI 参数解析 |
| `DatasetPipeline` | typed backend dispatch、discovery/manifest helper、public SAM convenience API | backend 生命周期、图像算法、URDF 几何 |
| `ObjectMaskResolver` | 消费已预加载的角色合法 seed，执行 text-first 的 query/seed/bbox 顺序并保存 attempt 审计 | propagation、artifact 发布、S1/S2/S3 子类树 |
| `UrdfWorkflow` | source selection 到 canonical publish 的完整用例 | mesh 渲染算法、CLI |
| `CanonicalMaskPublisher` | 唯一 canonical NPZ 原子发布边界 | manifest/provenance、SAM/URDF 算法决策 |
| `UrdfCanonicalEpisodePublisher` | URDF derived episode 的 lineage、整树验证和原子 rename | canonical NPZ 编码、URDF 几何 |
| `AlohaUrdfRenderer` | mesh/scene/render 资源生命周期 | finger 搜索策略、batch |
| `FingerPoseFitter` | 候选生成、评分、选择与诊断 | renderer 所有权、发布 |

真正需要替换的边界用少量 Protocol，例如 `DatasetReader`、`VisionLanguageClient`、
`ObjectSeedMaskBackend`（text query masks 与 SAM box mask）、`MaskTracker`、
`ArtifactRepository`、`ProgressSink`。不要为每个 helper、每个 stage 或每一级 fallback 创建接口。

## 7. S1–S3 open-set mask resolution

这里的 S1、S2、S3 是失败救回方案的能力标签，不是 State Loop、Qwen、SAM 三个 pipeline
stage。默认 pick-place 和 target-only profile 当前都显式启用完整链路，`--data-path` 也继承
对应 mode profile。

三部分在架构中的位置不同：

- S1：扩展 query bank、追加配置驱动的 curated aliases，并只在合法 seed 间重试；
- S2：使用 mode-specific open-set semantic/QC prompt，让外观 query 和完整性判定适应域偏移；
- S3：所有 text query × legal seed 都未通过后，才允许 Qwen bbox → SAM box mask。

S2 不是在 S1 失败后才执行的一个运行时步骤。完整 profile 从 semantic planning 开始就使用 S2
prompt，实际 attempt engine 的顺序仍是 text-first，然后才是 bbox：

```text
open-set SemanticPlan
  -> ordered semantic queries + curated aliases
  -> semantic seed 上的 text candidates
  -> rejected/ambiguous 时尝试其余合法 seed
  -> 所有 text attempts rejected/ambiguous
  -> 同一 seed 顺序执行 Qwen bbox -> SAM box mask
  -> 与 text candidates 相同的 mechanical gate + visual QC
  -> first passed seed mask，或 fail closed
  -> native propagation -> temporal QC
```

以下是结构重构不可改变的顺序约束：

- `SemanticStatus.NO_CLEAR_SEED` 是前置失败：该角色直接 rejected，不产生 text 或 bbox attempt；
- application 必须为每个角色预加载所有 `seed_eligible` 且 role 合法的 RGB seed，缺失时按输入/合同
  error fail closed，不能把缺失 seed 当作普通 rejected；
- text candidate 必须先于 bbox；bbox 不能抢占已通过的 text candidate；
- 只能使用 `LoopContext` 为该角色声明的合法 seed，Qwen seed 始终先尝试；
- `rejected/ambiguous` 可以进入下一 attempt，合同、服务或候选生成 `error` 必须立即停止该角色；
- bbox 坐标不得自动修正或 clamp，SAM 生成的真实 mask 必须通过普通机械检查和 visual QC；
- 第一个 passed attempt 立即停止；不得按面积选最大、合并候选或把非空等同于成功；
- 每个 method/seed attempt 都保留 query、候选、prompt、raw response 和 provenance；
- S4 的方向扩框、触边专用放行和传播强制修正不属于目标架构，遗留字段继续 fail closed。

建议只增加一个小型 `ObjectMaskResolver` 来拥有合法 seed、attempt ordering 和停止条件，并继续
使用 `MaskQCAttempt`、`BboxLocalization` 等不可变值对象记录审计。text proposal、bbox
localization、mechanical gate 和 visual decision 可以拆成函数或窄端口；S2 只是一组 prompt/policy
配置，不是运行时类。不要创建 `S1Fallback`、`S2Fallback`、`S3Fallback` 继承树，也不要复制三套
object pipeline。

## 8. 唯一公共产物边界

SAM 和 URDF producer 最终都构造同一个已校验的 typed `CanonicalMaskBundle`，其中包含：

- 固定四通道 bool mask；
- `frame_encoding`；
- annotation/QC status；

channel provenance 与 producer-specific quality metadata 不属于 NPZ DTO；它们继续保存在各自的
manifest/provenance artifact 中，由对应信任边界组装和验证。

`CanonicalMaskBundle` 是跨模块唯一的 canonical NPZ DTO；不要再并行引入一个语义相同的
`CanonicalMasks`。codec 可以是无状态函数/模块，publisher 才是有生命周期的发布服务。

`CanonicalMaskPublisher` 是唯一负责以下行为的中立组件：

- 校验固定通道顺序、shape、dtype、status 和 frame encoding；
- 原子写入 canonical 八键 `masks.npz`。

SAM artifact service 组装自身 manifest/provenance，并把 bundle 交给中立 publisher。URDF 的
source lineage 与内容寻址由 `SourceLineageValidator` 负责：无 source run contract 的 source
使用 `robotwin_derivation_source_lineage_v1`；带 immutable contract 和 completion receipt 的
incremental source 使用 `robotwin_derivation_source_lineage_v2`。frozen-source 模式可消费任一
版本；当前 source contract writer 写 v2、reader 兼容 v1/v2，completion receipt 只适用于
lineage v2。
`UrdfCanonicalEpisodePublisher` 组装并验证整棵 derived tree，在 staging 中调用中立 NPZ
publisher 后执行原子 rename。这样既统一公共格式，又不弱化两个 producer 的不同信任边界。

fallback method 只影响 QC/attempt provenance，不改变 canonical 四通道 schema。publisher 应
接收已经通过 resolution 和 propagation 的对象结果，不根据 `text_query` 或 `bbox_fallback`
分叉公共 mask 格式。

## 9. 分阶段迁移

### 阶段 0：冻结行为

- 为 loop JSON、canonical NPZ、manifest/provenance、summary 建 golden contract 测试；
- 固定 v2 只读、v3 新写的兼容规则；
- 固定 resume、tamper、active/inactive arm 和 target-only N/A 语义；
- 固定默认 profile 的完整 S1–S3 开关、attempt 顺序、error 停止和 S4 拒绝语义；
- 固定 `NO_CLEAR_SEED` 不产生 attempts、所有合法 seed 均可供 resolver 使用，以及 text/bbox
  attempt 的 nested/flat provenance；
- 记录当前 Ruff/mypy 历史基线，不把旧债误判为本次回归。

### 阶段 1：修正依赖方向

- 将 renderer 与 URDF batch engine 的生产实现迁入 `src/`；
- `scripts/` 暂时保留兼容 re-export；
- 消除 application 对 `scripts` 的动态导入；
- 收窄包级 eager import，保持 frozen-source URDF 的 optional dependency 边界。

### 阶段 2：拆 dataset 编排

- 从 `dataset_runtime.py` 依次抽出 discovery、run request/plan、SAM workflow、URDF workflow、
  streaming 和 render coordination；
- 让 `DatasetPipeline` 成为真正 coordinator；
- 每次只移动一个职责，保持 summary 和 CLI 行为等价。

### 阶段 3：统一 publication

- 建立 canonical mask codec、validator、bundle 和 publisher；
- SAM 与 URDF 分别接入；
- 把 stage 中的文件写入移到 publication/adapters；
- 保持所有 public schema 和 lineage identity 不变。

### 阶段 4：拆算法大模块

- 按 feature cohesion 拆 object QC、SAM gripper 和 finger fitting；
- object QC 拆分后仍由一个 resolver 拥有 text → seed → bbox 的全局顺序；
- 纯计算保持函数，资源所有者保留为类；
- 不在结构迁移中调整 prompt、阈值或 fallback policy。

### 阶段 5：统一 timeline

- 对两套 detector 做同输入对拍；
- 统一事件类型和窗口推导；
- v1/v2/v3 兼容逻辑只留在 codec/adapter；
- 最后删除兼容 alias 和重复 detector。

### 阶段 6：清理迁移遗留

- 迁移仓内调用和测试后，删除空转 facade、星号导入、`sys.modules` 替换和废弃参数别名；
- 最后处理 `models/`/`domain/` 命名和 Ruff/mypy 全量清债。

## 10. 架构验收结果

当前结构按以下标准验收：

- `src/` 不导入 `scripts/`；
- 核心 script launcher 只负责启动/兼容转发；package 内 CLI 层负责参数、依赖组装、UI 和退出码；
- `DatasetPipeline`/`SamWorkflow` 不再转发到巨型 runtime；`EpisodePipeline` 明确收窄为
  standalone/分阶段 façade；
- stage 算法不直接发布 canonical artifacts；
- 默认 pick-place、target-only 和 path-mode profile 继续执行完整 S1–S3 ladder；
- object resolution 保持 text-first、legal-seed-only、error-stop 和统一 QC；
- `MaskQCAttempt` 历史与候选 artifacts 在模块迁移后完整保留，S4 字段继续拒绝；
- bbox localization 的 confidence 只进入 provenance；实际 SAM mask 仍通过与 text 相同的 mechanical
  gate、visual QC 和既有 visual-QC confidence threshold；
- canonical mask 的最终 NPZ load/validation 只有一个 DTO/reader/validator 和一个 `CanonicalMaskPublisher` 事实来源；renderer 的历史跨-run candidate selection 仍有受控 metadata pre-read，选中后必须回到 strict codec/fallback 合同；
  producer-specific metadata 与 URDF 整树 lineage 校验仍留在各自信任边界；
- timeline detector 和事件类型只有一个当前实现；
- SAM 与 frozen-source URDF 仍保持各自 optional dependency 边界；
- 所有既有 CPU 测试、integration contract、resume/tamper 测试通过；
- 新代码的类型和 lint 不增加历史债务，并已将 Ruff/mypy 清零。

以上结构与 CPU/static 条目已通过。真实 backend 已完成左右臂各一个单 episode 和正式
`just process` 入口验收；coverage subset、full batch 与更大范围人工 review 仍须逐级进行，且
不属于上述 669 个测试或本轮四个 smoke run 所能证明的范围。兼容层允许保留的前提是
只有委托、存在真实调用方且删除条件已记录；这不允许在兼容层继续新增第二套算法或 schema。

## 11. 已确认的产品事实

重构没有改变以下行为，只把文档修正为与现有实现一致：

1. 默认 gripper backend 是 URDF；SAM backend 必须显式请求；
2. 新写 `loop.json` 是 `robotwin_loop_context_v3`；统一 codec 只读兼容 v1/v2/v3；
3. 新写 canonical `masks.npz` 是严格八键 `robotwin_visible_masks_v3`，v2 仅只读兼容。

若产品决定改变默认 backend 或 format version，必须另立行为/schema 变更任务和迁移说明，不能
把它解释为本次结构重构的延续。
