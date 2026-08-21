# 精简重构实施记录与兼容指南（供 AI 使用）

> 状态：结构、CPU/static 与重构后真实默认 URDF 单 episode/正式入口验收完成
>
> 日期：2026-08-20
>
> 面向人的架构说明见 [refactoring_architecture.md](refactoring_architecture.md)。
>
> 当前运行合同仍以 [architecture.md](architecture.md) 和代码/测试为准；本文保留迁移约束，
> 并在第 13 节记录最终 owner、验收证据和仍保留的 compatibility shim。

## 1. 使用方式

继续清理兼容层或开展下一次结构任务前，AI 必须：

1. 完整阅读仓库根目录 `AGENTS.md`；
2. 阅读 `docs/architecture.md`、本文和本次任务涉及模块的测试；
3. 运行 `git status --short --branch`，保留所有用户已有修改和未跟踪文件；
4. 明确本次只做一个迁移阶段，不将结构迁移、算法调整和 schema 变化混在一起；
5. 先建立或确认 characterization/contract tests，再移动高风险逻辑；
6. 使用小提交和 compatibility shim 实施 strangler migration；
7. 在删除旧入口前，用 `rg` 证明仓内调用与测试已经迁移。

本文中的目标文件名可以按实际依赖微调，但职责边界、兼容合同、迁移顺序和验收条件不可在
没有新证据时自行放宽。

### 1.1 精简代码的判定标准

“精简”指减少同一职责的实现数量和调用路径，不是把安全校验或类型删掉。每个迁移阶段都要
明确一份 canonical owner：timeline detector、object-mask resolver、canonical mask codec/
publisher、episode/workflow coordinator 各保留一个；旧入口只能是有删除条件的委托 shim。
先用 `rg`、依赖图和 characterization test 证明重复，再合并实现；合并后用旧新输出对拍和
focused tests 证明行为不变。若只是把同一逻辑复制到新文件、增加 facade、增加无状态 class 或
新增一层同义 DTO，不算精简。

## 2. 初始审查快照

最初的架构审查基于 `master@48d866b`。加入默认 open-set fallback 后，本文再次核对的工作分支为
`feat/default-open-set-s1-s3@fb31b6c`。以下规模和质量数字来自最初审查，应视为历史基线：

- `src/robotwin_annotation_v2/` Python 代码约 19,712 行；
- `scripts/` Python 代码约 6,173 行；
- `just test`：400 passed；
- `just test-all`：407 passed；
- `just lint`：当前已有 90 个 Ruff 问题；
- `.venv/bin/python -m mypy src`：当前已有 242 个 strict mypy 问题。

以上数字不是 fallback 提交后的重新验收结果，也不是永久预期。实施时应重新运行并记录当前
结果。重构阶段至少要求：测试不退化，Ruff/mypy 错误数不增加；静态检查全量清债应作为后续
独立工作。

审查时存在用户已有未跟踪文件。未来执行者必须以当时的 `git status` 为准，不能清理、覆盖
或顺手提交与本次阶段无关的文件。

### 2.1 实施完成快照

迁移从 `feat/default-open-set-s1-s3@fb31b6c` 开始，到已提交的 `318fc81` 完成 R0–R8 主体，
随后在本分支继续完成 `DatasetPipeline` 反向委托、public renderer seam、canonical publisher/
lineage owner 和文档闭环。`fb31b6c..318fc81` 共 50 个小提交；318fc81 之后的收尾仍属于同一
迁移，应以包含这些收尾的分支状态判断最终结果。

本分支 2026-08-20 验收记录：

- `just test`：662 passed；
- `just test-all`：669 passed；
- `just lint`：通过；
- `.venv/bin/python -m mypy src`：0 issues；
- `git diff --check`：通过；
- `rg` 检查 `src/` 对 `scripts/` 的导入：0；
- import-boundary tests 证明 public `DatasetPipeline`、timeline detector 和 frozen-source 路径不会
  因兼容 runtime 或重型 optional backend 被意外加载。

Python 物理行数以 `fb31b6c` 为 baseline、以结构收尾提交 `5ea5d9d` 为结果；不计之后工作树中
其他并行任务的未提交文件：

| 范围 | baseline | 当前 | 变化 |
| --- | ---: | ---: | ---: |
| production（`src/` + `scripts/`） | 25,885 | 28,424 | +2,539（+9.81%） |
| tests | 13,655 | 17,326 | +3,671（+26.88%） |
| 合计 | 39,540 | 45,750 | +6,210（+15.71%） |

因此本次不能称为“总代码缩减”。精简发生在职责和入口：`dataset_runtime.py`、`sam_stage.py`、
`gripper_stage.py` 与两个 renderer launcher 合计从 9,575 行降到 1,523 行，减少 8,052 行
（-84.09%）；逻辑迁入 canonical owner，并补充 strict typing、contract/lineage 校验和测试后，总代码
仍净增长。不得为了追求 LOC 数字删除兼容期 shim 或信任边界复验。

2026-08-20 又完成以下重构后真实数据验收；所有命令均省略 `--gripper-backend`，实际选择默认
`urdf`，并执行 live Qwen 3.5-27B、SAM3、CUDA/EGL、canonical publish/validation 和 review
render：

| 入口 | mode | exact run ID | episode | active arm | 结果 |
| --- | --- | --- | ---: | --- | --- |
| direct CLI | `pick_place` | `refactor-post-real-pick-place-urdf-7152-20260820` | 7152 | right | passed |
| direct CLI | `target_only` | `refactor-post-real-target-only-urdf-0-20260820` | 0 | left | passed |
| `just process` | `pick_place` | `just-process-refactor-pick-place-urdf-7152-20260820` | 7152 | right | passed |
| `just process` | `target_only` | `just-process-refactor-target-only-urdf-0-20260820` | 0 | left | passed |

四个 run 都生成 exact-run overlay/review sheet。两份正式入口结果还独立重跑 publisher validator：
canonical `masks.npz` 为严格八键 `robotwin_visible_masks_v3` 四通道，source object channels
逐像素一致，inactive gripper channel 全零；target-only 的 `receiver_0` 全零且 annotation/QC 均为
`not_applicable`；默认 URDF 为 `arx5_description_isaac_gripper.urdf`，资产、source lineage、
provenance、Qwen prompt/raw、semantic plan、mask QC 与 SAM native track 均完整。

以上新增证据只覆盖所列左/右臂各一个 episode。历史 coverage20 仍是重构前证据；coverage subset、
full batch 和像素级视觉质量签字不能由这四个 smoke run 推出。

## 3. 不可破坏的行为合同

### 3.1 输入与 frame authority

- Parquet 连续 `frame_index` 是有效帧数 `T` 的唯一 authority。
- RGB/depth 允许存在额外尾帧，但不得因此扩张 mask 时间轴。
- SAM discovery 要求 Parquet、RGB video、sidecar；URDF 还要求同 camera depth video。
- 显式请求的 episode 有任一不合格时 fail closed；自动发现的 partial policy 保持现状。
- 不得把 dataset、checkpoint、模型、URDF 的机器绝对路径写入新配置或源码。

### 3.2 Timeline 与窗口

Pick-place 事件顺序：

```text
t_move_start <= t_close_start < t_close_done < t_open_start < t_open_done
```

所有窗口均 inclusive：

```text
loop      = [move_start, open_done]
target    = [move_start, open_start - 1]
receiver  = [close_done, open_done]
gripper   = [move_start, open_done]
```

Target-only：

- 不得伪造 release/open 事件；
- target 普通段 `[remove_start, close_end]`；
- target hold 段 `[close_end + 1, T - 1]`；
- receiver 为 not applicable，而不是 failed 或 not annotated。

模式必须与事件类型匹配；semantic frame ID 必须唯一并保留 purpose、seed eligibility 和
eligible roles。

### 3.3 Qwen semantic plan

- Qwen server 是独立 OpenAI-compatible 外部服务；client/server ownership 不能混入 stage。
- `pipeline/qwen_stage.py` 负责 prompt 读取、请求构造和 semantic JSON 校验；
  `adapters/qwen_client.py` 负责 HTTP transport 与 OpenAI response envelope；
  `application/episode_pipeline.py` 负责 loop、plan、rendered prompt 和 raw response artifact；
  `application/managed_qwen.py` 负责 endpoint 复用、本地进程启动与清理。
- 当前若干 pipeline stage 仍复用 client adapter 的 completion DTO/image encoding helper；严格
  ports 反转尚未完成，不能把这层直接依赖描述成已验收事实。
- target/receiver 在一次请求中联合判断；target-only 响应只能包含 target。
- 保持各 Qwen 边界当前的 parser policy，不得在结构迁移中顺手统一宽严程度。
- semantic plan 和 mask visual-QC 继续执行现有字段、候选、seed、confidence 合同；是否进一步
  拒绝重复 key/非有限 JSON 常量属于单独行为决策。
- S3 bbox localization 使用严格 JSON：exact fields、duplicate key、非有限数和非法坐标均拒绝。
- `status=ok` 时 category query 必填。
- query 为 1–4 个小写英文词；不得在 Python 中生成任务特定对象名。
- general fallback 必须位于 semantic `recommended_order` 的末位；随后才可追加配置驱动的
  curated aliases。只允许沿用当前 parser 的窄去重/顺序补全，不得扩大成任意同义词生成。
- 真正的合同/解析错误保存 rendered prompt/raw response 后 fail closed，不能静默修复语义；不要
  在结构迁移中把现有 semantic parser 的窄 canonicalization 改成新的宽松策略。

### 3.4 Object QC、S1–S3 fallback 与 SAM propagation

#### 名称与默认 profile

这里的 S1/S2/S3 是 open-set 失败救回方案的能力标签，不是仓库中的 Pipeline Stage 1/2/3。

- `MaskConfig` 字段级默认值仍是 fallback disabled，便于显式构造最小配置和单元测试；
- 当前 `configs/pilot_move_pillbottle_pad.yaml` 和
  `configs/pilot_adjust_bottle_target_only.yaml` 都显式启用完整 S1–S3；
- 声明了匹配 `profile` 的 `EXTRACT_MANIFEST.json` 可用
  `--data-path --pick-place/--target-only` 加载对应 profile；缺少 `profile` 的旧 extract 会 fail
  closed，须改用 positional dataset root 与显式 `--config`；
- 其他 baseline/实验 profile 仍可显式关闭某一层；不能把“字段安全默认关闭”写成“所有默认入口
  都关闭”；
- `qwen.allow_query_fallback` 必须继续为 false；open-set query 扩展由
  `mask.qc_query_fallback_enabled` 控制。

当前完整 profile 的关键合同：

```text
open-set mode-specific semantic prompt
open-set mode-specific mask-QC prompt
qc_max_candidates = 8
qc_query_fallback_enabled = true
qc_seed_fallback_enabled = true
qc_bbox_fallback_enabled = true
qc_bbox_max_tokens = 180
qc_max_attempts = 2
```

`qc_max_attempts` 是同一个 Qwen 请求的临时服务重试上限，不是 query、seed 或 fallback 层数。

实验 profile 与默认 profile 的开关基线必须可区分，且每次实验使用独立 output/run 前缀：

| profile | query | seed | bbox |
| --- | ---: | ---: | ---: |
| `open_set_mask_fallback_failures.yaml`（S1） | true | true | false |
| `open_set_mask_fallback_appearance.yaml`（S2） | true | true | false |
| `open_set_mask_fallback_bbox.yaml`（S3） | true | true | true |
| 两个默认 pilot/path profile | true | true | true |

这张表描述当前配置基线，不授权迁移时改变默认值；实验 runner 的 batch summary、失败清单和
`passed=false` 退出语义也必须保持。

#### 三种能力的位置

- S1：semantic query bank + 最多 3 个配置驱动 curated aliases，并在多个合法 seed 间尝试；实际
  候选仍由同一个 text proposal/QC engine 生成；
- S2：mode-specific open-set semantic/QC prompt，允许有属性约束的外观 fallback query，并加强
  对可见完整部件、同步运动结构和夹爪接触证据的判定；
- S3：所有 text query × legal seed 均未通过后，执行 Qwen bbox → SAM box mask。

S2 不是一个排在 S1 后面的 runtime attempt。完整 profile 从 semantic planning 开始就使用 S2
prompt；runtime resolver 的候选顺序只有 text attempts 和最后的 bbox attempts。

#### 必须保持的 attempt 顺序

对每个 required object role：

1. 先检查 semantic precondition；`NO_CLEAR_SEED` 直接得到 rejected，attempt 列表为空，不进入
   text 或 bbox；target-only 不创建 receiver role 的 candidates；
2. 按 `recommended_order` 读取最多 4 个 semantic query；
3. query fallback 开启时，从 `configs/open_set_query_aliases.yaml` 追加最多 3 个 task/role-aware
   alias，规范化并去重；catalog 缺失或格式错误时 fail closed；
4. application 预加载该 role 全部 `seed_eligible` 且 role 合法的 RGB seed；首先使用 semantic plan
   选择的 seed，seed fallback 开启时再按 `LoopContext.seed_candidates(role)` 的稳定顺序追加其余
   合法 seed；缺失合法 seed 是输入/合同 error，不得伪装成 rejected；
5. 每个 seed 生成不超过 `qc_max_candidates` 的实际候选 mask。最多 7 个 text query；符合条件的
   receiver blue-region prior 也占候选槽位，总上限仍为 8；
6. 对候选执行空 mask、面积、component、duplicate 检查，并把实际轮廓交给普通 visual QC；
7. `passed` 立即停止；`rejected/ambiguous` 才能尝试下一合法 seed；任何 request、parser、prompt、
   candidate generation 或 shape `error` 都立即停止当前角色，不得静默进入下一 seed/bbox；
8. 只有所有 text seed attempts 都是 rejected/ambiguous，才按相同 seed 顺序进入 bbox fallback；
9. bbox response 必须是 exact strict JSON 和原始归一化 `xyxy`；不得 clamp、扩张或修正坐标；
10. 合法 bbox 原样交给 SAM box prompt；得到的 `BBOX` 二值 mask 必须通过与 text candidate 完全
    相同的 mechanical gate 和 visual QC（既有 visual-QC confidence threshold 仍适用；bbox
    localization confidence 不单独 gate）；
11. 第一个 passed attempt 成为 selected seed mask；全部失败则保留完整历史并 fail closed。

不得恢复“最大面积自动选择”“候选并集”“逐帧 text mask”或固定 envelope 裁剪。bbox localization
confidence 当前只进入 provenance，不单独决定接受；最终安全门是实际 SAM mask 的普通 visual QC。

S3 localization parser 只接受未包裹 Markdown fence 的 raw JSON，字段必须恰为
`status`、`bbox_xyxy`、`confidence`、`reason`：

- duplicate key、extra/missing field、`NaN`/`Inf`、非法 status 或非有限值均为 `error`，立即停止；
- `status=ok` 要求四个有限 normalized 数，且 `0 <= x0 < x1 <= 1`、`0 <= y0 < y1 <= 1`；
- `status=ambiguous` 才能继续下一个 bbox seed；`status=not_visible` 映射为 rejected；
- 非 `ok` 状态的 bbox 必须为 `null`；任何坐标都不得 clamp、扩张或自动纠正。

#### Attempt provenance

`MaskQCAttempt` 和 `MaskQCAttemptMethod` 是正式审计合同：

- method 目前只有 `text_query`、`bbox_fallback`；
- `(method, seed_frame_id)` 在一个 role report 中唯一；
- 每次 attempt 保存候选、selected query、confidence、reason、model、raw response、rendered prompt
  和 method-specific provenance；
- bbox provenance 保存 localization、原始 bbox、SAM prompt 类型和 `coordinates_clamped=false`；
- passed attempt 的 selection 必须精确指向该 attempt 的候选；rejected/ambiguous attempt 不得伪造
  selection；`qc_max_attempts` 的服务重试不创建额外的 `MaskQCAttempt`；
- nested attempt artifacts 保存每个 seed/method 的候选；flat candidate artifacts 继续指向最终
  role decision 使用的候选，迁移时两者都不能丢失。

配置中已删除的 S4 字段必须继续拒绝。禁止重新加入方向扩框、触边专用放行、降低完整性标准或
传播后的静态 envelope 强制修正。

#### Propagation

- 只传播 identity QC passed 的 selected seed mask，不关心它来自 text 还是 bbox；
- temporal QC 只判断连续性，不能替代 seed identity QC；
- 至少两类严重 temporal 信号才 quarantine；单信号进入 review；
- target 严格 temporal QC 只覆盖普通段到 close completion；hold 像素仍正常发布；
- quarantined channel 不得为了覆盖率发布已知坏像素；
- 普通 episode error 记录后继续；fatal CUDA error 终止 resident worker；
- batch 内继续复用一个 `Sam3Adapter`，每 episode session/temp resource 单独清理。

### 3.5 Gripper

- 只发布 active arm visible gripper；inactive channel 全零且为
  `not_annotated/not_run`，不能当成负样本。
- target/receiver 像素优先于 gripper。
- SAM gripper 复用已保存的 object native tracks，不重新运行 object propagation。
- target-only 在加载不需要的 SAM gripper 模型前继续拒绝该组合。
- URDF 从 source `loop.json` 读取事件 authority，不从 Parquet 再猜一遍。
- URDF derived run 逐像素继承 source object channels，丢弃 source 旧 gripper channels。

### 3.6 Canonical `masks.npz`

新写格式固定为 `robotwin_visible_masks_v3`，必须严格包含八个 key：

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

数据合同：

```text
masks.dtype          == bool
masks.shape          == [4, T, H, W]
instance_names       == target_0, receiver_0, gripper_left, gripper_right
roles                == target, receiver, gripper, gripper
frame_encoding.dtype == uint8
frame_encoding.shape == [4, T]
```

Encoding：`0=absent`、`1=visible`、`2=target_grasp_hold`。编码 2 只允许出现在 `target_0`，且
每个 channel/frame 的非零 encoding 必须与 bool mask presence 完全一致。

旧 `robotwin_visible_masks_v2` 仅只读兼容，loader 合成 0/1 encoding；任何新发布仍写 v3。

状态语义不可合并：

- target-only receiver：`not_applicable`，全零，不创建 receiver artifact 目录；
- 未运行 gripper：`not_annotated/not_run`，全零；
- 已运行失败与 quarantine 保持各自现有语义。

### 3.7 Artifact、lineage 与 publication

结构迁移期间保持现有 public format version 和字段：

- 新 `LoopContext` 当前写 `robotwin_loop_context_v3`；v1/v2 仅兼容读取；
- `MaskRun` 当前为 `robotwin_mask_run_v2`；
- provenance 当前为 `robotwin_frame_provenance_v2`；
- process summary 当前为 `robotwin_process_dataset_summary_v1`；
- canonical mask 为 v3。

Fallback 细节仍留在现有审计边界：`mask_qc.json` 保持 `robotwin_mask_qc_v2`，
`run_manifest.json` 的 `algorithm.automatic_query_fallback` 继续为 `false`，
`mask_qc_fallback_used` 仍只是 bool；method、seed、query、raw response 和 attempt history 不得
被压缩进 summary 或这个 bool。`frame_provenance.json` 继续记录 selected seed/query/QC 字段。

SAM/URDF 的 `gripper_qc` 保持相同字段集合：

```text
backend, status, qc_status, active_arm, selected_candidate,
confidence, reason, forced_fallback, nonempty_frames, quality
```

URDF source lineage 必须继续验证：

- 引用位于 source run 内；
- regular file；
- 无 symlink 和目录逃逸；
- path、SHA-256 和 byte size 一致；
- run/dataset/task/camera/episode/frame identity 一致；
- source/backend/asset/config/implementation identity 一致。

Publication 继续使用同父目录 staging、全树验证、atomic rename。fresh run 不覆盖已有
destination；resume 只跳过完整且未篡改的 episode，不能“修复”或重写被篡改产物。

同一个 validator 可以复用，但 source preflight、publish 前复验、staging 验证、render 前
验证这些调用时机不能合并删除。

### 3.8 外部服务与 optional dependencies

Managed Qwen：

- 已健康的服务保持外部 ownership，不查询 GPU，不停止服务；
- 自动启动仅限允许的 localhost endpoint；
- 只清理自己启动的 process group；启动失败也要清理；
- GPU physical/logical mapping 行为保持现有合同。

Optional dependencies：

- frozen-source URDF 路径不得因顶层 import 强制要求 torch/SAM/OpenCV；
- core、sam3、urdf extras 的部署边界保持不变；
- 包级 `__init__.py` 不得 eager import 重型 optional module。

### 3.9 CLI 与渲染

迁移期间保持已有 CLI alias、参数组合、默认值、退出码和 summary 状态，特别包括：

- `--data-path/--data_path`；
- `--target-only/--target_only`；
- `--output-format` 兼容 alias；
- positional `-` sentinel 行为；
- URDF fresh/frozen-source、`--dry-run`、`--resume`、`--force` 的组合限制；
- exact run ID render 选择。

path-mode 通过 mode profile 自动带入 S1–S3，不新增会静默覆盖配置的 fallback CLI flags；
`--validate-only` 只校验输入，不触碰 Qwen/SAM 服务。

Renderer 只读 canonical masks，不修复 mask；对象覆盖 gripper；held target 继续使用黄色；RGB
尾帧如保留则没有 overlay，并记录 trailing frame 元数据。

## 4. 已解决的文档冲突

| 事项 | 重构前文档描述 | 当前实现/测试 | 已确认合同 |
| --- | --- | --- | --- |
| 默认 gripper backend | `docs/README.md` 曾称 SAM | CLI 与 `AnnotationSpec` 默认 URDF | 默认 URDF；SAM 必须显式请求 |
| 新 `loop.json` 版本 | `architecture.md` 称 v1 | `LoopContext.to_json()` 写 v3 | 新写 v3，旧读兼容 |
| lineage 描述 | 文档主要描述 v1 | 无 source contract 时使用 lineage v1；带 contract/receipt 时使用 lineage v2 | validator 统一验证 v1/v2；frozen-source 可消费任一版；contract writer 写 v2、reader 兼容 v1/v2；receipt 仅适用于 lineage v2 |

`docs/README.md` 与 `docs/architecture.md` 已按右列修正。若以后改变默认 backend 或 schema，
必须单独形成行为变更任务、迁移说明和测试；不得把它称为兼容层清理。

## 5. 重构前热点与迁移目标（历史）

本节保留初始审查的规模、路径和目标映射，用于解释提交顺序。文中的“当前”均指
`fb31b6c` 附近的重构前状态；实施后的 owner 以第 13 节为准。

### 5.1 `application/dataset_runtime.py`

当前约 3371 行，关键热点：

- `process_urdf_source_run()`：约 795 行；
- `process_dataset()`：约 451 行；
- `process_live_urdf_pipeline()`：约 294 行；
- `_run_streaming_source_urdf_workers()`：约 229 行；
- `_parse_args()`/`_run_from_args()`：CLI 与 application 混合；
- `SamRuntime`：多个 `Callable[..., Any]` 绑定同级模块私有符号。

目标映射：

| 当前责任 | 建议目标 |
| --- | --- |
| discovery / dynamic manifest | `application/discovery.py` |
| SAM dataset lifecycle | `application/sam_workflow.py` |
| frozen/live URDF orchestration | `application/urdf_workflow.py` |
| multiprocessing wire message DTO/decoder | `application/streaming.py` |
| worker entry、Pipe/Queue transport、并发协调与 GPU handoff | `application/urdf_runtime.py` |
| renderer coordination | `application/rendering.py` 或窄 adapter |
| argparse / path-mode dispatch | `cli/process_dataset.py` |
| run request/record/summary | `models/process_run.py` |

不要一次切开整个文件。先抽纯 discovery 与 typed records，再抽一个完整 workflow；旧函数保持
兼容委托，直到内部调用和测试迁移完成。

### 5.2 `scripts/render_urdf_gripper_masks.py`

当前约 2654 行，包含：

- CLI/`RunConfig`；
- canonical mask loader/composer；
- JSON、hash、asset identity；
- ffmpeg decode/overlay；
- episode render/quality/artifact validation；
- run contract/checkpoint/resume；
- `IncrementalUrdfEpisodeWorker`；
- `run_experiment()`。

`IncrementalUrdfEpisodeWorker` 与 `run_experiment()` 都有 load → render → quality → save →
overlay → validate 生命周期。先提取一个共享 `UrdfEpisodeProcessor` 或等价 application
service，让两条路径调用同一实现；不要创建一个同时负责 CLI、contract、renderer、publisher
的 God class。

推荐目标：

```text
adapters/urdf/data.py          existing loader consolidation
adapters/urdf/renderer.py      geometry resource owner
adapters/urdf/finger_fit.py    candidate/score/select
adapters/urdf/runner.py        private backend episode product
application/urdf_batch.py      checkpoint/resume/batch lifecycle
cli/render_urdf.py             argparse only
```

原脚本先变成 compatibility re-export；测试迁移后才进一步收窄。

### 5.3 Canonical mask 与 publisher

当前逻辑分布在：

- `mask_schema.py`；
- `pipeline/sam_stage.py::save_sam_artifacts()`；
- `urdf_gripper_publisher.py`；
- `scripts/render_coverage20_videos.py::_load_masks()`；
- `scripts/render_urdf_gripper_masks.py::load_four_channel_masks()`。

目标建立：

```text
CanonicalMaskBundle       唯一 immutable/validated NPZ DTO
CanonicalMaskCodec        v2 reader + v3 reader/writer + schema validation（无状态模块/函数）
CanonicalMaskPublisher    SAM/URDF 共用的 canonical NPZ 原子 publish
UrdfCanonicalEpisodePublisher  URDF derived tree staging/validation/atomic rename
SourceLineageValidator    source trust boundary, separate from public codec
```

不要同时引入语义相同的 `CanonicalMasks` 和 `CanonicalMaskBundle`。可以先让所有 reader 使用新
codec，再逐个迁移 writer；不要在一个提交中同时切换 SAM writer 和 URDF publisher。

### 5.4 `pipeline/sam_stage.py`

`save_sam_artifacts()` 约 397 行，混合：

- role diagnostics；
- PNG/NPZ 写入；
- gripper result；
- canonical four-channel composition；
- manifest/provenance。

目标：

- `sam_stage` 只返回 object typed results；
- 不再导入 `ArtifactStore`；
- 不再依赖 `GripperStageResult`；
- application 组合 object/gripper producer output；
- publisher 负责 canonical files。

迁移期间必须保证现有 role-specific diagnostics 路径、内容和引用不变。

### 5.5 `pipeline/mask_qc.py`

当前约 1460 行，混合 proposal、image helpers、prompt、response parsing、retry、decision 和
artifact save。建议按 feature cohesion 拆为：

```text
object_mask/planner.py       semantic query、alias、合法 seed 的有序计划
object_mask/proposals.py     text proposal 与 bbox -> SAM box proposal
object_mask/qc.py            shared mechanical gate 与 visual QC
object_mask/resolver.py      唯一的 text -> seed -> bbox 状态机
object_mask/artifacts.py     attempt artifact adapter/publication side only
```

`resolver.py` 必须知道所有 proposal 方法、seed、状态和停止条件；不能把 query、seed、bbox 拆成
互不知情的策略类，再由外层按文件顺序猜测下一步。proposal、metrics、strict bbox parser 可以
是纯函数；SAM/Qwen 能力用最小窄端口表达。不要为每一种 fallback 创建继承子类，一个 resolver
和少量不可变 DTO 已足够。

### 5.6 `pipeline/gripper_stage.py`

当前约 1770 行，混合 state load、3D projection、ROI、candidate、panel、Qwen QC、SAM
propagation、composition。建议目标：

```text
gripper/sam/geometry.py      projection/rotation/ROI pure functions
gripper/sam/candidates.py    candidate construction and quality gate
gripper/sam/qc.py            prompt/response/selection
gripper/sam/annotator.py     stage orchestration
```

Parquet/state 读取应由 dataset port 提供，不留在几何模块。`SamGripperAnnotator` 可以是有依赖
的 application service；数学函数不要变成方法。

### 5.7 Timeline

当前需对拍的两组事实来源：

- `models/timeline.py` 与 `urdf_gripper_data.ActiveGripperLoop`；
- `pipeline/state_loop.py` 与 `urdf_gripper_data.py` 中的 filter/transition/loop detection。

最终唯一当前领域类型为 `PickPlaceEvents | TargetOnlyEvents`。旧 v1/v2/v3 JSON 差异进入
`LoopContextCodec`/adapter。此阶段可能改变 mask window，必须最后实施，并对所有已有 fixture
做旧新 detector 同输入逐字段比较。

### 5.8 其他脚本

- `render_coverage20_videos.py`：提取 canonical archive reader、video encoder 和 review-sheet
  builder；脚本只保留 CLI。
- `render_open_set_bad_cases.py`：停止调用 renderer 私有 `_load_masks`/`_sha256`。
- `prepare_target_only_dataset.py`：先补 selection、staging rollback、validation 的
  characterization tests，再提取 planner/materializer；不要先大搬家。
- `materialize_native_tracking_run.py`：复用现有 temporal QC/composition，先补 provenance 测试。
- `serve_qwen.py`：model/backend/HTTP implementation 可迁入 `adapters/qwen_server.py`；保持旧
  script symbols 的临时 re-export，因为测试可能通过 `runpy` 访问。
- `manage_qwen_process.py` 已较薄，优先级低。

## 6. 建议的最小类型与类

只补跨模块、高风险边界：

```python
@dataclass(frozen=True)
class ProcessRequest: ...

@dataclass(frozen=True)
class EpisodeRecord: ...

@dataclass(frozen=True)
class ProcessSummary: ...

@dataclass(frozen=True)
class CanonicalMaskBundle: ...

@dataclass(frozen=True)
class UrdfRunPlan: ...
```

有状态或生命周期类限制为：

- `EpisodePipeline`；
- `DatasetPipeline`；
- `SamWorkflow`/`UrdfWorkflow`；
- `ObjectMaskResolver`（仅拥有 object attempt ordering、停止条件和审计收集）；
- `CanonicalMaskPublisher`；
- `UrdfCanonicalEpisodePublisher`；
- `AlohaUrdfRenderer`；
- `FingerPoseFitter`；
- 必要的 resident worker。

窄端口示例：

```python
class DatasetReader(Protocol): ...
class VisionLanguageClient(Protocol): ...
class ObjectSeedMaskBackend(Protocol): ...
class MaskTracker(Protocol): ...
class ArtifactRepository(Protocol): ...
class ProgressSink(Protocol): ...
```

端口应靠近消费方，方法数保持最少。不要把所有接口放进一个大型 `interfaces.py`，也不要引入
DI container。`ObjectSeedMaskBackend` 至少要显式区分 `text_query_masks(...)` 与 `box_mask(...)`，
不要用隐式 cast 把 bbox 能力伪装成 text backend。构造函数显式注入即可。`QueryBank`、
`BboxLocalization`、`MaskQCAttempt` 等
已有 frozen DTO 直接复用；不要再包一层同义 model。

## 7. 详细迁移阶段

以下步骤保留为迁移审计记录。最终状态：

| 阶段 | 状态 | 当前结果 |
| --- | --- | --- |
| R0 | 完成 | loop/mask/schema、CLI、fallback、resume/tamper 和 optional-import 合同已由 focused tests 冻结 |
| R1 | 完成 | renderer 与 URDF batch 生产实现进入 package；`src/ -> scripts/` 导入为零；剩余 launcher seam 见 13.2 |
| R2 | 完成 | `CanonicalMaskBundle`/reader 统一 v2/v3 校验，旧 reader seam 委托或有明确兼容条件 |
| R3 | 完成 | discovery、typed process record/summary、streaming message 已拆出，public discovery 不返回 `Any` |
| R4 | 完成 | `DatasetPipeline`、`SamWorkflow`、`UrdfWorkflow` 持有正式 dataset dispatch/编排；`episode_pipeline.py` 提供具体 executors，`EpisodePipeline` 收窄为 standalone façade；runtime 反向委托 |
| R5 | 完成 | SAM/URDF 共用中立 `CanonicalMaskPublisher`；URDF 整树发布与 source lineage 各有唯一 owner |
| R6 | 完成 | object resolver、artifact writer、SAM gripper、temporal QC 和 finger fitting 已按 feature cohesion 拆分 |
| R7 | 完成 | 当前 detector/type/codec 统一；历史 type alias 只为剩余调用方保留 |
| R8 | 完成（保留受控 shim） | Ruff 与 mypy 清零；星号导入、dataset launcher module replacement 和两个 renderer `sys.modules` proxy 已移除；其余 shim 已登记调用方和删除条件 |

### R0：行为冻结与决策记录

- 允许改动：测试、golden fixture、本文档冲突的事实说明。
- 禁止改动：算法、schema、默认 backend、module move。

必须补强：

- loop v3 writer + v1/v2/v3 reader；
- mask v2 reader + v3 exact eight-key writer；
- target-only receiver N/A；
- inactive gripper not annotated；
- SAM/URDF public schema 等价；
- source bitwise inheritance；
- atomic publish/resume/tamper；
- CLI aliases/default/invalid option combinations；
- frozen URDF import without SAM/OpenCV。

S1–S3 在 R0 必须单独冻结，至少覆盖：

- 两个默认 path-mode profile 的 open-set prompt、`qc_max_candidates=8`、query/seed/bbox 开关；
- `MaskConfig` 安全默认关闭、S1/S2 实验 profile 与 S3 bbox profile 的显式差异；
- `qwen.allow_query_fallback=true` 拒绝、S4 已删除字段拒绝，以及 fallback 开关的依赖校验；
- semantic query/alias/seed 顺序、每 seed 候选上限和 blue-region 保留槽位；
- `NO_CLEAR_SEED` 无 attempts、`passed` 立即停止、`rejected/ambiguous` 才继续、任何 `error`
  立即停止且不进入 bbox；
- strict bbox JSON、原始 normalized `xyxy`、不 clamp，以及 text/bbox 的完整 nested/flat attempt
  artifacts 和 provenance。

退出条件：所有 golden contract 可独立指出字段、dtype、shape、status 和失败策略。

### R1：收窄 import 与迁移脚本生产代码

- 允许改动：纯机械 module move、兼容 re-export、内部 direct import。
- 禁止改动：业务分支、返回 payload、CLI 参数。

步骤：

1. 收窄 `application/__init__.py`、`pipeline/__init__.py` eager export；
2. 内部调用改为从具体模块导入；
3. 将 shared renderer implementation 移入 `src`；
4. 将 URDF batch engine 移入 `src`；
5. application 只导入包内 API；
6. scripts 保持 compatibility wrapper；
7. 用 import smoke test 验证 core/frozen URDF 不加载重型 optional modules。

退出条件：`rg` 证明 `src/` 不再导入 `scripts`，既有 script CLI 和 test patch seam 仍可用。

### R2：统一 canonical mask codec（先 reader）

- 允许改动：增加 typed DTO/codec，将多个 reader 委托给它。
- 禁止改动：切换 public writer、改变错误宽松度。

步骤：

1. 从现有 reader 测试归纳 v2/v3 exact rules；
2. 实现唯一已校验的 `CanonicalMaskBundle` DTO 和 codec；
3. 迁移 shared renderer reader；
4. 迁移 URDF runner reader；
5. 迁移 publisher source reader；
6. 保留旧函数为兼容 wrapper；
7. 对同一 fixture 比较旧新返回的数组和 metadata。

退出条件：v2/v3 读取与拒绝行为等价，writer 尚未变化。

### R3：拆 dataset runtime 的低风险部分

- 允许改动：discovery、typed request/record/summary、UI message types。
- 禁止改动：episode 执行顺序、backend lifetime、summary JSON。

步骤：

1. 提取 `DiscoveryResult`、dataset path builder 和 dynamic manifest；
2. 统一重复 episode path helpers；
3. 为 process request/record/summary 建 dataclass + mapper；
4. 为 subprocess messages 建 tagged union；
5. 旧 `dataset_runtime` 函数委托新模块；
6. 输出 JSON 与迁移前 golden 逐字段比较。

退出条件：`DatasetPipeline.discover()` 不返回 `Any`，summary 对外字节/结构等价。

### R4：建立真实 workflow 与 Pipeline

- 允许改动：把编排代码迁入 `SamWorkflow`、`UrdfWorkflow`、真实 Pipeline。
- 禁止改动：模型算法、publisher、ordinary/fatal failure 分类。

先迁 SAM：

1. resume scan；
2. Qwen health；
3. resident SAM backend acquire/release；
4. per-episode Qwen → `ObjectMaskResolver`（text query → legal seed → bbox/SAM）→ object native
   propagation → optional gripper；
5. 保持 `passed`/`rejected`/`ambiguous`/`error` 分类、selected seed 和 attempt artifacts；
6. fatal CUDA stop；
7. render/summary。

再迁 URDF：

1. source selection；
2. live source ownership；
3. serial/streaming backend；
4. publish/validate；
5. canonical render；
6. partial/incomplete classification。

退出条件：`DatasetPipeline`/workflow 持有真实 dataset 依赖和顺序，episode module 持有具体
executors；旧 runtime 只剩兼容委托和待迁 CLI。

### R5：统一 canonical writer/publisher

- 允许改动：建立 bundle/publisher，逐个 backend 切换。
- 禁止改动：public schema、路径、lineage、atomicity。

步骤：

1. 从 SAM writer 构造 `CanonicalMaskBundle`，但先比较新旧 payload；
2. 切换 SAM writer；
3. 全量运行 SAM artifact tests；
4. 将 URDF source validation 与 public payload construction 分离；
5. 让 URDF publisher 使用同一 codec/validator；
6. 保留 lineage validator 和各信任边界复验；
7. 对 source object channels 做 bitwise equality；
8. 对完整 published tree 做 file-set/hash 比较。

退出条件：公共 mask reader/writer/validator 只有一个事实来源，SAM/URDF public contract 完全
一致，tamper/resume 测试全绿。

### R6：拆 stage 算法与 artifact I/O

- 允许改动：按 feature cohesion 移动纯函数与小 service。
- 禁止改动：阈值、prompt、fallback、候选顺序、mask composition。

顺序：

1. `mask_qc` 的纯 metrics、query/alias planner 和 strict bbox parser；
2. 保留一个 resolver，先对拍 text → legal seed → bbox 的全局顺序和 error-stop，再拆 proposal/QC
   函数；
3. mask QC attempt artifact saving；
4. SAM stage artifact saving；
5. gripper geometry/candidates/QC；
6. URDF finger fitting。

拆 object QC 时，不能让每个 proposal strategy 自己决定“是否进入下一 seed”或“是否进入 bbox”；
这些决定只能由 resolver 作出。S2 的 open-set prompt/alias catalog 是配置资产，不应复制为一个
新的 S2 类。

对数值函数使用 fixed-array golden tests，要求 exact equality 或现有测试指定的 tolerance。

退出条件：pipeline stage 不依赖 `ArtifactStore` 或具体 HTTP client；renderer 只管理渲染资源，
fitter 只管理搜索策略。

### R7：统一 timeline

- 允许改动：统一 detector/type/codec。
- 禁止改动：阈值、窗口定义、source authority。

步骤：

1. 为 State Loop 与 URDF detector 建同输入对拍；
2. 将 pure detection 提取为唯一实现；
3. 让 Stage 1 调用唯一 detector；
4. 让 URDF 只解析 authoritative loop artifact；
5. v1/v2/v3 compatibility 留在 codec；
6. 比较 events/windows/semantic seeds/canonical encoding；
7. 最后移除 `ActiveGripperLoop`、`LoopEvents` 等不再需要的兼容类型。

退出条件：仓库只剩一个当前 timeline detector 和一个当前事件类型体系。

### R8：删除迁移 shim 与静态质量清债

只有满足以下条件才能删除 shim：

- `rg` 找不到仓内旧 import/call；
- tests 不再 monkeypatch 旧私有 module global；
- CLI contract tests 覆盖稳定 public seam；
- 至少一个发布版本或用户明确允许删除兼容入口。

最后处理：

- `scripts/process_dataset.py` 的 `sys.modules` 替换；
- `run_target_receiver.py` 星号导入；
- `target_receiver_only` 等废弃参数；
- 空转 facade；
- package barrel export；
- `models/`/`domain/` 命名整理；
- Ruff/mypy 历史债务。

## 8. 测试与验证阶梯

### 8.1 每个小改动

```bash
git diff --check
.venv/bin/python -m pytest tests/unit/<focused_test>.py -q
just test
```

检查：

- `git status --short` 仅包含本阶段预期文件；
- 无意外格式化或用户文件变化；
- 新增代码不引入新的 Ruff/mypy 错误。

### 8.2 每个模块边界迁移

```bash
just test-all
just lint
.venv/bin/python -m mypy src
git diff --check
```

Ruff/mypy 若仍有历史错误，保存前后计数和涉及文件；本阶段不得新增。涉及 optional dependency
时增加 import smoke tests。

### 8.3 重点测试集合

编排：

```bash
.venv/bin/python -m pytest \
  tests/unit/test_process_dataset.py \
  tests/unit/test_pipeline_cli.py \
  tests/unit/test_managed_qwen.py -q
```

S1–S3 object resolution：

```bash
.venv/bin/python -m pytest \
  tests/unit/test_open_set_queries.py \
  tests/unit/test_mask_qc.py \
  tests/unit/test_mask_qc_attempts.py \
  tests/unit/test_config.py \
  tests/unit/test_qwen_stage.py -q
```

重点断言：默认 path-mode profile、query/alias/seed 顺序、候选上限、`NO_CLEAR_SEED`、error-stop、
strict bbox parser、text-first、统一 visual QC、attempt provenance 和 S4 rejection；retry 次数
不得增加 `MaskQCAttempt` 数量。

Canonical/publication：

```bash
.venv/bin/python -m pytest \
  tests/unit/test_mask_schema.py \
  tests/unit/test_sam_stage.py \
  tests/unit/test_urdf_canonical_publisher.py \
  tests/unit/test_render_urdf_gripper_masks.py \
  tests/unit/test_render_coverage20_videos.py -q
```

Timeline：

```bash
.venv/bin/python -m pytest \
  tests/unit/test_models.py \
  tests/unit/test_state_loop.py \
  tests/unit/test_urdf_gripper_data.py \
  tests/integration/test_coverage20_dataset.py \
  tests/integration/test_target_only_dataset.py -q
```

### 8.4 真实 backend 验收

只有相关结构阶段完成且 CPU tests 全绿后执行：

1. 单 episode fake/backend smoke；
2. 一条 left-arm 与一条 right-arm；
3. frozen-source URDF dry-run、resume、tamper；
4. coverage subset；
5. full batch；
6. exact-run overlay/review sheet 人工检查。

必须核对：

- 默认 pick-place、target-only、path-mode 均加载预期的 S1–S3 profile；text-first 和 error-stop
  在真实服务下与 CPU contract test 一致；
- source object channels bitwise equal；
- inactive arm 全零且 status 正确；
- canonical NPZ 严格八键；
- frame count 服从 Parquet；
- lineage/hash/implementation identity；
- resume 不重写完整 episode；
- held target 颜色和对象/gripper 覆盖顺序。

## 9. 测试迁移注意事项

当前测试中有较多对 `_private` 符号、script module、`runpy` 和 monkeypatch module global 的直接
绑定。这些测试是迁移阻力，不一定是公共 API。

处理顺序：

1. 先增加 public behavior/contract test；
2. 保留旧 private wrapper；
3. 将算法单测迁到新的稳定模块；
4. 将 CLI 测试改为 public parse/run seam；
5. `rg` 确认无调用后删除 wrapper。

不要为了让搬移快速通过而批量删除测试，也不要永久保留一个空壳模块只为 monkeypatch 私有
global。兼容 shim 必须有明确删除条件。

以下脚本在初始审查时 characterization 较弱，迁移前优先补测试：

- `prepare_target_only_dataset.py` 的 selection/arm quota/staging rollback；
- `materialize_native_tracking_run.py` 的 composition/provenance；
- `benchmark_temporal_tracking.py` 的 aggregation/report。

## 10. 严格禁止事项

结构重构中禁止：

- 全仓重写或一次性搬完所有大模块；
- 修改 mask schema、format version、status 语义或路径布局；
- 修改 prompt、阈值、query 顺序、fallback 默认值或 temporal QC policy；
- 让 bbox 抢在 text candidate 前，或把 `error` 静默降级为下一个 seed/bbox attempt；
- 对非法 bbox 做 clamp、扩张、固定 envelope 修正，或恢复 S4 的方向/触边/传播放行；
- 按面积选最大候选、合并候选、把非空 mask 当作通过，或丢弃非 selected attempt 的 provenance；
- 为 S1、S2、S3 建独立 Pipeline/Strategy 继承树，或把 S2 prompt policy 复制成运行时 stage；
- 修改默认 backend，除非用户另行明确要求；
- 在 Python 中硬编码 task-specific object name；
- 为每个 task/backend 建继承子类；
- 引入 DI container、event bus、ORM、全量 Pydantic schema 或万能 Stage/Manager；
- 为每个纯函数创建 class/Protocol；
- 为追求 DRY 删除不同信任边界的复验；
- eager import torch/SAM/OpenCV/URDF renderer 到 core/frozen-source path；
- 用宽泛 `except Exception` 改变 ordinary/fatal failure 分类；
- 把 `Any` 从一个模块机械搬到另一个模块并称为完成类型化；
- 在结构提交中全仓 format，制造无关 diff；
- 删除用户数据、artifacts、run output 或未跟踪文件；
- 使用旧 run 作为“最佳结果”隐式回退进行验收。

## 11. 每个重构 PR/提交的说明模板

```text
Scope
- 本次只迁移哪个职责
- 明确未改变的算法/schema/default

Old -> New
- 旧模块/符号
- 新模块/符号
- compatibility shim 和删除条件

Contracts checked
- timeline / NPZ / manifest / lineage / CLI 中涉及哪些

Validation
- focused tests
- just test / just test-all
- Ruff/mypy 前后基线
- 若适用，backend smoke 与 exact run ID

Risk and rollback
- 最大风险
- 可恢复到哪个旧委托入口
```

## 12. 完成定义

只有同时满足以下条件，才可宣称本次精简重构完成：

- `src/` 对 `scripts/` 的反向依赖为零；
- scripts/CLI 只负责解析、组装、UI 和退出码；
- `DatasetPipeline` 与 workflow 承担正式 dataset dispatch/编排；`EpisodePipeline` 仅承担
  standalone/分阶段 CLI 顺序；
- `dataset_runtime.py` 不再是永久 God module，最终可删除或仅保留窄兼容入口；
- pipeline 计算与 canonical publication 解耦；
- canonical mask codec/validator/publisher 只有一个当前事实来源；
- timeline detector/type 只有一个当前事实来源；
- object-mask query/seed/bbox 顺序只有一个 resolver owner，S1–S3 不产生平行类树；
- 重复 reader/writer、重复 timeline 规则和空转 facade 已删除或有明确删除条件；
- SAM 与 URDF public artifacts 保持同一合同；
- frozen-source URDF 仍不要求 SAM runtime；
- fail-closed、lineage、atomic publish、resume/tamper 行为完整保留；
- 所有 unit/integration tests 通过；
- Ruff/mypy 无新增债务，并有后续清零计划；
- compatibility shim 已删除，或记录明确的剩余调用方与删除条件；
- 文档中的默认 backend、format version 和实际实现一致。

本分支按“shim 可以保留，但必须是委托且有调用方/删除条件”这一条完成结构验收。这里的
core script/CLI 指本次迁移范围内的 dataset、episode、renderer 和 URDF batch 入口；数据准备、
离线 materialization、benchmark、模型服务、实验与 review utility 不因本次重构自动变成同一条
production pipeline。具体包括 `prepare_target_only_dataset.py`、
`materialize_native_tracking_run.py`、`benchmark_temporal_tracking.py`、`serve_qwen.py`、
`run_open_set_failure_experiment.py` 和 `render_open_set_bad_cases.py`；其现有调用方、测试边界与
迁移触发条件见 13.4。它们仍然较厚不构成“core pipeline script thinness”未完成，但也不能借此
豁免各自的 characterization、原子写入或服务生命周期合同。

## 13. 实施结果与兼容层清单

### 13.1 Canonical owners

| 合同 | 唯一当前 owner | 兼容边界 |
| --- | --- | --- |
| dataset discovery | `application/discovery.py` | `dataset_runtime.discover_episodes()` 只委托 |
| SAM dataset 用例 | `DatasetPipeline` → `SamWorkflow` | `dataset_runtime.process_dataset()` 保留旧参数/monkeypatch seam 后反向委托 |
| URDF dataset 用例 | `UrdfWorkflow` | `dataset_runtime.py` 只组装 frozen/live/streaming hooks 和 CLI policy |
| URDF runtime dependencies | `application/urdf_runtime.py` | `dataset_runtime.py` 仅保留旧符号 alias 与 hook/CLI 组装；alias 指向 canonical runtime |
| 单 episode executor / standalone 顺序 | `application/episode_pipeline.py` module-level executors；standalone façade 为 `EpisodePipeline` | 正式 dataset 顺序由 `SamWorkflow` 调用这些 executors |
| object attempt 顺序 | `ObjectMaskResolver` | `mask_qc._role_query_candidates()` 仅为旧测试委托 planner |
| object-mask QC artifact publication | `application/mask_qc_artifacts.py` | `pipeline/object_mask/artifacts.py` 与旧 package barrel 只委托 |
| canonical mask DTO/codec | `adapters/canonical_masks.py` | v2 是正式只读兼容；renderer/URDF 的历史宽松 archive fallback 单独登记在 13.2，不得新增规则 |
| canonical NPZ publication | `adapters/canonical_publication.py::CanonicalMaskPublisher` | `write_canonical_masks()` 是旧函数入口，SAM/URDF 已直接使用 publisher |
| URDF 整树 publication | `UrdfCanonicalEpisodePublisher` | 旧 publish/validate 函数委托该 owner |
| source lineage | `SourceLineageValidator` | 无 source contract 为 lineage v1，带 contract/receipt 为 lineage v2；frozen-source 可消费任一版本；当前 contract writer 写 v2、reader 兼容 v1/v2，旧 validate 函数均委托该 owner |
| timeline detector/type | `pipeline/timeline_detector.py` + `PickPlaceEvents` / `TargetOnlyEvents` | v1/v2/v3 JSON 由 codec 读取；旧 Python 类型名只是 alias |
| loop JSON codec | `adapters/loop_context_codec.py` | 新写 v3；v1/v2/v3 只读兼容 |
| public renderer | `adapters/rendering.py` 的 public API | 私有名称 wrapper 仅服务旧 import/test |
| URDF geometry/FK renderer | `urdf_gripper_renderer.py::AlohaUrdfRenderer` | `application/urdf_batch.py` 在执行边界懒加载；finger search 由 `adapters/urdf/finger_fit.py` 持有，未来目录移动不得复制实现 |

### 13.2 Compatibility inventory

以下清单是完成状态的一部分，不是未登记 TODO。删除任何一项前都要同时满足对应条件，并重新
运行 contract tests；“保留一个 release”从包含本次重构的首次发布版本开始计算。

`scripts/render_coverage20_videos.py` 与 `scripts/render_urdf_gripper_masks.py` 已不在清单中：二者
现在只是 `from package_module import main` 加 `if __name__ == "__main__"`，没有 module proxy 或
生产符号 re-export；测试只保留轻量的 `main` delegation assertion。

| 兼容面 | 仍在使用者 | 当前委托关系 | 删除条件 |
| --- | --- | --- | --- |
| `application/dataset_runtime.py` 与 `scripts/process_dataset.py` 的动态属性入口 | 正式 `just process` launcher、`run_open_set_failure_experiment.py`、`test_process_dataset.py`、`test_process_terminal_ui.py` | SAM 调用反向进入 `DatasetPipeline`/`SamWorkflow`；URDF 进入 `UrdfWorkflow`，其进程/GPU/选择逻辑由 `application/urdf_runtime.py` 唯一实现；本模块保留 CLI、hook 组装与旧 patch seam | 建立稳定 package CLI/streaming seam，实验 runner 和测试改用 public API，至少保留一个兼容发布后再删除转发符号；launcher 文件名可继续保留 |
| `scripts/run_target_receiver.py` | 分阶段 CLI、`test_pipeline_cli.py` | `__getattr__` 转发 `application.episode_pipeline`，不含 stage 实现 | 测试与仓内 import 迁到 package API，并完成一个兼容发布；保留可执行 launcher，删除 module re-export |
| `application/episode_pipeline_api.py` | `test_pipeline_cli.py` | re-export canonical `EpisodePipeline` | 仓内 import 清零并保留一个兼容发布后删除文件 |
| lazy package barrel exports (`application/__init__.py`、`pipeline/__init__.py`、`adapters/__init__.py`) | 多个脚本、unit/integration tests 和可能的外部 import | lazy import 具体 owner，避免加载重型 optional dependency | 仓内全部改为 concrete imports，并在下一次 public API 版本变更中宣布；不可改回 eager import |
| `LoopEvents`、`ActiveGripperLoop`、`ActiveGripperEvents` | 多个 timeline/object/SAM tests；`urdf_gripper_data.py` 的旧签名和 URDF tests | 分别 alias 到 `PickPlaceEvents` / 当前 `TimelineEvents`，没有第二个 detector/type 实现 | 生产签名、测试和外部文档全部改用当前类型；保留 JSON 旧读不受 alias 删除影响；一个兼容发布后移除 export |
| `AuthoritativeLoopContext` | `urdf_gripper_data.py`、loop codec/URDF tests | alias 到 `adapters.loop_context_codec.AuthoritativeLoopContext`，URDF 继续从 authoritative `loop.json` 读取事件 | URDF/public callers 与 tests 改用 codec 类型并完成一个兼容发布；旧 JSON 读取合同不变 |
| `UrdfGripperEpisodeData.loop` 与 `EpisodePlan.loop` properties | 旧 episode/plan callers 与 tests | 委托当前 `events`/timeline owner，不复制事件检测 | 仓内调用改用明确的 `events` 字段并完成一个兼容发布 |
| `PickPlaceEvents.start`、`end`、`inclusive_window`、`target_window` properties | timeline/codec/URDF callers 与 tests | 当前 `PickPlaceEvents` 的只读便利属性；窗口计算仍只有 timeline owner | 所有调用改用显式事件字段/窗口 API，并完成一个兼容发布后删除便利属性 |
| `SemanticPlan.target`、`SemanticPlan.receiver` properties | Qwen/SAM callers 与 tests | 通过 `for_role()` 委托 `role_plans` 中的 typed role plan，不创建第二份语义状态 | 仓内调用改用 `role_plans`/`for_role()` 并完成一个兼容发布 |
| streaming short aliases（`Event`、`Error`、`SourceEpisode`、`SourceResult`、`UrdfEpisode`、`UrdfResult`）与 legacy tuple decoders | `application/urdf_runtime.py`、streaming protocol tests 和可能的旧 package callers | short names 直接 alias 到对应 `*Message` `NamedTuple`；`decode_ready_episode()`/`decode_message()` 接受旧 plain tuple wire shape，`try_decode_message()` 对 malformed legacy message 返回 `None`，没有第二套 coordinator protocol | producers/consumers 全部改用 canonical message classes、旧 tuple queue 已迁移并完成一个兼容发布；若收紧 malformed-message policy，须先冻结并显式版本化新 wire contract |
| `pipeline/gripper_stage.py` 与 `pipeline` barrel 中的旧 gripper exports | gripper 单测及历史 package import | 算法转发到 `pipeline/gripper/sam/*`；本文件仅保留 object-track reader 和 export facade | object-track reader 迁到明确 adapter、测试改用 concrete modules、`rg` 无仓内旧 import，再经一个兼容发布删除 facade |
| `urdf_gripper_renderer.py` 的 finger-fit re-exports | `test_urdf_gripper_renderer.py` 与可能的旧 package callers；`AlohaUrdfRenderer` 内部也使用 `FingerPoseFitter` 名称 | constants/types/functions 委托 `adapters/urdf/finger_fit.py`；URDF parse/FK/renderer 本身仍是该模块的 canonical 实现 | 内部使用和测试迁到 concrete finger-fit module、仓内旧 re-export 调用清零，并完成一个兼容发布；不得在移动 geometry renderer 时复制 fitting 实现 |
| `mask_qc._role_query_candidates()` | `test_open_set_queries.py` | 委托 `object_mask.planner.plan_role_queries()` | 测试改测 public planner 且 `rg` 无调用后删除 |
| `pipeline.object_mask.artifacts.save_mask_qc_artifacts()` | `test_mask_qc.py`、`test_mask_qc_attempts.py` 与旧 package barrel | 委托 `application.mask_qc_artifacts.save_mask_qc_artifacts()`；pipeline compatibility module 不再持有 `ArtifactStore` 依赖 | 测试与仓内调用改用 application owner，并完成一个兼容发布后删除 wrapper/export |
| renderer 私有 `_load_masks`、`_sha256`、`_output_video_name` | renderer characterization tests；adapter 内尚有一个旧名字调用 | 分别委托 `load_masks`、`file_sha256`、`output_video_name` | 内部调用与测试全部迁到 public 名称，并保留一个兼容发布后删除；v2 reader 本身继续保留 |
| renderer `_load_masks_compat()` 宽松 archive fallback | `load_masks()` 在严格 codec 拒绝历史 v2/v3 archive 后调用；reader characterization tests 冻结 extra key、可转换 mask dtype、缺失 `qc_status` 和旧非 canonical label 行为 | 当前 canonical archive 始终先调用 `read_canonical_masks()`；fallback 只维持重构前 renderer 可读的历史私有产物，不是新 writer 或 schema authority | 盘点/迁移仍需渲染的历史 archive，提供转换器或明确 EOL，并将调用方与 characterization tests 改为严格 `CanonicalMaskBundle` 后删除；删除前不得扩大接受范围 |
| renderer `_candidate()` selection metadata pre-read | `select_best_masks()` 的跨 run 历史“best current”选择与 renderer tests | 只预读 `roles`、`annotation_status`、`qc_status` 计算候选分数；选中后仍由 `load_masks()` 通过 strict canonical codec（再到已登记的历史 fallback）加载，不能作为 schema/payload reader 使用 | “best current”历史选择功能 EOL，或 codec 提供轻量 typed metadata API 后迁移并删除直接 archive 预读 |
| `urdf_batch.load_four_channel_masks()` / `_load_four_channel_masks_compat()` 与 publisher source-reader fallback | URDF batch 内部、URDF/reader characterization tests、历史 private product archive | canonical 输入先调用 `read_canonical_masks()`；fallback 只维持已冻结的旧 archive 错误/读取合同 | URDF batch 内部改用 `CanonicalMaskBundle`，旧 private product format 明确 EOL 或有迁移器，characterization tests 随之更新后删除；不能提前删除 v2 canonical 读取 |
| `canonical_masks.write_canonical_masks()` | canonical codec compatibility tests、可能的旧外部调用 | 延迟构造并委托中立 `CanonicalMaskPublisher` | 仓内调用清零、一个兼容发布且 release note 指向 publisher 后删除 |
| `validate_source_run_contract()`、`validate_derivation_source_episode()`、`validate_source_episode_completion_receipt()` | `application/urdf_runtime.py`、`application/urdf_batch.py`、publisher/process tests | 委托单例 `SourceLineageValidator`；不会跳过任何 trust-boundary 复验 | runtime/batch hooks 与测试改为注入 validator，仓内旧函数调用清零，并保留一个兼容发布后删除 |
| `publish_urdf_episode()`、`validate_published_urdf_episode()` | runtime 的 `UrdfWorkflowRuntime` hooks、publisher/process tests | 委托 `UrdfCanonicalEpisodePublisher` | workflow runtime 直接持有 publisher、tests 改用 class API、仓内旧调用清零，并保留一个兼容发布后删除 |
| `target_receiver_only` 参数 alias | dataset runtime tests/旧调用 | 显式映射到 `object_source_only`，不创建第二条执行路径 | 调用方迁移、CLI/API deprecation 周期完成且 contract tests 固定新参数后删除 |
| `load_urdf_gripper_episode(authoritative_loop=...)` | `test_urdf_gripper_data.py` 与旧 pick-place callers | 仅把 `ActiveGripperLoop` 适配为 `authoritative_events` 与其 `inclusive_window`；与 normalized 参数同时传入时拒绝，不重新检测 timeline | 调用方改传 `authoritative_events` + `authoritative_gripper_window`，仓内旧参数调用清零并完成一个兼容发布 |

### 13.3 验证结论与外部证据边界

结构完成结论由 669 个 CPU unit/integration tests、Ruff、strict mypy、import checks 与
`git diff --check` 支持。它证明迁移前冻结的 JSON/NPZ/status/failure/lineage/atomicity 合同在
可离线验证范围内保持，并证明 canonical owner 与 shim 委托方向。

第 2.1 节列出的四个 exact run 进一步证明：在当前机器、checkpoint、默认 URDF 和两条代表
episode 上，模型服务、CUDA/EGL、真实 depth/URDF、publication/lineage 与正式 `just process`
入口可工作。人工 review sheet 只做了 smoke 级可读性检查；它们不证明 coverage subset/full batch、
跨机器兼容或像素级视觉质量。部署签字仍须按任务范围记录模型/资产 identity、episode 集合和人工
review 结果，不能把单 episode smoke 外推成全数据集结论。

### 13.4 Thick standalone utilities（不属于 core pipeline thinness）

这些脚本是有独立输入、输出和生命周期的工具，不是 `just process`、单 episode、canonical
renderer 或 URDF batch 的生产实现藏回 `scripts/`。本轮没有为了表面一致性把它们机械包装成
空壳；只有触发下表条件时才启动各自的迁移，并先补足所列测试。

| 脚本 | 独立职责 | 现有调用方/测试 | 迁移触发条件 |
| --- | --- | --- | --- |
| `prepare_target_only_dataset.py` | data prep：规划、选择、校验并可物化 versioned target-only dataset | `docs/datasets.md` 提供 plan/materialize/validate-only 手工命令；当前没有 focused unit test | 当它成为受支持的定期 dataset workflow、出现第二个 materializer/selection consumer，或要改变 selection/staging 时；先补 selection、arm quota、copy/hash validation 和 staging rollback characterization，再抽 planner/materializer |
| `materialize_native_tracking_run.py` | offline materialization：从已保存 native tracks 构造新 mask run，不重跑 SAM3 | 当前只作为人工离线脚本使用；仓内没有直接调用方或 focused unit test | 当它进入正式 publication/resume 流程、出现第二个 native-track consumer，或需要改变 composition/provenance 时；先冻结 source identity、temporal QC、composition、provenance、partial-write rollback，再迁 package service |
| `benchmark_temporal_tracking.py` | benchmark：比较 published masks 与 SAM3 native tracks并汇总报告 | 当前只作为人工 benchmark 使用；仓内没有直接调用方或 focused unit test | 当报告成为 CI/release gate、多个 benchmark 复用 aggregation，或 schema 要稳定对外时；先补 fixed-array metrics、group aggregation、空集合和 report serialization tests，再抽 analytics/report 模块 |
| `serve_qwen.py` | service utility：本地 Qwen OpenAI-compatible HTTP server 与模型生命周期 | `manage_qwen_process.py` 以脚本路径启动；`test_qwen_server.py` 通过 `runpy` 验证 request/parser seam | 当 server 需要被多个 launcher 复用、成为正式 package service，或 backend/HTTP 实现继续扩展时；迁入 `adapters/qwen_server.py`，保留薄脚本入口和既有 runpy/import seam 一个兼容发布，并重验 ownership/signal cleanup |
| `run_open_set_failure_experiment.py` | experiment runner：仅运行声明的 open-set 失败 episode，管理独立 run 前缀和 batch summary | `docs/open_set_mask_fallback_s1_s3.md` 的 validate/smoke/batch 命令；`test_open_set_failure_experiment.py`；当前仍调用 `dataset_runtime.process_dataset()` | 当 `dataset_runtime` compatibility API 删除、实验升级为正式 workflow，或第二个实验复用其 selection/summary 时；改用 `DatasetPipeline` public API，同时冻结 declared-failure selection、独立 output/run prefix、`passed=false` 退出和 summary schema |
| `render_open_set_bad_cases.py` | review utility：从显式 source episode 清单生成 bad-case overlay 与审计 manifest | `docs/open_set_mask_fallback_s1_s3.md` 的 review 命令；`test_render_open_set_bad_cases.py`；已使用 renderer public API | 当第二个 review 工具复用 selection/report、该输出成为正式 renderer 子命令，或 manifest schema 对外稳定时；抽 review planner/report builder，保留 explicit-source/no-ranking、hash identity 和 overwrite 行为测试 |
