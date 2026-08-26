# Visual prompting: robotwin\_50 experiment

# RoboTwin 50 个 coarse task 分类

统计范围：RoboTwin 2\.0 完整数据集，共 50 个 coarse task；每个 task 550 条 episode，共 27,500 条。

## 1\. 分类口径

### 单臂 / 双臂

14 维 action 的划分如下：

```Plaintext
左臂：action[0:6]（pose）+ action[6]（gripper）
右臂：action[7:13]（pose）+ action[13]（gripper）
```

每只手逐帧计算：

```Plaintext
score = ||Δxyz|| + 0.05 × ||Δrpy||
```

若至少 3 帧满足 `score > 0.002` 且累计 score `>= 0.05`，或 gripper 指令发生明显切换，则该臂视为有效参与。

- **全量单臂**：该 task 的 550 条 episode 均只有一只手有效参与。

- **全量双臂**：该 task 的 550 条 episode 均为左右手共同参与。

- **混合**：同一 task 同时包含单臂和双臂 episode。

### 任务语义

任务语义与单/双臂独立分类，共 3 个一级类别：

1. **P\&P 家族**

    - **严格 P\&P**：对象 A 最终被放到、放入、挂到或堆叠到独立实体 B。

    - **相对放置**：对象 A 被移到 B 的左、右、旁边或远处；B 是位置参照，不是 receiver。

2. **Target\-only**：只改变一个语义对象或其自身部件的状态、姿态或位置，没有第二个语义对象或外部 receiver。

3. **其他**：工具接触、多对象抓持、倾倒等不属于上述两类的任务。

## 总体统计

### 单臂 / 双臂

|task\-level 标签|task 数|episode 级数量|
|---|---|---|
|全量单臂|25|13,750（这些 task 的全部 episode）|
|全量双臂|15|8,250（这些 task 的全部 episode）|
|混合|10|单臂 1,420；双臂 4,080|
|**合计**|**50**|**单臂 15,170；双臂 12,330**|

episode 总体比例：单臂 55\.16%，双臂 44\.84%。

### 任务语义

|语义类别|task 数|episode 数|占全部 episode|
|---|---|---|---|
|严格 P\&P|25|13,750|50\.00%|
|P\&P\-其他|5|2,750|10\.00%|
|**P\&P 家族合计**|**30**|**16,500**|**60\.00%**|
|Target\-only|14|7,700|28\.00%|
|其他工具/多对象/倾倒|6|3,300|12\.00%|
|**合计**|**50**|**27,500**|**100\.00%**|

### 语义类别 × 单/双臂 task 数

|语义类别|全量单臂|全量双臂|混合|
|---|---|---|---|
|严格 P\&P|9|9|7|
|P\&P\-其他|3|0|2|
|Target\-only|11|3|0|
|其他|2|3|1|
|**合计**|**25**|**15**|**10**|

## 3\. 任务分类与分组实验结果

混合任务括号内为“单臂 episode 数 / 双臂 episode 数”。为避免在一张 50 行大表中混排不同
实验口径，P\&P、Target\-only 和 Contact\-press 分开汇总；未纳入这三组抽样的任务另表保留。
这里的 11 个 task 在 coarse taxonomy 中都属于 Target-only；运行时 prompt profile 切分为
8 个普通 target-only 和 3 个 contact-press，完整分类见
[RoboTwin 50 task taxonomy](robotwin_50_task_taxonomy.md)。

### 单臂严格 P\&P

|task|action 分类|2026\-08\-20 基线（通过/20）|2026\-08\-24 晚间（通过/20）|Δ（晚间−基线）|
|---|---|---:|---:|---:|
|`move_pillbottle_pad`|全量单臂|20/20|19/20|−1|
|`move_stapler_pad`|全量单臂|16/20|18/20|+2|
|`place_container_plate`|全量单臂|18/20|20/20|+2|
|`place_empty_cup`|全量单臂|17/20|18/20|+1|
|`place_fan`|混合（549 / 1）|19/20|19/20|0|
|`place_mouse_pad`|全量单臂|18/20|19/20|+1|
|`place_object_scale`|全量单臂|20/20|17/20|−3|
|`place_object_stand`|全量单臂|19/20|18/20|−1|
|`place_phone_stand`|全量单臂|13/20|11/20|−2|
|`place_shoe`|全量单臂|17/20|19/20|+2|
|**抽样合计**|—|**177/200**|**178/200**|**+1**|

基线来自 run `pick-place20-full-default-urdf-20260820`，晚间重跑来自
`pick-place20-qwen38-api-20260824-v1`；通过数按 episode 统计。

### 普通 Target\-only profile（8 tasks）

|task|action 分类|2026\-08\-25 Target\-only（完成/20）|
|---|---|---:|
|`adjust_bottle`|全量单臂|20/20|
|`move_playingcard_away`|全量单臂|20/20|
|`open_laptop`|全量单臂|17/20|
|`open_microwave`|全量单臂|10/20|
|`rotate_qrcode`|全量单臂|14/20|
|`shake_bottle`|全量单臂|20/20|
|`shake_bottle_horizontally`|全量单臂|20/20|
|`turn_switch`|全量单臂|18/20|
|**抽样合计**|—|**139/160**|

### Contact\-press profile（3 tasks）

|task|action 分类|2026\-08\-25 有效 A/B contact arm（完成/20）|
|---|---|---:|
|`click_alarmclock`|全量单臂|11/20|
|`click_bell`|全量单臂|17/20|
|`press_stapler`|全量单臂|20/20|
|**抽样合计**|—|**48/60**|

Target\-only 与 Contact 表中的“完成”均指 object source 已完成且已到达最终 canonical 输出的
episode 数，不是 task 级 `passed` 状态。Target\-only 有 4/8 个 task 达到 20/20，Contact 有
1/3 个 task 达到 20/20；Contact 表采用有效配对 A/B 的 contact arm，较早独立运行的
`47/60` 只作为历史结果保留在第 4 节。

### 未纳入以上抽样的任务

|task|语义分类|action 分类|
|---|---|---|
|`beat_block_hammer`|其他|全量单臂|
|`blocks_ranking_rgb`|P\&P（相对放置）|混合（9 / 541）|
|`blocks_ranking_size`|P\&P（相对放置）|混合（10 / 540）|
|`dump_bin_bigbin`|其他|混合（323 / 227）|
|`grab_roller`|Target\-only|全量双臂|
|`handover_block`|P\&P（严格）|全量双臂|
|`handover_mic`|Target\-only|全量双臂|
|`hanging_mug`|P\&P（严格）|全量双臂|
|`lift_pot`|Target\-only|全量双臂|
|`move_can_pot`|P\&P（相对放置）|全量单臂|
|`pick_diverse_bottles`|其他|全量双臂|
|`pick_dual_bottles`|其他|全量双臂|
|`place_a2b_left`|P\&P（相对放置）|全量单臂|
|`place_a2b_right`|P\&P（相对放置）|全量单臂|
|`place_bread_basket`|P\&P（严格）|混合（269 / 281）|
|`place_bread_skillet`|P\&P（严格）|全量双臂|
|`place_burger_fries`|P\&P（严格）|全量双臂|
|`place_can_basket`|P\&P（严格）|全量双臂|
|`place_cans_plasticbox`|P\&P（严格）|全量双臂|
|`place_dual_shoes`|P\&P（严格）|全量双臂|
|`place_object_basket`|P\&P（严格）|全量双臂|
|`put_bottles_dustbin`|P\&P（严格）|混合（3 / 547）|
|`put_object_cabinet`|P\&P（严格）|全量双臂|
|`scan_object`|其他|全量双臂|
|`stack_blocks_three`|P\&P（严格）|混合（2 / 548）|
|`stack_blocks_two`|P\&P（严格）|混合（90 / 460）|
|`stack_bowls_three`|P\&P（严格）|混合（13 / 537）|
|`stack_bowls_two`|P\&P（严格）|混合（152 / 398）|
|`stamp_seal`|其他|全量单臂|

以上 29 个任务未纳入本节三组实验抽样；这表示未测试，不表示失败。

## 4\. 当前实验与 TODO

### 算法流程简述

当前默认的 pick-place 和 target-only path profile 使用同一条主链路；两者的主要差别是前者
要求 `target + receiver`，后者只要求 `target`：

```text
数据发现与合同校验
  → State Loop：从 action 得到活动手、抓取/释放事件、合法 seed 和角色输出窗口
  → Qwen semantic plan：识别角色，并生成有序 query bank
  → S1–S3 object-mask resolution：text-first，必要时最后使用 bbox
  → SAM3 从已通过 QC 的 seed 双向传播
  → temporal QC：检查覆盖率、缺帧、IoU、质心跳变和面积跳变
  → 默认 URDF + depth/sidecar 渲染左右夹爪
  → 合并 target、receiver、gripper_left、gripper_right 四通道
  → 原子发布八键 masks.npz，随后做校验、overlay 和 review sheet
```

这里的 S1/S2/S3 是 open-set mask 失败救回的**能力标签**，不是三个顺序执行的 pipeline
stage：

|能力|核心做法|运行时位置与安全边界|
|---|---|---|
|S1：多 query、多 seed|semantic query bank 最多提供 4 个 query，再追加最多 3 个 task/role-aware curated aliases；先试 Qwen 选择的 seed，再试 State Loop 声明的其他合法 seed|每个 seed 上生成真实 SAM mask，依次做空 mask、面积、连通域、重复候选和 Qwen visual QC；第一个 `passed` 立即停止，不能按面积选最大或合并候选|
|S2：开放集外观语义与严格 QC|从 semantic planning 开始就使用 mode-specific open-set prompt，允许 `silver object`、`white bar` 等带属性的外观 query；同时用可见完整部件、同步运动和夹爪接触证据检查候选身份与完整性|S2 不是“S1 失败后再跑一次”；它是贯穿 semantic plan 和 visual QC 的 policy，实际 resolver 顺序仍然只有 text attempts → bbox attempts|
|S3：Qwen bbox → SAM box mask|只有全部 text query × 合法 seed 都是 `rejected/ambiguous` 后，才请求 Qwen 定位 bbox，并把原始 bbox 交给 SAM box prompt|只接受 exact raw JSON 和合法归一化 `xyxy`；不 clamp、不扩框、不自动修正。SAM 得到的真实 mask 仍须通过与文本候选完全相同的机械检查和 visual QC|

任何服务、parser、prompt、输入合同或候选 shape 错误都只影响当前 episode：该 episode 保留
诊断并停止发布，不静默进入下一种 fallback。seed 通过后，SAM3 只在角色窗口内发布 native
propagation；严重 temporal 异常会 `quarantine` 并清空该角色的发布 mask。每条通过校验的
episode 独立发布固定四通道 bool mask；target 在抓持窗口使用 `frame_encoding=2` 保留“被持有
但不可见”的语义。更完整的实验和审计口径见
[Open-set mask 失败救回：S1–S3](open_set_mask_fallback_s1_s3.md)。

历史的 52 条已知失败切片中，S1 新救回 23 条，S2 再救回 17 条，S3 再救回 4 条，累计
`44/52（84.62%）`。这是失败切片的阶段性实验，不是下面最新 200 条抽样的成功率，也不是严格
单变量消融。

### 实验范围与当前结果

- **单臂严格 P\&P（2026\-08\-20 基线）**：10 个任务，每个任务选择 20 条 episode，共 200 条。
  该次运行中 object source 与默认 URDF 的 episode 级完成数为 **177/200（88\.5%）**：12 条
  semantic mask reject、10 条 temporal quarantine、1 条可复现的 Qwen bbox JSON 截断错误。
  汇总见 `artifacts/runs/pick-place20-full-default-urdf-20260820-collection-summary.json`。

- **单臂严格 P\&P（2026\-08\-24 晚间 Qwen API 重跑）**：先有一次启动尝试
  `20260824T131847Z-9e79a670`（约 21:18），10 个 task 全部在 episode 处理前因
  `live URDF mode is fresh-only; --dry-run/--resume require --source-run-dir` 失败；该次不计入
  通过率。随后实际运行 `pick-place20-qwen38-api-20260824-v1`，约 21:20–02:20（日志耗时
  4:59:18），仍为 10 task × 20 episode、`cam_high`、URDF backend；Qwen runtime/model 为
  `api / qwen3.8-max`，SAM 使用 GPU 2，EGL 使用 GPU 0。object-source episode 级通过为
  **178/200（89\.0%）**，相对基线 **+1 条（+0\.5 个百分点）**。通过的 episode 按条独立
  进入 canonical 处理；失败 episode 只保留 frozen diagnostics。汇总与完整日志分别见
  `artifacts/runs/pick-place20-qwen38-api-20260824-v1-collection-summary.json` 和
  `artifacts/runs/pick-place20-qwen38-api-20260824-v1.log`；逐 task 的 `run_manifest.json`、
  `mask_qc.json` 和 `qwen_failure.json` 位于
  `artifacts/runs/_sources/pick-place20-qwen38-api-20260824-v1-<task>-object-source/`。

昨晚逐任务的通过数和相对基线的 `Δ` 已并入第 3 节的单臂严格 P\&P 表；下方只保留失败
episode 的原因分类，避免重复维护同一组统计。

昨晚的 22 条未完成 episode 可按以下类型复核：

1. **11 条 receiver temporal quarantine**：`move_stapler_pad/8275`、`place_mouse_pad/17098`，
   以及 `place_phone_stand/19258、19259、19268、19288、19298、19300、19424、19545、19799`。
   这 11 条全部是 receiver 角色，触发 `low_adjacent_iou_p05` 与/或 `large_centroid_jump_p95`
   （部分还触发窗口覆盖/内部缺帧），不是 target 语义识别失败。
2. **5 条 query bank 合同拒绝**：`7188、8272、18184、18200、18855`。原始 Qwen 输出超出
   4 词限制或包含当前 validator 禁止的 `object`/`with` 描述词，因而在进入 SAM 前停止该 episode。
3. **2 条 target mask QC reject**：`place_empty_cup/16001、16380`，候选只覆盖杯子边缘/把手，
   未覆盖完整主体。
4. **2 条 Qwen API 临时不可用**：`place_fan/16815`、`place_object_stand/18700`，健康探针虽
   通过，但 mask QC 请求返回 connection refused。
5. **1 条 semantic seed 失败**：`place_object_scale/18209`（`semantic_plan_no_clear_seed`）。
6. **1 条 bbox fallback 合同错误**：`place_shoe/19849`，QC 返回 `selected_candidate='A'`，
   但该阶段只接受 `BBOX`。

与基线相比，昨晚运行将 Qwen 从本地 `qwen3.5-27b` 切换为 API `qwen3.8-max`；语义 prompt
版本仍为 `object_roles_semantic_v2`。因此 178/200 与 177/200 的差异只能作为运行结果对比，
不能解释为单一模型变量带来的提升。当前“episode 级达到 85%”已达到；canonical 产物按
episode 独立发布，失败 episode 保留诊断，不阻断同 task 的成功 episode。

- **单臂 Target\-only（2026\-08\-25 Qwen API 全量运行，已完成）**：按互斥任务分组统计时，
  排除归入 Contact\-press 的 `click_alarmclock`、`click_bell` 和 `press_stapler`，剩余 8 个任务，
  每个任务 20 条 episode，共 160 条；输入位于
  `/DATA/disk8/xuran/add_mask_robotwin/dataset/target_only_20_v2`。run id 为
  `target-only220-qwen38-api-20260824-v1`（名称沿用 2026\-08\-24 的预定 run id，实际于
  2026\-08\-25 10:18 启动），Qwen runtime/model 为 `api / qwen3.8-max`，SAM 使用 GPU 1，
  EGL 使用 GPU 3；约 13:50 完成，总耗时约 3:32。object source 与最终 canonical episode
  均完成 **139/160（86\.88%）**，严格全 20 条通过的 task 为 4/8。原始 run 同时执行了上述
  3 个 Contact\-press 任务，因此未去重的 runner 原始汇总为 188/220；该数不用于分组总计。

  逐 task 完成数见第 3 节。其中 `open_laptop/8850` 超过 4 词，
  `open_laptop/9212` 含禁止词 `with`；`rotate_qrcode/22548` 也超过 4 词；
  另有 1 条 mask-QC reject，其余 17 条未完成均记为 `sam_incomplete`。所有 139 条
  source-completed episode 随后都完成 URDF、canonical publication/validation 和 review
  render；分组中 4 个 task 含 source-excluded episode，不代表这 139 条 canonical 产物失败。
  日志和最终 summary 分别为
  `/tmp/target-only220-qwen38-api-20260824-v1.log`、
  `/tmp/target-only220-qwen38-api-20260824-v1-runner-summary.json`；逐 task 产物位于
  `.worktrees/contact-press-profile/artifacts/runs/target-only220-qwen38-api-20260824-v1-<task>/`。

- **Contact-press 专用 profile（2026\-08\-25 Qwen API 并行运行，已完成）**：从 Target-only
  中拆出 `click_alarmclock`、`click_bell`、`press_stapler` 三个直接接触 action-site 的 task，
  每个 task 20 条，共 60 条。run id 为 `contact-press60-qwen38-api-20260824-v1`（同样沿用
  预定命名），实际约 10:18–11:03 运行；Qwen runtime/model 为 `api / qwen3.8-max`，SAM
  使用 GPU 4，EGL 使用 GPU 5。object source 与最终 canonical episode 均完成
  **47/60（78\.33%）**，严格全 20 条通过的 task 为 1/3。

  逐 task 完成数见第 3 节。`click_bell/2750` 因 query bank 含禁止词 `hand` 被拒绝，
  `click_bell/2777` 因
  `general_fallback_query` 超过 4 个小写英文词被拒绝。所有 47 条 source-completed episode
  随后均完成 URDF、canonical publication/validation 和 review render。`click_alarmclock` 与
  `click_bell` 的 task summary 仍为 `passed=false`，原因是 streaming URDF worker 按原始 20 条
  episode 收尾，把没有 source receipt 的条目标记为 `SourceEpisodeUnavailable` 并抛出
  `UrdfBatchIncompleteError`；这不是已经移除的 URDF 可见非空率 QC。contact runner summary
  为 `/tmp/contact-press60-qwen38-api-20260824-v1-runner-summary.json`，逐 task 产物位于
  `.worktrees/contact-press-profile/artifacts/runs/contact-press60-qwen38-api-20260824-v1-<task>/`。

  上述 `47/60` 是较早的独立 contact-profile 运行，不是普通 target-only 与 contact-press 的
  配对对照，不能单独作为 profile 选择依据。

### Target-only vs Contact-press 配对决策

2026-08-25 在同一批 60 条 episode 上完成了显式 profile A/B：普通 target-only 为
**46/60（76.67%）**，contact-press 为 **48/60（80.00%）**；配对 exact McNemar 检验
`p=0.803619`，总体流水线完成率没有显著差异。人工复核总视频和 review sheet 后，语义差异
很明确：普通 target-only 倾向于标完整闹钟、铃或订书机，contact-press 倾向于标按钮或最小
完整按压部件。因此 click/press 的选择依据是 action-site 标注正确性，而不是 `+3.33` 个百分点
的聚合差值；这类任务采用 `contact_press` profile。实验口径、逐任务结果、失败原因和媒体路径
见 [Contact-press target-only profile](experiments/contact-press-profile.md)。

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=YjQ5YWNjZDM2ZjVjZjQ4YTk1MmUzMjkyMGIzMzcyY2JfNjZmZmY0ZmRlY2FkMDcwM2FiZTk3MWQyNGFlZTcxNDRfSUQ6NzY3NTE5MzYyMzEzODUzNjY5MF8xNzg3MjA2OTc2OjE3ODcyOTMzNzZfVjM)

### TODO List

* [x] 将严格 P\&P 的 mask 成功率提高到至少 85%：同一批 200 条数据的 2026\-08\-24
  Qwen API 重跑达到 178/200（89\.0%），超过 170/200 目标；逐 task 结果和 22 条失败分类见上文。

* [ ] 为 target 物品提供两种 mask 标注 pre\-target mask：表示操作前物品所在位置，提供物品初始位置信息。 post\-target mask：表示抓取后随机械臂移动的 target 物品。 在输出 metadata 中明确标注两种 mask 类型。

* [ ] 为 wrist 相机生成 mask 梳理 wrist 相机的视角、遮挡和运动同步问题。 实现并验证 wrist 相机 mask 生成流程。 预留更多开发和验证时间。
