# Contact-press target-only profile

本页记录 `target_only` 中接触/按压类任务的独立 profile 实验。实现保持
`annotation_mode=target_only` 不变，只把 target 的语义合同作为正交 profile；因此不会把
按压目标误当成被夹爪抓取并搬运的物体。

## 问题与假设

原 target-only prompt 将 target 定义为首次闭合后被夹爪抓取、移动并稳定持有的完整物体。
这适合抓取操作，不适合已经完成配对验证的 `click`、`press`：seed 可能仍在接近阶段，正确
目标可能是按钮、按压面或其他固定接触 action site。`open`、`turn` 虽然也涉及 articulated
action site，但尚无同口径 profile A/B，因此本次 8+3 决策不切换它们。假设是：

- 接触类 prompt 明确“即将接触/驱动”，并允许 seed 尚未接触；
- mask-QC 用接近轨迹和预期接触证据，不要求 target 随夹爪运动或被持有；
- grasp/manipulation 语义、时间线、SAM 传播和 URDF QC 保持原合同；
- 只放宽错误的“必须已接触/持有”语义，不放宽候选完整性、身份或背景混入检查。

## Profile 路由

运行时读取静态 manifest 的 `task_kind`，不根据任务名猜测：

| `task_kind` | profile | 语义 |
| --- | --- | --- |
| `single_movable_target`（含 conditional） | `grasp_manipulation` | 首次闭合后抓取/持有的 target |
| `contact_action_site` | `contact_press` | 即将接触、按压或驱动的功能部件 |
| `articulated_action_site` | `grasp_manipulation` | 保持普通 target-only，等待独立 A/B 决策 |
| 缺失 | `grasp_manipulation` | 兼容旧 manifest，保留旧行为 |

只有 `contact_action_site` 自动选择 `contact_press`，即当前 11-task 集合中的
`click_alarmclock`、`click_bell`、`press_stapler`。`contact_press` 只允许与 `target_only`
mode 组合；`pick_place` 和未声明 task kind 的旧 target-only 数据不改变。

## Prompt 合同

contact profile 使用三份专用模板：

- `configs/prompts/target_only_contact_press_semantic_open_set.txt`：识别最小完整 action
  site；seed 可以早于实际接触，不要求抓取、持有或同步运动。
- `configs/prompts/target_only_contact_press_mask_candidate_qc_open_set.txt`：检查正确
  预期接触实例、完整可见轮廓和污染；未接触本身不是 reject 理由。
- `configs/prompts/target_only_contact_press_bbox_localization.txt`：bbox fallback 同样
  定位即将驱动的功能部件，而不是整机或抽象接触点。

`grasp_manipulation` 继续使用原 target-only templates 和原 QC 规则；本实验不改变 URDF
阈值、gripper encoding 或 grasp 时间窗口。

## 已有基线

在同一 `target_only_20_v2` action-site 审计集合（3 个 contact-action + 3 个 articulated task，
共 120 episodes）上，旧流程最终完成
`78/120 = 65.00%`。只统计 object/source 阶段时为 `85/120 = 70.83%`；两者差异主要由
后续 temporal、URDF 和其他发布阶段失败造成，不能把 source 率当作最终成功率。

旧流程失败分布（episode 可有多个审计原因）：

| 阶段/原因 | 数量 |
| --- | ---: |
| mask QC reject | 17 |
| Qwen schema/service | 12 |
| SAM seed | 4 |
| temporal quarantine | 2 |
| URDF QC | 7 |

### JSON 审计抽样

固定审计 JSON 的人工复核标签为 `A=7`、`B=4`、`C=6`（合计 17 个 mask-QC reject）。标签
只用于本实验的可追溯分类，不把它们解释为新的成功率或像素真值；每条记录仍以原始
`mask_qc.json`、候选 mask 和视频为准。

## 验证方法

验证拆成三个互不替代的实验：

1. **显式 profile A/B（已完成）**：在固定 press 子集的同一批 60 条 episode 上，分别强制
   使用普通 `grasp_manipulation` target-only 和 `contact_press` 配置，完整执行
   semantic → SAM candidate/QC → native propagation → URDF → canonical publication。
2. **固定候选 A/B（待运行）**：对原 17 个 mask-QC reject 读取已归档的 loop、seed、候选
   mask 和 attempt 顺序，只替换 QC prompt，重新请求 Qwen；不运行 SAM、不生成新候选。
3. **Articulated profile A/B（待运行）**：对 `open_laptop`、`open_microwave`、`turn_switch`
   的固定 60 条分别显式运行普通 target-only 和 contact profile；在证据完成前保持普通
   target-only 路由。

所有成功率均以最终 canonical episode 为单位；不得用 source 可用数或 task 级 `passed` 状态
替代最终完成数。

实现提交（按依赖顺序）：

- `64c0b5d` — route contact-press target prompts/profile；
- `0bd639c` — localize contact-press action sites；
- `3ee9313` — accept loop event key variants；
- `132a860` — merge `fix/target-only-first-close`；
- `b4603ec` — align the latest first-close semantics and contracts。

## 2026-08-25 显式 profile A/B（60 episodes）

### 可比性与成功定义

实验只覆盖独立物化的 3 个 `contact_action_site` task：`click_alarmclock`、`click_bell`、
`press_stapler`，每类 20 条。`open_laptop`、`open_microwave`、`turn_switch` 属于
`articulated_action_site`，不在本次 60 条集合内。

两臂均运行提交 `7619299`，并对每个 task 显式传入同一个
`target_only_20_v2_contact_press/<task>` 数据根。普通 target-only 强制加载
`configs/process_target_only_qwen38_api.yaml`，contact-press 强制加载
`configs/process_contact_press_qwen38_api.yaml`；没有使用会按 `task_kind` 自动切 profile 的
`--data-path` 入口。60/60 原始视频 SHA-256 一致，60/60 rendered prompt SHA-256 不同，符合
只改变 profile prompt 合同的预期。

这是有效的完整 profile A/B，但不是冻结模型响应和候选的严格确定性单变量实验：两臂使用
不同 GPU，并分别请求远端 Qwen API；服务和 query 合同噪声仍会进入完成率。因此统计结论只
描述这次完整流水线运行，语义判断还必须结合逐 episode 视频复核。

本节的“完成”要求最终 `process_summary.json` 记录为 `completed`/`skipped_complete`，且对应
canonical overlay 存在。两臂所有 source-completed episode 都通过了 URDF、canonical
publication、validation 和 render；未完成项都在 object-source 阶段被排除。

### 成功率与配对结果

| task | 普通 target-only | contact-press | contact − target |
| --- | ---: | ---: | ---: |
| `click_alarmclock` | 16/20（80%） | 11/20（55%） | −5（−25 pp） |
| `click_bell` | 18/20（90%） | 17/20（85%） | −1（−5 pp） |
| `press_stapler` | 12/20（60%） | 20/20（100%） | +8（+40 pp） |
| **合计** | **46/60（76.67%）** | **48/60（80.00%）** | **+2（+3.33 pp）** |

配对矩阵为：两者都成功 39 条、仅普通 target-only 成功 7 条、仅 contact-press 成功 9 条、
两者都失败 5 条；agreement 为 73.33%。exact McNemar 双侧检验 `p=0.803619`，因此不能从
这 60 条推断 contact-press 带来总体完成率提升。任务间方向相反，比聚合差值更重要。

### 失败结构与人工视频复核

普通 target-only 的 14 条失败由 10 条 Qwen query-bank 合同拒绝和 4 条 `sam_incomplete`
组成。其中 `press_stapler` 有 7 条都因 `shape_category_query` 使用禁止词 `object` 被拒绝，
另有 1 条因普通 target-only 的 grasp/hold 合同与桌面按压动作不匹配而未完成。contact-press
的 12 条失败由 2 条 Qwen query-bank 合同拒绝和 10 条 `sam_incomplete` 组成；9 条
`sam_incomplete` 集中在 `click_alarmclock`。

人工复核总视频和 review sheet 后，两个 profile 的语义差异稳定可见：

- 普通 target-only 按“被抓取/持有的完整物体”解释 target，倾向于输出整个闹钟、铃或订书机；
- contact-press 按“即将被驱动的最小完整功能部件”解释 target，倾向于输出按钮或按压面；
- `click_alarmclock` 中有 5 条普通 target-only 成功而 contact-press 失败：2 条没有清晰 action-site
  seed（2232、2620），2 条按钮被侧视角遮挡（2256、2748），1 条候选只覆盖整机而被正确拒绝
  （2386）。接受整机 mask 虽会提高流水线完成率，但不满足按钮/action-site 标注合同。

因此本实验的决策依据是标注语义与人工视频复核，而不是总体 `+3.33 pp`：
`contact_action_site` 应使用 `contact_press`；`single_movable_target` 继续使用
`grasp_manipulation`。闹钟后续应改善按钮可见性、seed 和 bbox 定位，不能通过放宽 QC 接受
整机 mask 来提高数字。

### 证据与产物

- 汇总和逐 episode 配对记录：
  `artifacts/contact_press_vs_target_only_ab_valid_20260825/summary.json`；
- 60 条总对比视频：
  `artifacts/contact_press_vs_target_only_ab_valid_20260825/videos/all_60_comparison.mp4`；
- 总 review sheet：
  `artifacts/contact_press_vs_target_only_ab_valid_20260825/comparison_review_sheet.jpg`；
- 普通 target-only collection summary：
  `artifacts/runs/contact-ab60-target-only-7619299-20260825-v1-collection-summary.json`；
- contact-press collection summary：
  `artifacts/runs/contact-ab60-contact-press-7619299-20260825-v1-collection-summary.json`。

早先 `artifacts/contact_press_vs_target_only_ab_20260825/` 中命名为 target-only/contact-press 的
两组 run 实际都被 `--data-path` 自动路由到 contact profile，只能作为同 profile 重跑审计，
不能用于本节结论。

## 剩余验证

| 实验 | 输出/判定 | 状态 |
| --- | --- | --- |
| Stage 1（120 episodes） | 6 个 action-site task 均为 20/20；合计 `120/120` | 已完成；只证明时间线和 sparse-frame 合同通过 |
| 显式 profile A/B（60） | target-only `46/60` vs contact-press `48/60`；人工复核选择 action-site 语义 | 已完成 |
| 固定候选 A/B（17 rejects） | replay JSON、`old→new` 转换计数、人工确认清单 | TODO：运行 `scripts/replay_contact_press_mask_qc.py` |
| articulated profile A/B（60） | 两臂完成率、配对矩阵、视频语义复核 | TODO；完成前不扩大 contact-press 路由 |

## 代码验证

合并 first-close 最新改动后，当前验证为 `733 passed, 1 skipped`；Ruff 和严格类型检查
`mypy` 均通过。另对 manifest 中 3 个 `contact_action_site` 和 3 个
`articulated_action_site` task 逐条构建 Stage-1 context，结果为 `120/120`，其中没有检测到
reopen。上述只证明代码、数据输入和时间线合同通过，不是 120 条全流程成功率。全流程和
replay 完成后，应在本页补上命令、run id、产物路径及实际统计。
