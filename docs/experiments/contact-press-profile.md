# Contact-press target-only profile

本页记录 `target_only` 中接触/按压类任务的独立 profile 实验。实现保持
`annotation_mode=target_only` 不变，只把 target 的语义合同作为正交 profile；因此不会把
按压目标误当成被夹爪抓取并搬运的物体。

## 问题与假设

原 target-only prompt 将 target 定义为首次闭合后被夹爪抓取、移动并稳定持有的完整物体。
这适合抓取操作，不适合 `click`、`press`、`open` 和 `turn`：seed 可能仍在接近阶段，正确
目标可能是按钮、开关、把手、盖板或其他 action site。假设是：

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
| `articulated_action_site` | `contact_press` | 即将驱动的把手、盖板或 articulated site |
| 缺失 | `grasp_manipulation` | 兼容旧 manifest，保留旧行为 |

`contact_press` 只允许与 `target_only` mode 组合；`pick_place` 和未声明 task kind 的旧
target-only 数据不改变。

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

实验分两步，保持候选和其他变量可复现：

1. **固定候选 A/B**：对原 17 个 mask-QC reject 读取已归档的 loop、seed、候选 mask 和
   attempt 顺序，只替换 QC prompt，重新请求 Qwen；不运行 SAM、不生成新候选。记录
   `old -> new` decision，并人工复核所有 `rejected -> passed`。
2. **120 全流程**：在同一 episode 列表和 contact profile 配置下重新执行
   semantic → SAM candidate/QC → native propagation → URDF → canonical publication；与
   旧 run 对齐 episode、stage、failure reason 和最终成功率。不得用 source 成功数替代完整
   pipeline 成功数。

实现提交（按依赖顺序）：

- `64c0b5d` — route contact-press target prompts/profile；
- `0bd639c` — localize contact-press action sites；
- `3ee9313` — accept loop event key variants；
- `132a860` — merge `fix/target-only-first-close`；
- `b4603ec` — align the latest first-close semantics and contracts。

## 结果登记（待补）

以下数值必须由实际产物填写；当前留空，避免把 projected 或未运行结果写成实验结果。

| 实验 | 输出/判定 | 状态 |
| --- | --- | --- |
| Stage 1（120 episodes） | 6 个 action-site task（3 contact-action + 3 articulated）均为 20/20；合计 `120/120` | 已完成；只证明时间线和 sparse-frame 合同通过 |
| 固定候选 A/B（17 rejects） | replay JSON 路径、`old→new` 转换计数、人工确认清单 | TODO：运行 `scripts/replay_contact_press_mask_qc.py` 并登记结果 |
| contact_press 全流程（120） | completed/120、各阶段失败分布、最终成功率、run/source 路径 | TODO：完成 120 条重跑后登记 |
| 成功率对比 | baseline `78/120` vs 新流程实际 completed 数；百分点变化 | TODO：仅依据全流程 summary 计算 |
| 误判结论 | A/B 翻转中确认的 QC 误判数；其余为真实候选/服务/时序失败 | TODO：逐条复核，不自动推断 |

### 当前可复现统计

当前独立物化的 press 子集只包含 `click_alarmclock`、`click_bell`、`press_stapler`，共 60 条
episode，路径为 `/DATA/disk8/xuran/add_mask_robotwin/dataset/target_only_20_v2_contact_press`；
`open_laptop`、`open_microwave`、`turn_switch` 不在该数据集内。

在 `press_stapler` 的 20 条 episode 上，合并后的 branch 使用已有冻结 source run 做
`URDF --dry-run --allow-partial-source`，结果为 `12` 条可规划、`8` 条按 source contract
排除，dry-run summary 的 `passed=true`。这是输入/规划合同检查，不是渲染完成数。

已有 `target-only20-v2-first-close-v4-20260821-full220` 完整 run 的最终成功率如下；这些是
旧 prompt 的 baseline，不是本 branch 新 contact prompt 的结果：

| 范围 | 完成 | 总数 | 成功率 |
| --- | ---: | ---: | ---: |
| `press_stapler` | 12 | 20 | 60.00% |
| 三个 contact-action task（click_alarmclock、click_bell、press_stapler） | 42 | 60 | 70.00% |
| 六个 action-site task（3 个 contact-action + 3 个 articulated） | 78 | 120 | 65.00% |

新 profile 的 Qwen/SAM 全流程尚未重跑：本机 Qwen endpoint 当前未监听，GPU 均被其他训练
任务占用，因此不能把上述 baseline 当作新 prompt 的提升结果。

## 代码验证

合并 first-close 最新改动后，当前验证为 `716 passed, 1 skipped`；Ruff 和严格类型检查
`mypy` 均通过。另对 manifest 中 3 个 `contact_action_site` 和 3 个
`articulated_action_site` task 逐条构建 Stage-1 context，结果为 `120/120`，其中没有检测到
reopen。上述只证明代码、数据输入和时间线合同通过，不是 120 条全流程成功率。全流程和
replay 完成后，应在本页补上命令、run id、产物路径及实际统计。
