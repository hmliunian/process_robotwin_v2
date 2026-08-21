# Open-set mask 失败救回：S1–S3 实现、复现与审查

更新日期：2026-08-18

## 1. 结论和口径

最终方案只保留 S1、S2、S3：

| 实验阶段 | 主要改动 | 新救回 | 累计完成 |
| --- | --- | ---: | ---: |
| S1 | 多 query、多合法 seed | 23 | 23/52 |
| S2 | 外观描述、开放集语义 prompt、curated aliases | 17 | 40/52 |
| S3 | 所有文本尝试失败后，Qwen bbox → SAM box mask | 4 | **44/52** |

这里的 S1/S2/S3 是本次“失败救回实验阶段”，不是仓库架构中的 Pipeline Stage 1/2/3。

本实验只运行原批次中已经确认的 52 个 mask/QC 失败 episode，没有重跑原来成功的 128 个。
因此需要同时保留两个统计口径：

- 已知失败切片：`44/52 = 84.62%` pipeline completed；
- 原 180 个 episode 合并计算：`128 + 44 = 172/180 = 95.56%`。

第二个数字满足“总体至少 90%”的目标，但不能替代第一个失败切片结果，也不能解释为像素级
准确率。52 个失败已排除单纯的 Qwen transient timeout；超时样本此前已通过延长 timeout 和重跑
单独处理。

S4 不属于最终方案，也不计入上述结果。它曾带来 6 个额外 pipeline completion，但依赖特定
方向扩框、边界样本二次放行和强制时序修正，规则过强，存在把不完整候选当作成功的风险，现已
从代码、配置、prompt 和对应测试中删除。

## 2. 原始失败是什么

原始批次：

```text
artifacts/batches/pick_place_20_urdf_full_20260814
```

失败声明：

```text
configs/open_set_mask_fallback_failures.yaml
```

52 个 episode 来自 9 个任务，共涉及 56 个失败角色：32 个 target、24 个 receiver。任务分布为：

| Task | 失败 episode 数 |
| --- | ---: |
| `move_stapler_pad` | 11 |
| `place_container_plate` | 4 |
| `place_empty_cup` | 4 |
| `place_fan` | 11 |
| `place_mouse_pad` | 2 |
| `place_object_scale` | 4 |
| `place_object_stand` | 8 |
| `place_phone_stand` | 7 |
| `place_shoe` | 1 |

47 个角色的失败是所有候选 mask 都为空，其中 44 个语义规划选择了 `frame 0`。

`frame 0` 只是视频第一帧，不是 pipeline 的某个阶段。首帧经常能看到物体，但不一定是类别
外观最清晰、最符合 SAM3 训练词汇、最容易区分 target/receiver 的帧。原方案又只使用 Qwen
推荐顺序前几个 query，所以“首帧 + 少数类别词”在开放集上很容易同时失败。

## 3. 简单诊断：遮挡还是词汇问题

诊断 probe 在 6 个代表失败角色上测试了：原 3 个 query、人工补充的 3 个 query，以及所有
合法 pre-grasp seed。这里的“6 个 query”只属于诊断 probe，并不是正式方案固定使用 6 个词。

| 例子 | 观察 | 主要解释 |
| --- | --- | --- |
| `018710 receiver` | `box` 在多帧都能得到正确 mask，原 query 全空 | 词汇/ontology 问题 |
| `014899 target` | 原 seed 上 `small container` 成功 | query 问题 |
| `017087 receiver` | `gray rectangle` 只在 frame 16 成功 | query × frame 交互 |
| `018209 target` | 清晰帧和 6 个 query 都为空 | 不是遮挡，text grounding 失败 |
| `016534 target` | 多个无遮挡帧和 6 个 query 都为空 | 资产外观/模型域偏移 |
| `019273 receiver` | 宽泛同义词得到非空 mask，但选中了 gripper/phone | 非空不等于正确身份 |

所以失败不是单一原因。主要成分是：

1. 任务名与 SAM3 熟悉词汇不一致；
2. 模拟资产外观与现实类别原型不一致；
3. 同一个 query 对不同 seed 的 grounding 结果不同；
4. 宽泛词可能转而命中错误实例；
5. seed mask 正确也不保证后续传播在遮挡后稳定。

详细 probe 记录只存在于历史实验分支 `experiment/open-set-mask-fallback`，不属于本分支的公共
文档合同。

## 4. 总体设计原则

最终实现遵守以下顺序和安全边界：

```text
Qwen semantic plan
  → query bank + curated aliases
  → 在 Qwen seed 上生成并检查文本 mask
  → rejected/ambiguous 时换下一个合法 seed
  → 所有 query × seed 都失败后才允许 bbox fallback
  → Qwen 定位 bbox
  → SAM 根据原 bbox 生成真实 mask
  → 与文本候选完全相同的机械检查和视觉 QC
  → 通过后做 native video propagation
  → 普通 temporal QC
  → pass/review 发布，quarantine 清空并失败
```

核心原则：

- **text-first**：bbox 不能抢在一个可用文本候选之前；
- **multi-seed**：只在 State Loop 已声明为该角色合法的 seed 候选中尝试；
- **actual-mask QC**：Qwen 检查 SAM 实际生成的轮廓，而不是只评价 query 文本；
- **fail-closed**：格式错误、身份不明、置信度不足、错误实例和 temporal quarantine 都不发布；
- 不按面积自动选最大候选，不合并多个候选，不把“非空”直接当成功；
- 每次尝试保留 seed、query、方法、候选图、Qwen 原始响应和 provenance，方便人工回查。

## 5. S1：多 query、多 seed

### 5.1 配置

S1 使用：

```text
configs/open_set_mask_fallback_failures.yaml
```

关键配置：

```yaml
mask:
  qc_enabled: true
  qc_max_candidates: 8
  qc_query_fallback_enabled: true
  qc_seed_fallback_enabled: true
  qc_max_attempts: 2
  qc_min_confidence: 0.70
  qc_min_area_fraction: 0.0001
  qc_max_area_fraction: 0.85
  qc_duplicate_iou_threshold: 0.98
```

S1 不启用 bbox fallback。

`qc_max_attempts: 2` 的含义是：同一个 Qwen 请求发生临时服务错误时最多请求两次。它不是两个
query，也不是两个 seed。

### 5.2 正式方案到底有多少 query

每个角色先从 semantic query bank 读取最多 4 个非空字段：

1. `category_query`
2. `color_category_query`
3. `shape_category_query`
4. `general_fallback_query`

实际顺序由 `recommended_order` 决定。打开 `qc_query_fallback_enabled` 后，再追加最多 3 个
task/role-aware curated aliases。归一化并去重后，最多是 7 个文本 query。

如果 receiver 的 query 中包含可靠的 `blue` 描述，程序可额外生成一个 saturated-blue planar
region 机械候选。总候选上限仍是 8，因此这时文本槽位最多为 7。这个蓝色区域也必须通过同一套
视觉 QC，不能因颜色或面积自动被接受。

所以：

- “6 个 query”是早期诊断 probe 的 3 + 3；
- 正式 S1 不是固定 6 个，而是最多 7 个去重文本 query；
- `qc_max_candidates: 8` 是每个 seed 的候选总上限，不表示一定会生成 8 个。

### 5.3 curated aliases 如何生成

别名逻辑位于：

```text
src/robotwin_annotation_v2/pipeline/open_set_queries.py
```

它不是无边界地枚举英文同义词，而是按任务、角色和已有颜色证据补充最多 3 个常见短语。例如：

| 角色/任务族 | 可能追加的短语 |
| --- | --- |
| phone stand receiver | `phone holder`, `phone dock`, `phone stand` |
| object stand receiver | `box`, `display platform`, `display base` |
| empty cup receiver | `drink coaster`, `cup coaster`, `beverage coaster` |
| stapler target | `desk stapler`, `office stapler`, `paper stapler` |
| fan target | `desk fan`, `table fan`, `electric fan` |
| bowl/container target | `small bowl`, `small container`, `rice bowl` |
| phone target | `smartphone`, `mobile phone`, `cell phone` |

如果已有 query 中能稳定提取颜色，receiver 别名会保留该颜色；所有别名都与 semantic query bank
去重。这个规则对当前九个 RoboTwin task family 有针对性，因此不能声称已经证明对任意新类别
泛化。

### 5.4 seed 尝试顺序

对每个角色，seed 顺序是：

1. 先尝试 Qwen semantic plan 选择的 seed；
2. 如果 QC 返回 `rejected` 或 `ambiguous`，按 `context.seed_candidates(role)` 的顺序尝试其余
   合法 seed；
3. 如果某次 QC `passed`，立即停止并使用该候选；
4. 如果发生请求/合同 `error`，fail-closed，避免在服务或解析异常时静默换方案；
5. 所有合法 seed 都没有通过时才进入 S3 bbox fallback（若配置已启用）。

每个 seed 上会批量生成该角色的文本候选，然后执行：

- 空 mask 检查；
- 面积上下限检查；
- 连通域统计；
- IoU ≥ 0.98 的重复候选标记；
- 候选轮廓图 + 最多 2 张动作上下文图的 Qwen visual QC。

S1 新增完成 23 个 episode。

## 6. S2：按真实外观做开放集回退

### 6.1 配置和 prompt

S2 使用：

```text
configs/open_set_mask_fallback_appearance.yaml
configs/prompts/target_receiver_semantic_open_set.txt
configs/prompts/mask_candidate_qc_open_set.txt
```

S2 保留 S1 的多 query、多 seed 和 curated aliases，同时替换 semantic/QC prompt。它不是只改
一个 prompt 的严格单变量实验。

### 6.2 semantic prompt 的变化

开放集 semantic prompt 要求：

- 逐一比较全部合法 seed，不能默认最小 frame id；
- 常见三维物体优先保留可检测的完整类别名；
- 当任务类别与模拟资产外观不一致时，允许最后一个 `general_fallback_query` 使用真实外观；
- 例如 `silver object`、`white bar`、`small red object`、`brown platform`；
- 裸 `object/thing/item/stuff` 仍然禁止，`object` 必须带真实视觉属性且只能放在 general fallback；
- 不允许用 cap、label、handle 等子部件代替完整物体；
- query 保持 1–4 个小写英文词，非空字段互不重复；
- target 的身份优先使用 close/hold 中被夹住并随夹爪移动的动作证据；
- receiver 先在 place context 中确定与 target 最终直接接触的对象，再回到合法 seed 找清晰视图。

这解决了“任务叫 stapler，但资产更像一根白条”“透明倾斜 fan 不像常见正面风扇”之类域偏移。

### 6.3 mask QC prompt 的安全合同

外观 query 更宽，必须同时收紧实际 mask 的身份/完整性检查。accept 前要求：

1. 候选是当前角色的正确实例；
2. 任务文本点名且画面可见的部件全部在轮廓内；
3. 轮廓外不存在与候选刚性连接、在上下文中同步运动的可见结构；
4. 轮廓外不存在被夹爪直接接触、夹住或带走的可见结构；
5. 不明显包含夹爪、背景或邻近其他物体；
6. 没有明确正确候选时必须返回 `reject_all` 或 `ambiguous`。

“不要凭常识补不可见结构”只适用于确实没有图像或动作证据的隐藏部分，不能用来忽略画面中
已经可见的漏标结构。边界相交的候选执行完全相同标准，不因为靠边而放宽完整性要求。

S2 新增完成 17 个，累计 40/52。代表例子包括：

- `016534 target`：`silver object`；
- `018209 target`：frame 48 的 `white bar`；
- `008251`、`008272`、`008292 target`：`blue object`；
- `018892 target`：`blue object`。

## 7. S3：Qwen bbox → SAM box mask

### 7.1 配置

S3 使用已经收敛为 S1–S3-only 的配置：

```text
configs/open_set_mask_fallback_bbox.yaml
configs/prompts/open_set_bbox_localization.txt
```

关键增量：

```yaml
mask:
  qc_bbox_fallback_enabled: true
  qc_bbox_prompt_template: prompts/open_set_bbox_localization.txt
  qc_bbox_max_tokens: 180
```

### 7.2 执行顺序

S3 的完整顺序是：

1. 先穷尽当前角色全部 text query × 合法 seed；
2. 对同一组 seed 逐个请求 Qwen 定位指定角色；
3. Qwen 只允许输出 `status`、`bbox_xyxy`、`confidence`、`reason`；
4. `bbox_xyxy` 必须是 `[x0, y0, x1, y1]` 的有限归一化坐标，满足
   `0 ≤ x0 < x1 ≤ 1`、`0 ≤ y0 < y1 ≤ 1`；
5. 非法 JSON、非法坐标、`ambiguous` 或 `not_visible` 都不能调用 SAM 或自动修正坐标；
6. 合法 bbox 原样传入 SAM box prompt，得到候选 `BBOX` 的真实二值 mask；
7. `BBOX` 再经过与文本候选相同的面积、重复和 visual QC；
8. 只有 visual QC 通过，才把该 seed mask 送去 native video propagation。

Qwen 的 bbox confidence 当前只记录在 provenance 中，不单独作为最终接受阈值；安全门是随后对
真实 SAM mask 的普通 QC。定位 prompt 明确分离 target 和 receiver，尤其 phone 任务中不能用
phone 替代 stand/holder。

S3 新增完成 4 个：

| Episode | 通过方式 |
| ---: | --- |
| `008284` | target `BBOX` |
| `016500` | target、receiver `BBOX` |
| `016549` | target `BBOX` |
| `016930` | target、receiver `BBOX` |

S3 不做候选方向扩张、不因触边进行特殊二次判定，也不在传播后用静态范围强行裁剪轨迹。

## 8. 传播与普通 temporal QC

通过 seed QC 后，SAM3 从选中 seed 双向做 native propagation，只在该角色的输出窗口发布。
固定 envelope 仅保存为 seed 诊断图，不参与逐帧修正。

普通 temporal QC 记录：

- 输出窗口 coverage 和内部缺帧；
- adjacent IoU p05；
- centroid jump p95；
- area-ratio jump p95。

默认严重阈值为：

```text
adjacent IoU p05 < 0.5
centroid jump p95 > 5 px
area-ratio jump p95 > 0.4
```

三类连续性信号中至少两类严重越界才 `quarantine`；单一信号为 `review`。`quarantine` 会清空
该角色的发布 mask；`review` 当前仍可形成 completion receipt，所以 receipt 不能替代人工验收。

44 个 S1–S3 completion 中有 5 个 temporal review：

- `017087 target`
- `019250 receiver`
- `019277 receiver`
- `019299 receiver`
- `016500 receiver`

因此可进一步拆成“39 个无时序告警 + 5 个需要人工看视频”。

## 9. S4 为什么删除

被删除的 S4 包含三类强规则：

1. 在普通 bbox 被拒绝后，按固定方向和比例改变 bbox；
2. 对触碰图像边界的单候选换特殊 prompt 再判一次，并降低边界不完整性的拒绝倾向；
3. 根据 seed 附近静态范围对传播轨迹做额外裁剪/重判。

这些规则能提高 completion 数，但把“怎样修正”写死在系统里，而不是让角色身份、完整轮廓和
动作证据决定结果。已观察到的风险案例中，候选漏掉任务明确点名的可见部分和夹爪接触结构，
说明额外 completion 不能视为可靠成功。

最终处理是：

- 删除 S4 的可执行配置字段，并对遗留 S4 字段显式 fail-closed；
- 删除特殊候选方法和生成逻辑；
- 删除触边专用 prompt、渲染和 retry；
- 删除传播强制修正、专属 quarantine 和持久化字段；
- 删除对应单元测试；
- 保留 S2/S3 的严格 open-set QC 条款，尤其是可见部件、同步运动和夹爪动作证据；
- 保留普通 temporal QC 的 fail-closed quarantine。

旧 review bundle 中仍有 `s4/` 和对应行，目的是保留实验审计证据，不代表最终方案继续使用。
查看最终结果时只统计 S1、S2、S3。

## 10. 代码与配置改动位置

以下实现已移植到本分支 `feat/default-open-set-s1-s3`：

| 文件 | 职责 |
| --- | --- |
| `src/robotwin_annotation_v2/config.py` | S1–S3 配置字段及依赖验证 |
| `src/robotwin_annotation_v2/models/semantic_plan.py` | 开放集 query bank schema |
| `src/robotwin_annotation_v2/models/mask_qc.py` | 每次 text/bbox attempt 的结构化记录 |
| `src/robotwin_annotation_v2/pipeline/open_set_queries.py` | task/role-aware curated aliases |
| `configs/open_set_query_aliases.yaml` | 任务相关 alias 数据，避免在 Python 中硬编码物体名 |
| `src/robotwin_annotation_v2/pipeline/bbox_localization.py` | bbox prompt 渲染、JSON 和坐标合同 |
| `src/robotwin_annotation_v2/pipeline/object_mask/resolver.py` | text query → legal seed → bbox 的全局调度和停止条件 |
| `src/robotwin_annotation_v2/pipeline/object_mask/{planner,proposals,qc}.py` | query 计划、SAM proposal 与 mechanical QC |
| `src/robotwin_annotation_v2/pipeline/mask_qc.py` | 具体候选生成与 Qwen visual-QC execution |
| `src/robotwin_annotation_v2/pipeline/sam_stage.py` | native propagation 与普通 temporal QC |
| `src/robotwin_annotation_v2/application/episode_pipeline.py` | 复用已选 seed mask、artifact 接线 |
| `src/robotwin_annotation_v2/application/mask_qc_artifacts.py` | object-mask QC diagnostics publication |
| `scripts/run_open_set_failure_experiment.py` | 只运行声明的失败 episode |
| `scripts/render_open_set_bad_cases.py` | 从显式 source episode 清单渲染 bad-case 视频 |

移植基线为 `master@6dd9cea`。最终代码保留了 master 的 managed-Qwen、`masks.npz v3`、
`frame_encoding` 和 target grasp-hold 合同；没有整体 cherry-pick 混有 S4 的历史提交。

## 11. 配置矩阵

| 配置项 | Baseline | S1 | S2 | S3 |
| --- | ---: | ---: | ---: | ---: |
| `qc_max_candidates` | 3 | 8 | 8 | 8 |
| query fallback | off | on | on | on |
| seed fallback | off | on | on | on |
| open-set semantic prompt | off | off | on | on |
| open-set mask-QC prompt | off | off | on | on |
| bbox fallback | off | off | off | on |
| S4 强规则 | off | off | off | **off** |

注意：

- `qwen.allow_query_fallback` 一直为 false；本实验使用的是
  `mask.qc_query_fallback_enabled`；
- S1 运行同时包含候选记录和基础 QC prompt 的小幅修订，因此 23 个不能当作严格单变量消融；
- S2 同时改变 semantic prompt 和 mask-QC prompt，17 个不能继续拆成两个独立贡献；
- 这是渐进式工程救回记录，不是随机种子、服务状态完全受控的学术消融实验。

## 12. 如何只跑失败 case

先启动 Qwen：

```bash
just serve-qwen
```

只验证配置和输入，不加载模型：

```bash
PYTHONPATH=src .venv/bin/python scripts/run_open_set_failure_experiment.py \
  --config configs/open_set_mask_fallback_bbox.yaml \
  --validate-only
```

S1 全部 52 个失败：

```bash
PYTHONPATH=src .venv/bin/python scripts/run_open_set_failure_experiment.py \
  --config configs/open_set_mask_fallback_failures.yaml \
  --sam-gpu 1 \
  --output-root /tmp/open_set_s1 \
  --run-id-prefix open-set-s1-20260818 \
  --ui plain
```

S2 建议只跑 S1 后仍失败的 episode；S3 建议只跑 S2 后的剩余 12 个：

```text
8262 8275 8284 8417 16130 16500 16549 16930 19298 19424 19545 19849
```

命令：

```bash
PYTHONPATH=src .venv/bin/python scripts/run_open_set_failure_experiment.py \
  --config configs/open_set_mask_fallback_bbox.yaml \
  --episode-ids 8262 8275 8284 8417 16130 16500 16549 16930 19298 19424 19545 19849 \
  --sam-gpu 1 \
  --output-root /tmp/open_set_s3 \
  --run-id-prefix open-set-s3-20260818 \
  --ui plain
```

使用 `--force` 会重新计算已完成 episode；不加时 runner 会按既有 run/receipt 语义处理。真实
复现实验前应确认 GPU 空闲和 18086 Qwen endpoint 健康，避免把服务问题重新混进语义结果。

## 13. 如何看 mask 图、视频和证据

审查入口：

```text
artifacts/reviews/open_set_mask_fallback_20260818/index.html
```

这是本机实验产物路径，不随 Git checkout 分发；fresh checkout 只能获得本文记录与可复现命令。

同目录还包含：

```text
manifest.json
index.csv
stage_overviews/
s1/
s2/
s3/
s4/   # 仅历史实验，不计入最终方案
```

每个 S1–S3 episode 推荐按以下顺序检查：

1. `candidate_seed_target.png`：是否覆盖真正被抓取的完整可见 target；
2. `candidate_seed_receiver.png`：是否是最终与 target 直接接触的独立对象/区域；
3. `temporal_contact_6frames.jpg`：检查 target start/seed/late、grasp boundary、receiver
   middle/end；
4. `mask_overlay.mp4`：逐帧检查漂移、跳实例、吸附夹爪、内部断帧、release 后错误延续；
5. `mask_qc.source.json`：检查 selected method、seed、query、attempt history 和 Qwen 原始原因；
6. target/receiver 的 `temporal_qc.source.json`：检查 coverage、internal gaps 和跳变指标；
7. `masks.source.npz`、completion receipt 和 SHA256：核对最终发布内容与审查副本一致。

overlay 图例：绿色 target、蓝色 receiver、红色 gripper。HTML 中的黄色高风险标记属于旧 S4；
最终审查应过滤 `stage in {S1,S2,S3}`。

当前分支的 bad-case renderer 不会 glob 或自动选择“最新”run。输入 JSON 的每一条必须
显式给出 `episode_dir` 和 `video_path`，并将源 run、状态与 SHA256 写入输出 manifest：

```bash
PYTHONPATH=src .venv/bin/python scripts/render_open_set_bad_cases.py \
  --manifest artifacts/reviews/open_set_s3_current_master_20260818/cases.json \
  --output-dir artifacts/reviews/open_set_s3_current_master_20260818/videos \
  --overwrite
```

## 14. 结果清单

### 14.1 S1 新救回 23 个

```text
008561 008799
014881 014899 015050 015294
016001 016246 016501
017050 017087
018698
018710 018722 018749 018855 018982 019109 019248
019250 019273 019277 019299
```

### 14.2 S2 新救回 17 个

```text
008251 008272 008292 008675 008798
016380
016511 016512 016529 016534 016548 016550 016683
018166 018209 018699
018892
```

### 14.3 S3 新救回 4 个

```text
008284 016500 016549 016930
```

### 14.4 截止 S3 仍失败 8 个

```text
008262 008275 008417 016130 019298 019424 019545 019849
```

不要求用高风险规则把这 8 个全部强行做成 completed；在总体 `172/180` 已超过 90% 时，保留
少量 fail-closed 比发布不完整或错误实例更合理。

## 15. 已知风险和不能过度解读的地方

- `44/52` 是 pipeline completion，不是 44 个经人工确认的像素级真阳性；
- `016246 receiver` 已发现明显可疑：期望是圆形木质 coaster，但候选像是覆盖了较大桌面区域，
  应在最终人工验收中单独复核，不能因 receipt completed 自动接受；
- 5 个 temporal review 仍需看完整视频；
- appearance query 越宽，选错同场实例的概率越高；
- 多个同义词最终仍使用同一个 SAM3 text-grounding 模型，失败高度相关；
- 正确 seed mask 不保证遮挡后的传播稳定；
- Qwen QC 有随机性，阶段结果来自多次 lane/retry，并非一次完全确定性运行；
- curated aliases 可能过拟合当前九个 task family；
- 只测试已知失败切片，不能证明未知开放集类别的全量泛化；
- 没有 simulator pixel ground truth，不能把结果称为 IoU、precision 或 recall。

## 16. 验证命令

S1–S3 改动提交或合并前，至少运行：

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/unit/test_open_set_failure_experiment.py \
  tests/unit/test_open_set_queries.py \
  tests/unit/test_mask_qc.py \
  tests/unit/test_mask_qc_attempts.py \
  tests/unit/test_render_open_set_bad_cases.py \
  tests/unit/test_sam_stage.py -q

just test
just lint
.venv/bin/python -m mypy src
git diff --check
```

随后再做：单 episode smoke、S3 四个 bbox case、一个 temporal review case，以及 exact-run
overlay/video 人工复核。测试通过只证明代码合同，没有替代对实际 mask 的视觉判断。
