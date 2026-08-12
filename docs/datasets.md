# RoboTwin pick-and-place 数据集兼容性

> 扫描日期：2026-08-07。这里的“兼容”指当前单 target、单 receiver、单 active arm、一次
> close/open loop 的 `cam_high` pipeline；不代表所有 episode 都有完整 depth，也不代表
> Qwen/SAM mask 一定通过。

## 1. 结论

RoboTwin 2.0 中有 9 类任务可直接套用当前角色/事件模型，共 4,950 个 episode。每类 550 条：
前 50 条 `clean_50`，后 500 条 `randomized_500`。9 类的全部 Parquet state 都通过当前“一次
完整 gripper loop”检测。

完整数据根：

```text
/DATA/disk8/xuran/robotwin2.0_full_aloha_rgbd_320_rgb_correct_v1
```

元数据在 `meta/episodes.jsonl`；task instruction 和环境代码分别位于本机 RoboTwin/FastWAM
checkout。运行 pipeline 不应修改完整数据根。

## 2. 可直接复用的任务

| 优先级 | 任务 | episode | target → receiver | 备注 |
| --- | --- | --- | --- | --- |
| A1 | `move_stapler_pad` | 8250–8799 | stapler → 彩色 mat | 与当前 pilot 最接近 |
| A1 | `place_mouse_pad` | 17050–17599 | mouse → 彩色 mat | 动作同构、语义清晰 |
| A1 | `place_empty_cup` | 15950–16499 | cup → coaster | 完整 RGBD，优先回归 |
| A1 | `place_container_plate` | 14850–15399 | bowl/cup → plate | receiver 轮廓清晰 |
| A2 | `place_object_scale` | 18150–18699 | mouse/stapler/bell → scale | 跨 target 类别 |
| A2 | `place_object_stand` | 18700–19249 | 多类物体 → display stand | distractor 较多 |
| A2 | `place_phone_stand` | 19250–19799 | phone → phone stand | 放置遮挡/对齐更难 |
| A2 | `place_shoe` | 19800–20349 | shoe → 蓝色 mat | 有方向约束 |
| A3 | `place_fan` | 16500–17049 | fan → 彩色 mat | 部分帧低对比，风险最高 |

每类 clean 是起始 ID 到 `start+49`，randomized 是 `start+50` 到结束。例如
`move_stapler_pad`：clean 8250–8299，randomized 8300–8799。

“直接复用”只保证角色和 state-loop 结构满足：

1. 一个被抓取并移动的 target；
2. 一个任务完成时与 target 直接接触的 receiver/区域；
3. 一次完整 close → open；
4. 不需要扩展固定 `target_0/receiver_0` 数据模型。

## 3. RGBD 完整性

检查了四路 depth：

```text
sidecars/videos/chunk-*/observation.depths.
  {cam_high,cam_left_wrist,cam_right_wrist,front_camera}/episode_*.mkv
```

| A1 任务 | 四路 depth 齐全 | 缺失 episode | 结论 |
| --- | ---: | --- | --- |
| `place_empty_cup` | 550/550 | 无 | 唯一完整 550 RGBD A1 集 |
| `place_mouse_pad` | 548/550 | 17071, 17402 | 使用 548 子集或补 depth |
| `place_container_plate` | 547/550 | 14941, 15022, 15360 | 使用 547 子集或补 depth |
| `move_stapler_pad` | 512/550 | 见下 | 不能称为完整 550 RGBD |

`move_stapler_pad` 缺失 38 条：

```text
8253, 8274, 8300, 8303, 8306, 8359, 8361, 8383, 8395, 8396,
8437, 8451, 8453, 8507, 8516, 8520, 8529, 8553, 8560, 8579,
8601, 8623, 8636, 8638, 8654, 8659, 8693, 8700, 8710, 8712,
8726, 8730, 8748, 8750, 8762, 8777, 8782, 8792
```

这四类的 Parquet、四路 RGB 和 HDF5 sidecar 均为 550/550；缺口只在 depth，且缺失
episode 的四个 camera depth 同时缺失。作为对照，
`move_pillbottle_pad_full550_original` 为 546/550，缺 7273、7473、7629、7696。

后端要求不同：

| backend | 必需输入 |
| --- | --- |
| 默认 `sam` | Parquet + selected RGB MP4 + HDF5 sidecar |
| `urdf` | 上述输入 + selected camera depth MKV |

因此 depth 缺失不阻止 SAM pipeline，却会被 URDF discovery 排除。自动发现存在排除项时，
URDF 默认 fail closed；明确接受完整子集时才传 `--allow-partial-source`。显式
`--episode-ids` 始终不能静默丢项。

## 4. 需要额外策略

| 任务 | episode | 额外结构 |
| --- | --- | --- |
| `place_bread_skillet` | 12650–13199 | 一手拿 skillet，另一手放 bread；receiver 会动 |
| `place_can_basket` | 13750–14299 | 放置后另一手抬 basket |
| `place_object_basket` | 17600–18149 | 放置后移动 basket |
| `put_object_cabinet` | 21450–21999 | 另一手开 drawer；articulated receiver |
| `place_bread_basket` | 12100–12649 | 单/双 bread 混合，仅 144/550 是单循环 |

这些任务至少需要动态 receiver、第二只 gripper 事件，或先筛出单物体 episode。

## 5. 不适合直接处理

- `move_can_pot`、`place_a2b_left/right`：目标放在另一物体旁，实际接触桌面，不符合当前
  receiver 直接接触定义。
- `handover_block`、`hanging_mug`：两次完整 gripper loop，当前检测为 0/550。
- `place_burger_fries`、`place_cans_plasticbox`、`place_dual_shoes`、
  `put_bottles_dustbin`：多 target 或双臂放置。
- `stack_blocks_*`、`stack_bowls_*`、`blocks_ranking_*`：连续多物体、多阶段事件。
- `adjust_bottle`、`pick_*`、`shake_*`、`open_*`、`click_*`、`press_*`、
  `turn_switch` 等：不是当前定义的标准 pick-and-place，缺少稳定 receiver 角色。

不要只因为 state detector 能返回一个 loop 就将上述任务标为兼容；角色语义和产物结构同样
必须满足。

## 6. 推荐迁移顺序

若只使用 SAM backend：

1. `move_stapler_pad`
2. `place_mouse_pad`
3. `place_empty_cup`
4. `place_container_plate`

这四类覆盖彩色平面、coaster 和 plate，同时保持单 target/receiver。每类先抽 20 条，平衡
clean/randomized、左右臂、颜色和物体型号；通过后再扩
`place_object_scale/object_stand`。

若 URDF/depth 是硬要求，顺序调整为：

1. `place_empty_cup`（550/550）
2. `place_mouse_pad` 的 548 子集
3. `place_container_plate` 的 547 子集
4. `move_stapler_pad` 的 512 子集

## 7. 已抽取数据和运行

当前已准备的严格 RGBD 数据集：

```text
/DATA/disk8/xuran/add_mask_robotwin/dataset/
  place_empty_cup_full550_original
  place_container_plate_full547_original
```

### 7.1 `place_empty_cup_full550_original`

episode 15950–16499，四路 RGB/depth 均为 550/550，已提供：

```text
configs/datasets/place_empty_cup_full550.json
configs/pilot_place_empty_cup.yaml
```

默认 SAM：

```bash
just serve-qwen
just config=configs/pilot_place_empty_cup.yaml \
  process ../dataset/place_empty_cup_full550_original
```

live URDF：

```bash
just config=configs/pilot_place_empty_cup.yaml \
  process ../dataset/place_empty_cup_full550_original \
  --gripper-backend urdf
```

已有 frozen source 只有 456/550 对象结果满足 QC；使用它做 URDF 子集时必须显式
`--source-run-dir` 和 `--allow-partial-source`。456 条只完成合同 dry-run，尚不能表述为正式
full render；已完成的左右臂 pilot 和数字见 [experiments.md](experiments.md#44-place_empty_cup_full550-扩展)。

### 7.2 `place_container_plate_full547_original`

原任务 episode 14850–15399；去除没有 replay geometry/depth 的 14941、15022、15360 后，
保留 547 条四路 RGB/depth 完整 episode。文件保持原始全局 episode id 和逐文件字节内容，
提取清单记录在数据根的 `EXTRACT_MANIFEST.json`。已提供：

```text
configs/datasets/place_container_plate_full547.json
configs/pilot_place_container_plate.yaml
```

live URDF 全量运行无需再传 `--episode-ids` 或 `--allow-partial-source`：

```bash
just config=configs/pilot_place_container_plate.yaml \
  process ../dataset/place_container_plate_full547_original \
  --gripper-backend urdf
```

这里的“无需 `--allow-partial-source`”仅指 547 条数据全部满足 depth 合同；默认仍会在任一
target/receiver source 未通过 QC 时 fail closed。若希望 source 阶段仍覆盖全部 547 条、随后
只对 QC 通过项继续 URDF，可显式追加 `--allow-partial-source`。

`scripts/process_dataset.py` 使用 Parquet 帧数为有效长度；原始 RGB/depth 多出的尾帧不会进入
mask。任务适配前还应检查 prompt 中 target/receiver 角色定义，以及新类别在 Qwen/SAM
candidate QC 中的表现。
