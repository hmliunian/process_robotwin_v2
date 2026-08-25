# RoboTwin 50 个 coarse task：单/双臂与任务语义分类

全量语义扫描日期：2026-08-17

11-task 抽取验收：2026-08-20；runtime profile 决策：2026-08-25
统计范围：RoboTwin 2.0 完整数据集，50 个 coarse task，每类 550 条 episode，共 27,500 条。

## 数据集路径

完整数据集根目录：

```text
/DATA/disk8/xuran/robotwin2.0_full_aloha_rgbd_320_rgb_correct_v1
```

本次使用的来源：

```text
/DATA/disk8/xuran/robotwin2.0_full_aloha_rgbd_320_rgb_correct_v1/meta/episodes.jsonl
/DATA/disk8/xuran/robotwin2.0_full_aloha_rgbd_320_rgb_correct_v1/data/chunk-*/episode_*.parquet
```

`episodes.jsonl` 提供 coarse task 和 episode text prompt；Parquet 的 `action` 字段用于单/双臂判定。

## 分类口径

### 单臂 / 双臂

统计单位先是 episode，再汇总到 task。左右臂身份可以在 episode 间变化，不影响单臂判定。

Parquet 的 14 维 action 按以下结构读取：

```text
left arm:  action[0:6]  pose, action[6]  gripper
right arm: action[7:13] pose, action[13] gripper
```

对每只手计算每帧 pose 活跃度：

```text
score = ||Δxyz|| + 0.05 × ||Δrpy||
```

满足至少 3 帧 `score > 0.002` 且累计 score `>= 0.05`，或 gripper 指令发生明显切换，视为该臂有效参与。数值抖动和初始化噪声不计入。

- **单臂 episode**：只有一只手有有效 action。
- **双臂 episode**：左右两只手都有有效 action，包括一手操作物体、另一手固定或移动 receiver。
- **单臂 task**：550 条 episode 全部单臂。
- **双臂 task**：550 条 episode 全部双臂。
- **混合 task**：同一 task 同时存在单臂和双臂 episode，不按多数票强行归类。

全量扫描中，非活跃臂与活跃臂之间存在明显 action 幅度间隔；加入或不加入 gripper 条件均未改变最终单/双臂标签。没有 episode 被判为零活跃臂。

### 任务语义

任务语义与单/双臂独立统计，分为四类：

1. **严格 P&P**：A 最终被放到、放入、挂到或堆叠到独立实体 B，存在明确的 `target → receiver` 关系。
2. **P&P-其他**：A 被移动到另一个明确语义对象 B 的左/右/旁边/远离，或多个对象按相对关系排列；B 是 reference，不是 receiver。
3. **Target-only**：只改变单个对象或其自身部件的状态、姿态或位置，没有第二个语义对象或外部 receiver；仅相对工作空间方向移动仍属于 Target-only。
4. **其他**：工具—对象接触、多对象抓持、倾倒等，不强行塞入前两类。

严格 P&P 的语义分类不等于当前单 target/单 receiver pipeline 已经可以直接处理；多 target、动态 receiver、articulated receiver 和多阶段任务仍需额外 profile。

## 总体统计

### 单臂 / 双臂

| task-level 标签 | task 数 | episode 级数量 |
|---|---:|---:|
| 全量单臂 | 25 | 13,750（这些 task 的全部 episode） |
| 全量双臂 | 15 | 8,250（这些 task 的全部 episode） |
| 混合 | 10 | 单臂 1,420；双臂 4,080 |
| **合计** | **50** | **单臂 15,170；双臂 12,330** |

episode 总体比例：单臂 55.16%，双臂 44.84%。

### 任务语义

| 语义类别 | task 数 | episode 数 | 占全部 episode |
|---|---:|---:|---:|
| 严格 P&P | 25 | 13,750 | 50.00% |
| P&P-其他 | 5 | 2,750 | 10.00% |
| **P&P 家族合计** | **30** | **16,500** | **60.00%** |
| Target-only | 14 | 7,700 | 28.00% |
| 其他工具/多对象/倾倒 | 6 | 3,300 | 12.00% |
| **合计** | **50** | **27,500** | **100.00%** |

### 语义类别 × 单/双臂 task 数

| 语义类别 | 全量单臂 | 全量双臂 | 混合 |
|---|---:|---:|---:|
| 严格 P&P | 9 | 9 | 7 |
| P&P-其他 | 3 | 0 | 2 |
| Target-only | 11 | 3 | 0 |
| 其他 | 2 | 3 | 1 |
| **合计** | **25** | **15** | **10** |

### 逐 task 联合分类表

语义类别和 action 单/双臂类别合并在同一张表中。对混合任务直接列出单臂与双臂 episode 数，不按多数票归入某一侧。完整数据集没有 task 级目录，因此每个 task 超链接使用相对于本文档的路径，指向该任务首条 episode 的主视角相机（`cam_high`）MP4，作为任务动作示例。

| task（首条 episode 的主视角 MP4 链接） | 语义类别 | action 类别 | 单臂 episode | 双臂 episode |
|---|---|---|---:|---:|
| [adjust_bottle](../../../robotwin2.0_full_aloha_rgbd_320_rgb_correct_v1/videos/chunk-000/observation.images.cam_high/episode_000000.mp4) | Target-only | 全量单臂 | 550 | 0 |
| [beat_block_hammer](../../../robotwin2.0_full_aloha_rgbd_320_rgb_correct_v1/videos/chunk-000/observation.images.cam_high/episode_000550.mp4) | 其他 | 全量单臂 | 550 | 0 |
| [blocks_ranking_rgb](../../../robotwin2.0_full_aloha_rgbd_320_rgb_correct_v1/videos/chunk-001/observation.images.cam_high/episode_001100.mp4) | P&P-其他 | 混合 | 9 | 541 |
| [blocks_ranking_size](../../../robotwin2.0_full_aloha_rgbd_320_rgb_correct_v1/videos/chunk-001/observation.images.cam_high/episode_001650.mp4) | P&P-其他 | 混合 | 10 | 540 |
| [click_alarmclock](../../../robotwin2.0_full_aloha_rgbd_320_rgb_correct_v1/videos/chunk-002/observation.images.cam_high/episode_002200.mp4) | Target-only | 全量单臂 | 550 | 0 |
| [click_bell](../../../robotwin2.0_full_aloha_rgbd_320_rgb_correct_v1/videos/chunk-002/observation.images.cam_high/episode_002750.mp4) | Target-only | 全量单臂 | 550 | 0 |
| [dump_bin_bigbin](../../../robotwin2.0_full_aloha_rgbd_320_rgb_correct_v1/videos/chunk-003/observation.images.cam_high/episode_003300.mp4) | 其他 | 混合 | 323 | 227 |
| [grab_roller](../../../robotwin2.0_full_aloha_rgbd_320_rgb_correct_v1/videos/chunk-003/observation.images.cam_high/episode_003850.mp4) | Target-only | 全量双臂 | 0 | 550 |
| [handover_block](../../../robotwin2.0_full_aloha_rgbd_320_rgb_correct_v1/videos/chunk-004/observation.images.cam_high/episode_004400.mp4) | 严格 P&P | 全量双臂 | 0 | 550 |
| [handover_mic](../../../robotwin2.0_full_aloha_rgbd_320_rgb_correct_v1/videos/chunk-004/observation.images.cam_high/episode_004950.mp4) | Target-only | 全量双臂 | 0 | 550 |
| [hanging_mug](../../../robotwin2.0_full_aloha_rgbd_320_rgb_correct_v1/videos/chunk-005/observation.images.cam_high/episode_005500.mp4) | 严格 P&P | 全量双臂 | 0 | 550 |
| [lift_pot](../../../robotwin2.0_full_aloha_rgbd_320_rgb_correct_v1/videos/chunk-006/observation.images.cam_high/episode_006050.mp4) | Target-only | 全量双臂 | 0 | 550 |
| [move_can_pot](../../../robotwin2.0_full_aloha_rgbd_320_rgb_correct_v1/videos/chunk-006/observation.images.cam_high/episode_006600.mp4) | P&P-其他 | 全量单臂 | 550 | 0 |
| [move_pillbottle_pad](../../../robotwin2.0_full_aloha_rgbd_320_rgb_correct_v1/videos/chunk-007/observation.images.cam_high/episode_007150.mp4) | 严格 P&P | 全量单臂 | 550 | 0 |
| [move_playingcard_away](../../../robotwin2.0_full_aloha_rgbd_320_rgb_correct_v1/videos/chunk-007/observation.images.cam_high/episode_007700.mp4) | Target-only | 全量单臂 | 550 | 0 |
| [move_stapler_pad](../../../robotwin2.0_full_aloha_rgbd_320_rgb_correct_v1/videos/chunk-008/observation.images.cam_high/episode_008250.mp4) | 严格 P&P | 全量单臂 | 550 | 0 |
| [open_laptop](../../../robotwin2.0_full_aloha_rgbd_320_rgb_correct_v1/videos/chunk-008/observation.images.cam_high/episode_008800.mp4) | Target-only | 全量单臂 | 550 | 0 |
| [open_microwave](../../../robotwin2.0_full_aloha_rgbd_320_rgb_correct_v1/videos/chunk-009/observation.images.cam_high/episode_009350.mp4) | Target-only | 全量单臂 | 550 | 0 |
| [pick_diverse_bottles](../../../robotwin2.0_full_aloha_rgbd_320_rgb_correct_v1/videos/chunk-009/observation.images.cam_high/episode_009900.mp4) | 其他 | 全量双臂 | 0 | 550 |
| [pick_dual_bottles](../../../robotwin2.0_full_aloha_rgbd_320_rgb_correct_v1/videos/chunk-010/observation.images.cam_high/episode_010450.mp4) | 其他 | 全量双臂 | 0 | 550 |
| [place_a2b_left](../../../robotwin2.0_full_aloha_rgbd_320_rgb_correct_v1/videos/chunk-011/observation.images.cam_high/episode_011000.mp4) | P&P-其他 | 全量单臂 | 550 | 0 |
| [place_a2b_right](../../../robotwin2.0_full_aloha_rgbd_320_rgb_correct_v1/videos/chunk-011/observation.images.cam_high/episode_011550.mp4) | P&P-其他 | 全量单臂 | 550 | 0 |
| [place_bread_basket](../../../robotwin2.0_full_aloha_rgbd_320_rgb_correct_v1/videos/chunk-012/observation.images.cam_high/episode_012100.mp4) | 严格 P&P | 混合 | 269 | 281 |
| [place_bread_skillet](../../../robotwin2.0_full_aloha_rgbd_320_rgb_correct_v1/videos/chunk-012/observation.images.cam_high/episode_012650.mp4) | 严格 P&P | 全量双臂 | 0 | 550 |
| [place_burger_fries](../../../robotwin2.0_full_aloha_rgbd_320_rgb_correct_v1/videos/chunk-013/observation.images.cam_high/episode_013200.mp4) | 严格 P&P | 全量双臂 | 0 | 550 |
| [place_can_basket](../../../robotwin2.0_full_aloha_rgbd_320_rgb_correct_v1/videos/chunk-013/observation.images.cam_high/episode_013750.mp4) | 严格 P&P | 全量双臂 | 0 | 550 |
| [place_cans_plasticbox](../../../robotwin2.0_full_aloha_rgbd_320_rgb_correct_v1/videos/chunk-014/observation.images.cam_high/episode_014300.mp4) | 严格 P&P | 全量双臂 | 0 | 550 |
| [place_container_plate](../../../robotwin2.0_full_aloha_rgbd_320_rgb_correct_v1/videos/chunk-014/observation.images.cam_high/episode_014850.mp4) | 严格 P&P | 全量单臂 | 550 | 0 |
| [place_dual_shoes](../../../robotwin2.0_full_aloha_rgbd_320_rgb_correct_v1/videos/chunk-015/observation.images.cam_high/episode_015400.mp4) | 严格 P&P | 全量双臂 | 0 | 550 |
| [place_empty_cup](../../../robotwin2.0_full_aloha_rgbd_320_rgb_correct_v1/videos/chunk-015/observation.images.cam_high/episode_015950.mp4) | 严格 P&P | 全量单臂 | 550 | 0 |
| [place_fan](../../../robotwin2.0_full_aloha_rgbd_320_rgb_correct_v1/videos/chunk-016/observation.images.cam_high/episode_016500.mp4) | 严格 P&P | 混合 | 549 | 1 |
| [place_mouse_pad](../../../robotwin2.0_full_aloha_rgbd_320_rgb_correct_v1/videos/chunk-017/observation.images.cam_high/episode_017050.mp4) | 严格 P&P | 全量单臂 | 550 | 0 |
| [place_object_basket](../../../robotwin2.0_full_aloha_rgbd_320_rgb_correct_v1/videos/chunk-017/observation.images.cam_high/episode_017600.mp4) | 严格 P&P | 全量双臂 | 0 | 550 |
| [place_object_scale](../../../robotwin2.0_full_aloha_rgbd_320_rgb_correct_v1/videos/chunk-018/observation.images.cam_high/episode_018150.mp4) | 严格 P&P | 全量单臂 | 550 | 0 |
| [place_object_stand](../../../robotwin2.0_full_aloha_rgbd_320_rgb_correct_v1/videos/chunk-018/observation.images.cam_high/episode_018700.mp4) | 严格 P&P | 全量单臂 | 550 | 0 |
| [place_phone_stand](../../../robotwin2.0_full_aloha_rgbd_320_rgb_correct_v1/videos/chunk-019/observation.images.cam_high/episode_019250.mp4) | 严格 P&P | 全量单臂 | 550 | 0 |
| [place_shoe](../../../robotwin2.0_full_aloha_rgbd_320_rgb_correct_v1/videos/chunk-019/observation.images.cam_high/episode_019800.mp4) | 严格 P&P | 全量单臂 | 550 | 0 |
| [press_stapler](../../../robotwin2.0_full_aloha_rgbd_320_rgb_correct_v1/videos/chunk-020/observation.images.cam_high/episode_020350.mp4) | Target-only | 全量单臂 | 550 | 0 |
| [put_bottles_dustbin](../../../robotwin2.0_full_aloha_rgbd_320_rgb_correct_v1/videos/chunk-020/observation.images.cam_high/episode_020900.mp4) | 严格 P&P | 混合 | 3 | 547 |
| [put_object_cabinet](../../../robotwin2.0_full_aloha_rgbd_320_rgb_correct_v1/videos/chunk-021/observation.images.cam_high/episode_021450.mp4) | 严格 P&P | 全量双臂 | 0 | 550 |
| [rotate_qrcode](../../../robotwin2.0_full_aloha_rgbd_320_rgb_correct_v1/videos/chunk-022/observation.images.cam_high/episode_022000.mp4) | Target-only | 全量单臂 | 550 | 0 |
| [scan_object](../../../robotwin2.0_full_aloha_rgbd_320_rgb_correct_v1/videos/chunk-022/observation.images.cam_high/episode_022550.mp4) | 其他 | 全量双臂 | 0 | 550 |
| [shake_bottle](../../../robotwin2.0_full_aloha_rgbd_320_rgb_correct_v1/videos/chunk-023/observation.images.cam_high/episode_023100.mp4) | Target-only | 全量单臂 | 550 | 0 |
| [shake_bottle_horizontally](../../../robotwin2.0_full_aloha_rgbd_320_rgb_correct_v1/videos/chunk-023/observation.images.cam_high/episode_023650.mp4) | Target-only | 全量单臂 | 550 | 0 |
| [stack_blocks_three](../../../robotwin2.0_full_aloha_rgbd_320_rgb_correct_v1/videos/chunk-024/observation.images.cam_high/episode_024200.mp4) | 严格 P&P | 混合 | 2 | 548 |
| [stack_blocks_two](../../../robotwin2.0_full_aloha_rgbd_320_rgb_correct_v1/videos/chunk-024/observation.images.cam_high/episode_024750.mp4) | 严格 P&P | 混合 | 90 | 460 |
| [stack_bowls_three](../../../robotwin2.0_full_aloha_rgbd_320_rgb_correct_v1/videos/chunk-025/observation.images.cam_high/episode_025300.mp4) | 严格 P&P | 混合 | 13 | 537 |
| [stack_bowls_two](../../../robotwin2.0_full_aloha_rgbd_320_rgb_correct_v1/videos/chunk-025/observation.images.cam_high/episode_025850.mp4) | 严格 P&P | 混合 | 152 | 398 |
| [stamp_seal](../../../robotwin2.0_full_aloha_rgbd_320_rgb_correct_v1/videos/chunk-026/observation.images.cam_high/episode_026400.mp4) | 其他 | 全量单臂 | 550 | 0 |
| [turn_switch](../../../robotwin2.0_full_aloha_rgbd_320_rgb_correct_v1/videos/chunk-026/observation.images.cam_high/episode_026950.mp4) | Target-only | 全量单臂 | 550 | 0 |

## 单/双臂逐 task 结果

### 全量单臂（25）

```text
adjust_bottle
beat_block_hammer
click_alarmclock
click_bell
move_can_pot
move_pillbottle_pad
move_playingcard_away
move_stapler_pad
open_laptop
open_microwave
place_a2b_left
place_a2b_right
place_container_plate
place_empty_cup
place_mouse_pad
place_object_scale
place_object_stand
place_phone_stand
place_shoe
press_stapler
rotate_qrcode
shake_bottle
shake_bottle_horizontally
stamp_seal
turn_switch
```

### 全量双臂（15）

```text
grab_roller
handover_block
handover_mic
hanging_mug
lift_pot
pick_diverse_bottles
pick_dual_bottles
place_bread_skillet
place_burger_fries
place_can_basket
place_cans_plasticbox
place_dual_shoes
place_object_basket
put_object_cabinet
scan_object
```

### 混合 task（10）

| task | 单臂 episode | 双臂 episode |
|---|---:|---:|
| `blocks_ranking_rgb` | 9 | 541 |
| `blocks_ranking_size` | 10 | 540 |
| `dump_bin_bigbin` | 323 | 227 |
| `place_bread_basket` | 269 | 281 |
| `place_fan` | 549 | 1 |
| `put_bottles_dustbin` | 3 | 547 |
| `stack_blocks_three` | 2 | 548 |
| `stack_blocks_two` | 90 | 460 |
| `stack_bowls_three` | 13 | 537 |
| `stack_bowls_two` | 152 | 398 |

少量异常/少数派 episode 的例子：

- `place_fan` 的唯一双臂 episode 是 `16568`，第二只手存在明显 pose action，不是数值噪声。
- `put_bottles_dustbin` 的 3 条单臂 episode 是 `20930`、`21131`、`21414`。
- `stack_blocks_three` 的 2 条单臂 episode 是 `24412`、`24525`。

## 任务语义逐类结果

### 严格 P&P（25）

```text
handover_block
hanging_mug
move_pillbottle_pad
move_stapler_pad
place_bread_basket
place_bread_skillet
place_burger_fries
place_can_basket
place_cans_plasticbox
place_container_plate
place_dual_shoes
place_empty_cup
place_fan
place_mouse_pad
place_object_basket
place_object_scale
place_object_stand
place_phone_stand
place_shoe
put_bottles_dustbin
put_object_cabinet
stack_blocks_three
stack_blocks_two
stack_bowls_three
stack_bowls_two
```

### P&P-其他 / 相对放置（5）

```text
blocks_ranking_rgb
blocks_ranking_size
move_can_pot
place_a2b_left
place_a2b_right
```

这些任务都有另一个明确语义对象 B，B 只作为相对位置参照，不作为 receiver 计入。

### Target-only（14）

```text
adjust_bottle
click_alarmclock
click_bell
grab_roller
handover_mic
lift_pot
move_playingcard_away
open_laptop
open_microwave
press_stapler
rotate_qrcode
shake_bottle
shake_bottle_horizontally
turn_switch
```

`handover_mic` 是“双臂 + target-only”的例子：只有一个场景物体，没有 receiver，但两只手都参与交接。

### 其他工具/多对象/倾倒（6）

```text
beat_block_hammer
dump_bin_bigbin
pick_diverse_bottles
pick_dual_bottles
scan_object
stamp_seal
```

- `beat_block_hammer`、`scan_object`、`stamp_seal`：工具—对象接触或 action-site 操作。
- `pick_diverse_bottles`、`pick_dual_bottles`：多对象抓持，没有 receiver。
- `dump_bin_bigbin`：受控对象是小垃圾桶，进入大桶的是被动 contents，保守地标为倾倒/内容转移，而不是标准 target→receiver P&P。

## 边界说明

### `handover_block`

主分类放在严格 P&P：不少 prompt 明确描述“交给另一只手后放到 blue pad”。但部分同义 prompt 只写交接、没有重复写出 pad，因此这是严格 P&P 类中置信度相对较低的一项。

### `dump_bin_bigbin`

主分类放在“其他/倾倒”。如果后续把桶内 contents 视为正式 target，也可以把它改入 P&P；这会使严格 P&P 从 25 变为 26、其他从 6 变为 5。

### `move_playingcard_away`

该任务只有 playing-card holder 一个语义对象；`outward/away` 是相对工作空间的移动方向，table 是背景坐标系，不构成第二个对象 B 或 receiver。因此归入 Target-only，可进一步标注为 `Target-only / relocate`。

### `stamp_seal`

部分 prompt 使用 `place ... onto` 的表面措辞，但任务结果是盖章/按压接触，不是把 seal 留在 receiver 上，因此归入工具接触类。

## 已发布子集复核

复核日期：2026-08-20（`target_only_20_v2` 扩展到严格单臂 Target-only 全集后复核）。

复核对象：

- [`pick_place_20` collection manifest](/DATA/disk8/xuran/add_mask_robotwin/dataset/pick_place_20/EXTRACT_MANIFEST.json)
- [`target_only_20_v2` collection manifest](/DATA/disk8/xuran/add_mask_robotwin/dataset/target_only_20_v2/EXTRACT_MANIFEST.json)

本次检查：

1. task 是否符合本报告的严格 P&P / Target-only 语义；
2. 每条抽取 episode 的 Parquet action 是否只有一只活跃臂；
3. selection、collection manifest、逐 task metadata 和实际 episode ID 是否一致；
4. 每条 episode 的 Parquet、HDF5 sidecar、`cam_high` RGB MP4 和 `cam_high` depth MKV 是否都存在。

### 总结

| 数据集 | task | episode | 语义符合 | 单臂 action 符合 | 四类文件齐全 | 结论 |
|---|---:|---:|---:|---:|---:|---|
| `pick_place_20` | 10 | 200 | 200/200 | 200/200 | 200/200 | **通过** |
| `target_only_20_v2` | 11 | 220 | 220/220 | 220/220 | 220/220 | **通过** |

### `pick_place_20`

10 个 task 均属于严格 P&P，每类 20 条；所有 task 都是 10 条左臂、10 条右臂，没有双臂或零活跃臂 episode：

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

`place_fan` 在完整 550 条中有一条双臂异常 episode `16568`，但它不在 `pick_place_20` 的抽样清单中；当前抽中的 20 条全部为单臂。因此 `pick_place_20` 同时满足“严格 P&P + 单臂 execution”。

### `target_only_20_v2`

| task | 本报告语义 | runtime profile | action 左/右/双臂 | 结论 |
|---|---|---|---:|---|
| `adjust_bottle` | Target-only | `grasp_manipulation` | 10/10/0 | 通过 |
| `click_alarmclock` | Target-only | `contact_press` | 10/10/0 | 通过 |
| `click_bell` | Target-only | `contact_press` | 10/10/0 | 通过 |
| `move_playingcard_away` | Target-only / relocate | `grasp_manipulation` | 10/10/0 | 通过 |
| `rotate_qrcode` | Target-only | `grasp_manipulation` | 10/10/0 | 通过 |
| `shake_bottle` | Target-only | `grasp_manipulation` | 10/10/0 | 通过 |
| `shake_bottle_horizontally` | Target-only | `grasp_manipulation` | 10/10/0 | 通过 |
| `open_laptop` | Target-only | `grasp_manipulation` | 10/10/0 | 通过 |
| `open_microwave` | Target-only | `grasp_manipulation` | 20/0/0 | 通过；该任务实际只使用左臂 |
| `press_stapler` | Target-only | `contact_press` | 10/10/0 | 通过 |
| `turn_switch` | Target-only | `grasp_manipulation` | 10/10/0 | 通过 |

`target_only_20_v2` 的 11 个 task、220 条 episode 全部满足“单个语义对象/action-site +
单臂 execution”。其中 `move_playingcard_away` 是单对象 relocate，不含第二个语义对象 B；
`open_microwave` 的 20 条全部使用左臂，仍满足单臂合同。Parquet、HDF5、`cam_high` RGB 和
`cam_high` depth 各 220 个，selection、collection 和逐 task manifest 一致。运行时仅
`click_alarmclock`、`click_bell`、`press_stapler` 三个任务采用 `contact_press`；其余 8 个
保持普通 target-only。选择依据和 60-episode 配对证据见
[Contact-press target-only profile](experiments/contact-press-profile.md)。
