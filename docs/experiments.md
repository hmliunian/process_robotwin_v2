# 实验结论与证据

> 本文只保留影响当前实现的结论、关键数字、失败模式和产物定位。架构行为见
> [architecture.md](architecture.md)。历史 branch/worktree、逐次运行日志和已删除脚本可从
> Git history 查找。

## 1. 证据范围

主要实验集：

```text
/DATA/disk8/xuran/add_mask_robotwin/dataset/
  move_pillbottle_pad_coverage20_original
```

20 个 episode：

```text
7152, 7156, 7157, 7163, 7168, 7179, 7181, 7185, 7187, 7188,
7274, 7317, 7335, 7367, 7424, 7464, 7571, 7621, 7673, 7674
```

前 10 条是 clean，后 10 条是 randomized；active arm 分布 10 left / 10 right。除特别说明，
相机为 `cam_high`，mask 没有像素级 simulator GT。因此“20/20 完成”“QC passed”“连续性
提高”只说明运行合同和视觉审查通过，不能解释为精确率/召回率真值。

### 1.1 可读 Pipeline 重构后的 Pick & Place 验收

2026-08-13 在重构提交 `a819bd8` 上执行完整一键流程：

```text
run id: pick-place-coverage20-readable-v1
dataset: move_pillbottle_pad_coverage20_original
annotation mode: pick_place
required roles: target, receiver
gripper backend: urdf
```

结果：

| 项目 | 结果 |
| --- | ---: |
| Qwen → Object SAM/QC → URDF → canonical publication | 20/20 completed |
| canonical validation | 20/20 passed |
| exact-run overlay video | 20/20 generated |
| excluded / fatal error | 0 / none |
| pipeline summary | `passed=true` |
| 总耗时 | 1:40:21 |

产物保存在：

```text
artifacts/runs/pick-place-coverage20-readable-v1/process_summary.json
artifacts/runs/pick-place-coverage20-readable-v1/rendered_videos/
artifacts/runs/_sources/pick-place-coverage20-readable-v1-object-source/
```

该 run 验证了重构后的默认 Pick & Place 路径仍执行完整业务链路，而不是只验证轻量 smoke：
target 和 receiver 均经过 Qwen、SAM3 candidate identity QC 与 native propagation；gripper
由 URDF/depth 生成；随后统一发布和验证 canonical 四通道 mask。20 个视频在 2026-08-14
完成人工检查并确认 Pick & Place 结果可接受。

### 1.2 Target-only close-and-hold 验收

2026-08-14 在分支 `feat/target-only-close-hold`、提交 `b36cfa3` 上完成 Target-only 全链路
验收：

```text
run id: target-only-full20-qwen18087-b36cfa3
source run id: target-only-full20-qwen18087-b36cfa3-object-source
dataset: /DATA/disk8/xuran/add_mask_robotwin/dataset/target_only_20/adjust_bottle
annotation mode: target_only
required roles: target
gripper backend: urdf
```

20 个 episode：

```text
0, 6, 10, 18, 21, 30, 34, 42, 48, 49,
50, 51, 171, 176, 287, 313, 415, 431, 547, 549
```

该 run 执行完整业务链路，而不是跳过模型的轻量路径：

```text
Qwen semantic plan
  -> SAM candidates + basic/identity QC + native propagation
  -> URDF/depth gripper
  -> canonical publication + validation
  -> exact-run overlay video
```

Target-only 使用 `remove_start < close_start < close_end` 三事件 close-and-hold 时间线；target
窗口为 `[remove_start, close_end]`，receiver 为全零且 `not_applicable`，活动 gripper 窗口为
`[remove_start, T-1]`。左右活动臂分布为 10 left / 10 right。

独立逐帧验收结果：

| 合同 | 结果 |
| --- | ---: |
| completed record / canonical publication | 20/20 |
| three-event timeline 与窗口 | 20/20 |
| canonical 七键四通道 `masks.npz` | 20/20 |
| target 精确覆盖闭区间 | 20/20 |
| target 在 `close_end` 后严格归零 | 20/20 |
| receiver 全零且所有合同层为 N/A | 20/20 |
| active/inactive gripper window clipping | 20/20 |
| 活动 gripper 末帧非空 | 20/20 |
| per-episode run manifest | 20/20 |
| frame provenance | 20/20 |
| URDF 0.90 quality gate | 20/20 |
| MP4 文件大小与 SHA-256 | 20/20 |

最低 active-window coverage 为 ep42 的 `125/138 = 0.905797`；最低 eligible coverage 也在
ep42，为 `125/126 = 0.992063`，均通过固定 `0.90` gate。summary records、backend selected
IDs、video manifest 与实际 20 个 MP4 的 ID 集完全一致，excluded 为 0，`passed=true`。

产物保存在：

```text
artifacts/target_only_validation/target-only-full20-qwen18087-b36cfa3/process_summary.json
artifacts/target_only_validation/target-only-full20-qwen18087-b36cfa3/rendered_videos/
artifacts/target_only_validation/_sources/target-only-full20-qwen18087-b36cfa3-object-source/
```

首次 streaming run 在 source 20/20 完成、URDF 16/20 后被中断；恢复过程复用 immutable
source，以 incremental resume 只补齐缺失 episode，再执行统一 canonical publication、validation
和视频生成。最终 exact-run 的 20 条逐帧合同与 MP4 哈希全部通过，因此中断与恢复不改变最终
结果或 source lineage。

该数据集同样没有像素级 simulator GT。上述 20/20、coverage 和哈希证明流程合同、时间窗口、
产物一致性与交付完整性，不应表述为 target 或 gripper 像素精确率/召回率真值。

## 2. target/receiver：语义、seed 和跟踪

### 2.1 query 设计结论

早期实验试图把物体扩写成长 referring expression。31 例中，长描述和当时的 expanded query
都只有 19/31 非空，且会让部分原本可分割实例退化为空。短候选 bank 能补覆盖，但过宽的
`food`、无颜色 `block` 等候选容易命中错误实例。

最终决策：

- Qwen 输出有序的 1–4 词短 query bank，而不是一条长描述；
- category 是必需的 head noun，颜色/形状仅在有稳定证据时加入；
- Qwen 事前排序，不按 SAM 非空或面积自动选；
- SAM 对前几个 query 生成真实 mask，再让独立 Qwen mask-QC 看轮廓选择实例；
- receiver 依据最终直接接触关系定义，不要求是承托物；
- 蓝色平面 proposal 只作为 receiver 候选，不能按最大面积自动获胜；
- 无合法候选、身份不明或 QC 服务失败时 fail closed。

### 2.2 candidate QC coverage20

正式对象 run：

```text
run id: coverage20-qc-contact-v5-native
artifacts/runs/coverage20-qc-contact-v5-native/
artifacts/rendered_videos/coverage20_qc_contact_v5_native/
```

结果：

| 项目 | 结果 |
| --- | ---: |
| `masks.npz` / `mask_qc.json` / manifest | 20/20 |
| target QC passed / status ok | 20/20 |
| receiver QC passed / status ok | 20/20 |
| exact-run overlay | 20/20 |

最后补跑的 `7187, 7317, 7464, 7571, 7674` 由常驻 SAM3 worker 完成，batch summary 无
fatal error。全量 review sheets 确认 target 跟随实际 bottle、receiver 跟随 blue pad；被
gripper 或已放置 bottle 遮住的缺失像素符合 visible-only 定义。

该轮同时验证：

- Qwen omission repair 可以补齐 query order，而不生成新 query；
- `teal white bottle` 等两色判别短语可保留；
- saturated-blue planar proposal 能补 text-only SAM 的 pad 空结果；
- transient Qwen QC 最多重试两次，不重新生成候选；
- 一个 batch 内只初始化一次 SAM3 backend，episode session 仍隔离。

### 2.3 为什么选择 SAM3 native propagation

旧最终组合是：

```text
native_track & same_frame_text & fixed_envelope
```

逐帧 text mask 间歇为空，固定 envelope 又裁掉位移或遮挡后的像素，导致闪烁。删除这两个
交集条件后，只保留 native track 并按 role window 裁剪：

| Role | 方法 | 非空帧/窗口帧 | 存在性切换 | 内部断帧 | 平均相邻 IoU |
| --- | --- | ---: | ---: | ---: | ---: |
| target | 旧三重交集 | 841/1269 | 117 | 201 | 0.929585 |
| target | SAM3 native | 1269/1269 | 0 | 0 | 0.968481 |
| receiver | 旧三重交集 | 1063/1240 | 55 | 63 | 0.980061 |
| receiver | SAM3 native | 1240/1240 | 0 | 0 | 0.971647 |

相邻 IoU 不是身份指标：一个静态错误实例也能接近 1。native tracking 解决传播连续性，不能
解决 seed 身份。

### 2.4 CoTracker probe

用 64 个 seed-mask 内点和本地 `scaled_offline.pth` 做代表 probe：

| Episode | 中位可见点比例 | 最大质心位移 | 人工结论 |
| ---: | ---: | ---: | --- |
| 7152 | 0.9766 | 2.649 px | 正确 |
| 7274 | 0.9062 | 3.040 px | 稳定跟错 |
| 7464 | 0.0156 | 41.483 px | 坏 seed/坏 track |
| 7621 | 1.0000 | 12.838 px | 跟错实例 |

CoTracker 能识别 7464 这种明显崩溃，却无法区分 7274 的稳定错误，还需额外模型把点轨迹
转回像素 mask。Cutie/XMem 当时未安装；已有 SAM3 native 达到合法轨迹的全时窗连续性，
没有证据支持增加新 checkpoint 依赖。因此 CoTracker 不进入主 pipeline。

### 2.5 身份错误与传播错误分离

早期 native release candidate 人工确认以下 target seed 错误：

| Episode | 错误 |
| ---: | --- |
| 7274 | 左上黄色瓶；实际抓取中间棕色瓶 |
| 7317 | 右上瓶；实际抓取中间橙色瓶 |
| 7464 | seed 含 robot/gripper，传播后落到 gripper part |
| 7571 | 细长绿色瓶；实际抓取相邻青白瓶 |
| 7621 | 左侧绿色瓶；实际抓取右侧橙标瓶 |

早期 release 状态是 target 14 valid / 5 quarantined / 1 failed，receiver 17 valid / 3
failed；31 条 valid 轨迹的窗口 coverage 均为 1、无切换和断帧。这个结果证明 tracker
连续，但同时证明 temporal QC 不能发现稳定错实例。后续 `coverage20-qc-contact-v5-native`
通过实际候选轮廓的 Qwen identity QC 将对象结果补到 20/20。

保留的历史产物：

```text
artifacts/runs/coverage20-sam3-native-v1/
artifacts/temporal_tracking_experiment/benchmark_v1.json
artifacts/temporal_tracking_experiment/benchmark_release_v1.json
```

## 3. SAM gripper：pose ROI 到固定 front45

### 3.1 pose ROI 可行性

RoboTwin state 提供 EEF pose 和 gripper opening，TCP 为 EEF local `+x` 的 `0.12 m`。第一版
3-D ROI 参数：

```text
axial_back/front = 0.025/0.060 m
closed/open half-width = 0.045/0.085 m
half-thickness = 0.050 m
margin = 3 px
```

coverage20 geometry audit：

| 指标 | 结果 |
| --- | ---: |
| 完成 episode | 20/20 |
| ROI 与图像相交 | 20/20 |
| first TCP-in-frame | frame 14..36 |
| episode median ROI area | 4190..6444 px |

ROI 能跟随正确 active gripper 并排除长 wrist/forearm，但抓取阶段也包含 target，因此只是
空间上界，不能直接作为像素 mask。

### 3.2 box-only 为什么不够

ep7152 和 ep7317 各 7 个关键帧：

- pose box-only：14/14 非空，却经常选择 bottle；
- `black robot gripper` + pose box：14/14 非空，gripper recall 更好，但部分帧仍混入 target；
- 精确 `candidate & ~target & ~receiver` 能去污染，但前提是 object track 身份正确；
- 若 box-only candidate 根本没包含 gripper，subtraction 不可能补回缺失像素。

ep7317 的第一次对象排除只删除 365/22757 candidate pixels，原因不是 subtraction 失效，
而是 generic `bottle` seed 选中了 distractor。用已知 tight target box 做控制实验后：

| Frame | Candidate | Target removed | Residual | Residual dark |
| ---: | ---: | ---: | ---: | ---: |
| 63 | 1887 | 888 | 999 | 97.1% |
| 92 | 1584 | 432 | 1152 | 97.9% |
| 120 | 2537 | 2122 | 413 | 90.3% |
| 121 | 2526 | 2018 | 508 | 92.3% |

这支持：

```text
pose ROI
  + 有真实 gripper recall 的 text+box candidate
  + identity-correct visible object tracks
  + exact object exclusion
= compact visible gripper candidate
```

### 3.3 native full-trajectory pilot

reviewed seed → native propagation → per-frame pose crop → object exclusion 的两条 pilot：

| 指标 | ep7152 | ep7317 |
| --- | ---: | ---: |
| active arm | right | left |
| usable frames | 138 | 140 |
| active window | 4..132 | 4..134 |
| nonempty active coverage | 117/129 (90.7%) | 131/131 (100%) |
| median nonempty area | 1663 px | 1599 px |
| adjacent IoU mean | 0.824 | 0.844 |
| native throughput | 11.99 fps | 12.15 fps |

native track 会延伸到 wrist/forearm，逐帧 pose intersection 能去除长 arm segment。该结果
仍是 `review_required`：没有 robot-part GT，ep7317 的 target 来自控制用 manual box，且两条
轨迹中 receiver subtraction 基本未被触发。

### 3.4 text+box seed policy

早期 mixed-prompt batch 让 Qwen 同时在 box-only/text+box 和 keyframe 之间选择：20 条中选
13 个 box-only、7 个 text+box，平均 active coverage 92.22%，mean adjacent IoU 0.805。

改为优先只生成 text+box，通过机械 gate 后让 Qwen 只选 keyframe；只有全部 text+box gate
失败才生成 box-only fallback。`videos_native_qwen_qc_text_box_v1` 的 20 条全部完成：

| 指标 | mixed baseline | text+box |
| --- | ---: | ---: |
| selected box/text+box | 13/7 | 0/20 |
| mean active coverage | 92.22% | 92.56% |
| mean adjacent IoU | 0.805 | 0.823 |
| minimum adjacent IoU | 0.704 | 0.786 |
| mean propagation throughput | 13.85 fps | 14.13 fps |

episode 7185 从 box-only frame 53 换为 text+box frame 94 后，IoU mean `0.783 -> 0.845`，
p05 `0.389 -> 0.574`，median area `750 -> 1304 px`。所有 review sheets 未见明显长
forearm 泄漏，但这仍是视觉证据，不是零 arm-pixel 保证。

### 3.5 “只剩指尖”的根因

旧 ROI 相对 EEF 的轴向范围为：

```text
[0.12 - 0.025, 0.12 + 0.060] = [0.095, 0.180] m
```

EEF 后 9.5 cm 的近端掌部/基座天然无法进入 mask。并且：

- 某些 raw SAM seed 在 crop/subtraction 前已经只有两指；
- seed 和每帧 track 都再次与同一窄 ROI 相交，传播无法补回掌部；
- gate 只检查暗色、连通性和 TCP 距离，tip-only mask 很容易通过；
- renderer 的 object overlap 为 0，排除“渲染阶段扣掉掌部”。

这促成三个当前决策：ROI 向 EEF 延伸、prompt/hard ROI 分离、横向宽度固定。长期更强的
几何先验则发展为独立 URDF backend。

### 3.6 固定 bbox 与 front45

当前参数：

```yaml
prompt back/front: 0.120/0.060 m
hard back/front:   0.120/0.045 m
fixed half-width:  0.085 m
```

EEF-relative axial range：prompt `[0.00, 0.18] m`，hard `[0.00, 0.165] m`。hard 只在
approach/指尖侧收紧 1.5 cm，不影响 wrist 侧覆盖。

F12-front60 的 20 条量化（front45 只做 final crop 微调，不能把下表误标为 front45 指标）：

| 指标 | F12-front60 |
| --- | ---: |
| episodes / failures | 20 / 0 |
| seed QC passed | 20/20 |
| final nonempty frames | 2570/2759 |
| aggregate coverage | 0.9315 |
| final area mean | 3569.6 px |
| adjacent IoU | 0.8662 |
| native pixels clipped by hard ROI | 11.64% |
| final pixels outside hard ROI | 0 |
| final target/receiver overlap | 0 |

相对旧动态 ROI：coverage `0.9246 -> 0.9315`，IoU `0.8184 -> 0.8662`，mean area
`1349.5 -> 3569.6 px`。新增像素 82.56% 位于新开放的 hard-ROI band。

front45 full20 两个 10-episode shard 均完成且 0 failure。选择路径为 17 text_box / 3
box_only、17 Qwen / 3 deterministic fallback；重点复核：

```text
7163: text_box + forced fallback
7179: box_only + forced fallback
7464: box_only + forced fallback
7571: box_only + Qwen
```

这些是 QC flags，不是 batch failure。可视化显示腕部连接和 gripper base 明显恢复，未见明显
长 arm leakage；7163/7464 后段最宽、最块状，仍是优先复核案例。面积和连续性无法证明
ROI 内绝无 forearm，严格机器验证需要 CAD/URDF wrist plane 或像素级 robot-part GT。

历史产物：

```text
artifacts/gripper_pose_roi_coverage20/videos_native_qwen_qc_text_box_v1/
artifacts/rendered_videos/coverage20_qwen_front45/
artifacts/gripper_roi_ablation/visualizations/F12_full20/
```

## 4. URDF gripper

### 4.1 核心假设与修正

目标是只替换 gripper SAM producer，保留 target/receiver。plain 2-D URDF silhouette 会包含
被 target、pad、table 或其他 robot link 遮挡的像素，因此必须用记录的 scene depth 做
visibility。

另一个关键发现：normalized gripper value 是 drive target，不是接触后的 realized qpos。
例如 ep7152 frame69 在 closed command 下渲染不匹配，而两指约 `0.0375 m` 时能对齐 depth。
因此每帧要拟合 `joint7/joint8`，不能直接转发 command。

最终算法采用 visual meshes、两臂完整 render、active-side `link6|link7|link8` 和固定
8 mm publication tolerance；无 SAM/Qwen、RGB threshold、object subtraction、morphology 或
temporal fill。

### 4.2 standalone pilot 与 full20

冻结实现：

```text
branch: experiment/urdf-gripper-mask-coverage20
generation commit: 16d8bc87af76ac16167cba80e06dd10e4915d1cc
```

双臂 pilot：ep7152 右臂 `116/117` eligible 非空（99.15%），ep7157 左臂 `149/149`
（100%）；approach/contact/transport/release/post-window、channel preservation、link membership
和 saved-vs-rerender review 均通过。

standalone full20：

```text
artifacts/urdf_gripper_mask_coverage20/
  coverage20-urdf-gripper-v1-16d8bc8/
```

| 指标 | 结果 |
| --- | ---: |
| episodes / failures / failure attempts | 20 / 0 / 0 |
| eligible nonempty | 2561/2570 (99.65%) |
| 最低单 episode | 125/128 (97.66%), ep7571 |
| visible gripper pixels | 6,929,725 |
| link6/link7/link8 acceptance | 92.21% / 85.94% / 85.28% |
| max fitted-q jump | 10.5 mm, ep7274 joint8 |
| first render / full resume validation | ~704.3 s / ~13.9 s |

20 条六张 contact sheets 全部通过；未发现 wrong arm、长 forearm、bottle body inclusion 或
release-opening 错误。最大 10.5 mm jump 来自 bounded search 失败后的合法 full-range
reacquisition，没有触发质量或视觉失败。

### 4.3 canonical coverage20 验收

canonical run：

```text
source:
  artifacts/runs/20260806T120824Z-2fc33b5c
output:
  artifacts/runs/coverage20-urdf-canonical-v1
```

2026-08-11 结果：

| 项目 | 结果 |
| --- | ---: |
| `process_summary.passed` | true |
| selected/completed/failed | 20/20/0 |
| failure attempts | 0 |
| active arms | 10 left / 10 right |
| authoritative mask frames | 2,940 |
| overlay | 20 MP4, 2,960 RGB frames, 320x240, 50 FPS |
| source lineage | 20 个唯一 episode digest |
| repository validation | 210 passed, 1 skipped |

每条视频保留一个无 overlay 的 RGB 尾帧并记录 `unmasked_trailing_frames=1`。public
target/receiver 与 source 逐像素相同，URDF active gripper 使用相同七键 NPZ 和共同
`gripper_qc` schema；几何产品、lineage 和 publisher identity 留在 `_backend/urdf` /
`derivation`，不形成第二套下游接口。

这个 run 是“URDF 可以替换 gripper producer 并保持 downstream contract”的正式验收证据。

### 4.4 `place_empty_cup_full550` 扩展

数据集 550 条；frozen source 中 456 条 target/receiver 满足 QC，94 条被排除（91 条对象
结果不完整，16027/16336/16345 无可发布 source mask）。因此自动 subset 必须显式
`--allow-partial-source`；显式 `--episode-ids` 仍 fail closed。

456 条最终 dry-run 约 115 s，通过 source/dataset/lineage 合同但未启动 5–7 小时正式 render。
左右臂 pilot：

| Episode | arm | eligible nonempty | 结果 |
| ---: | --- | ---: | --- |
| 15950 | right | 111/112 (99.11%) | passed |
| 15955 | left | 144/144 (100%) | passed |

两条 target/receiver 均与 source 逐像素一致，overlay 和六张 review sheets 已生成。该证据
支持扩展可行性，但不能表述为 456 条 formal run 已完成。

## 5. Active-wrist phase-seed（未实施）

### 5.1 提案

根据 active arm 路由到 `cam_left_wrist` 或 `cam_right_wrist`：

| role | seed search window | tracking window |
| --- | --- | --- |
| target | `[close_start, close_done]` | `[0, frame_count-1]` |
| receiver | `[open_start, open_done]` | `[0, frame_count-1]` |

target/receiver 使用独立 seed、独立 SAM session 和独立 full-video track；不生成 gripper。
动态 wrist 视角允许物体离开 FOV，因此 full-video QC 不能要求每帧非空。

### 5.2 coverage20 可见性与 seed smoke

- 20/20 target 在 close window 可见；
- 20/20 receiver 在 open window 可见；
- ep7163 receiver 最难，只剩画面上方一小段；7424/7621 也只有部分区域；
- visible-only seed 必须允许 image-boundary truncation、gripper/target occlusion。

release midpoint 的 receiver text-SAM smoke：

| Episode | seed | `blue square pad` | `blue pad` | `blue square` |
| ---: | ---: | ---: | ---: | ---: |
| 7152 | 125 | 0 | 0 | 0 |
| 7163 | 140 | 0 | 0 | 0 |
| 7424 | 139 | 0 | 0 | 2857 px |
| 7621 | 130 | 0 | 0 | 0 |

结论：可见不等于 text-SAM 可分割，不能使用“release midpoint + 单 query + 空则失败”。
saturated-blue planar proposal 在 20/20 中非空，但可能选中 target 蓝色区域或蓝绿背景，仍需
跨帧 Qwen candidate QC。

用蓝色可见区域做 receiver full-video 双向传播：

| Episode | camera | seed | nonempty / total |
| ---: | --- | ---: | ---: |
| 7152 | right wrist | f125, 5821 px | 72/138 |
| 7163 | left wrist | f140, 1117 px | 63/153 |

tracker 在可见时能保持区域，离开 FOV 时自然为空，不会把窄条补成 amodal receiver。这支持
phase-seed 方向，但还缺 target seed smoke、20 条 Qwen mask-QC、camera policy、role-specific
window 和 segment-aware temporal QC 实现。

### 5.3 如果继续实施

必须独立于当前 `cam_high` 默认行为，至少完成：

- active-arm → wrist-camera routing 与 provenance；
- target/receiver 不共享 seed candidates；
- anchor-window QC + 每个连续 nonempty segment diagnostics；
- out-of-view 空段不算 internal failure，也不能 forward-fill；
- no-gripper profile 仍写四通道，gripper 为全零/not_annotated；
- 先验收 7152、7163、7181、7571，再扩 coverage20。

当前工作树没有实现这些行为，不能用本文直接推断 CLI 已支持 `active_wrist`。

## 6. 决策汇总

| 问题 | 采用 | 拒绝/限制 |
| --- | --- | --- |
| object query | Qwen 短候选 bank + actual-mask QC | 长 referring expression、按面积自动选 |
| object propagation | SAM3 native bidirectional + role window | 逐帧 text mask、fixed envelope 交集 |
| temporal QC | 多信号 quarantine | 用 IoU 判断身份 |
| SAM gripper seed | text + projected prompt ROI | pose box-only 作为主 producer |
| SAM gripper crop | 固定 half-width、back120、front45 hard ROI | 随 opening 收窄、同一 prompt/hard ROI |
| object contamination | identity-correct track 的精确 subtraction | 用 subtraction 修复没有 gripper recall 的候选 |
| geometry gripper | URDF visual mesh + depth visibility + q fitting | plain silhouette、collision mesh、直接使用 drive target |
| wrist view | phase-specific multi-frame candidates，尚待实现 | 单 midpoint/单 query、每帧必须非空 |

## 7. 当前复现入口

优先使用现有 canonical CLI，不使用已删除的历史实验脚本：

```bash
# default Qwen/SAM objects + SAM gripper
just serve-qwen
just process DATASET_ROOT OUTPUT_ROOT --run-id NEW_RUN

# live objects + URDF gripper
just process DATASET_ROOT OUTPUT_ROOT \
  --gripper-backend urdf --run-id NEW_RUN

# frozen source + URDF gripper
just process DATASET_ROOT OUTPUT_ROOT \
  --gripper-backend urdf \
  --source-run-dir SOURCE_RUN \
  --run-id NEW_RUN
```

需要重现某个历史实验的逐帧中间量时，应 checkout 文中对应 commit/branch；当前主线已删除
`generate_gripper_mask_video_*`、ROI sweep 和独立 review-sheet 脚本，其结论已固化到配置、
pipeline stage 和本文件。
