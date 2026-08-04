# Qwen gripper mask：固定 3D bbox 实验

## 结论

本轮实验将夹爪 ROI 定义为“腕部 + 完整夹爪”，不再根据夹爪开合量动态改变
3D bbox 宽度。最终候选 F12 在 coverage20 上完成 20/20，0 batch failure；相比旧
ROI，夹爪基座、导轨和腕部连接明显恢复，重点静态检查未见 mask 沿腕部后方继续
扩展成长机械臂。

推荐固定参数：

```text
tcp_offset_m        = 0.120
axial_back_m        = 0.120
axial_front_m       = 0.060
fixed_half_width_m  = 0.085
half_thickness_m    = 0.050
margin_px           = 3
```

prompt ROI 和最终 hard ROI 当前使用相同几何。横向 `closed_half_width_m` 与
`open_half_width_m` 都固定为 `0.085 m`。

## 分支与实验环境

- 分支：`experiment/qwen-mask-gripper-roi`
- 基线提交：`eada22f`
- worktree：`.worktrees/process_data_v2-qwen-mask-gripper-roi`
- coverage20 使用 GPU6/GPU7 两个 10-episode shard 并行运行。

实验没有修改主工作区已有的未提交文件。

## 为什么使用固定 bbox

旧几何会根据 gripper opening 在 closed/open lateral width 之间插值。夹爪闭合后，
bbox 随之变窄，容易把掌部附近的导轨或连接结构裁掉。即使 SAM3 native track
短暂恢复这些像素，最终 hard ROI 相交仍会再次删除它们。

固定横向 half-width `0.085 m` 后，开合只改变真实夹爪姿态，不再改变允许 mask
存在的横向空间上界，因此导轨不会因为抓取动作而被几何裁剪。

## 为什么 axial back 使用 0.12 m

EEF 到 TCP 的固定 offset 是 `0.12 m`。令 `axial_back_m=0.12`，bbox 近端正好从
TCP 回到 EEF 平面：

```text
EEF-relative axial range
= [tcp_offset - axial_back, tcp_offset + axial_front]
= [0.12 - 0.12, 0.12 + 0.06]
= [0.00 m, 0.18 m]
```

这允许腕部、掌部、导轨和两指形成连通区域，同时不主动把 bbox 延伸到 EEF 后方的
长机械臂。`axial_front_m` 保持 `0.06 m`，避免为了补腕部而向物体一侧继续放宽。

## 小样本参数对照

| Variant | Back/front | Half-width | Seed clean mean | Final area mean | Adjacent IoU |
| --- | --- | ---: | ---: | ---: | ---: |
| A / baseline | 0.025/0.060 m | dynamic | 1414.2（20 条） | 1349.5 | 0.8184 |
| F10 | 0.100/0.060 m | 0.085 m fixed | 3545.0（5 条） | 3516.2 | 0.8712 |
| F12 | 0.120/0.060 m | 0.085 m fixed | 4614.8（5 条） | 3683.9 | 0.8727 |

F10 已能恢复大部分掌部；F12 向近端补到 EEF 平面后，腕部与完整夹爪的连接更稳定。
F12 小样本的 5 条 episode 在 full20 重跑中，candidate、seed frame、clean pixels 和
最终 track 均完全一致。

## F12 coverage20 结果

| Metric | Result |
| --- | ---: |
| Shards | 2/2 completed |
| Episodes | 20/20 |
| Batch failures | 0 |
| Seed QC passed | 20/20 |
| Final nonempty frames | 2570/2759 |
| Aggregate coverage | 0.9315 |
| Seed clean pixels mean / median | 3787.4 / 4222.0 |
| Final area mean | 3569.6 px |
| Adjacent IoU, analysis audit | 0.8662 |
| Native clipped by hard ROI | 1,300,268 pixel-frames（11.64%） |
| Final pixels outside hard ROI | 0 |
| Final target/receiver overlap | 0 |

与 20 条旧基线 A 相比：

- coverage：`0.9246 -> 0.9315`
- adjacent IoU：`0.8184 -> 0.8662`
- final area mean：`1349.5 -> 3569.6 px`
- gained/lost/net：`6,319,316 / 193,960 / +6,125,356 pixel-frames`
- `82.56%` 的新增像素位于新开放的 hard-ROI band 内。

## Seed QC 边缘情况

20 条都通过 seed QC，但 prompt/选择路径并非完全相同：

- 17 条 `text_box`，3 条 `box_only`。
- 17 条由 Qwen 选择，3 条使用 deterministic forced fallback。
- 需要重点复核的并集为 `7163, 7179, 7464, 7571`。
  - `7163`：text-box + forced fallback。
  - `7179, 7464`：box-only + forced fallback。
  - `7571`：box-only，但由 Qwen 正常选择。

这些不是 batch failure；对应图中以橙色 `QC FLAG` 标记，避免把 bbox 改善与 seed
策略的边缘行为混在一起。

## 可视化入口

- [F12 full20 A/F12 对比索引](../artifacts/gripper_roi_ablation/visualizations/F12_full20/index.md)
- [F12 full20 量化报告](../artifacts/gripper_roi_ablation/analysis/F12_full20.md)
- [固定 bbox 几何投影 sweep](../artifacts/gripper_roi_ablation/fixed_bbox_geometry_w075_b080_b100_b120/fixed_gripper_bbox_sweep.jpg)
- [A/S/F10/F12 对比索引](../artifacts/gripper_roi_ablation/visualizations/fixed_bbox_AS_F10_F12/index.md)

full20 索引包含 9 张原生分辨率 contact-sheet 对比：

- 重点样例：`7152, 7188, 7274, 7317, 7674`
- QC 边缘样例：`7163, 7179, 7464, 7571`

每条都链接 A/F12 review MP4、contact sheet、selected seed panel 和 manifest。
静态检查中，5 条重点样例都明显补回夹爪基座和腕部；未见明显沿腕部后方延伸成
长机械臂。`7163` 和 `7464` 的后段 mask 最宽、较块状，是视频复核优先级最高的
两个边界案例；当前仍主要落在“腕部 + 完整夹爪”的目标范围内。

## 代码改动

`scripts/generate_gripper_mask_video_qwen_qc.py` 新增：

- 独立 prompt/hard axial back/front 参数。
- `--roi-fixed-half-width-m` 固定横向 half-width。
- manifest/NPZ 中显式记录 prompt ROI、hard ROI 和 geometry。
- resume 时校验 ROI policy，避免不同几何混写同一个 batch。

`scripts/generate_gripper_mask_video_preview.py` 的 ROI track builder 支持显式 geometry。
另外新增量化分析、固定 bbox sweep 和 A/B 对比渲染脚本及单元测试。

## 证据边界

量化结果可以证明最终 mask 全部位于 hard ROI 内、hard ROI 确实裁掉了一部分 native
预测，并且最终与 target/receiver track 零重叠。面积、连续性和新增 band 像素本身
不是 robot-part 标注，因此不能仅凭指标严格证明 ROI 内绝无前臂像素。

当前“没有明显长机械臂泄漏”的判断来自重点和边缘 episode 的可视化检查。若后续需要
机器可验证的严格 wrist cut，应增加 CAD/URDF wrist plane 或像素级 robot-part GT。
