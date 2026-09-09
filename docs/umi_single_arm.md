# UMI 单臂物体标注

选择清单及对象定义：[`configs/datasets/umi_single_arm.yaml`](../configs/datasets/umi_single_arm.yaml)。
任务名只用于选择配置，实际开关方向、操作实例由视频决定。

| 任务 | 段数 | target | receiver |
| --- | ---: | --- | --- |
| push_real | 10 | 被驱动的完整抽屉把手；无独立把手时用前面板 | 不适用 |
| pull_real | 10 | 被驱动的完整门把手；无独立把手时用门板 | 不适用 |
| turn_over_real | 10 | 被翻转的完整物体 | 不适用 |
| scoop_real | 10 | 完整舀取工具 | 工具进入的独立容器 |
| stir_real | 10 | 完整搅拌工具 | 被搅拌的完整容器 |
| wipe_real | 9 | 完整抹布或海绵 | 被擦拭的独立物体或有物理边界的独立表面 |

清单按每段多时刻画面审核操作侧，共 59 段；这不是逐帧人工标注。
`wipe_real/01af5625e4b77cedb925ca5ef69cd4a4.mcap` 中另一手可能扶住托盘，按单臂要求保守排除。

## 转换与运行

将 `UMI_SOURCE_ROOT` 设为原始 UMI 目录，以下命令以 `push_real` 为例；其他任务替换任务名：

```bash
just convert-umi "$UMI_SOURCE_ROOT/push_real" artifacts/datasets/umi_single_arm/push_real \
  --task-config configs/datasets/umi_single_arm.yaml
just process artifacts/datasets/umi_single_arm/push_real \
  --run-id umi-push-origin-v1 --sam-worker-gpus 6 --qwen-max-in-flight 1 \
  --object-source-only --ui plain
```

转换器拒绝覆盖已有输出。各任务共享 Origin 的语义规划、候选检查、框定位兜底及 SAM 原生跟踪；
前三组使用 `video_object`，后三组使用 `tool_use`，均由生成的 manifest 自动绑定。
不需要 joint state、机器人模型或夹爪开合事件；窗口来自有效视频首尾帧，不伪造抓取时间。
四通道 `masks.npz` 合同保持不变，左右夹爪均为 `not_annotated`，无 receiver 的任务为
`not_applicable`。视频模式不生成“已持物”的特殊像素编码。

三个源视频共 23 个 H.264 包没有解码出对应帧（翻转 7、搅拌 8 + 8）。输出保留 15,810 个
有效 RGB 帧；`source_frame_index` 和 metadata 中的源帧索引保留精确映射，不直接截掉头尾来凑长度。
视频按采样率重新编码，源时间戳保留在 Parquet；跨缺帧处的物理时间间隔不等于一个普通帧间隔。

## 质量与产物

候选检查使用操作中段证据及先前识别的实例线索；小目标附局部放大图，以避免混淆相邻把手。
UMI 的低相邻 IoU 与中心位移可能是同一次相机／物体运动，因此两个运动信号同时触发时保留
`review`；若面积变化也超限则仍隔离。原有机器人数据配置不变。`review` 不代表逐帧验收通过。

每个 run 保留：

- `<task>/episode_<id>/cam_high/masks.npz`：四通道输出及标注状态。
- 同目录下的 `mask_qc.json`、`run_manifest.json` 和角色目录 `temporal_qc.json`：检查证据与失败原因。
- `rendered_videos/episode_<id>_cam_high_overlay.mp4`：完成段的叠加视频；`review_sheets/` 为抽帧检查图。
- `process_summary.json`：任务完成／失败记录。进程结束不等于全部通过；失败或隔离段不可当成有效空 mask 使用。

试跑目录名含 `smoke`，不作为最终训练输入。批次结果和复核结论另行记录，不将生成数据提交到 Git。

## 2026-09-09 批次结果

59 段全部结束，41 段产出所需对象通道及叠加视频；其中 12 段自动质检通过、29 段带时序警告，
另外 18 段未通过。自动质检通过也不等于逐帧人工验收。

| 任务 | 自动质检通过 | 需时序复核 | 未通过 | 所需对象输出／总数 |
| --- | ---: | ---: | ---: | ---: |
| push_real | 1 | 8 | 1 | 9/10 |
| pull_real | 3 | 7 | 0 | 10/10 |
| turn_over_real | 5 | 3 | 2 | 8/10 |
| scoop_real | 1 | 3 | 6 | 4/10 |
| stir_real | 0 | 2 | 8 | 2/10 |
| wipe_real | 2 | 6 | 1 | 8/9 |

本地[结果索引](../artifacts/umi_single_arm_review/index.html)包含每段源文件、操作侧、
最终 run、mask、原视频、叠加视频和失败原因；同目录下有 `results.json`、`results.csv`。
全部 59 段的[整体 review sheet](../artifacts/umi_single_arm_review/review_sheets/overall_index.jpg)
按任务排列，中期取 50% 帧；同目录的 `early.jpg`、`late.jpg` 分别取 25% 和末帧。
失败段展示带红色状态的原始 RGB，并保留失败标记。可复现生成：

```bash
.venv/bin/python scripts/render_batch_review_sheets.py \
  artifacts/umi_single_arm_review/results.json \
  --datasets-root artifacts/datasets/umi_single_arm
```

按此索引选取结果，不要直接合并所有 run 目录：

- 默认使用 `umi-<task 去掉 _real 并将下划线替换为连字符>-origin-v1`。
- `pull_real` 的 episode 0、1、6 改用 `umi-pull-origin-v2`，修正跨视角把手身份等问题。
- `turn_over_real` 的 episode 0、3、5、6 改用 `umi-turn-over-origin-v2`；其中 5、6 仍未通过。
- `smoke`、`initial` 目录及上述被替代的 v1 段只保留诊断用途。

已抽查全部 41 段产物的早期和末期叠加图，并对部分难例追加动作中段检查；不是逐帧人工质检。
例如 [turn_over_real episode 4](../artifacts/runs/umi-turn-over-origin-v1/rendered_videos/episode_000004_cam_high_overlay.mp4)
可查看完整物体翻转的叠加视频。搅拌细工具与小把手仍容易定位或跟踪失败；stir episode 0 末帧
工具 mask 的离散小块、wipe episode 7 手臂遮挡处的 receiver 边界已在索引中列为复核备注。
原始跟踪保留供复核：隔离可能来自实际漂移，也可能来自真实大幅运动，不能仅凭隔离状态认定
所有像素都错，更不能把失败角色的空输出当作训练标注。

验证结果：

- 59 段数据 preflight 通过；所有已有 mask 的维度、实例顺序、空夹爪通道和 0/1 编码通过检查。
- 41 段叠加视频逐帧解码通过，共 10,083 帧，长度与 mask 一致；索引的 190 个链接均存在。
- target-only 的 receiver 为空；失败段保留状态，不计入上述完整对象输出。
- `just test-all`：838 passed，1 skipped；`just lint` 和 `git diff --check` 通过。
- 修改源码的严格类型检查通过；全量 mypy 仍有既存的
  `pipeline/gripper/sam/annotator.py:251` OpenCV `fillConvexPoly` 颜色参数类型错误，未改动该路径。
