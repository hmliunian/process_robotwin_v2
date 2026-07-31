# 视频 Mask 跟踪实验

## 本次改动（简要）

- Stage 3 改为单帧 SAM3 seed，加 native video tracker 双向传播，不再逐帧重新分割；
- 增加时序 QC，并将多项严重异常或人工确认身份错误的结果标为 `quarantined`；
- renderer 支持用 `--run-id` 固定输入，并使用彩色外轮廓和黑色衬边增强可见性；
- 增加 coverage20 benchmark、review sheet 和 native-track run 物化脚本。

## 结论

coverage20 的主跟踪器选择 **SAM3 native video propagation**。SAM3 只在 seed frame 接收
一次 text 生成的 mask prompt，后续帧由 video tracker 双向传播；最终像素只按角色时间窗裁剪。
CoTracker3 不作为 mask 生成器，也不参与逐帧重分割。

原实现的闪烁主要来自最终组合：

```text
native_track & same_frame_text & fixed_envelope
```

native track 本身连续，而逐帧 text mask 会间歇为空，固定 envelope 也会错误裁掉遮挡或轻微位移
后的像素。因此新实现删除这两个像素交集条件，并保留 envelope 仅作 seed 诊断。

## 方法对比

### 当前组合与 SAM3 native

下表只统计旧 run 中具有可用 seed/native track 的角色：target 19 条，receiver 17 条。

| Role | 方法 | 非空帧/窗口帧 | 存在性切换 | 内部断帧 | 平均相邻 IoU |
|---|---|---:|---:|---:|---:|
| target | 旧三重交集 | 841/1269 | 117 | 201 | 0.929585 |
| target | SAM3 native | 1269/1269 | 0 | 0 | 0.968481 |
| receiver | 旧三重交集 | 1063/1240 | 55 | 63 | 0.980061 |
| receiver | SAM3 native | 1240/1240 | 0 | 0 | 0.971647 |

相邻 IoU 不是越高越好：稳定跟错一个静态物体也可能接近 1。它只用于衡量已选实例的时间连续
性，不能代替身份验证。

### CoTracker3 probe

使用 64 个 seed-mask 内点和本地 `scaled_offline.pth` 做了代表样本 probe：

| Episode | 中位可见点比例 | 最大质心位移 |
|---|---:|---:|
| 7152（正确） | 0.9766 | 2.649 px |
| 7274（稳定跟错） | 0.9062 | 3.040 px |
| 7464（坏 seed/坏 track） | 0.0156 | 41.483 px |
| 7621（跟错实例） | 1.0000 | 12.838 px |

CoTracker 能发现 7464 这种明显崩溃，却无法区分 7274 这种稳定的错误静态实例；点轨迹还需要
额外模型才能转回像素 mask。因此没有把它加入主 pipeline。Cutie/XMem 在当前环境中未安装，
现有 SAM3 native 已达到全时窗连续性，没有证据支持先增加新的模型和 checkpoint 依赖。

## Temporal QC

每个角色保存 `temporal_qc.json`，包含：

- 时间窗覆盖率、存在性切换、内部断帧；
- 相邻 IoU mean/p05；
- 质心跳变 p95、面积比例跳变 p95；
- 相对 seed mask 的最大质心距离。

默认阈值为相邻 IoU p05 `< 0.5`、质心跳变 p95 `> 5 px`、面积比例跳变 p95 `> 0.4`。
三类严重信号至少两类同时出现才 quarantine；单一信号或内部空窗只进入 review，避免把真实
遮挡直接判失败。7464 target 同时触发三类信号，会自动隔离。

## 身份错误与传播错误必须分开

传播器会稳定传播错误 seed。人工对比 close-done 抓取对象后，以下 target 被确认选错，并记录
在 `configs/datasets/move_pillbottle_pad_coverage20_identity_review.json`：

| Episode | 结论 |
|---|---|
| 7274 | 跟踪左上黄色瓶，实际抓取中间棕色瓶 |
| 7317 | 跟踪右上瓶，实际抓取中间橙色瓶 |
| 7464 | seed 包含机械臂/夹爪，传播后落到夹爪部件 |
| 7571 | 跟踪细长绿色瓶，实际抓取相邻青白瓶 |
| 7621 | 跟踪左侧绿色瓶，实际抓取右侧橙标瓶 |

release candidate 将这些角色标为 `quarantined`，不会为了追求覆盖率发布已知错误像素。后续应
在 seed candidate/identity QC 阶段修复它们，而不是更换 tracker。

## Release candidate 产物

版本化 mask run：

```text
artifacts/runs/coverage20-sam3-native-v1/
```

20 段全长视频：

```text
artifacts/rendered_videos/coverage20_sam3_native_v1/
```

指标与 review sheets：

```text
artifacts/temporal_tracking_experiment/benchmark_v1.json
artifacts/temporal_tracking_experiment/benchmark_release_v1.json
artifacts/temporal_tracking_experiment/release_candidate_review_v1/
```

release 状态：

- target：14 valid、5 quarantined、1 failed（原始 text seed 为空）；
- receiver：17 valid、3 failed（原始 text seed 为空）；
- 31 条 valid 角色轨迹全部达到窗口覆盖率 1.0、存在性切换 0、内部断帧 0，temporal QC
  全部为 pass。

## 复现命令

对比旧发布 mask 与已保存 native track：

```bash
.venv/bin/python scripts/benchmark_temporal_tracking.py \
  --selection-manifest artifacts/rendered_videos/coverage20_best_current/manifest.json \
  --output artifacts/temporal_tracking_experiment/benchmark_v1.json
```

不重跑 GPU，从已保存 native tracks 物化一个新 run（默认拒绝覆盖已有 run）：

```bash
.venv/bin/python scripts/materialize_native_tracking_run.py \
  --selection-manifest artifacts/rendered_videos/coverage20_best_current/manifest.json \
  --output-run-id coverage20-sam3-native-v1 \
  --identity-review configs/datasets/move_pillbottle_pad_coverage20_identity_review.json
```

固定精确 run 渲染，避免 best-run 选择器退回旧结果：

```bash
.venv/bin/python scripts/render_coverage20_videos.py \
  --run-id coverage20-sam3-native-v1 \
  --output-dir artifacts/rendered_videos/coverage20_sam3_native_v1
```

生成 early/late review sheets：

```bash
.venv/bin/python scripts/build_tracking_review_sheets.py \
  --render-manifest artifacts/rendered_videos/coverage20_sam3_native_v1/manifest.json \
  --output-dir artifacts/temporal_tracking_experiment/release_candidate_review_v1
```

mask、视频和 benchmark JSON 都是 gitignored 运行产物；Git 只保存实现、配置、测试和本实验
说明。
