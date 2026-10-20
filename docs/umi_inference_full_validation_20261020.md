# UMI 完整视频传播验证与端到端测试准备

日期：2026-10-20。分支：`feat/umi-inference-fast`。

已从十帧检查扩展为固定 10 条完整视频、2,399 个不同 RGB 帧。13 个有批准 seed 的角色，
共 3,542 个角色帧，优化前后原始传播 mask 逐像素一致，平均及最小 IoU 均为 1。
其中双方为空的 51 帧单独统计，其余 3,491 帧的 IoU 也全部为 1。

本轮传播耗时合计 **250.00 → 155.36 秒，减少 37.86%**。这是一次固定 seed 的 SAM T1
传播对照，不能解释为整条任务或异步流水线的加速比例。真实 Qwen 端到端三组测试尚未运行。

## 完整视频结果

输入为[固定清单](../configs/datasets/inference_timing_umi_10tasks.json)中的原十条视频，
640×480、`cam_high`。输入视频、extract manifest 和 episode metadata 已与原实验的
`input_identity.json` 校验一致。每条均使用完整窗口 `[0, frame_count - 1]`。

耗时列为该视频所有可比较角色的传播调用之和，不包括 Qwen、seed 分割、模型加载和渲染。

| 任务 | episode | RGB 帧 | 比较角色 / 必需角色 | 原生传播 s | T1 传播 s | 最小 IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pick_up | 0 | 166 | 1 / 1 | 13.264 | 7.526 | 1 |
| put_down | 101 | 176 | 1 / 1 | 12.344 | 7.686 | 1 |
| pull | 203 | 170 | 1 / 1 | 11.919 | 7.569 | 1 |
| push | 301 | 212 | 1 / 1 | 14.780 | 9.300 | 1 |
| pour | 500 | 124 | 2 / 2 | 17.531 | 11.021 | 1 |
| wipe | 700 | 519 | 2 / 2 | 72.905 | 45.368 | 1 |
| rotate | 928 | 155 | 1 / 1 | 10.903 | 6.790 | 1 |
| scoop | 1000 | 221 | 1 / 2 | 15.504 | 9.645 | 1 |
| stir | 1101 | 500 | 2 / 2 | 69.877 | 43.534 | 1 |
| turn_over | 1203 | 156 | 1 / 1 | 10.970 | 6.924 | 1 |
| 合计 | — | 2,399 | 13 / 14 | 249.999 | 155.362 | 1 |

`scoop.receiver` 在历史 QC 中被拒绝，没有批准的 seed，本轮明确保留为 skipped，缺少该角色
的 221 帧对比，不能声称 14 个角色都通过。`pull.target` 和 `scoop.target` 虽在历史时序 QC 中隔离，仍有通过候选 QC
的 seed，因此本轮比较其原始 track；原始传播一致不等于生产流程通过。

五个非零 seed 为 pick_up 94、put_down 150、wipe.target 444、scoop.target 157、
stir.target 71，包含 seed 之前的反向传播。最长的 wipe 519 帧和 stir 500 帧都完整覆盖。
51 个双方空帧全部来自 scoop.target；没有单边空变化，变化像素总数为 0。
IoU 衡量与优化前输出的一致性，没有人工逐帧真值。

## 对照设置与计时边界

- baseline 冻结原工作区 adapter，保留用户已有的文本候选单图补丁；传播使用原生路径。
- fast 使用当前集成的 adapter，`tracking_batch_size=1`，仅取跟踪所需视觉特征。
- 两者共享同一 predictor、权重及数值设置，逐角色使用相同历史 seed 和窗口，交替执行顺序。
- GPU 7：NVIDIA H200；PyTorch `2.7.1+cu128`；CPU 8 线程；TF32 开启。
- SAM checkpoint SHA256：
  `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`。
- 每个传播调用起止均 CUDA synchronize，含该调用内部的完整视频 session 加载和关闭。
- 单次运行、没有额外预热，首轮可能含首次 kernel/cache 成本；该耗时用于观察，不是稳态统计。
- 两个变体共用一次模型加载，加载 10.576 秒；完整对照进程 435.12 秒，均不计入传播之和。
- 本轮离线运行，不读取 Qwen 密钥、不调用 API，没有重跑候选分割或 QC 决策。

GPU pmon 在第三个角色附近启动，但采样进程无法连接 NVIDIA driver，`gpu.log` 没有有效
占用记录，因此无法核对测试期间同卡争用，速度仅作为本轮观察。输入、seed、每帧 JPEG、
checkpoint 和执行代码均保存 SHA256。

## 原始产物与复现

ignored 产物目录为 `artifacts/umi_pipeline_full_20261020/tracking_full/`：

- `metadata.json`：输入和源码指纹、硬件、时间及参数。
- `summary.json`：14 个必需角色的完整账本、13 个角色的耗时和逐帧 IoU。
- `<task>/<role>/baseline.npz`、`fast.npz`、`per_frame.csv`：原始 mask 与逐帧对比。
- `analysis/independent_checks.json`：独立重算全部 NPZ / CSV 与指纹的审计结果。
- `gpu.log`：监控失败记录；父目录的 `tracking_full.time` 和 `.log` 保存进程耗时及模型日志。

这些比较 NPZ 为 `[T,H,W]` 原始 track，未替换生产四通道 canonical `masks.npz`。
已有的十帧候选分割和 batch4 检查另见[十帧报告](umi_inference_fast_10frames_20261020.md)。

设置 `UMI_SEED_RUN` 为原十任务实验产物根，`UMI_FRAMES_ROOT` 为完整 JPEG 根后执行：

```bash
env PYTHONPATH=src:. OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python scripts/benchmark_umi_tracking_full.py \
  --manifest configs/datasets/inference_timing_umi_10tasks.json \
  --seed-run "$UMI_SEED_RUN" --frames-root "$UMI_FRAMES_ROOT" \
  --baseline-adapter artifacts/umi_inference_fast_10frames/baseline_sam3_adapter.py \
  --checkpoint checkpoints/sam3/sam3.pt --gpu 7 \
  --output artifacts/umi_tracking_full_repeat
```

## 端到端对比的待完成部分

已准备 `scripts/benchmark_umi_pipeline.py`，三组均使用同一固定清单和渲染要求，分别在同一
GPU 的新进程运行。保留失败、隔离和未执行记录，整批墙钟包含失败成本及渲染 drain。

| mode | SAM | 生命周期 | 调度 |
| --- | --- | --- | --- |
| baseline | 冻结旧 adapter | 每任务加载 / 释放 | 串行 |
| resident | S1 + T1 | 全批加载 / 释放各一次 | 串行 |
| async | S1 + T1 | 全批加载 / 释放各一次 | 有界规划预取、后台渲染 |

baseline 为复用相同 core/renderer 的受控重建，不是逐字节复现旧 CLI。它每任务进行健康
检查，新模式全批一次；应单列健康检查、加载及各阶段耗时。单条服务时间、预取延迟与
整批墙钟分别报告，不能用 `整批耗时 / 10` 代替单条延迟。完整进程计时另用 `/usr/bin/time`。

实际 Qwen 可能改变语义 query、seed、QC 和救援次数。离线脚本
`scripts/summarize_umi_pipeline_benchmark.py` 已准备比较 canonical 与 native 两层逐帧 IoU，
单列缺失、隔离、双方空和决策差异，避免把少完成任务或空 mask 当成同质量加速。

本次联网测试启动被自动审批拒绝，原因是尚缺把这些视频的抽帧和提示发送至配置中具体
阿里云 Qwen 服务的明确授权。三组没有产生真实端到端数据，也没有替换为回放耗时。
待确认外发授权后，用 `--mode baseline/resident/async` 各选新输出目录运行并补齐结果。

## 检查

新增离线质量分析测试 4 项通过，覆盖缺失必需 mask、隔离后的双空、单边空、截断视频与
缺失整条任务。三个新增脚本的 Ruff、格式及 strict mypy 检查通过。生产推理实现本轮未改。
独立审计重算 26 个 NPZ 与 13 份 CSV，全部一致；2,399 张 JPEG、13 个 seed、10 份历史
manifest、5 个源码文件、权重和固定清单的指纹均匹配。
