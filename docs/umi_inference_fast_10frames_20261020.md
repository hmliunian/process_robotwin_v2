# UMI 快速推理集成与 10 帧对照验证

日期：2026-10-20。开发分支：`feat/umi-inference-fast`，从 `feat/umi-h200-batch` 新建。

本次 H200 实测中，优化前后两个角色的 10 帧传播 mask 逐像素一致，平均及最小 IoU 均为
1.000000。SAM 阶段平均耗时由 7.756 秒降至 6.807 秒（S1+T1，减少 12.23%）；启用可选
batch4 后为 6.553 秒（S1+T4，减少 15.51%）。同卡存在其他进程，三轮波动较大，结果只
代表本样本、本次运行；尚不能据此承诺完整数据集或异步流水线的加速比例。

## 集成范围

- S1：文本、bbox、文本+bbox 候选均可按单帧加载，保留多帧候选搜索及原回退路径。
- T1：传播仅提取跟踪所需视觉特征，保留逐帧记忆更新和正反向传播。
- T4：真实视觉编码 batch4，独立开关；默认快路径采用 batch1。
- 常驻模型：每张 GPU 一个 worker，跨任务及分辨率组复用 SAM。
- 异步调度：预取 Qwen 规划、后台渲染，队列有界，SAM 在 worker 主线程顺序运行。
- 结果与恢复：保留失败记录，校验输入和实现身份，恢复时补齐未完成工作。

旧入口默认 `--inference-mode legacy`；`resident` 为常驻串行，`async` 开启阶段重叠。
快路径默认 `--tracking-batch-size 1 --prefetch 2 --render-buffer 2`，batch4 显式选择。
QC 规则、多候选重试及 canonical 四通道产物合同保持原样。本报告中的 NPZ 是微基准保存
的两角色比较数组，不是 canonical `masks.npz`。

## 样本与比较方法

| 项目 | 本次设置 |
| --- | --- |
| 任务 / episode / camera | `pour_640x480` / `000500` / `cam_high` |
| 输入 | 原视频前 10 帧，原始帧号 `0..9`，640×480；不缩放、不跨全视频抽样 |
| 角色 | `target`：glass cup；`receiver`：metal pot |
| 候选分割 | 两角色 × 候选帧 `0/5/9` ×（4 个文本 + bbox + 文本+bbox），共 36 个 mask |
| 传播 | 两个历史 QC 通过的 frame 0 seed，固定窗口 `[0,9]`，共 20 个角色帧 |
| 重复 | 每个变体各 1 次预热、3 次正式运行；每轮轮换变体执行顺序 |
| 硬件 / 软件 | NVIDIA H200，GPU 1，PyTorch `2.7.1+cu128`，CPU 8 线程 |

语义查询和顺序来自历史 `semantic_plan.json`。bbox 由历史 seed 的外接范围构造，仅用于
控制输入的候选一致性测试，不经过本次 Qwen 定位。每轮传播都使用相同的历史 seed，不
依赖本轮候选的选择。这样分别验证候选分割与传播能否保持原输出。

基线冻结的是任务开始时原工作区的 `sam3_adapter.py`，已包含用户此前未提交的**文本
候选单图补丁**。因此此处 S1 新增的是 bbox、文本+bbox 的单图加载，不能把原实验中完整
S1 的收益再次计入。三个变体共用同一 predictor、checkpoint 和精度设置，顺序执行。

| 记录名 | 种子分割 | 视频传播 |
| --- | --- | --- |
| `baseline` | 冻结基线，已有文本单图补丁 | 原生路径 |
| `fast_t1` | 完整 S1 | T1，视觉编码 batch1 |
| `fast_t4` | 完整 S1 | T1 + 视觉编码 batch4 |

计时包围实际 GPU 调用，起止均做 CUDA synchronize。`seed_s` 为全部 36 个候选的耗时，
`propagation_s` 为两个角色各 10 帧的传播耗时，`sam_total_s` 为二者之和。Qwen、模型加载、
预热、保存和渲染均不计入比较；该候选压力测试也未模拟生产流程通过 QC 后提前停止。

## IoU 与候选一致性

IoU 按每角色每帧的 `|baseline ∩ optimized| / |baseline ∪ optimized|` 计算，衡量优化前后
的一致性，**不代表相对于人工真值的准确率**。双方均为空时约定 IoU=1，另计空 mask，
避免将空输出当成有效分割。

下表由保存的首轮 NPZ 独立重算；三轮全部 `records.json` 也逐条核对。

| 原帧号 | target：S1+T1 | receiver：S1+T1 | target：S1+T4 | receiver：S1+T4 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 1 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 2 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 3 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 4 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 5 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 6 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 7 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 8 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 9 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

传播的 20 个角色帧均非空；每种优化三轮合计比较 60 个角色帧，变化像素数和
presence changes 均为 0。每角色平均 IoU、最小 IoU、仅非空平均 IoU 均为 1.000000。

每轮 36 个候选也逐像素一致，三轮合计每变体 108 个候选。每轮有 6 对双方空候选，均为
target 的部分文本查询；其余 30 个非空候选的平均和最小 IoU 同为 1.000000。单图加载
保留了三个候选帧及三类请求的原输出，没有因减少加载而减少候选搜索。

已目视检查 [T1 review sheet](../artifacts/umi_inference_fast_10frames/paired_final/review_t1.jpg)
和 [T4 review sheet](../artifacts/umi_inference_fast_10frames/paired_final/review_t4.jpg)：
各包含原图、基线、优化、像素差异四行；差异行无变化像素。原图与检查图只保存在
ignored artifacts 中，不进入 Git。

## 速度结果

单位为秒，`均值 ± SD`；SD 为 3 次重复的总体标准差（`ddof=0`）。

| 变体 | 36 候选分割 | 两角色传播 | SAM 合计 | 合计时间减少 | 合计速度比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 基线 | 5.855 ± 0.086 | 1.901 ± 0.386 | 7.756 ± 0.405 | — | 1.000× |
| S1+T1 | 5.502 ± 1.661 | 1.305 ± 0.307 | 6.807 ± 1.966 | 12.23% | 1.139× |
| S1+T4 | 5.450 ± 1.310 | 1.103 ± 0.051 | 6.553 ± 1.360 | 15.51% | 1.184× |

传播阶段平均时间分别减少 31.38% 和 42.00%。候选分割平均时间分别减少 6.02% 和 6.91%，
但其波动大于平均差值；本次未单独运行 S1-only，不能把合计收益全部归因于 T1 或 S1。
T4 相对 T1 的传播平均时间减少 15.48%，此结论同样只适用于本次短窗口。

保留每轮完整结果，不剔除慢轮：

| 轮次（执行顺序） | 变体 | 候选分割 | 传播 | SAM 合计 |
| --- | --- | ---: | ---: | ---: |
| 1（1） | baseline | 5.743296 | 1.623394 | 7.366690 |
| 1（2） | fast_t1 | 3.993942 | 1.067128 | 5.061070 |
| 1（3） | fast_t4 | 5.230635 | 1.087019 | 6.317654 |
| 2（1） | fast_t1 | 4.697268 | 1.108673 | 5.805940 |
| 2（2） | fast_t4 | 7.152819 | 1.171129 | 8.323948 |
| 2（3） | baseline | 5.953573 | 1.633638 | 7.587210 |
| 3（1） | fast_t4 | 3.967613 | 1.049861 | 5.017474 |
| 3（2） | baseline | 5.867411 | 2.446557 | 8.313968 |
| 3（3） | fast_t1 | 7.815927 | 1.738262 | 9.554189 |

GPU 在启动检查时空闲，但随后有外部 PID `2866265`。独立解析 `gpu.log`：共 34 次 pmon
采样，外部进程在 34/34 次出现，17/34 次有正 SM 活动，最高 98%。当前进程采样到的最大
显存为 7548 MiB，这不是精确峰值，也没有按变体拆分。应将该结果理解为共享 H200 下的
观察值；三轮不足以评估统计显著性，后续若需稳定容量规划，应在空闲卡上复测。

此前 GPU 0 的首次试跑同样保留于 `artifacts/umi_inference_fast_10frames/paired/`。
其 SAM 合计均值为 baseline 7.400510、T1 6.539988、T4 6.580355 秒，三轮输出均精确一致。
首次试跑与正式运行使用相同 SAM adapter/features；之后仅修正 benchmark 回调绑定与
JSON 字典封装类型。正式表采用修正后脚本的 GPU 1 重跑，不混合两次运行的绝对耗时，
也不按较快的一次择优报告。

## 源身份与独立核验

正式记录：`artifacts/umi_inference_fast_10frames/paired_final/`。

| 文件 | SHA-256 |
| --- | --- |
| 冻结 `baseline_sam3_adapter.py` | `b7a69096a4f1edd15e1576d4f4cdb61c7241aab129c74e8ab5358d4960d2de96` |
| `scripts/benchmark_umi_fast.py` | `c211ff6b34c8c2528b863f4980373c6f2a04cb70bd63bb9fe432ee584288167c` |
| `adapters/sam3_adapter.py` | `d01afe1594d715600f31eb73f6e304fa919e4f7f44933584e04f4ec37d7f27c2` |
| `adapters/sam3_tracking_features.py` | `a6dd1e3b85364e7c62be127aa9ea448b6460095f3178cf7d3f7392417182c95d` |
| SAM checkpoint | `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e` |

冻结基线 hash 与原工作区文件相同；当前三个源文件 hash 与正式 metadata 相同，10 张
输入 JPEG 的 hash 也逐一匹配 metadata。独立核验重新从三个 NPZ 计算交并集、空 mask
计数和差异像素，并检查全部 9 条正式记录的唯一性、候选及传播一致性。计时均值和 SD
从逐轮记录重算，与 `summary.json` 一致。

- `metadata.json`：任务、帧号、请求、环境、输入与源代码 hashes。
- `records.json` / `summary.json`：逐轮记录与汇总；正式 records 为 `{"records": [...]}`。
- `baseline.npz` / `fast_t1.npz` / `fast_t4.npz`：首轮保存的 bool candidates `[36,480,640]`
  与 masks `[2,10,480,640]`；后两轮输出一致性记录在 records 中，未另存 NPZ。
- `analysis/independent_checks.json`：独立核验结果。
- `analysis/frame_iou.csv` / `analysis/timings.csv`：逐帧 IoU 与原始时间表。

上述原始结果及快照保留在 ignored artifacts 中，不提交视频、图片、checkpoint 或运行
产物。首次试跑的 `paired/records.json` 为裸 list，读取时应与正式封装区分。

## 复现与覆盖边界

先设置 `FRAMES_DIR`、`SEED_ARTIFACTS`、`BASELINE_ADAPTER`、`SAM_CHECKPOINT`、`OUTPUT_DIR`
及 `SAM_GPU`；输出目录必须不存在。帧目录应包含上述样本的 `000000.jpg` 至 `000009.jpg`，
seed 目录应包含对应历史 manifest、semantic plan 和两个通过 QC 的 frame 0 seed；基线
快照需匹配上表 hash。参数只通过环境变量传入，不在配置内写入机器绝对路径。

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONPATH=src:. \
  .venv/bin/python -m scripts.benchmark_umi_fast \
  --frames "$FRAMES_DIR" \
  --seed-artifacts "$SEED_ARTIFACTS" \
  --baseline-adapter "$BASELINE_ADAPTER" \
  --checkpoint "$SAM_CHECKPOINT" \
  --output "$OUTPUT_DIR" \
  --gpu "$SAM_GPU" --repeats 3
```

CPU 验证已运行单元测试：923 passed、1 skipped；Ruff 及变更文件严格 mypy 通过。
全量 `mypy src` 仍报告基线已有的 `pipeline/gripper/sam/annotator.py:251`
OpenCV `fillConvexPoly` 参数类型错误，该文件未修改。
异步重叠、跨组常驻、多 GPU 分发、并发限制与恢复通过 CPU 流程测试验证；本轮真实 GPU
对照只覆盖 SAM 候选分割和固定 seed 传播，未实测这些调度优化的完整数据吞吐。

10 帧来自同一个视频开头；本轮真实 GPU 对照未覆盖长视频、非零 seed 的反向跟踪、明显
遮挡及其他分辨率。因此当前支持的结论是：该短样本输出一致，SAM 阶段观察到提速。
全流程、长序列和异步吞吐仍需独立验证，不能与历史实验的加速比相乘。
