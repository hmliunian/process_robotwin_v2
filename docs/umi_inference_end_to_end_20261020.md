# UMI 完整视频端到端效率与 IoU

2026-10-20，开发分支 `feat/umi-inference-fast`。固定十条完整视频，共 2,399 RGB 帧、14 个
必要对象角色。真实调用现有阿里云 `qwen3.8-max` 和本地 SAM3，保留候选 QC、bbox 救援、
时序 QC、canonical 保存、叠加视频和检查图。没有使用回放接口或删去失败任务。
测量执行基于 `d9e5dc0`，随后仅新增离线统计与报告，执行依赖的源码指纹未变。

**本轮基线整批 692.30 秒，常驻串行 532.17 秒；没有联网错误的异步复测为 396.89 秒。**
异步复测比常驻串行少 25.42%，日志确认阶段重叠 80.93 秒。基线输出就绪 10 条，常驻串行
和异步复测各 9 条，异步复测相对基线的发布 mask 平均 IoU 为 **0.865275**。因此目前
证明了可观察的速度收益和真实异步重叠，尚未证明端到端输出等质或完全一致。

## 整批效率

三组依次使用同一 GPU 7（NVIDIA H200），CPU / OMP / MKL / OpenCV 各 8 线程，TF32 开启。
异步首轮遇到短暂连接错误，因此保留首轮并以相同配置完整复测一次，不覆盖或挑选首轮结果。

| 模式 | runner 整批 s | 完整进程 s | 每次尝试摊销 s | 输出就绪 / 尝试 | 就绪条 / min | SAM 加载次数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 旧 SAM 逐条加载基线 | 692.30 | 694.62 | 69.23 | 10 / 10 | 0.867 | 10 |
| S1+T1 常驻串行 | 532.17 | 534.20 | 53.22 | 9 / 10 | 1.015 | 1 |
| S1+T1 异步首轮 | 368.76 | 370.84 | 36.88 | 7 / 10 | 1.139 | 1 |
| S1+T1 异步复测 | 396.89 | 398.99 | 39.69 | 9 / 10 | 1.361 | 1 |

每次尝试摊销为整批耗时除以全部十次尝试，**不是独立单条任务延迟**；就绪吞吐的分母同样
保留失败成本。异步复测相对基线就绪吞吐增加 56.99%，相对常驻串行增加 34.09%。
四轮均尝试全部十条，没有遗漏、未执行或待渲染任务。

runner 墙钟从 main 开始，包含导入重型运行库、输入指纹准备、健康检查、模型加载、全部
任务及失败成本、渲染 drain 和释放；顶层 Python imports 及最终落盘的完整进程时间另由
`/usr/bin/time` 测量。execution 墙钟另为 685.01 / 525.45 / 362.07 / 389.76 秒。

基线为冻结原工作区 adapter、每任务重新加载模型的受控串行重建，使用与快路径相同的
pipeline core 和 renderer；不是逐字节复现旧 CLI。原工作区已有文本候选单图补丁，不能
重复计入这一部分 S1 收益。基线每条健康检查一次，其他组全批一次。

## 单条任务服务时间

下表相加 planning、SAM/QC/save、render 三个互不嵌套的父阶段。它排除模型启动和任务排队，
异步预取后的等待另计在 `plan_to_render_wall_s`，原始 CSV 同时保留这两种时间。

| 任务 / episode | RGB 帧 | 基线服务 s | 常驻串行服务 s | 异步复测服务 s | 串行 / 异步复测就绪 |
| --- | ---: | ---: | ---: | ---: | --- |
| pick_up / 0 | 166 | 34.20 | 27.20 | 24.43 | 是 / 是 |
| put_down / 101 | 176 | 30.58 | 22.18 | 26.47 | 是 / 是 |
| pull / 203 | 170 | 51.95 | 41.95 | 40.95 | 是 / 是 |
| push / 301 | 212 | 39.97 | 31.75 | 32.20 | 是 / 是 |
| pour / 500 | 124 | 47.57 | 41.07 | 35.97 | 是 / 是 |
| wipe / 700 | 519 | 136.23 | 105.34 | 104.22 | 是 / 是 |
| rotate / 928 | 155 | 37.86 | 31.29 | 32.31 | 是 / 是 |
| scoop / 1000 | 221 | 96.74 | 99.03 | 94.09 | 否 / 否 |
| stir / 1101 | 500 | 107.29 | 92.33 | 84.90 | 是 / 是 |
| turn_over / 1203 | 156 | 25.78 | 22.80 | 22.55 | 是 / 是 |

例如 pour 基线含健康检查和模型生命周期的单条墙钟为 54.90 秒；其共同口径服务时间为
47.57 → 41.07 → 35.97 秒。异步复测从规划线程开始至渲染完成为 79.62 秒，包含预取后
等 GPU 的时间，不能用 39.69 秒的整批摊销替代这一延迟。

## 异步确已执行

- 同一主线程持有并顺序调用 SAM；没有把 tracker 改成多线程，也没有异步发布 masks。
- Qwen 规划预取容量 2，规划线程 2，请求并发上限 4。
- 后台渲染线程 1，容量 2；mask/QC 保存同步，渲染只读取已保存产物。
- 异步复测规划与 SAM/QC/save 父阶段重叠 58.71 秒，渲染重叠 54.11 秒；去重后的并集
  交集为 **80.93 秒**，两项不能直接相加。串行两组的对应重叠均为 0。
- 该父阶段还包含 QC 网络等待和文件保存，80.93 秒不是 GPU kernel 利用时间。

## 失败与工作量差异

常驻串行和异步复测均为 `scoop.receiver` 未通过候选 QC：候选包含支撑凳，未独立覆盖
方形陶瓷碗。失败任务的规划、SAM/QC/save 和检查图服务成本分别为 99.03 和 94.09 秒，
已保留在整批分母；没有降低 QC 门槛使其通过。

异步首轮额外有 put_down 的 QC 请求、pull 的语义规划请求失败。错误在同一约四秒窗口内
表现为连接被远端关闭和连接被拒绝，随后请求恢复。首轮失败父阶段的去重服务成本为
112.24 秒；整批时间并非正常完成相同工作量的速度。完整复测没有联网请求错误。

| 实际工作量 / 阶段 | 基线 | 常驻串行 | 异步首轮 | 异步复测 |
| --- | ---: | ---: | ---: | ---: |
| Qwen completion 次数 / 错误 | 42 / 0 | 53 / 0 | 46 / 3 | 48 / 0 |
| SAM 传播调用次数 | 14 | 13 | 11 | 13 |
| SAM 候选函数调用次数 | 58 | 65 | 52 | 55 |
| 模型加载合计 s | 75.35 | 10.29 | 10.34 | 10.31 |
| 传播合计 s | 261.18 | 157.63 | 145.43 | 157.56 |
| Qwen completion 合计服务 s | 175.82 | 196.60 | 178.77 | 185.64 |
| canonical 保存合计 s | 15.73 | 15.83 | 14.89 | 15.85 |
| QC 产物保存合计 s | 27.36 | 28.63 | 23.63 | 25.23 |

各阶段为 inclusive service time，有嵌套或并行，不能把全表相加求墙钟。Qwen 温度虽为 0，
语义选词、seed、候选接受及 bbox 救援仍可能不同；传播次数也随失败变化。

## IoU 的两种口径

固定历史 seed 的[完整视频 SAM 传播对照](umi_inference_full_validation_20261020.md)
已验证 13 个角色、3,542 个角色帧全部逐像素一致，非空帧 IoU 也为 1。这隔离了 Qwen 决策，
没有人工逐帧真值。

本次真实端到端对照同时重跑 Qwen 和 QC，不能沿用固定 seed 的 IoU=1 作为其结论。
canonical 与 native 分层计算，排除 gripper 和不适用 receiver；双方空 IoU 约定为 1 并单列，
单边空为 0，缺失产物标为 unavailable，不伪造空 mask，也不忽略失败任务的状态。

下表以每角色每帧的交并比均匀平均，包含失败后发布的空通道。native 是时序隔离前的
原始 track；没有批准 seed 的角色缺少 native，不能将其当成全空且一致。

| 对基线 | canonical 平均 IoU | 非空 union IoU | canonical 可比角色帧 | 缺 canonical 角色 | native 平均 IoU / 可比角色帧 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 常驻串行 | 0.899715 | 0.898283 | 3,763 | 0 | 0.955852 / 3,542 |
| 异步首轮 | 0.856657 | 0.854635 | 3,593 | 1 | 0.963070 / 3,196 |
| 异步复测 | 0.865275 | 0.863424 | 3,763 | 0 | 0.919263 / 3,542 |

三组双方空角色帧分别为 53 / 50 / 51；其余非空 union 帧包含单边空的 IoU=0。
异步首轮 pull 规划失败，缺少该角色的 170 帧，标记 unavailable，不进入 IoU 均值。
常驻串行和异步复测各缺少 scoop.receiver 的 native；首轮另缺 put_down 和 pull。

作为补充，只比较双方整条输出都就绪的交集：基线→常驻串行为 9 条 / 3,321 角色帧，
IoU=0.976063；基线→异步首轮为 7 条 / 2,975 帧，IoU=0.982518；基线→异步复测为
9 条 / 3,321 帧，IoU=0.926205。常驻串行→异步复测相同九条的 IoU=0.939793。
常驻串行交集含 12 个双方空帧，非空为 3,309 帧；其他三项交集均全为非空 union。
交集 canonical/native 均值相同；这些交集高分不能替代主表中的失败和低 IoU。

已定位的具体决策差异：

- rotate 两次均用 frame 0 的同一 RGB 和相同定位提示，但 Qwen 返回 bbox 右边界分别为
  0.566 / 0.562；seed mask 变化 49 像素，seed IoU=0.983057，最终 track IoU=0.9517。
  单看通用 query 名称和 seed 帧号相同，不能认定 SAM 输入相同。
- 异步首轮 rotate 的 bbox 与基线一致，seed PNG 也逐像素一致；复测的右边界为 0.551，
  再次产生不同 seed。它说明实时定位响应变化确实混入端到端 IoU。
- pull 在异步复测中改用 frame 48 的 `metal handle`，基线为 frame 0 的 bbox 救援；
  该角色平均 IoU 仅 0.0151，不能由整体平均值掩盖。
- stir.receiver 的最终 seed 从基线 214 变为快路径的 0；scoop.receiver 则由通过变为拒绝。

这些例子说明真实 Qwen 决策和输入存在变化，但不能据此把所有端到端差异都归因到同一个
环节。当前原入口仍默认 legacy，异步为显式选项；本轮没有将其切换为默认，也没有放松 QC。

## 产物、环境与复现

原始产物在 ignored `artifacts/umi_pipeline_full_20261020/`，四轮目录分别为 `baseline/`、
`resident/`、`async/`、`async_repeat/`，均保留完整选择、任务结果、源码和输入指纹、
Qwen 原始响应、候选 QC、native/canonical masks、overlay、检查图、GPU 日志和分阶段时间。
各轮 `*.time` 为独立完整进程计时，首轮失败产物未被复测覆盖。

PyTorch `2.7.1+cu128`；SAM checkpoint SHA256
`9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`。
四轮配置、输入、prompt、权重及 79 个执行源码指纹相同，只有 baseline 的 SAM 优化开关不同。
GPU pmon 有效采样 216 / 166 / 114 / 123 次，只见各轮自身 PID；采样未见同卡外部进程，
不能保证采样间隔内完全独占。

设置本地 `UMI_INPUTS` 和 `UMI_KEY_FILE` 后，完整基线复现命令为：

```bash
env PYTHONPATH=src:. OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  QWEN_API_KEY_FILE="$UMI_KEY_FILE" \
  .venv/bin/python scripts/benchmark_umi_pipeline.py \
  --manifest configs/datasets/inference_timing_umi_10tasks.json \
  --dataset-root "$UMI_INPUTS" --gpu 7 --mode baseline \
  --baseline-adapter artifacts/umi_inference_fast_10frames/baseline_sam3_adapter.py \
  --output artifacts/umi_pipeline_repeat/baseline
```

常驻串行和异步分别改为 `--mode resident`、`--mode async`，每轮选择新的输出目录；
`async_repeat` 的实际 mode 同样为 async，所有参数和环境不变。基线快照保存在本地 ignored
产物中，其 SHA256 为 `b7a69096a4f1edd15e1576d4f4cdb61c7241aab129c74e8ab5358d4960d2de96`。

时序明细由以下离线命令生成，无模型/API 调用：

```bash
env PYTHONPATH=src:. .venv/bin/python scripts/summarize_umi_pipeline_timings.py \
  --root artifacts/umi_pipeline_full_20261020 \
  --modes baseline resident async async_repeat
```

输出 `analysis/timing_summary.json` 和 `analysis/per_episode_timings.csv`，包括单条服务、
预取延迟、原基线完整单条墙钟、失败成本和去重重叠。新增汇总及质量分析的 16 项 CPU 测试
通过，Ruff / 格式 / strict mypy 通过。本轮未改生产推理或 QC 代码。

完整质量明细在 `analysis/quality.json` / `per_frame_iou.csv`；复测另存
`analysis/async_repeat_quality/`，输入映射保存在 `analysis/async_repeat_inputs/source_mapping.json`。
`analysis/independent_quality_audit.json` 从原始 NPZ 独立重算全部对照和交集，逐行核对 CSV，
并检查 39 份已有 canonical 合同；`input_code_gpu_audit.json` 核对输入、代码和监控。
`analysis/decision_evidence.json` 保存 bbox 坐标、seed 指纹和 QC 候选差异证据。

已目视检查以下代表检查图，未做人工逐帧验收：

- [pick_up 基线 frame 141](../artifacts/umi_pipeline_full_20261020/baseline/runs/pick_up_640x480/pick_up_640x480/episode_000000/cam_high/.pipeline/rendered_videos/review_sheets/target_early.jpg)。
- [pick_up 异步复测 frame 141](../artifacts/umi_pipeline_full_20261020/async_repeat/runs/pick_up_640x480/pick_up_640x480/episode_000000/cam_high/.pipeline/rendered_videos/review_sheets/target_early.jpg)。
- [wipe 异步复测 frame 518](../artifacts/umi_pipeline_full_20261020/async_repeat/runs/wipe_640x480/wipe_640x480/episode_000700/cam_high/.pipeline/rendered_videos/review_sheets/target_late.jpg)。
