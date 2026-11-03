# UMI 嵌套数据批处理

`configs/datasets/umi_h200.yaml` 排除搬运 `A_003`，为其他 15 类动作声明物体角色。
每段从 `task.json` 读取实际任务文本，不能只用目录名代替对象定义。

流程为多视角单臂筛选、保留原始分辨率的 MCAP 转换、现有 SAM worker 并行处理。
单臂筛选使用全部头部预览及左右腕部各 4 帧，低于 0.9 置信度、双臂、无动作、证据缺失
或接口失败均不进入转换。已知需要排除的片段保存在 YAML 中，理由可追溯。
模型筛选并不保证整段无短暂辅助动作，因此所有成功输出仍需视觉复核。

环境沿用 `.venv`、`configs/process.yaml` 及其 Qwen 凭据环境变量；变量未设置时，
启动脚本读取已有的 `secrets/qwen_api_key.txt`。输入 MCAP 只读，全部生成文件写入批次目录。
SAM 进程需要实际 CUDA 设备访问权限；仅 `nvidia-smi` 成功不足以证明 PyTorch 可用。

```bash
# SOURCE_ROOT 和 BATCH_DIR 由运行环境指定，后者需要足够空间。
.venv/bin/python -m scripts.process_umi_batch "$SOURCE_ROOT" "$BATCH_DIR" \
  --phase screen --workers 4 --limit-per-action 2
.venv/bin/python -m scripts.process_umi_batch "$SOURCE_ROOT" "$BATCH_DIR" \
  --phase prepare --workers 4
.venv/bin/python -m scripts.run_umi_batch "$BATCH_DIR" \
  --smoke --gpus 4 5 6 7 --qwen-workers 4

# 抽查 smoke 结果后启动全量。GPU 编号应按运行环境调整。
.venv/bin/python -m scripts.process_umi_batch "$SOURCE_ROOT" "$BATCH_DIR" \
  --phase all --workers 4 --gpus 4 5 6 7
```

同一批次不要并发启动两个筛选或 SAM 驱动，也不要在 SAM 读取输入时重建同一组输入。
不同阶段可独立重启：筛选复用相同输入和模型的成功判断、重试接口失败；转换复用已完成
的逐段缓存；SAM 复用现有 completion scan 跳过完整产物。源文件、任务规则或筛选 prompt
改变时应使用新的批次目录。成功转换的每段 video 和 sidecar 在分组输入中使用链接，
Parquet 的 `index` 按组连续排列，`source_frame_index` 保持原始解码来源。

主要结果：

- `inventory.json`：全部候选的源 ID、任务文本、文件指纹及冻结规则。
- `screening/*.json`：逐段左右手判断、图像证据哈希、模型原始回答或失败原因。
- `screening_status.json`：筛选进度；`batch_plan.json`：所有候选的准备状态和实际分辨率分组。
- `converted/<source-id>/`、`inputs/<action>_<resolution>/`：转换缓存和 pipeline 输入。
- `smoke/`、`runs/`：小样本和全量结果，均有 `batch_status.json`、`episode_results.json`、
  `index.html`；每组有原生四通道 masks、叠加视频和三个时刻的 review sheets。

`needs_temporal_review` 表示 masks 和 overlay 已生成，仍需视觉复核；`failed` 表示模型、
转换后处理或渲染未完成。被排除、待确认、尚未筛选的片段保留原状态，不计为成功输出。
UMI 只处理 object source；两个 gripper 通道为空，不补造抓取、持有或释放时刻。

## 快速推理

`scripts/run_umi_batch.py` 增加显式选择的快速模式。默认仍为 `legacy`；其他数据工作流的
默认参数保持原样。这里的 `legacy` 对应已提交分支的原路径；性能对照另行冻结包含原
工作区文本单图补丁的基线，避免重复计算已经存在的收益。

| 模式 | 行为 |
| --- | --- |
| `legacy` | 原有按组 worker 调度，使用 process profile 的 SAM 参数。 |
| `resident` | 各 GPU 的 SAM 跨动作及分辨率组常驻；启用三类 seed 单图请求和 T1；规划与渲染同步。 |
| `async` | 在 resident 基础上预取规划并后台渲染，SAM 始终由所属进程的同一线程调用。 |

先用新输出目录运行，避免把旧产物当作新实现的结果：

```bash
# PROCESS_CONFIG 指向实际使用的 process.yaml，权重和凭据由环境及现有配置提供。
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONPATH=src:. \
  .venv/bin/python -m scripts.run_umi_batch "$BATCH_DIR" \
  --config "$PROCESS_CONFIG" --gpus 0 \
  --inference-mode async --tracking-batch-size 1 \
  --prefetch 2 --render-buffer 2 --cpu-threads 8 \
  --qwen-workers 4 --output "$FAST_OUTPUT"
```

将 `--inference-mode` 改成 `resident` 可保留同样的模型生命周期做串行对照。
`--tracking-batch-size 0` 使用原传播路径，`1` 使用 T1，`4` 使用 T4；快速模式始终开启
种子单图优化。T4 的额外收益需要按实际负载确认，当前默认候选为 T1。
已有完整准备流程也接受 `scripts/process_umi_batch.py --phase all --inference-mode async`。

### SAM 任务队列

`async` 模式可将 Qwen/候选质检生产者与完整视频工作分开，例如：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=src:. \
  .venv/bin/python -m scripts.run_umi_batch "$BATCH_DIR" \
  --config "$PROCESS_CONFIG" --gpus 0 --inference-mode async \
  --episode-workers 16 --qwen-workers 16 --prefetch 16 \
  --video-workers 2 --render-buffer 2 --cpu-threads 4 --output "$FAST_OUTPUT"
```

`--episode-workers` 是每 GPU 的候选/Qwen 生产者及待传播任务总容量，默认 1 保留原调度；
大于 1 只适用于 async。`--video-workers` 默认 2，限制每 GPU 同时持有完整解码目录或
完整传播结果的任务数，包括正在保存和等待保存的结果。生产者只保留候选帧、少量上下文
图像及 QC 证据，不在等待 Qwen 时持有整段 JPEG。每条视频的 QC/fallback 依赖顺序不变。
两次对照须使用不同的新输出目录；`process_umi_batch.py --phase all` 透传这两个参数。

所有候选生成和传播请求进入有界 FIFO，由每卡唯一 SAM owner 串行执行。传播完成后独立
writer 保存 QC 图片和 canonical masks，完成后才标记 masks_ready。渲染队列只保留路径
及小型结果记录，积压不会占生产者名额，批次仍等待所有保存和渲染任务完成才返回。视频
名额和生产者名额避免大数组无界积压，但内存需求仍随视频长度、分辨率及 QC 尝试数变化。

`--qwen-workers` 是整批全局请求上限，不是每 GPU 的上限。队列模式将规划请求限制在
最多一半容量（至少一个），请求上限大于 1 时为 QC/bbox 留出 token；QC 可以使用所有空闲 token。运行中的
请求不被抢占。预取和渲染执行容量按 GPU worker 限制；GPU 分组仍按帧数和角色数静态
分配，不提供跨 GPU 任务迁移。并发参数应结合实际 API 延迟和限流逐档调整。

fatal CUDA 会停止接收模型调用并唤醒等待者；中断先关闭模型队列，再回收线程，已提交的
写盘任务会排空。完成的 masks 可恢复复用，尚未保存的阶段会重做。已发出的 Qwen HTTP
请求仍等待响应或现有超时；没有新增 HTTP 取消机制。

每个 `.pipeline/worker-*/episode_results.json` 记录 `staged`、`episode_workers`、
`video_workers`、`peak_episodes`（生产者/就绪任务）、`peak_videos`（传播/保存）、
`peak_ready_videos` 和 `sam_queue`。`timings.json` 包含 `sam_prepare`、`sam_propagate`、
`sam_save` 以及每次模型方法调用的 job ID。`model_call_s` 包含方法内部 CPU 工作；
`queue_wait_s` 是请求等待 owner 的累计时间，`empty_wait_s` 是 owner 取任务的等待时间。
稀疏取帧仍需解码压缩视频，但只输出候选 JPEG；ffmpeg 每个进程使用一个解码/编码线程，
避免大量生产者自动扩张编解码线程。上述计时都不是 CUDA kernel 时长。性能验收应同时比较输出一致性、就绪产物吞吐、队列
空等和 GPU 采样，并区分启动、稳定处理及收尾阶段。

使用相同命令重启会校验完整选择、输入、权重、配置、prompt、代码及调度参数，再继续处理。
有效 masks 可只补渲染。改变 subset、GPU 分配或实现应使用新的输出目录。
旧输出没有快速模式身份文件时会拒绝复用，不覆盖已有推理结果。批次状态及分组索引仍由
协调端更新，原有分辨率合并入口可读取快速模式的 `episode_results.json`。

固定十条完整视频的速度、IoU 和失败记录见
[合并验证摘要](umi_inference_end_to_end_20261020.md)。固定 seed 的传播结果一致；真实 Qwen
端到端结果存在差异，尚未证明等质替换。`needs_temporal_review` 仍要求视觉复核。

### 2026-11-03 队列对照验证

在同一张 H200 上依次比较 `async --episode-workers 1` 和 `3`。输入为 `cam_high` 的
`pick_up_640x480/0`、`pull_640x480/203`、`pour_640x480/500`、`scoop_640x480/1000`，
共 681 帧。实际执行 SAM3、完整视频解码、候选筛选、fallback、传播、保存和 review 渲染；
Qwen 规划与 24 次 QC/bbox 请求使用已保存的响应及逐次耗时重放，无远程 API 调用。
两轮均为 T1、Qwen 请求上限 4、prefetch 2、render buffer 2、CPU threads 4。

| 指标 | 原 async，1 条在途视频 | SAM 队列，3 条在途视频 |
| --- | ---: | ---: |
| 总墙钟时间，含模型生命周期和渲染 | 183.145 s | 156.830 s |
| SAM 方法调用数 | 37 | 37 |
| QC/bbox 请求数 | 24 | 24 |
| masks 和 review 就绪 | 3/4 | 3/4 |
| 测试进程 SM 活跃采样均值 | 16.1% | 20.2% |
| 测试进程显存峰值，pmon 采样 | 10,178 MiB | 10,178 MiB |
| SAM 方法与候选 QC/bbox 重叠 | 0 s | 16.204 s |
| SAM 方法与保存重叠 | 0 s | 6.302 s |

本次观察到用时减少 14.4%，同等就绪产物的吞吐提高 16.8%。9 个 NPZ 的全部数组、
249 个 PNG、4 份候选 QC 报告及 24 次请求内容一致；canonical masks 差异像素为 0。
`scoop` 的 receiver 在两轮均被相同 QC 规则拒绝，保留失败，未跳过其重试工作。

这是四条视频的一组对照。同卡存在另一个推理进程，其 SM 采样均值由 19.5% 变为 39.1%，
所以不能把本次百分比当成独占 GPU 或八卡生产吞吐承诺。SM 数值来自约 1 Hz 的进程采样，
不是 kernel trace；SAM 重叠时间包括方法内部 CPU 工作。末尾仅剩 `scoop` 时仍有连续 QC
等待，队列不能消除同一视频内部的依赖。

本地实验保存在 `artifacts/sam_queue_validation/`：`comparison.json` 包含逐项比较结果，
`baseline/` 和 `queued/` 保留 timings、pmon、请求哈希、mask 和渲染产物，
`timeline.png`/`timeline.svg` 展示阶段重叠；`replay_benchmark.py` 和 `analyze.py` 可复现实验。
这些生成文件按仓库规则不提交。

验证覆盖队列线程归属、QC/保存等待时交错执行、渲染背压、普通错误、fatal CUDA、两种
中断来源、已完成 masks 保留、恢复身份和 spawned worker 的实际文件渲染。全量单元测试
为 932 passed、1 skipped，另有 2 个终端 UI 测试失败，已在原分支复现。Ruff 与变更模块
的严格 mypy 检查通过；全量 mypy 仍有原分支已有的 `gripper/sam/annotator.py:251`
OpenCV `fillConvexPoly` 类型签名错误。


### 2026-11-03 分阶段供给验证

将原有十条固定视频各重复三次，得到 30 个独立任务、7,197 帧（不是 30 条不同视频）。
基线使用 `c4281ee` 的 `episode-workers=3 / qwen-workers=4 / prefetch=2`；新版使用
`episode-workers=16 / qwen-workers=16 / prefetch=16 / video-workers=2`。两轮均为单卡
H200、T1、CPU threads 4、render buffer 2，依次执行真实 SAM、保存及渲染；语义规划和
QC/bbox 使用相同历史响应及请求延迟回放。

| 指标 | 原队列 | 分阶段生产者/消费者 |
| --- | ---: | ---: |
| 总墙钟时间 | 714.709 s | 814.363 s |
| SAM owner 等待任务 | 129.213 s | 40.323 s |
| SAM 方法执行占整批时间 | 79.2% | 92.8% |
| 稳定区间 SAM 方法执行占比 | 83.5% | 96.2% |
| SAM 方法累计墙钟时间 | 566.214 s | 755.637 s |
| 回放 Qwen 同时在途请求峰值 | 4 | 13 |
| SAM 待执行请求峰值 | 2 | 15 |
| masks 与 review 就绪 | 27/30 | 27/30 |
| 测试进程 SM 活跃采样均值 | 44.4% | 42.4% |
| 同卡其他进程 SM 活跃采样均值 | 8.0% | 31.8% |
| 进程 RSS 峰值 | 8,703 MiB | 10,408 MiB |
| 测试进程显存峰值，pmon 采样 | 12,610 MiB | 15,518 MiB |

稳定区间取第 6 个任务保存完成至第 24 个任务保存完成，不含启动及最后六个任务的收尾。
SAM 方法时间包含内部 CPU 工作，不能解释为 CUDA kernel 时长。新版完整视频/待保存结果
峰值为 2，准备完成、等待传播的任务峰值为 12。空队列等待减少 68.8%，但本轮总耗时增加
13.9%，吞吐降低 12.2%；**本次没有测出整体提速**。同卡其他进程在两轮中的负载明显不同，
同时相同 204 次 SAM 调用自身耗时增长，不能仅用这组共享 GPU 对照归因调度的速度收益或
回归。检查时全部八张 GPU 都有其他任务；没有停止这些任务来取得独占资源。

69 个 NPZ 全部数组、1,455 个 PNG、30 份 QC 报告及 114 次 QC/bbox 请求内容/响应一致，
canonical masks 差异像素为 0。三份 `scoop` receiver 失败均保留。该实验确认供给及资源
边界生效，独占 GPU 上的吞吐和 SAM 内部 CPU/CUDA 时间仍需单独验证。

另一次仅访问真实 Qwen 的并发探测，使用十条源视频各两次，共 20 个语义请求：实际 HTTP
在途峰值 16，HTTP 错误 0，20 份语义计划均通过解析；整轮 16.194 s，平均请求 6.935 s。
它验证此次接口可以接受这批并发请求，不是完整流水线或多卡性能对照。

生成记录保存在 `artifacts/sam_supply_validation/`：`comparison.json`、两轮原始 timings、
pmon、产物、`summary.png`/`summary.svg`、逐视频 `timeline.png`/`timeline.svg`，以及
`live_qwen/summary.json` 和实际语义响应。回放及分析脚本也在此目录，按产物规则不提交。
相关 72 个测试、Ruff 和变更模块严格 mypy 通过；全量单元记录为 940 passed、1 skipped、
2 个已在原分支复现的终端 UI 失败，后续新增的三个中断场景包含在上述 72 项验证中。

## 合并同一任务的不同分辨率

批次结束后可以生成独立的任务视图，按任务并行重绘 review sheets：

```bash
.venv/bin/python -m scripts.merge_umi_runs "$BATCH_DIR/runs" "$MERGED_DIR" --workers 4
```

输出目录必须不存在。`pick_up_640x352`、`pick_up_640x480` 等分组归入 `pick_up/`，
episode、原图和叠加视频通过软链接引用，保留原始尺寸、ID、状态和来源路径。
每任务 `episode_results.json` 包含实际处理的片段；根清单和任务网页还保留筛选排除及
待确认记录。每任务重绘早期、中期、末期三张 review sheet，失败段使用带标记的原图。
`index.html` 是任务总览，`review_sheets/index.html` 集中展示合并后的检查图。
`merge_status.json` 在所有任务完成后才标记 `finished`；失败的处理结果不会因此变成成功。
