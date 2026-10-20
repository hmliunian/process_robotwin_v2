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

多个 GPU 使用 `--gpus 0 1`。任务按输入帧数和角色数估算工作量，分配到跨组 worker；
各 worker 有自己的规划及渲染队列。`--qwen-workers` 是整批请求上限，不是每 GPU 的上限。
单卡限制规划线程数可为 QC 留出容量；多卡没有全局 QC 优先级。预取和渲染容量均按 worker
限制，队列只传路径及小型元数据。`--cpu-threads` 设置各 worker 的 PyTorch/OpenCV 线程。

使用相同命令重启会校验完整选择、输入、权重、配置、prompt、代码及调度参数，再继续处理。
有效 masks 可只补渲染。改变 subset、GPU 分配或实现应使用新的输出目录。
旧输出没有快速模式身份文件时会拒绝复用，不覆盖已有推理结果。批次状态及分组索引仍由
协调端更新，原有分辨率合并入口可读取快速模式的 `episode_results.json`。

固定十条完整视频的速度、IoU 和失败记录见
[合并验证摘要](umi_inference_end_to_end_20261020.md)。固定 seed 的传播结果一致；真实 Qwen
端到端结果存在差异，尚未证明等质替换。`needs_temporal_review` 仍要求视觉复核。

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
