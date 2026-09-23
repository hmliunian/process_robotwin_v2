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
