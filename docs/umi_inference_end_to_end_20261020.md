# UMI 快速推理：合并验证摘要

2026-10-20，固定 10 条完整视频、2,399 RGB 帧、14 个必要物体角色，真实调用现有
`qwen3.8-max` 与 SAM3。单张 H200，CPU / OMP / MKL / OpenCV 各 8 线程，TF32 开启。
保留候选 QC、bbox 救援、时序 QC、canonical 保存、叠加视频和检查图。

完整实验工具、清单及原报告归档在 `perf/umi-inference-fast-validation`（`e9218de`）；
合并分支保留生产优化和本摘要。测量执行基于 `d9e5dc0`，执行依赖源码在四轮间未变。

## 完整视频结果

| 模式 | 整批秒 | 输出就绪 / 尝试 | 就绪条 / min | 对基线 canonical IoU | 可比角色帧 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 冻结旧 SAM、逐条加载基线 | 692.30 | 10 / 10 | 0.867 | — | — |
| S1+T1 常驻串行 | 532.17 | 9 / 10 | 1.015 | 0.899715 | 3,763 |
| S1+T1 异步首轮 | 368.76 | 7 / 10 | 1.139 | 0.856657 | 3,593 |
| S1+T1 异步复测 | 396.89 | 9 / 10 | 1.361 | 0.865275 | 3,763 |

整批时间包含准备、模型加载、真实 API、失败成本、保存、渲染 drain 和释放；不代表独立
单条延迟。基线使用冻结原 adapter 和相同 pipeline core/renderer，每条重新加载模型；
已包含原工作区文本单图补丁，不能重复计其 S1 收益，也不等于逐字节重放旧 CLI。

异步复测比常驻串行少 25.42%，但四轮间 Qwen 决策、传播工作量及输出就绪数存在差异，
**尚未证明端到端输出等质，不能将上述耗时比解释为同质量下的纯调度加速。**
基线加载 SAM 10 次，其余各 1 次；本轮未测多 GPU 扩展效率。

## IoU 与失败边界

IoU 衡量优化前后输出一致性，不是人工真值准确率；按必要角色逐帧平均，排除 gripper
及不适用 receiver。双方空计 1 并单列，单边空计 0，缺文件记 unavailable。
常驻串行 / 异步首轮 / 复测的非空 union IoU 为 0.898283 / 0.854635 / 0.863424，
双方空帧为 53 / 50 / 51。首轮缺 pull 的 170 帧 canonical，不进入均值但保留失败。
native IoU 分别为 0.955852 / 0.963070 / 0.919263，可比角色帧为 3,542 / 3,196 / 3,542。

双方整条输出就绪的交集：基线→串行 9 条、3,321 帧（3,309 非空），IoU 0.976063；
基线→异步首轮 7 条、2,975 非空帧，IoU 0.982518；基线→复测 9 条、3,321 非空帧，
IoU 0.926205。交集高分不能替代主表中的失败及低 IoU。

固定历史 seed 的独立全视频传播对照覆盖 13 角色、3,542 角色帧，逐像素一致、IoU=1；
其中 3,491 非空帧的 IoU 也为 1，51 帧双方空。scoop.receiver 无批准 seed，明确跳过。
该单轮仅传播耗时 250.00→155.36 秒（减少 37.86%），不含 Qwen、候选 QC 和渲染。

常驻串行和异步复测均因 scoop.receiver 候选混入支撑凳、漏掉独立碗边界而被语义 QC
拒绝；target 茶杯仍保留。首轮另有 put_down QC、pull 规划连接错误，复测无连接错误。
实时语义、seed 和 bbox 输入也发生变化：rotate 同 frame 0 / 同定位提示的 bbox 右边界
由 0.566 变为 0.562，seed 相差 49 像素；pull 复测改用 frame 48 的文本 seed。
因此不能把所有差异归因于 SAM 优化，基线自动通过也不等于人工正确。

## 生产入口

S1 为三类候选按单帧加载，保留多候选重试；T1 为传播视觉快路径，batch4 独立可选。
`legacy` 仍为默认，`resident` 跨组常驻且串行，`async` 预取规划并后台渲染。
SAM 始终由所属 worker 的同一线程调用，QC/传播/保存同步。复测规划与 SAM 父阶段重叠
58.71 秒、渲染重叠 54.11 秒，去重交集 **80.93 秒**；这不是 GPU kernel 利用时间。

设置 `BATCH_DIR`、`PROCESS_CONFIG`、全新 `FAST_OUTPUT`，沿用既有权重和凭据环境：

```bash
env PYTHONPATH=src:. OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  .venv/bin/python -m scripts.run_umi_batch "$BATCH_DIR" \
  --config "$PROCESS_CONFIG" --gpus 0 \
  --inference-mode async --tracking-batch-size 1 \
  --qwen-workers 4 --prefetch 2 --render-buffer 2 --cpu-threads 8 \
  --output "$FAST_OUTPUT"
```

改为 `--inference-mode resident` 可做常驻串行对照；batch4 使用 `--tracking-batch-size 4`。
输入、配置、权重及实现身份参与恢复校验；策略或选择改变时使用新的输出目录。

## 验证与溯源

合并前精简后 CPU 单元测试 921 passed、1 skipped；全量 Ruff、修改文件的格式和 strict mypy
检查通过。生产优化、并发、恢复和失败路径回归保留，实验统计测试随工具归档。
独立审计核对 39 份 canonical、逐帧 CSV、输入及代码 hashes；四轮 pmon 有效采样
216 / 166 / 114 / 123 次，未采到同卡外部 PID，不能保证采样间隔内完全独占。
原始结果与审计保留在 ignored `artifacts/umi_pipeline_full_20261020/`；
完整四轮、首轮错误、固定 seed 对照和决策证据均保留，视频、图片、权重及运行产物不入 Git。
