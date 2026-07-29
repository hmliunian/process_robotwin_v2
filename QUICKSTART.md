# 快速开始指南

## 当前状态

✅ **核心框架已完成**（~1100 行代码，21 个文件）

- Domain 层：业务实体和规则（无外部依赖）
- Application 层：用例编排
- Ports 层：接口定义
- 测试：Domain 和 Policy 单元测试

## 快速验证

```bash
# 1. 进入项目目录
cd /DATA/disk8/xuran/add_mask_robotwin/process_data_v2

# 2. 安装依赖（开发模式，不含 GPU 依赖）
just install  # 或 uv sync --extra dev

# 3. 运行单元测试
just test-fast  # 或 uv run pytest tests/unit -v

# 4. 类型检查
just typecheck  # 或 uv run mypy src/robotwin_annotation_v2

# 5. 代码格式检查
just lint  # 或 uv run ruff check src tests
```

## 项目结构一览

```
process_data_v2/
├── 配置
│   ├── pyproject.toml              # uv 项目配置
│   ├── uv.toml                     # PyTorch index
│   └── justfile                    # 命令快捷方式
│
├── 源码 (src/robotwin_annotation_v2/)
│   ├── domain/                     # ✅ 完成
│   │   ├── models.py               # 不可变值对象
│   │   ├── policies.py             # 角色策略
│   │   └── errors.py               # 领域异常
│   │
│   ├── ports/                      # ✅ 完成
│   │   ├── dataset.py              # 数据访问接口
│   │   ├── vision.py               # 视觉服务接口
│   │   └── artifacts.py            # 存储接口
│   │
│   ├── application/                # ✅ 完成
│   │   ├── prepare_keyframes.py   # 主用例
│   │   └── review_keyframes.py    # 审批用例
│   │
│   ├── adapters/                   # 📋 待实现
│   ├── bootstrap/                  # 📋 待实现
│   └── cli/                        # 📋 待实现
│
└── 测试 (tests/)
    ├── unit/                       # ✅ 2 个测试文件
    ├── contract/                   # 📋 待添加
    └── integration/                # 📋 待添加
```

## 核心概念速查

### Domain 对象

```python
# 引用一个 episode
ref = EpisodeRef(coarse_task="move_pillbottle_pad", episode_id="007152")

# 定义实例槽位
slot = InstanceSlot(name="target_0", role=AnnotationRole.TARGET)

# 定义帧窗口
window = FrameWindow(first=10, last=50)  # [10, 50] 包含边界

# 视觉提示
prompt = VisualPrompt(
    text="white pill bottle",
    bbox=Box(x_min=0.2, y_min=0.3, x_max=0.8, y_max=0.9)
)

# 关键帧请求
request = KeyframeRequest(
    request_id="007152_target_0_r001",
    episode=ref,
    slot=slot,
    anchor_kind=AnchorKind.PRE_GRASP_VISIBLE,
    allowed_window=window,
    visual_query="white pill bottle",
    revision=1,
)
```

### 策略模式

```python
# 语义计划
semantic = SemanticPlan(
    episode=ref,
    target_query="white pill bottle",
    receiver_query="blue square pad",
    has_static_receiver=True,
)

# 时间线
timeline = InteractionTimeline(
    episode=ref,
    move_start=10,
    close_start=50,
)

# 生成所有角色的请求
registry = RolePolicyRegistry()
requests = registry.get_requests(semantic, timeline)
# → [target_0 请求, receiver_0 请求]
```

### 用例流程

```python
# 准备关键帧
use_case = PrepareKeyframes(
    episode_repo=...,
    semantic_planner=...,
    timeline_detector=...,
    frame_source=...,
    keyframe_selector=...,
    grounding_service=...,
    segmenter=...,
    artifact_repo=...,
    policy_registry=RolePolicyRegistry(),
)

run_id = use_case.execute(ref)
# → "kf-20260729-abc123"
```

## 设计优势

| 传统方式 | Process Data V2 |
|---------|-----------------|
| `pipeline.py` 上千行 | 分层，每层职责清晰 |
| 难以测试（需要 GPU） | Domain/App 层可用 fake 测试 |
| 关键帧和传播混在一起 | 阶段隔离（Phase 1/2/3） |
| 难以定位故障 | 每阶段独立验证 |
| 修改风险大 | 添加新策略不改旧代码 |
| 可追溯性弱 | 完整 provenance 链 |

## 下一步

### 立即可做（不需要 GPU）

1. ✅ 运行单元测试验证核心逻辑
2. 📋 添加 fake adapters（用于测试）
3. 📋 用 fake adapters 测试完整 PrepareKeyframes 流程

### 需要 GPU 和数据

4. 📋 实现 RoboTwin dataset adapter
5. 📋 实现 Qwen HTTP client（连接现有 serve_qwen.py）
6. 📋 实现 SAM3 adapter
7. 📋 实现文件系统 artifact 存储
8. 📋 CLI 入口
9. 📋 运行第一个真实 episode

## 常用命令

```bash
# 列出所有命令
just

# 开发循环
just fmt          # 格式化代码
just lint         # 检查代码
just typecheck    # 类型检查
just test-fast    # 快速测试

# 检查 GPU
just check-gpu

# 查看最近的 run
just list-runs

# 检查某个 run 的 manifest
just inspect-run kf-20260729-abc123
```

## 文档

- `README.md` - 项目概览
- `PROGRESS.md` - 实施进度
- `process_data_v2_architecture_design.md` - 完整设计文档
- `configs/pilot_move_pillbottle_pad.yaml` - 配置示例

## 与 process_data 的关系

- **复用数据**: 读取 `../process_data/data/`
- **复用服务**: HTTP 连接 Qwen 服务（端口 18086）
- **复用权重**: 共享 SAM3/CoTracker checkpoints
- **独立环境**: 自己的 `.venv`，不污染 v1

## 问题排查

### 依赖安装慢
```bash
# 检查网络（需要下载 torch）
export ALL_PROXY=socks5://10.0.3.219:7890
just install
```

### 测试失败
```bash
# 查看详细输出
uv run pytest tests/unit -vv

# 只运行特定测试
uv run pytest tests/unit/test_domain_models.py::test_episode_ref -v
```

### 类型错误
```bash
# 详细类型检查
uv run mypy --show-error-codes src/robotwin_annotation_v2
```
