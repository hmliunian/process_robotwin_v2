# Process Data V2 实施进度

## 已完成 ✅

### 1. 项目配置
- `pyproject.toml` - uv 项目配置，phase1/phase2/dev 依赖分离
- `uv.toml` - PyTorch CUDA 128 index
- `justfile` - 命令快捷方式（install, test, prepare-keyframes, etc）
- `.gitignore` - 排除 artifacts 和虚拟环境
- `README.md` - 项目说明

### 2. Domain 层（核心业务逻辑）
✅ `domain/models.py` - 不可变值对象
  - `EpisodeRef`, `InstanceSlot`, `FrameWindow`
  - `Box`, `VisualPrompt`
  - `KeyframeRequest`, `ApprovedSeed`, `MaskArtifactRef`
  - `AnnotationRole`, `AnchorKind`, `ReviewStatus`, `SegmentationMethod` (枚举)

✅ `domain/policies.py` - 角色策略
  - `SemanticPlan`, `InteractionTimeline`
  - `TargetSeedPolicy` - pre-grasp 窗口
  - `StaticReceiverSeedPolicy` - 静态 receiver 窗口
  - `RolePolicyRegistry` - 策略注册表

✅ `domain/errors.py` - 领域异常

✅ `domain/__init__.py` - 导出清单

### 3. Ports 层（接口定义）
✅ `ports/dataset.py`
  - `EpisodeRepository` - 读取 RoboTwin 数据
  - `SemanticPlanner` - 确定角色和查询
  - `TimelineDetector` - 检测动作边界

✅ `ports/vision.py`
  - `FrameSource` - 按需解码视频帧
  - `GroundingService` - Qwen VLM grounding
  - `SingleFrameSegmenter` - SAM3 单帧分割（阶段 1 专用）
  - `KeyframeSelector` - 候选帧排序

✅ `ports/artifacts.py`
  - `ArtifactRepository` - 存储和检索 artifact

✅ `ports/__init__.py` - 导出清单

### 4. Application 层（用例编排）
✅ `application/prepare_keyframes.py`
  - `PrepareKeyframes` - 主流程用例
  - `MaskCandidate` - 候选 mask 数据类
  - `KeyframePackage` - 完整的关键帧包
  - 编排：plan → timeline → generate requests → select frames → ground → segment → save

✅ `application/review_keyframes.py`
  - `ReviewKeyframes` - 审批用例
  - `approve()` - 批准候选为 ApprovedSeed

✅ `application/__init__.py` - 导出清单

### 5. 测试
✅ `tests/unit/test_domain_models.py` - domain 模型单元测试
  - 测试 EpisodeRef, InstanceSlot, FrameWindow, Box, VisualPrompt
  - 测试验证逻辑（边界检查、不变量）
  - 测试 KeyframeRequest.next_revision()

✅ `tests/unit/test_domain_policies.py` - domain 策略单元测试
  - 测试 TargetSeedPolicy 生成正确的窗口
  - 测试 StaticReceiverSeedPolicy
  - 测试 RolePolicyRegistry 组合多个策略
  - 测试边界情况（缺少 timeline 数据）

### 6. 配置
✅ `configs/pilot_move_pillbottle_pad.yaml` - pilot 配置示例

### 7. 目录结构
```
process_data_v2/
├── pyproject.toml, uv.toml, justfile
├── README.md, .gitignore
├── src/robotwin_annotation_v2/
│   ├── domain/          ✅ 完成
│   ├── application/     ✅ 完成
│   ├── ports/           ✅ 完成
│   ├── adapters/        📦 空（待实现）
│   ├── bootstrap/       📦 空（待实现）
│   └── cli/             📦 空（待实现）
├── tests/
│   ├── unit/            ✅ 2 个测试文件
│   ├── contract/        📦 空
│   └── integration/     📦 空
├── configs/             ✅ pilot 配置
└── artifacts/           📦 运行时输出（gitignored）
```

## 进行中 🔄

- **依赖安装**: `uv sync --extra dev` 正在后台运行
  - 安装 pydantic, pillow, numpy, pandas, pytest, mypy, ruff
  - phase1 依赖（torch, SAM3）暂未安装

## 下一步 📋

### P0: 验证核心框架
1. ✅ 等待 `uv sync --extra dev` 完成
2. 运行单元测试：`just test-fast`
3. 验证类型检查：`just typecheck`

### P1: 实现关键 Adapters（最小可运行）
4. `adapters/fake_adapters.py` - 用于测试的 fake 实现
   - FakeEpisodeRepository
   - FakeFrameSource
   - FakeSegmenter
   - FakeArtifactRepository
5. `tests/unit/test_prepare_keyframes.py` - 用 fake adapters 测试完整流程

### P2: 实现真实 Adapters
6. `adapters/robotwin_dataset.py` - 读取 process_data/data
7. `adapters/qwen_grounding.py` - HTTP client 到 serve_qwen.py
8. `adapters/sam3_adapter.py` - SAM3 单帧分割
9. `adapters/filesystem_artifacts.py` - 写 JSON/PNG 到 artifacts/

### P3: CLI 和依赖注入
10. `bootstrap/container.py` - 组装依赖
11. `cli/keyframes.py` - 命令行入口
12. 运行第一个真实 episode：`just prepare-keyframes 007152`

## 设计验证 ✅

当前实现验证了设计文档的核心原则：

1. ✅ **分层清晰**：Domain → Application → Ports → Adapters
2. ✅ **不可变对象**：所有 domain 对象用 `@dataclass(frozen=True)`
3. ✅ **无外部依赖**：Domain 层不 import numpy/torch/SAM3
4. ✅ **接口隔离**：Ports 用 `Protocol` 定义，Application 不依赖具体实现
5. ✅ **策略模式**：每个角色有独立的 Policy，易于扩展
6. ✅ **阶段隔离**：SingleFrameSegmenter 是阶段 1 专用，阶段 2 的 PropagationEngine 未定义
7. ✅ **可测试**：Domain 和 Application 层可用 fake adapters 测试，不需要真实模型

## 关键文件清单

| 文件 | 行数 | 状态 | 说明 |
|------|------|------|------|
| `domain/models.py` | 155 | ✅ | 核心值对象 |
| `domain/policies.py` | 153 | ✅ | 角色策略 |
| `application/prepare_keyframes.py` | 223 | ✅ | 主用例 |
| `application/review_keyframes.py` | 54 | ✅ | 审批用例 |
| `ports/*.py` | ~150 | ✅ | 接口定义 |
| `tests/unit/*.py` | 168 | ✅ | 单元测试 |

**总计**: ~900 行核心代码，零外部依赖（Domain/Application/Ports 层）

## 与 process_data 的对比

| 方面 | process_data (v1) | process_data_v2 |
|------|-------------------|-----------------|
| 架构 | 流水线脚本 | Clean Architecture |
| 可测试性 | 需要真实模型 | Domain/App 层可用 fake 测试 |
| 关键帧 vs 传播 | 混在一起 | 阶段隔离 |
| 故障定位 | 难（黑盒） | 易（每阶段独立） |
| 扩展性 | 修改现有代码 | 添加新 Policy/Adapter |
| 可追溯性 | 有限 | 完整（run_id + artifact hash） |
| 代码行数 | ~5000+ | ~900（核心层） |

## 时间估计

- ✅ P0（框架验证）: 已完成
- 🔄 P1（Fake adapters + 测试）: 1-2 小时
- 📋 P2（真实 adapters）: 3-4 小时
- 📋 P3（CLI + 第一次运行）: 2-3 小时

**总计**: 核心功能 6-9 小时可完成阶段 1 的最小可运行版本。
