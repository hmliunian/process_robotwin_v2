# 🎯 当前状态与下一步计划

**更新时间**: 2026-07-29  
**Qwen 服务**: ✅ 运行中（cuda:0, qwen3.5-27b）  

---

## ✅ 已完成

### Phase 1: 核心框架
- ✅ Domain/Ports/Application 层完整实现
- ✅ Fake Adapters + 22 个测试全部通过
- ✅ 完整文档（10 个文件）

### 执行工具任务
- ✅ 成功读取 data_one_task 数据集
- ✅ 提取 9 帧关键帧并可视化
- ✅ 验证 pyav 视频解码工作正常

### Qwen Limitation 分析
- ✅ 识别 3 个核心 limitation
- ✅ 设计改进方案（6 个原则）
- ✅ 完成详细设计文档

---

## 🔍 Qwen Limitation 总结

### 核心问题
1. **Bbox 精度**: Qwen bbox 不精确，text+bbox 可能比 text_only 更差
2. **假阴性**: reject_all 不可靠，可能拒绝正确的 mask
3. **角色推断**: 无法从单帧判断 target vs receiver（需要时序+任务先验）

### 解决方案
> **Qwen 定位为"语义候选生成器"，而不是最终裁判**

- ✅ 角色由任务+状态定义
- ✅ 宽窗口多帧选择
- ✅ Box 作为 soft prompt，保留 text_only
- ✅ reject_all → needs_human_review
- ✅ 时序验证角色一致性
- ✅ 人工最终确认

---

## 📋 下一步：实现真实 Adapters

### P0: 核心 Adapters（必需）

#### 1. RoboTwinDatasetAdapter
```python
# src/robotwin_annotation_v2/adapters/robotwin_dataset.py
class RoboTwinEpisodeRepository:
    """读取 RoboTwin data_one_task 数据集。"""
    
    def load_metadata(self, ref: EpisodeRef) -> dict:
        # 读取 episodes.jsonl
        # 返回 length, tasks, etc.
    
    def load_state(self, ref: EpisodeRef) -> dict:
        # 读取 parquet
        # 返回 gripper state
```

**依赖**: pandas, pyav（已有）  
**测试**: `tests/integration/test_robotwin_dataset.py`

#### 2. RoboTwinFrameSource
```python
class RoboTwinFrameSource:
    """使用 pyav 解码视频并提取帧。"""
    
    def read_frame(self, ref: EpisodeRef, frame_index: int) -> Image.Image:
        # 使用 pyav 解码
        # 缓存常用帧
    
    def get_dimensions(self, ref: EpisodeRef) -> tuple[int, int]:
        # 返回 (240, 320)
```

**依赖**: pyav（已有）  
**测试**: 已在 `tools/test_data_one_task.py` 验证

#### 3. QwenGroundingClient（改进版）
```python
# src/robotwin_annotation_v2/adapters/qwen_grounding.py
class QwenGroundingClient:
    """改进的 Qwen HTTP client with role-aware prompts."""
    
    def __init__(self, base_url: str = "http://localhost:18086"):
        self.base_url = base_url
    
    def ground(
        self,
        frames: list[Image.Image],  # 多帧输入
        text_query: str,
        role: AnnotationRole,
        exclusions: list[str],
    ) -> GroundingResult:
        # 构造角色感知 prompt
        # 调用 Qwen API
        # 返回结构化结果
```

**依赖**: requests（已有）  
**服务**: ✅ 运行中（http://localhost:18086）  
**测试**: `tests/integration/test_qwen_grounding.py`

#### 4. TimelineDetector
```python
class GripperStateTimelineDetector:
    """从 gripper state 检测动作边界。"""
    
    def detect(self, ref: EpisodeRef, state: dict) -> InteractionTimeline:
        # 分析 gripper openness
        # 检测 move_start, close_start, hold, open_start
        # 返回 InteractionTimeline
```

**算法**: 已在 `process_data` 中实现（可复用）  
**测试**: `tests/unit/test_timeline_detector.py`

#### 5. SemanticPlanner
```python
class TaskTextSemanticPlanner:
    """从任务文本提取 target/receiver query。"""
    
    def plan(self, ref: EpisodeRef) -> SemanticPlan:
        # 读取 episodes.jsonl 的 tasks
        # 解析 "move bottle to pad" → target="bottle", receiver="pad"
        # 返回 SemanticPlan
```

**算法**: 简单关键词匹配（可用 LLM 增强）  
**测试**: `tests/unit/test_semantic_planner.py`

#### 6. SAM3Adapter
```python
class SAM3SingleFrameSegmenter:
    """SAM3 single-frame segmentation."""
    
    def __init__(self, checkpoint_path: str):
        # 加载 SAM3 模型
        self.model = load_sam3(checkpoint_path)
    
    def segment(
        self,
        frame: Image.Image,
        prompt: VisualPrompt,
        method: SegmentationMethod,
    ) -> np.ndarray:
        # 调用 SAM3
        # 返回 bool mask
```

**依赖**: torch, SAM3（已有）  
**Checkpoint**: `/DATA/disk8/xuran/add_mask_robotwin/process_data/checkpoints/sam3/sam3.pt`  
**测试**: `tests/integration/test_sam3_adapter.py`

#### 7. FilesystemArtifactRepository
```python
class FilesystemArtifactRepository:
    """保存 artifacts 到文件系统。"""
    
    def create_run(self, config: dict) -> str:
        # 创建 artifacts/data_one_task/run-20260729-xxx/
        # 保存 run_manifest.json
    
    def save_request(self, run_id: str, request: KeyframeRequest, data: dict):
        # 保存 episode_xxx/target_0/request.json
        # 保存 candidates/*.png
```

**依赖**: 标准库（Path, json）  
**测试**: `tests/integration/test_filesystem_artifacts.py`

---

## 🏗️ 实现顺序

### 阶段 1: 数据访问（1-2 小时）
1. RoboTwinEpisodeRepository
2. RoboTwinFrameSource
3. TimelineDetector
4. SemanticPlanner

**输出**: 可以读取数据集并生成 KeyframeRequest

### 阶段 2: 视觉服务（2-3 小时）
5. QwenGroundingClient（改进版）
6. SAM3Adapter
7. KeyframeSelector（简单实现：均匀采样）

**输出**: 可以生成 mask 候选

### 阶段 3: 存储与组装（1 小时）
8. FilesystemArtifactRepository
9. Bootstrap container（依赖注入）
10. CLI 入口

**输出**: 可以运行完整流程

### 阶段 4: 第一次运行（0.5 小时）
11. 运行 episode 000000
12. 生成 mask overlay
13. 验证结果

**输出**: 第一个 mask！

---

## 🎯 当前优先级

### 立即开始（P0）
```bash
# 1. 创建 RoboTwinEpisodeRepository
# 2. 创建 RoboTwinFrameSource（复用 test_data_one_task.py 代码）
# 3. 创建简单的 TimelineDetector（先用固定值）
# 4. 创建简单的 SemanticPlanner（先用硬编码）
```

### 短期目标
- 运行 episode 000000
- 生成 target_0 的 3 个候选 mask
- 保存 contact sheet

### 验收标准
```
artifacts/data_one_task/run-xxx/
└── episode_000000/
    └── target_0/
        ├── request.json
        ├── candidates/
        │   ├── text_only.png
        │   ├── box_only.png
        │   └── text_box.png
        └── contact_sheet.png
```

---

## 📊 资源清单

### 已有资源
- ✅ Qwen 服务：http://localhost:18086
- ✅ SAM3 checkpoint：`../process_data/checkpoints/sam3/sam3.pt`
- ✅ 数据集：`../process_data/data_one_task`
- ✅ 环境：`.venv` 链接到 process_data（已有所有依赖）

### 需要创建
- 📋 7 个 Adapter 实现文件
- 📋 7 个集成测试文件
- 📋 1 个 Bootstrap container
- 📋 1 个 CLI 入口

---

## 💡 关键设计决策

### 1. Qwen 改进
- ✅ 角色感知 prompt
- ✅ 多帧输入，Qwen 选最佳
- ✅ 结构化输出（GroundingResult）
- ✅ Box 作为 soft prompt

### 2. 候选策略
- ✅ 始终保留 text_only
- ✅ 多候选并列（不预先过滤）
- ✅ reject_all → needs_human_review

### 3. 时序验证
- 📋 Phase 1: 先不做（快速迭代）
- 📋 Phase 2: 添加角色一致性检查

---

## 🚀 下一个命令

```bash
# 开始实现第一个 Adapter
cd /DATA/disk8/xuran/add_mask_robotwin/process_data_v2

# 创建 RoboTwinEpisodeRepository
# （复用 tools/test_data_one_task.py 的代码）
```

---

**状态**: 📋 准备开始 Phase 2 实现  
**下一步**: 实现 RoboTwinEpisodeRepository  
**目标**: 运行第一个 episode，生成第一个 mask！
