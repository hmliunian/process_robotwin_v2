# Process Data V2 执行摘要

**状态：设计评审 | 目标：阶段1关键帧验证 | 预计周期：2-3周**

---

## 核心问题：为什么需要 V2？

`process_data/` 的核心痛点：
- **无法回答基本问题**："这张mask是哪一帧、哪个角色、怎么生成的？"
- **质量无法追溯**：不知道mask是关键帧问题还是传播问题
- **无法渐进式验证**：必须跑完整个视频才能发现关键帧选错了

**V2 的核心理念**：先验证关键帧质量，只有被批准的seed才能用于视频传播。

---

## 阶段1范围：只做关键帧，不做视频传播

### ✅ 当前要做的
```
任务: move_pillbottle_pad
相机: cam_high
角色: target_0（药瓶）、receiver_0（垫子）

生成 → 为每个角色找到最清晰的单帧
     → 用3种方法生成mask候选（box_only/text_only/text_box）
     → 自动几何检查 + Qwen语义审核
     
输出 → 可视化contact sheet（原图+3种候选overlay）
     → 结构化的候选包（JSON + PNG）
     → 人工审批接口（approve/reject/request_revision）
```

### ❌ 明确不做的
- ❌ 视频传播（接口已定义，但不实现）
- ❌ 全视频mask的NPZ输出
- ❌ 时序连续性检查
- ❌ 抓取成功/放置成功的QC
- ❌ gripper关键帧（属于阶段1B，target/receiver稳定后再加）

**关键区别**：阶段1的"通过"≠"视频标注通过"，而是**"传播所需的seed已可信、可追溯"**。

---

## 三个核心设计原则

### 1. Keyframe First, 传播 Second
```
❌ 旧流程：选帧 → 传播 → QC失败 → 不知道是帧选错还是传播错
✅ 新流程：选帧 → 人工确认seed正确 → 用已批准seed传播
```

### 2. 角色独立的证据链
不再用"一个episode的大dict"混装所有角色：
```python
# ❌ 旧方式
results = {
    "target_mask": ..., 
    "receiver_mask": ...,  # 不知道来自哪一帧
    "qc_passed": True       # 不知道是谁通过的
}

# ✅ 新方式
target_seed = ApprovedSeed(
    slot="target_0",
    frame_index=49,
    mask_sha256="abc123...",
    approved_by="human_reviewer",
    approval_time="2026-07-29T10:30:00Z"
)
```

### 3. 拒绝和失败是一等结果
```
✅ APPROVED       → 生成ApprovedSeed，可用于传播
✅ REJECTED       → 显式记录原因，不伪造mask
✅ NEEDS_REVIEW   → 保留候选，等待人工决策
❌ 静默fallback  → 绝不允许
```

---

## 关键对象速查

| 对象 | 用途 | 示例 |
|------|------|------|
| `InteractionTimeline` | 动作时间边界，限定搜索窗口 | `close_start=54` → target窗口`[3,53]` |
| `KeyframeRequest` | "为谁在哪找什么类型的帧"的工作单 | `target_0 / PRE_GRASP_VISIBLE` |
| `MaskCandidate` | 单帧mask + 生成方法 + 指标 | frame 49的`text_box`药瓶mask |
| `ApprovedSeed` | **唯一**可用于传播的已确认mask | 人工批准的target_0@frame49 |

**记住**：只有`ApprovedSeed`才能进入阶段2的传播引擎。

---

## 架构分层（依赖只能向内）

```
┌─────────────────────────────────────┐
│  CLI / 人工审批UI                    │
└─────────────┬───────────────────────┘
              ▼
┌─────────────────────────────────────┐
│  application（用例编排）             │
│  PrepareKeyframes, ReviewKeyframes  │
└──────────┬──────────────────────────┘
           │ 调用 Protocol（ports）
    ┌──────┴──────┐
    ▼             ▼
┌─────────┐  ┌──────────────────┐
│ domain  │  │ ports (接口定义)  │
│ 实体+规则│  │ 无具体实现        │
└─────────┘  └─────┬────────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
  RoboTwin适配器      Qwen/SAM3适配器
```

**禁止的依赖**：
- domain不能import SAM3/Qwen/numpy
- application不能硬编码文件路径
- CLI不能直接调用Qwen API

---

## 阶段1工作流程

```
1. 加载episode上下文
   ├─ 语义规划（哪些角色需要mask）
   └─ 时间线检测（动作边界）

2. 为每个角色创建KeyframeRequest
   ├─ target_0:    窗口=[3, 53]  （抓取前可见）
   └─ receiver_0:  窗口=[0, 100] （完整可见，不强制与target同帧）

3. 选帧 + 生成候选
   ├─ 选择最清晰的候选帧（可以多个）
   ├─ Qwen grounding → query + tight bbox
   └─ SAM3生成3种方法的mask
       ├─ box_only
       ├─ text_only
       └─ text_box（text和bbox在同一次SAM3请求）

4. 自动审查
   ├─ 几何检查（空mask、面积异常、bbox重叠度）
   ├─ 语义检查（Qwen判断是否正确物体）
   └─ 输出建议，但不自动批准

5. 人工审批（必需）
   ├─ 查看contact sheet（原图 + 3种overlay）
   └─ 三选一：
       ├─ approve(candidate_id)   → 生成ApprovedSeed
       ├─ reject_all(reason)      → 标记为REJECTED
       └─ request_revision(...)   → 生成新revision的request

6. 输出artifact
   artifacts/keyframes/runs/<run_id>/
     move_pillbottle_pad/episode_007152/cam_high/
       ├─ target_0/
       │   ├─ frame_000049.rgb.png
       │   ├─ text_box.mask.png
       │   ├─ text_box.overlay.png
       │   ├─ contact_sheet.png
       │   ├─ qc.json
       │   └─ review_r001.json
       └─ receiver_0/
           └─ ...
```

**关键**：流程到第5步人工审批为止，不会自动进入传播。

---

## 阶段1验收标准（10条pilot）

在`move_pillbottle_pad/cam_high`的10条episode上：

1. ✅ 每条都有target和receiver的可审阅artifact
2. ✅ 每个`APPROVED` seed可追溯到：
   - 精确帧号 + RGB hash
   - Qwen的query和bbox
   - SAM3的method（box_only/text_only/text_box）
   - candidate mask的SHA-256
   - QC报告 + 人工reviewer身份
3. ✅ `APPROVED` mask经人工逐张确认是正确物体（不是"bbox内的漂亮区域"）
4. ✅ `reject_all`不会伪造selected mask
5. ✅ 测试证明**从未调用**`propagate_in_video`
6. ✅ 旧`process_data/output/`完全不被覆盖

**不验收的内容**（因为不属于阶段1）：
- ❌ 视频传播质量
- ❌ 全视频连续性
- ❌ 抓取/放置成功率

---

## 后续阶段（仅供理解完整图景）

### 阶段1B：gripper关键帧（仍然单帧）
- 加入`gripper_left/right`的两个anchor
- `pre_close_open`（闭合前）+ `post_open`（释放后张开）
- 与target/wrist/forearm做exclusion

### 阶段2：视频传播
```python
# 接口已定义，阶段1不实现
class PropagationEngine(Protocol):
    def propagate(self, request: TrackRequest) -> VideoTrack:
        # 输入必须是ApprovedSeed
        # 输出是新的VideoTrack artifact
        # 不能回写keyframe mask
        ...
```

### 阶段3：全视频QC
- 时序连续性
- 抓取共动
- 放置关系
- 最终NPZ导出

---

## 与旧系统的关系

| 方面 | process_data | process_data_v2 |
|------|--------------|-----------------|
| 项目关系 | 保留，作为参考 | 独立项目，不import旧代码 |
| 输出目录 | `process_data/output/` | `process_data_v2/artifacts/` |
| 冲突 | **绝不覆盖旧输出** | 使用run_id隔离 |
| 复用 | 可参考数据格式和行为 | 通过独立adapter读取RoboTwin数据 |

---

## 技术债务预防

阶段1明确**禁止**的模式（这些是旧系统的痛点）：

❌ 上帝模块：一个`v2_pipeline.py`同时处理读数据、调模型、写NPZ、QC、渲染  
❌ 跨职责dict：同一个dict在模块间传递，不断补字段  
❌ 隐式fallback：失败时静默切换到备选方案，不记录原因  
❌ 文件存在性判断：用`skip-existing`代替artifact version/status  
❌ 角色大分支：target/receiver/gripper在同一函数里用大量if/else处理  
❌ 混淆阶段产物：keyframe包里预先写入video mask的占位字段  

✅ 替代方案：
- 分层架构，依赖只向内
- 类型化的domain对象（dataclass + Enum）
- 显式的状态转换和拒绝结果
- artifact的version和review_status字段
- 策略对象（TargetSeedPolicy/ReceiverSeedPolicy/GripperAnchorPolicy）
- 阶段间用Protocol隔离，当前阶段只定义接口不实现

---

## 实施顺序

### P0：骨架（1-2天）
- 独立`pyproject.toml`
- domain对象 + 状态机
- 用fake adapter写单元测试

### P1：关键帧最小闭环（当前目标，1-2周）
1. RoboTwin单帧读取 + state timeline
2. Target/receiver policy
3. Qwen grounding + SAM3 single-frame
4. Candidate artifact + contact sheet
5. 人工approve/reject/revision
6. **3条消融 → 10条验收**

### P1B：gripper单帧（3-5天）
- 双anchor policy
- Finger/palm exclusion QC

### P2 & P3：传播和视频QC
在P1验收后再开始，避免过早优化。

---

## 快速决策检查清单

如果你需要快速评审本设计，重点看这些：

- [ ] 同意"先验证关键帧，再传播"的理念？
- [ ] 同意阶段1只做target/receiver单帧，不做视频传播？
- [ ] 同意人工审批是必需的（Qwen只做推荐）？
- [ ] 接受独立项目，不在旧pipeline旁叠加功能？
- [ ] 认可ports-and-adapters的分层约束？
- [ ] 同意拒绝结果是一等公民，不静默兜底？

如需更多技术细节，请参阅完整设计文档：`process_data_v2_architecture_design.md`

---

**问题讨论**：xuran  
**审批状态**：待确认  
**下一步**：确认设计 → 开始P0骨架实施
