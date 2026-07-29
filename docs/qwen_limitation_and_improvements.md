# Qwen Limitation 分析与改进方案

## 🔍 Qwen 的核心 Limitation

### 1. Bbox 精度问题
- **问题**: Qwen 的 bbox 不是精确检测框
- **实例**: 药瓶框偏小，导致 text+bbox 会裁掉瓶盖/瓶身
- **根因**: Qwen 是 VLM，不是专门的目标检测器
- **影响**: text+bbox 方法可能比 text_only 更差

### 2. Mask Review 假阴性
- **问题**: Qwen 的 mask review 会产生假阴性
- **实例**: 药瓶的 text_only 目视正确，但 Qwen 仍输出 reject_all
- **根因**: 小图（320×240）、阴影、遮挡、相近物体降低判断稳定性
- **影响**: 不能盲目信任 Qwen 的 reject_all

### 3. 角色推断局限
- **问题**: Qwen 无法仅凭单帧可靠判断 target vs receiver
- **根因**: 角色本质是**动作关系**，不只是外观类别
- **实例**: 药瓶和 pad 在静止帧中没有明显的角色标记
- **影响**: 需要时序信息和任务先验

---

## ✅ 改进方案：Qwen 定位为"语义候选生成器"

### 核心原则
> **Qwen 负责"语义候选"，而不是独自决定角色和最终 mask**

---

## 📋 Target 和 Receiver 的更好标识方案

| 环节 | Target（被移动物） | Receiver（目的地/承接物） |
|------|-------------------|------------------------|
| **角色先验** | 来自任务文本：move bottle to pad 中 bottle 是 target | pad 是 receiver |
| **选帧策略** | 选择夹爪闭合前、物体完整可见的早期帧 | 选择开始阶段、未被瓶子遮挡的稳定帧 |
| **动作证据** | 应靠近闭合夹爪，hold 后与夹爪一起移动 | 应在抓取前已存在，通常相对桌面静止 |
| **Qwen Prompt** | "将被左夹爪抓取并移动的白色药瓶，不要选 pad/阴影" | "药瓶最终要放到的蓝色方形 pad，不要选药瓶" |
| **分割候选** | text_only、text+bbox、扩框候选并列 | 同上 |
| **最终决定** | 候选 sheet 人工/Qwen 复核，但人工可覆盖 | 同上 |

---

## 🎯 推荐的 V4.1 原则

### 1. 角色由任务+状态定义，Qwen 不负责推断
```python
# ✅ 正确：任务解析先定义角色
semantic = SemanticPlan(
    episode=ref,
    target_query="white pill bottle",  # 从任务文本提取
    receiver_query="blue square pad",  # 从任务文本提取
    has_static_receiver=True,
)

# ❌ 错误：让 Qwen 从零推断"哪个是 target"
# Qwen 无法仅凭外观判断动作角色
```

### 2. 宽范围多帧选择
```python
# ✅ 正确：宽窗口，选清晰 seed
target_window = FrameWindow(first=move_start, last=close_start - 1)  # [10, 49]

# ❌ 错误：只给临近抓取、已经遮挡的帧
target_window = FrameWindow(first=close_start - 5, last=close_start)  # [45, 50]
```

### 3. Qwen 返回结构化信息
```python
@dataclass
class GroundingResult:
    """Qwen grounding 返回的完整信息。"""
    refined_query: str           # 精确外观描述
    selected_frame: int          # Qwen 选中的最佳帧
    bbox: Box                    # 粗略 bbox（仅作参考）
    rationale: str               # 为什么它是 target/receiver
    exclusions: list[str]        # 排除对象（例如 target 排除 pad）
    confidence: float            # Qwen 的置信度
```

### 4. Box 只是 Prompt Hypothesis
```python
# ✅ 正确：保留 text_only，不因超出 box 就拒绝
candidates = [
    segment(prompt=VisualPrompt(text=query), method=TEXT_ONLY),      # 保留
    segment(prompt=VisualPrompt(bbox=bbox), method=BOX_ONLY),
    segment(prompt=VisualPrompt(text=query, bbox=bbox), method=TEXT_BOX),
]

# ❌ 错误：只因 mask 超出 Qwen box 就自动拒绝
if mask_area_outside_box > threshold:
    reject_candidate()  # 错误！text_only 可能是对的
```

### 5. 角色一致性检查（时序验证）
```python
def verify_target_role(mask_sequence: list[np.ndarray], timeline: InteractionTimeline) -> bool:
    """验证 target 角色的时序一致性。"""
    # Target 应该：
    # 1. pre-grasp 可见
    # 2. close 附近接近夹爪
    # 3. hold 后位置变化
    
    pre_grasp_visible = check_visibility(mask_sequence[:timeline.close_start])
    near_gripper_at_close = check_proximity_to_gripper(mask_sequence[timeline.close_start])
    moves_during_hold = check_position_change(mask_sequence[timeline.hold_start:timeline.hold_end])
    
    return all([pre_grasp_visible, near_gripper_at_close, moves_during_hold])

def verify_receiver_role(mask_sequence: list[np.ndarray], timeline: InteractionTimeline) -> bool:
    """验证 receiver 角色的时序一致性。"""
    # Receiver 应该：
    # 1. pre-grasp 可见
    # 2. 抓取期间相对静止
    # 3. release 后 target 接近/落在其上
    
    pre_grasp_visible = check_visibility(mask_sequence[:timeline.move_start])
    static_during_grasp = check_static(mask_sequence[timeline.move_start:timeline.hold_end])
    target_lands_on_receiver = check_spatial_relation(mask_sequence[timeline.open_start:])
    
    return all([pre_grasp_visible, static_during_grasp, target_lands_on_receiver])
```

### 6. reject_all → needs_human_review
```python
# ✅ 正确：转为人工审查
if qwen_review_result == "reject_all":
    candidates[i].status = ReviewStatus.NEEDS_HUMAN_REVIEW
    candidates[i].note = "Qwen 建议拒绝，但保留供人工确认"

# ❌ 错误：直接删除候选
if qwen_review_result == "reject_all":
    del candidates[i]  # 错误！可能是假阴性
```

---

## 🏗️ 实现架构

### Qwen Grounding Service 改进

```python
class QwenGroundingService:
    """Qwen grounding service with structured output."""
    
    def ground(
        self,
        frames: list[Image.Image],  # 多帧输入
        text_query: str,
        role: AnnotationRole,
        exclusions: list[str],
    ) -> GroundingResult:
        """
        Qwen grounding with role-aware prompt.
        
        Args:
            frames: 候选帧列表（从宽窗口选择）
            text_query: 基础查询（例如 "bottle"）
            role: target 或 receiver
            exclusions: 排除对象（例如 target 排除 "pad", "shadow"）
        
        Returns:
            GroundingResult with frame selection, refined query, bbox, rationale
        """
        
        # 构造角色感知 prompt
        if role == AnnotationRole.TARGET:
            prompt = f"""
            任务：找到将被机械臂抓取并移动的物体。
            
            描述：{text_query}
            
            要求：
            1. 选择物体完整可见、未被遮挡的帧
            2. 物体应该靠近将要闭合的夹爪
            3. 不要选择：{', '.join(exclusions)}
            4. 返回：最佳帧索引、精确外观描述、粗略边界框
            """
        else:  # RECEIVER
            prompt = f"""
            任务：找到物体将被放置到的目标位置/承接物。
            
            描述：{text_query}
            
            要求：
            1. 选择目标位置完整可见、未被遮挡的帧
            2. 目标位置应该在抓取开始前已存在
            3. 不要选择：{', '.join(exclusions)}
            4. 返回：最佳帧索引、精确外观描述、粗略边界框
            """
        
        # 调用 Qwen HTTP API
        response = self.client.post(
            url=f"{self.base_url}/ground",
            json={
                "frames": [encode_image(f) for f in frames],
                "prompt": prompt,
            }
        )
        
        return GroundingResult(
            refined_query=response["refined_query"],
            selected_frame=response["selected_frame"],
            bbox=Box(**response["bbox"]),
            rationale=response["rationale"],
            exclusions=exclusions,
            confidence=response["confidence"],
        )
```

---

## 📊 对比：当前 vs 改进

| 方面 | 当前方法 | 改进方法 |
|------|---------|---------|
| **角色判断** | Qwen 从单帧推断 | 任务+状态先验 |
| **选帧** | 固定窗口单帧 | 宽窗口多帧，Qwen 选最佳 |
| **Bbox 使用** | 作为硬约束 | 作为 soft prompt |
| **Text_only** | 可能被 box 误导丢弃 | 始终保留 |
| **reject_all** | 直接删除 | 转为 needs_human_review |
| **时序验证** | 无 | 角色一致性检查 |

---

## 🎯 最可靠的组合（针对 pillbottle_pad）

```
任务语义确定 bottle/pad 角色
  ↓
宽范围 early frame 选帧（move_start 到 close_start）
  ↓
Qwen 精确文本 + 粗略 bbox
  ↓
SAM3 多候选（text_only, box_only, text+box）
  ↓
角色一致性时序检查（可选）
  ↓
人工确认（contact sheet）
```

---

## 🚀 实现优先级

### P0: 立即改进
1. ✅ Qwen prompt 添加角色信息和排除对象
2. ✅ 保留 text_only 候选，不因超出 box 拒绝
3. ✅ reject_all → needs_human_review

### P1: 短期改进
4. 📋 宽窗口多帧输入，Qwen 选最佳帧
5. 📋 结构化 GroundingResult（rationale, confidence）
6. 📋 任务解析自动提取 target/receiver

### P2: 长期增强
7. 📋 角色一致性时序检查
8. 📋 Qwen 不确定时请求更多上下文帧
9. 📋 多模态融合（Qwen + SAM3 + 夹爪状态）

---

## 💡 关键洞察

> **Qwen 的价值在于语义理解和外观精化，而不是精确定位和角色推断。**

- ✅ Qwen 擅长：文本精化（"bottle" → "white bottle with orange cap"）
- ✅ Qwen 擅长：多帧选择（选择最清晰、最完整的帧）
- ❌ Qwen 不擅长：精确 bbox（比专门检测器差）
- ❌ Qwen 不擅长：单帧角色判断（需要时序+任务先验）

**因此，让每个组件做它最擅长的事：**
- 任务解析 → 角色先验
- 夹爪状态 → 时序窗口
- Qwen → 语义理解 + 帧选择
- SAM3 → 精确分割
- 人工 → 最终确认

---

**创建日期**: 2026-07-29  
**基于**: 实际 ep0 观察和用户洞察  
**状态**: 设计完成，待实现
