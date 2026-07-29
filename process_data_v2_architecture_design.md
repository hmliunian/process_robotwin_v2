# `process_data_v2` 架构设计：以关键帧为中心的可扩展 Mask 标注系统

> **状态：设计评审稿；尚未在 `process_data_v2/` 实现任何代码。**
> **当前唯一实施目标：阶段 1——产出、审阅并确认可信的单帧 keyframe mask。**
> 后续的视频传播与整段视频 QC 只在接口和数据契约中预留，**本轮不实现、不运行，也不以它们验收阶段 1**。

---

## 1. 要解决的问题与本次边界

`process_data/` 已经同时承载了文本解析、事件检测、Qwen grounding、SAM3、CoTracker、
mask 后处理、QC、渲染和许多历史实验入口。功能虽多，但一个脚本/函数往往跨越多个职责；
同一份 `dict` 在不同模块之间不断补字段，导致人很难回答下面最基本的问题：

- 这张 mask 是哪一帧、哪个角色、由什么提示生成的？
- 它是否已经被人/模型审核为正确的关键帧 mask？
- 这是关键帧本身的问题，还是视频传播后才产生的问题？
- 后续传播时，究竟应该使用哪一个已经确认的 seed？

`process_data_v2` 不应继续在旧流水线旁边叠加一个更大的 `pipeline.py`。新设计以
**“可审核的关键帧标注包（Keyframe Package）”** 为第一等产物：先确认每个角色的 seed
mask，只有被批准的 seed 才能进入未来的视频传播。

### 1.1 本轮明确做与不做

| 项目 | 阶段 1（当前） | 阶段 2 以后 |
|---|---|---|
| 任务语义、角色槽位、动作边界 | 做；只作为 keyframe 选择的上下文 | 复用，不重新猜测 |
| target / receiver 单帧 mask | 做；生成候选、QC、人工确认 | 作为传播 seed |
| gripper 单帧锚点 | 接口预留；在 target/receiver 验收后作为阶段 1B 增量加入 | 用双锚点传播和动态 ROI |
| `text + bbox` 单帧 SAM3 | 做 | 可继续作为重新播种手段 |
| `propagate_in_video` / CoTracker | **禁止调用** | 做 |
| 全视频 `masks[T,H,W]`、NPZ | **不写出** | 做 |
| 连续性、抓取成功、放置成功 QC | **不做** | 阶段 3 做 |
| keyframe 几何/语义/人工 QC | 做，且是阶段 1 的验收核心 | 保留 |

因此，阶段 1 的“通过”不表示“视频标注通过”；它只表示：**未来传播所需的 seed 已经可信、
可追溯、可复现。**

### 1.2 首个 pilot 和逐步收窄的范围

首个可运行范围固定为：

```text
coarse_task = move_pillbottle_pad
camera      = cam_high
实例         = target_0（药瓶）、receiver_0（pad）
```

这与 `docs/v4_1_keyframe_mask_design.md` 一致。目标和 receiver 不要求使用同一帧：
药瓶应选夹爪闭合前无遮挡帧，pad 应选自身完整、未被遮挡帧。

在这两个角色稳定后，仍属于**阶段 1**的下一小步是加入：

```text
gripper_left / gripper_right
anchor_kind = pre_close_open | post_open
```

夹爪和物体的视觉定义、候选生成和 QC 规则不同，不能为了“统一”而把它们塞进 target 的逻辑。
但它们将使用同一套 `KeyframeRequest → Candidate → Review → ApprovedSeed` 契约。

---

## 2. 总体原则与关键决策

1. **Keyframe first，传播 second。** 传播服务的输入只能是已批准的 seed；它不能为了掩盖
   失败而默默重新选择关键帧。
2. **动作锚点与 mask seed 分开。** `close/open/move` 是由 state 得到的事件边界；mask seed
   是在合法窗口内视觉上最清晰的一帧。两者可能相邻，但不是同一个概念。
3. **一个角色实例一份独立的证据链。** 不再用“本 episode 的一个大结果对象”混装角色、
   phase、mask、QC 和渲染状态。
4. **单帧的语义正确性优先于几何漂亮。** bbox 内、面积合理只能排除明显错误；不能证明
   分到的是药瓶或 pad。阶段 1 必须保存候选 overlay，并允许 `reject_all`。
5. **Qwen 是 reviewer/grounder，不是不可质疑的真值。** 初期只有人工确认才能把 seed
   提升为 `APPROVED`；以后若要开放自动批准，必须单独配置并保留同样的证据。
6. **外部模型都放在 adapter 后面。** domain/application 不 import SAM3、Qwen、CoTracker、
   OpenCV 或 `numpy` 图像操作代码。
7. **输出是不可变 run artifact。** 每次运行有 `run_id`、配置/model/input hash；批准或拒绝
   是新的 review revision，不覆盖历史候选，也绝不覆盖 `process_data/output/`。
8. **不依赖旧代码作为运行时核心。** `process_data/` 只作为行为和数据格式的参考。v2 通过
   独立 adapter 读取 RoboTwin 数据，不能 import 旧的巨型 pipeline 来“复用”。

特别保留 v4.1 已验证的技术决策：对 `text_box` 候选，SAM3 必须在**同一次**
`add_prompt` 请求里收到 `text + bounding_boxes`；不能先发 text、再发 bbox，也不能在阶段 1
调用视频传播 API。

---

## 3. 术语和状态模型

### 3.1 四类容易混淆的对象

| 名称 | 含义 | 示例 |
|---|---|---|
| `InteractionTimeline` | state/VLM 给出的动作上下文和合法时间窗，不含像素 mask | `close_start=54`、target seed window=`[3,53]` |
| `KeyframeRequest` | “为哪个角色在什么窗口寻找什么类型的关键帧”的工作单 | `target_0 / PRE_GRASP_VISIBLE` |
| `MaskCandidate` | 某个候选帧、某种提示方法生成的一张单帧 mask 及指标 | frame 49 的 `text_box` 药瓶 mask |
| `ApprovedSeed` | 已经通过 review 的、允许视频传播消费的不可变选择 | `target_0` 在 frame 49 的 mask |

`InteractionTimeline` 不是 QC 结果；`MaskCandidate` 不是已确认标注；只有
`ApprovedSeed` 才是未来传播的输入。

### 3.2 生命周期

```text
DRAFT request
  → CANDIDATES_GENERATED
  → AUTO_REVIEWED
  → NEEDS_HUMAN_REVIEW ──→ REJECTED
          │
          └────────────────→ APPROVED → ApprovedSeed（阶段 2 的唯一合法输入）
```

- 模型/几何检查可以建议 `selected_candidate`，但不能跳过 `NEEDS_HUMAN_REVIEW`。
- `reject_all`、空 mask、无合法 seed frame 都是正常且显式的 `REJECTED`/`BLOCKED` 结果，
  不是静默 fallback。
- 修正 query、bbox 或 frame 后创建新的 request revision；旧 artifact 保留。

---

## 4. 分层架构（面向对象，但不过度抽象）

采用轻量的 **ports-and-adapters / clean architecture**。核心依赖方向只能向内：

```text
CLI / 配置 / 人工 review UI
             │
             ▼
┌─────────────────────────────────────────────┐
│ application：Use Case / Workflow             │
│ PrepareKeyframes, ReviewKeyframes, ...       │
└──────────────────┬──────────────────────────┘
                   │ 依赖 Protocol（ports）
        ┌──────────┴──────────┐
        ▼                     ▼
┌───────────────┐      ┌──────────────────────┐
│ domain        │      │ ports                │
│ 实体、状态、   │      │ Dataset, Grounder,   │
│ policy、规则   │      │ Segmenter, Store...  │
└───────────────┘      └──────────┬───────────┘
                                  │
                     ┌────────────┴────────────────────────┐
                     ▼                                     ▼
          RoboTwin / 文件系统 adapter          Qwen / SAM3 / 渲染 adapter
```

### 4.1 每层允许与禁止的内容

| 层 | 负责什么 | 绝不负责什么 |
|---|---|---|
| `domain` | ID、状态转换、角色规则、合法窗口、审批不变量 | 文件路径、JSON、HTTP、模型调用、图像 ndarray 操作 |
| `application` | 编排一个可解释的用例，调用 port，构造 artifact | 硬编码 Qwen/SAM3 API、CLI 参数解析 |
| `ports` | 用 `Protocol` 定义外部能力的输入/输出 | 具体模型和磁盘实现 |
| `adapters` | RoboTwin 读取、Qwen grounding、SAM3 单帧分割、文件落盘、渲染 | 业务状态机和角色决策 |
| `cli` | 参数、依赖组装、退出码、显示 run id | annotation 业务逻辑 |

这不是为了把每个函数都包成 class。**纯计算**（例如 bbox IoU、mask 面积、JSON schema
验证）可保留为小型无状态函数；只有具有状态、策略差异或外部依赖的职责才建对象。

---

## 5. 核心领域对象

领域对象使用不可变 `dataclass`、`Enum` 和显式类型，而不是跨模块传递
`dict[str, Any]`。序列化/反序列化只发生在 adapter 边界。

```python
# domain/models.py（示意，不是本轮实现）
class AnnotationRole(StrEnum):
    TARGET = "target"
    RECEIVER = "receiver"
    GRIPPER = "gripper"

class AnchorKind(StrEnum):
    PRE_GRASP_VISIBLE = "pre_grasp_visible"
    STATIC_RECEIVER_VISIBLE = "static_receiver_visible"
    PRE_CLOSE_OPEN = "pre_close_open"
    POST_OPEN = "post_open"

class ReviewStatus(StrEnum):
    DRAFT = "draft"
    NEEDS_HUMAN_REVIEW = "needs_human_review"
    APPROVED = "approved"
    REJECTED = "rejected"

@dataclass(frozen=True)
class InstanceSlot:
    name: str                  # target_0, receiver_0, gripper_left
    role: AnnotationRole
    arm: Literal["left", "right"] | None = None

@dataclass(frozen=True)
class FrameWindow:
    first: int                 # inclusive
    last: int                  # inclusive

@dataclass(frozen=True)
class KeyframeRequest:
    request_id: str
    episode: EpisodeRef
    slot: InstanceSlot
    anchor_kind: AnchorKind
    allowed_window: FrameWindow
    visual_query: str
    exclusions: tuple[str, ...]
    revision: int

@dataclass(frozen=True)
class ApprovedSeed:
    request_id: str
    candidate_id: str
    frame_index: int
    slot: InstanceSlot
    mask_artifact: MaskArtifactRef
    approval_revision: int
```

其中 `MaskArtifactRef` 只是 artifact 的稳定引用和 hash，不把图像数组塞进 domain 对象。像素
mask 和几何统计属于 vision adapter 的输出 `SegmentationCandidate`，再由 application 写入
artifact。

### 5.1 角色不靠大量 `if/else`，而靠策略对象

三种角色确实有不同的 seed 规则，因此使用小而明确的策略，而不是一个越来越长的
`generate_pilot_masks()`：

```python
class KeyframePolicy(Protocol):
    def create_requests(
        self, semantic: SemanticPlan, timeline: InteractionTimeline
    ) -> list[KeyframeRequest]: ...

class TargetSeedPolicy(KeyframePolicy):
    # pre-grasp、可见、与夹爪重叠尽量小

class StaticReceiverSeedPolicy(KeyframePolicy):
    # receiver 完整且无遮挡；不要求与 target 同帧

class GripperAnchorPolicy(KeyframePolicy):
    # 后续 1B：pre-close open + post-open 两个 anchor，带 target/wrist exclusion
```

`RolePolicyRegistry` 根据 `InstanceSlot` 选择 policy。新增“动态 receiver”或“relation place”
时新增对应 policy，不修改 target/receiver 的已有逻辑。

### 5.2 关键对象的拥有关系

```text
EpisodeContext
 ├─ SemanticPlan                 # target/receiver 的文本角色，不含 mask
 ├─ InteractionTimeline          # 事件边界和窗口
 └─ KeyframePackage
     ├─ KeyframeRequest (1..N)
     │   ├─ FrameCandidate (0..N)
     │   ├─ GroundingEvidence
     │   ├─ SegmentationCandidate (0..N)
     │   └─ KeyframeReview
     └─ ApprovedSeed (0..N)
```

一个 episode 没有 `ApprovedSeed` 是有效结果；系统不能为了凑齐 channel 而伪造 mask。

---

## 6. 外部能力接口（Ports）

下表是 v2 的稳定边界。实现可以替换，但 application 对它们的调用方式不变。

| Port | 关键方法 | 阶段 1 实现 | 后续用途 |
|---|---|---|---|
| `EpisodeRepository` | `load_context(ref)` | RoboTwin metadata/state/video 索引 adapter | 全流程共用 |
| `FrameSource` | `read(frame_index)` | 按需解码单帧，带 cache | 视频传播可读取 clip |
| `SemanticPlanner` | `plan(context)` | Qwen/规则 adapter | 复用，不重新猜角色 |
| `TimelineDetector` | `detect(context)` | gripper/EEF state adapter | 给 propagation/QC 时间窗 |
| `KeyframeSelector` | `rank(request, frames)` | 清晰度、遮挡、合法窗口策略 | 重试时复用 |
| `GroundingService` | `ground(frame, request)` | Qwen 输出 query + tight bbox | 重新播种 |
| `SingleFrameSegmenter` | `segment(frame, prompt, method)` | SAM3 one-frame adapter | 阶段 1 专用 |
| `KeyframeReviewer` | `review(request, candidates)` | 几何 + Qwen reviewer；人工为最终 gate | 可复审历史包 |
| `ArtifactRepository` | `save/load(package)` | 文件系统、JSON、PNG/NPZ/RLE adapter | 所有阶段共用 |
| `PropagationEngine` | `propagate(track_request)` | **仅定义 Protocol** | 阶段 2 才实现 |
| `VideoQCService` | `evaluate(video_run)` | **仅定义 Protocol** | 阶段 3 才实现 |

`SingleFrameSegmenter` 和未来的 `PropagationEngine` 必须是两个 port。这样可从类型和测试上
防止“为了生成 keyframe 顺便跑了一遍全视频”。

一个建议的 prompt 类型如下：

```python
@dataclass(frozen=True)
class VisualPrompt:
    text: str | None
    bbox_xyxy_normalized: Box | None
    positive_points: tuple[Point, ...] = ()
    negative_points: tuple[Point, ...] = ()

class SegmentationMethod(StrEnum):
    BOX_ONLY = "box_only"
    TEXT_ONLY = "text_only"
    TEXT_BOX = "text_box"
```

阶段 1 对每个 instance 可以导出 `box_only / text_only / text_box` 三种候选；`TEXT_BOX` 的
adapter 将 text 与 bbox 合为一次 SAM3 请求。候选比较本身不改变原图，也不触发传播。

---

## 7. 阶段 1 的用例流程

### 7.1 `PrepareKeyframes`：唯一的主流程

```text
EpisodeRef
  │
  ├─ EpisodeRepository.load_context
  ├─ SemanticPlanner.plan                    → SemanticPlan
  ├─ TimelineDetector.detect                 → InteractionTimeline
  ├─ RolePolicyRegistry.create_requests      → KeyframeRequest[]
  │       （每个角色有自己的合法 seed window）
  ├─ KeyframeSelector.rank                   → FrameCandidate[]
  ├─ GroundingService.ground                 → query + tight bbox evidence
  ├─ SingleFrameSegmenter.segment × methods  → MaskCandidate[]
  ├─ KeyframeReviewer.auto_review            → 建议 / reject_all / QC report
  ├─ ArtifactRepository.save_candidate_package
  └─ 返回 NEEDS_HUMAN_REVIEW，不自动进入传播
```

关键点：

- 先获得 `InteractionTimeline`，是为了限定搜索窗口，并不是开始做全视频 QC。
- `KeyframeSelector` 可以提出多个 frame；grounding 和分割候选均要记录各自 frame，
  不允许把不同帧的 bbox/mask 混在一起。
- 一个 receiver 可选比 target 更早或更晚的帧；不共享“episode 的唯一 seed frame”。
- 任意关键步骤失败只产生可读 failure artifact（例如 `no_clear_frame`、`reject_all`），
  不写一个假 selected mask。

### 7.2 `ReviewKeyframes`：审批与版本化

人工 review UI/CLI 读取 contact sheet，并只能做三种动作：

```text
approve(candidate_id, reviewer, note)
reject_all(reason, reviewer)
request_revision(reason, reviewer)  # 生成下一 revision 的 request，不改旧记录
```

`approve` 会创建 `ApprovedSeed`；阶段 2 查询时只接受这个对象。Qwen 的推荐结果、阈值、
reviewer、时间、候选 hash 都进入 `review.json`。

### 7.3 阶段 1 的 keyframe 内容定义

| slot / anchor | 合法候选窗口 | 要生成的内容 | 必须排除/注意 |
|---|---|---|---|
| `target_0 / pre_grasp_visible` | `[t_move_start, t_close_start)` | 完整可见药瓶、精确 query、tight bbox、单帧候选 mask | 夹爪主体、邻近同类、背景；不要求抓取后继续可见 |
| `receiver_0 / static_receiver_visible` | 优先 action 前、receiver 完整可见的窗口 | 完整 pad 区域、内部点、单帧候选 mask | target/夹爪遮挡；不强迫与 target 同帧 |
| `gripper_<arm> / pre_close_open`（1B） | 接近 close 前 | 两指及必要短掌部的 anchor | target、wrist/forearm、另一臂 |
| `gripper_<arm> / post_open`（1B） | `t_open_done` 附近 | 释放后重新张开的第二 anchor | 同上；不能用一个宽大 arm bbox 替代 |

`target_0` 和 `receiver_0` 是 1A 的最小闭环。只有它们的 contact sheet 和人工 review
稳定后，才引入视觉上明显更难的 gripper anchors；不会让 gripper 问题掩盖物体 seed 的问题。

---

## 8. 阶段 1 的 QC：只检查关键帧，不越界检查整段视频

`KeyframeReviewer` 组合三个独立 checker，输出结构化 `KeyframeQCReport`。每条 finding 必须
带 `severity`、`evidence`、`candidate_id` 与可读 reason。

| Checker | 当前是否执行 | 典型问题 | 结论 |
|---|---:|---|---|
| `RequestContractChecker` | 是 | frame 不在窗口、角色/相机/尺寸不匹配 | hard fail |
| `PromptContractChecker` | 是 | `text_box` 没有在同一 SAM3 request 发送 text+bbox | hard fail |
| `MaskGeometryChecker` | 是 | 空 mask、面积极端、主连通域异常、与 bbox 交集太小 | reject / warn |
| `RoleLocalChecker` | 是 | 药瓶 mask 覆盖背景；pad 不完整；gripper 含长前臂 | reject / needs review |
| `SemanticCandidateReviewer` | 是 | 几何过关但分到错误物体 | 建议候选或 `reject_all` |
| `HumanApprovalChecker` | 是（初期） | Qwen 不确定、三个候选都错、部分遮挡 | 最终 approve/reject |
| `TemporalContinuityChecker` | **否** | mask 是否连续、是否漂移 | 阶段 3 |
| `CoMotion/PlaceChecker` | **否** | 是否抓住、是否放到 pad | 阶段 3 |
| `VideoCoverageChecker` | **否** | 全视频是否有缺口 | 阶段 3 |

几何检查只能作**拒绝器**，不是正确性证明。例如“mask 大部分在 Qwen bbox 内”只能说明它
没有明显跑远，不能证明那就是药瓶。因此，阶段 1 artifact 始终保留原图、每种候选 overlay、
contact sheet 与 reviewer 的理由。

### 8.1 阶段 1 的验收标准

以 `move_pillbottle_pad / cam_high` 的 10 条 pilot 为准：

1. 每条 episode 均有 target 与 receiver 的可审阅 artifact，成功或显式拒绝均可；
2. 每个 `APPROVED` seed 都可追溯到 exact frame、RGB hash、query、bbox、SAM3 method、
   candidate mask hash、QC 和人工 reviewer；
3. `APPROVED` mask 由人工逐张确认是目标物/完整 receiver，而非“bbox 内的漂亮区域”；
4. `reject_all` 不会输出伪造 selected mask；
5. 自动与单元/contract 测试证明阶段 1 **从未调用** `propagate_in_video`，也不生成
   `[T,H,W]` 全视频 mask 或 video QC 结论；
6. 旧的 `process_data/output/key_masks*` 和 annotations 完全不被覆盖。

---

## 9. Artifact 契约与目录布局

所有输出按 run 隔离，默认不纳入 Git。下面是计划中的结构，不代表本轮创建这些文件：

```text
process_data_v2/
  artifacts/                                      # .gitignore
    keyframes/
      runs/<run_id>/
        run_manifest.json
        move_pillbottle_pad/
          episode_007152/
            cam_high/
              episode_context.json                # 输入引用/哈希、image size
              semantic_plan.json
              interaction_timeline.json
              keyframe_package.json               # requests、状态、引用
              target_0/
                request_r001.json
                frame_000049.rgb.png
                grounding.json
                box_only.mask.png
                text_only.mask.png
                text_box.mask.png
                box_only.overlay.png
                text_only.overlay.png
                text_box.overlay.png
                contact_sheet.png
                qc.json
                review_r001.json
              receiver_0/
                ...
```

`run_manifest.json` 至少记录：

```json
{
  "format": "robotwin_keyframe_run/v1",
  "run_id": "kf-20260729-...",
  "algorithm_version": "process_data_v2-keyframe-v1",
  "stage": "keyframe",
  "video_propagation": false,
  "video_qc": false,
  "input": {"dataset_root": "...", "episode_hash": "..."},
  "models": {"qwen": "...", "sam3_checkpoint": "..."},
  "config_sha256": "..."
}
```

`keyframe_package.json` 不嵌入 mask 像素；它引用文件和 SHA-256。每一个 candidate 至少有：

```json
{
  "candidate_id": "target_0-r001-f0049-text_box",
  "slot": "target_0",
  "anchor_kind": "pre_grasp_visible",
  "frame_index": 49,
  "image_size_hw": [240, 320],
  "query": "white pill bottle with orange label",
  "bbox_xyxy_normalized": [0.46, 0.32, 0.61, 0.62],
  "method": "text_box",
  "video_propagation": false,
  "mask_sha256": "...",
  "metrics": {"area_fraction": 0.012, "bbox_overlap": 0.91},
  "review_status": "needs_human_review"
}
```

原则：**阶段 1 写的是单帧证据包，不是 `masks.npz` 的半成品。** 全视频数据格式和
`frame_provenance` 留给阶段 2 的独立 artifact schema，避免把两类产物混淆。

---

## 10. 未来视频传播如何接入，而不破坏阶段 1

阶段 2 的接口现在定义、以后实现：

```python
@dataclass(frozen=True)
class TrackRequest:
    approved_seed: ApprovedSeed
    tracking_window: FrameWindow
    strategy: PropagationStrategyName
    exclusions: tuple[MaskArtifactRef, ...]

class PropagationEngine(Protocol):
    def propagate(self, request: TrackRequest) -> VideoTrack: ...
```

### 10.1 传播服务的硬边界

- 输入必须是 `ApprovedSeed`，不能直接消费某个 Qwen bbox 或 `MaskCandidate`。
- 输出是新的 `VideoTrack` artifact，逐帧记录 `FrameMask`、`MaskProvenance` 和失败片段；
  绝不回写/修改 keyframe mask。
- 若传播漂移，产生 `RecoveryRequest`（需要新关键帧或第二 anchor），而不是在传播模块中
  偷偷重新 grounding。
- `VideoQCService` 只读取 `VideoTrack + InteractionTimeline + ApprovedSeed`；它不负责
  重新定义关键帧正确性。

### 10.2 首轮传播策略的预留

| 角色 | 阶段 2 的候选 strategy | 阶段 1 需要先准备什么 |
|---|---|---|
| target | SAM3 在 `target_window` 内传播；必要时多 anchor | `pre_grasp_visible` 已批准 seed |
| 静态 receiver | `StaticMaskReplicator` 在固定相机有效窗口复制已批准 mask | 静态 receiver seed + camera 静态性证据 |
| gripper | 双 anchor + CoTracker 动态 finger ROI + wrist/target exclusion | `pre_close_open`、`post_open` 两个已批准 seed |
| 动态 receiver / relation place | 新的 role policy 和 strategy | 不进入当前 pilot |

因此“考虑视频 mask 扩展”不等于现在提前把传播、persistence 和全视频 QC 混进关键帧代码；
而是确保 seed 的 ID、窗口、角色、anchor kind、exclusion 和 revision 都足以让后续模块使用。

### 10.3 阶段 3 的视频 QC（现在只定义责任）

未来 `VideoQCService` 将分为：

1. **结构 QC**：时间窗外为空、active/inactive arm、artifact 完整性；
2. **时序 QC**：coverage、断裂、漂移、provenance 比例；
3. **跨角色 QC**：target/gripper 排斥、动态 ROI、receiver 稳定性；
4. **任务结果 QC**：抓取共动、release 后 target 与 receiver 的关系；
5. **人工复核选择**：只从 QC 标记的可疑帧生成 review package。

这与当前 keyframe QC 的输入、失败原因和验收目标完全不同，必须保持成两个服务。

---

## 11. 计划中的目录与模块划分

`process_data_v2` 初始项目结构建议如下。`robotwin_annotation_v2` 是新的 Python package 名，
避免与旧 `robotwin_annotate` 发生 import/产物混淆。

```text
process_data_v2/
  pyproject.toml
  README.md
  src/robotwin_annotation_v2/
    domain/
      models.py                 # EpisodeRef, Slot, Window, Request, Review state
      policies.py               # Target/Receiver/Gripper keyframe policy
      errors.py
    application/
      prepare_keyframes.py      # PrepareKeyframes use case
      review_keyframes.py       # approve/reject/revision use case
      dto.py
    ports/
      dataset.py
      vision.py                 # frame source, grounding, single-frame segmenter
      artifacts.py
      propagation.py            # Protocol only in phase 1
      video_qc.py               # Protocol only in phase 1
    adapters/
      robotwin_dataset.py
      qwen_grounding.py
      sam3_single_frame.py
      filesystem_artifacts.py
      image_rendering.py
      human_review.py           # CLI/HTML review input adapter
    bootstrap/
      container.py              # 唯一 composition root
      settings.py
    cli/
      keyframes.py              # thin CLI only
  configs/
    pilot_move_pillbottle_pad.yaml
    keyframe_thresholds.yaml
  tests/
    unit/
    contract/
    integration/
    fixtures/
  artifacts/                    # ignored runtime output
```

### 11.1 禁止重新出现的结构问题

- 不建 `v2_pipeline.py` 这种同时读数据、调模型、写 NPZ、跑 QC、渲染的上帝模块。
- 不让 CLI 脚本直接 import SAM3/Qwen 并拼接业务规则。
- 不让任何外部 JSON `dict` 直接流入 domain；adapter 必须校验并转成 typed DTO/value object。
- 不让通用 `target/receiver/gripper` 三角色在同一个长函数里由大量 task 特判处理。
- 不以 `skip-existing` 之类的文件存在性代替 artifact version/status 判断。
- 不让阶段 1 的 package 依赖或输出阶段 2 的全视频 mask。

---

## 12. 测试和可维护性要求

| 测试层 | 重点 | 不需要真实模型 |
|---|---|---:|
| `unit/domain` | window 不变量、状态转换、policy 产生正确 request | 是 |
| `unit/application` | fake ports 下的流程顺序、拒绝不产生 seed | 是 |
| `contract/sam3` | `TEXT_BOX` 在一次 request 中发送 text+bbox；阶段 1 无传播调用 | adapter mock/spy |
| `contract/artifact` | JSON schema、hash、revision、不可变引用 | 是 |
| `integration/pilot` | 真实 Qwen/SAM3 生成 2–3 条 contact sheet | 否 |
| `manual acceptance` | 10 条 × target/receiver 的视觉审核 | 否 |

每个阶段 1 运行还应输出一个简短 summary：`approved / needs_review / rejected / blocked` 数量和
原因聚合，帮助先修 keyframe 问题，而不是被全视频 QC 指标淹没。

---

## 13. 实施顺序（等待本设计确认后）

### P0：项目骨架与纯领域测试

1. 在空的 `process_data_v2/` 建独立 `pyproject`、package、测试框架和 `.gitignore`；
2. 实现 domain value objects、状态机、role policies、ports；
3. 用 fake adapter 写 `PrepareKeyframes` 的单元测试；此时不接模型、不读旧 annotations。

### P1：关键帧最小闭环（当前目标）

1. 实现 RoboTwin 单帧读取、state timeline、target/receiver policy；
2. 实现 Qwen grounding 和 SAM3 `SingleFrameSegmenter`（`box_only/text_only/text_box`）；
3. 实现 candidate artifact、overlay/contact sheet、geometry + semantic auto review；
4. 实现人工 approve/reject/revision；
5. 在 `move_pillbottle_pad/cam_high` 的 3 条失败样本做消融，再扩到 10 条。

**P1 的完成条件就是第 8.1 节；不会开始视频传播。**

### P1B：夹爪关键帧（仍然只做单帧）

1. 增加 `GripperAnchorPolicy` 和两个 anchor kind；
2. 增加 finger/palm 与 target/wrist/forearm exclusion 的 keyframe QC；
3. 对少量人工样本确认双 anchor 可用。

### P2：视频传播

在 P1/P1B seed 被批准后，才实现 `PropagationEngine`、`VideoTrack` 与逐帧 provenance；
先从 target 和静态 receiver 开始，再做 gripper 动态 ROI。

### P3：全视频 QC 与最终导出

仅当已有真实 video track 后，增加连续性、共同运动、place、渲染与 NPZ/下游导出。

---

## 14. 本设计需要保持的默认决策

若按本稿执行，默认采用以下决策：

1. `process_data_v2` 是独立、干净的项目；旧 `process_data` 不作为运行时依赖；
2. 当前 pilot 只做 `move_pillbottle_pad + cam_high + target_0/receiver_0` 的关键帧；
3. 阶段 1 的输出是 versioned keyframe artifacts，不是全视频 NPZ；
4. Qwen review 只做候选筛选，**人工 review 才生成 `ApprovedSeed`**；
5. 任何未通过的结果明确保留为 `needs_review/rejected/blocked`，绝不静默兜底；
6. 阶段 2/3 的接口和数据字段现在预留，但实现工作在 P1 验收后才开始。

这使当前最重要的问题——“关键帧内容是否正确”——成为一个小、清楚、可验收的闭环；
同时不会堵死之后的 video mask 传播与全视频 QC 扩展。
