# `process_data_v2` 简化架构设计

> **状态：已确认，当前生效架构；代码按 P0–P4 实施。**
> 本架构将流程收缩为三个主阶段，并在 SAM seed 与传播之间加入候选 mask QC：
> `State Loop → Qwen Semantic Plan → SAM Candidates → Qwen Mask QC → SAM Propagation`。
> 当前实验范围：`move_pillbottle_pad / cam_high / target_0 + receiver_0`。
> 测试数据集：`/DATA/disk8/xuran/add_mask_robotwin/dataset/move_pillbottle_pad_coverage20_original`。
> 实施进度：P0–P4 已完成；run `coverage20-qc-contact-v5-native` 的 20 条全量
> Qwen/SAM regression、overlay 渲染和视觉抽查均已完成。

本文已经替代旧的“单帧 keyframe 候选 → 人工审批 → 后续传播”设计。旧文档只保留在 Git
历史中，不再作为实现依据。

---

## 1. 当前要解决的问题

给定一个 RoboTwin episode，自动输出：

- target：被机械臂抓取并移动的物体；
- receiver：任务完成时应与 target 直接接触的完整物体或目标区域；
- 两个角色在各自活动时间窗口内的 visible mask。

当前方案的重点不是建立一个通用标注平台，而是先把一个小实验跑通，并让每个阶段都能独立检查输入和输出。

## 2. 明确边界

本版本做：

- 从机器人 state 中提取一次机械臂操作 loop；
- 根据 loop 选择少量语义关键帧；
- 通过 Qwen 联合确定 target / receiver 的语义，并为每个角色生成有序的
  SAM3-native 短 query 候选池；
- 使用 SAM3 生成 seed mask 并进行视频传播；
- 对 query bank 的实际 seed mask 候选进行 Qwen 实例身份检查，并 fail closed；
- 生成 target / receiver 的 visible-only mask；
- 保存每个阶段的中间结果和来源信息。

本版本不做：

- 人工选择或确认 mask；
- Qwen 输出精确 bbox 作为默认输入；
- Qwen 逐帧重新框或逐帧修补 mask；
- gripper mask；
- hidden / amodal mask 补全；
- 多任务、多相机、动态相机的通用化处理。

gripper state 仍然可以用于判断动作边界；这不等于本版本要生成 gripper mask。

---

## 3. 总体架构

```text
RoboTwin episode
      │
      ▼
┌──────────────────────────────┐
│ Stage 1: State Loop           │
│ 读取 state，提取动作边界和帧窗 │
└──────────────┬───────────────┘
               │ LoopContext
               ▼
┌──────────────────────────────┐
│ Stage 2: Qwen Semantic Plan   │
│ client 组 prompt，server 推理 │
│ 输出角色语义、seed、短 query 池 │
└──────────────┬───────────────┘
               │ SemanticPlan
               ▼
┌──────────────────────────────┐
│ Stage 3: SAM                  │
│ seed candidates → Qwen mask QC│
│ → native propagation          │
│ → role window → temporal QC   │
└──────────────┬───────────────┘
               │ MaskRun
               ▼
       target / receiver masks
```

三个阶段之间只传递三个主要对象：

```text
LoopContext → SemanticPlan → MaskQCResult → MaskRun
```

`run_pipeline.py` 可以作为一个很薄的编排入口，只负责按顺序调用三个阶段，不在其中实现 Qwen、SAM3 或 mask 算法细节。

---

## 4. 阶段一：State Loop Extraction

### 4.1 职责

Stage 1 只读取 episode metadata、state 和必要的视频长度信息，回答：

- 哪一只机械臂是本次操作的 active arm；
- 机械臂何时开始接近或移动；
- 何时完成夹取；
- 何时开始释放、何时完成释放；
- target 和 receiver 各自应该在哪个窗口输出 mask；
- 哪些帧应该交给 Qwen 作为语义上下文。

Stage 1 不判断哪个视觉物体是 target，也不调用 Qwen 或 SAM3。

### 4.2 当前 loop 定义

当前 `move_pillbottle_pad` 实验使用以下事件：

```text
pre_grasp
  → move_start
  → close_start
  → close_done
  → transport / hold
  → open_start
  → open_done
```

一个完整的机械臂操作 loop 是：

```text
[t_move_start, t_open_done]
```

在这个 loop 内，当前状态 detector 输出五个边界事件，并划分为四个状态阶段：

| 阶段 | 时间范围 | 含义 |
|---|---|---|
| `approach / move` | `[t_move_start, t_close_start)` | 机械臂开始接近目标，夹爪仍未完成闭合 |
| `close / grasp` | `[t_close_start, t_close_done]` | 夹爪闭合并稳定完成抓取 |
| `hold / transport` | `(t_close_done, t_open_start)` | 目标被夹持并向 receiver 移动 |
| `open / release` | `[t_open_start, t_open_done]` | 夹爪打开并完成释放 |
| **完整 loop** | `[t_move_start, t_open_done]` | 以上四个阶段的合并窗口 |

当前没有单独的 `hold_start` 或 `place_start` 事件；如果后续确实需要，再从 state 中增加，不在本轮预先扩展。

角色输出窗口：

```text
target   : [t_move_start, t_close_done]
receiver : [t_close_done, t_open_done]
```

seed 候选窗口与输出窗口不是同一个概念：

- target seed：优先选择 close 前无遮挡的早期帧；
- receiver seed：优先选择动作前完整可见的 receiver 帧；
- receiver 可以使用早期帧作为 seed，但只在放置阶段输出。

### 4.3 Qwen 输入帧类别

Stage 1 生成少量带用途标签的帧，不把整个视频发送给 Qwen：

| purpose | 用途 | 是否允许作为 seed |
|---|---|---:|
| `pre_grasp_seed_candidate` | 目标/接收物无遮挡的早期帧 | 是 |
| `post_grasp_context` | 判断哪个物体随夹爪移动 | 否 |
| `place_context` | 判断物体最终放置到哪里 | 否 |

每个帧必须保存原始 frame id，不能只使用在数组中的位置：

```json
{
  "frame_id": 0,
  "purpose": "pre_grasp_seed_candidate",
  "eligible_roles": ["target", "receiver"]
}
```

### 4.4 Stage 1 输出：`LoopContext`

```json
{
  "episode": {
    "task": "move_pillbottle_pad",
    "episode_index": 7152,
    "episode_id": "007152",
    "camera": "cam_high"
  },
  "frame_count": 138,
  "events": {
    "active_arm": "right",
    "t_move_start": 4,
    "t_close_start": 55,
    "t_close_done": 67,
    "t_open_start": 119,
    "t_open_done": 132
  },
  "windows": {
    "loop": [4, 132],
    "target_0": [4, 67],
    "receiver_0": [67, 132]
  },
  "semantic_frames": [
    {
      "frame_id": 0,
      "purpose": "pre_grasp_seed_candidate",
      "eligible_roles": ["target", "receiver"]
    },
    {
      "frame_id": 68,
      "purpose": "post_grasp_context",
      "eligible_roles": ["target"]
    },
    {
      "frame_id": 120,
      "purpose": "place_context",
      "eligible_roles": ["receiver"]
    }
  ]
}
```

上面的数值仅用于说明 schema；实际事件帧由 state detector 计算，不在代码中写死任何 episode 的数字。

---

## 5. 阶段二：Qwen Semantic Plan

### 5.1 Client / Server 分工

```text
Qwen Server
  - 加载 Qwen 模型
  - 启动 HTTP / OpenAI-compatible endpoint
  - 接收多模态请求
  - 返回原始模型响应

Qwen Client
  - 读取 prompt 配置文件
  - 填充 task、LoopContext 和帧目录
  - 编码并发送图像
  - 解析严格 JSON
  - 保存 rendered prompt、raw response 和 hash
```

Server 是独立运行的基础设施，不由每个 episode 的 client 重复启动。实验入口可以在运行前启动 server，并在 client 调用前检查 health endpoint。

### 5.2 Qwen 输入

Qwen 一次联合判断 target 和 receiver，输入包括：

- 任务文本或 coarse task；
- `LoopContext` 中的动作边界；
- 带 `frame_id` 和 `purpose` 的 sparse RGB 帧；
- target / receiver 的角色定义。

target 和 receiver 不应该由两个完全独立的请求决定，否则可能出现角色交换或两个角色指向同一个物体。

### 5.3 Prompt 配置与实验依据

这里不再沿用 v4.1 的“把物体扩写成更长视觉描述”思路：31 例实验中，v4.1 与当前 expanded
query 都只有 19/31 个非空结果，而且长描述会让部分原本非空的物体退化为空。后续 v4.2
说明短候选 bank 能补充覆盖，但 `food`、无颜色 `block` 等宽泛候选也会命中错误物体。
因此这里同时采用“Qwen 动态短候选”和“只信任 Qwen 事前排序、不按 SAM 结果自动选”两条
约束。

prompt 模板放在配置文件中，例如：

```yaml
qwen:
  endpoint: "http://127.0.0.1:18086/v1/chat/completions"
  model: "qwen3.5-27b"
  prompt_template: "configs/prompts/target_receiver_semantic.txt"
  max_tokens: 800
  query_selection: "first_recommended"
  allow_query_fallback: false
```

prompt 模板可以修改，但模板中的输出字段和类型必须保持稳定。具体的物体类别、颜色、形状等
不能写死在模板中，由 Qwen 根据当前 episode 动态生成。模板只规定输出合同和选择规则；
详细视觉理由保存在 `SemanticPlan`，不发送给 SAM3。

建议 prompt 模板如下：

```text
你是机器人操作视频的语义规划器。你的工作是联合确定 target 和 receiver，
为每个角色选择一个可见 seed frame，并生成供 SAM3 使用的短英文 query 候选池。
你不画框、不输出 mask，也不评价 SAM3 的 mask。

任务描述：
{task_text}

相机：{camera}

事件上下文：
- move_start: {move_start}
- close_done: {close_done}
- open_done: {open_done}

下面是带原始 frame id 和用途标签的图像：
{labeled_multimodal_frames}

角色定义：
- target：随后被夹爪抓取并移动的物体。
- receiver：任务完成时应与 target 直接接触的完整物体或目标区域。receiver 不要求位于
  target 下方或承托 target，核心判断依据是二者的直接接触关系。

receiver 的身份判断与 seed 选择分成两个步骤：先用 `place_context` 确定任务完成时与 target
直接接触的对象或区域，再回到 receiver 允许的 seed 候选中选择同一对象最清晰的一帧。seed
中二者不需要已经接触；只要该对象在任一允许候选中清晰可见，即使它在 `place_context` 中被
target 或夹爪部分遮挡，也不能仅因此返回 `no_clear_seed`。

请利用任务文本和多帧动作关系联合判断 target 与 receiver，不要只凭一张静态图猜测。

对每个角色：
1. 只能从允许的 seed 候选中选择一个完整、清晰、遮挡最少的 seed frame。必须逐一比较所有
   允许候选，不得因为 frame id 最小或排在输入最前就默认选择。若多个候选接近，优先选择
   跨相邻候选外观稳定、曝光正常且夹爪尚未接触主体的帧，并在 reason 中写明比较依据；
2. 输出一个有序的 query candidate bank，而不是一条长 referring expression；
3. 每条非空 query 必须是 1–4 个小写英文词组成的、指向完整物体的单数名词短语；
4. 必须保留完整类别名（例如 bottle、pad、basket）。颜色词或形状词不能充当 head noun，
   例如 square、blue square 都不是完整物体 query，必须补上真实对象类别。可加入一个颜色、
   形状或材质修饰词；如果两个紧凑修饰词都确有区分力，可以使用类似 blue square pad 的
   短语，但总词数仍不得超过 4；
5. 只使用多帧中稳定可见、能区分实例的属性，不猜测无法确认的属性；
6. 禁止冠词/所有格、品牌或 OCR 文字、数字、动作、位置、空间关系、比较级以及包含
   with 的长属性串；禁止只描述 cap、logo、label、handle 等子部件；
7. category_query 必须是简单、常见、可直接检测的无修饰完整类别，通常使用物体 head noun；
   不得把任务文本中的用途、商品亚型或尺寸词拼成类别，例如 medicine bottle、pill bottle、
   compact bottle 的 category_query 都应是 bottle。有颜色或形状证据时分别填写
   color_category_query、shape_category_query。若规则 4 的第二个紧凑线索不可缺少，
   color_category_query 可同时保留它；可选的 general_fallback_query 必须是更一般但仍有
   视觉意义的完整物体类别，且永远排在最后；没有合理上位类别时填写 `null`，禁止使用
   `object`、`thing`、`item`、`stuff` 等空泛词；
8. `recommended_order` 排的是预计 SAM3 分割鲁棒性，不是描述详细程度。对于 target 这类常见
   三维可操作物体，若提交帧中没有第二个同类别实例，第一项必须是无修饰类别；只有确实存在
   多个同类别实例并需要消歧时，才让颜色/形状候选优先；
9. 对 `pad`、`mat` 等薄平面接触区域，如果颜色和形状都稳定清晰，应保留“颜色 + 形状 +
   类别”的紧凑组合并优先排序，不机械拆成两条更弱的提示；具体短语仍由 Qwen 根据图像生成；
10. 所有非空候选必须互不相同。若某个可选字段只能生成与已有候选相同的短语，或没有独立的
    真实视觉依据，必须将该字段写为 null，绝不能复制短语来凑满字段；recommended_order 只列
    所有非空候选字段且不重复。输出前按小写并合并空格后再次检查候选是否重复；
11. 不返回 bbox，不返回 mask，不评价任何 mask。不要为了让两个角色的文字不同而编造
   视觉属性；如果身份仍有歧义，在 reason 中说明。

只返回一个 JSON 对象，不要输出 Markdown 或额外说明：
{
  "target": {
    "status": "ok" | "no_clear_seed",
    "seed_frame_id": <原始 frame id 或 null>,
    "category_query": "<1-4 lowercase English words 或 null>",
    "color_category_query": "<1-4 lowercase English words 或 null>",
    "shape_category_query": "<1-4 lowercase English words 或 null>",
    "general_fallback_query": "<1-4 lowercase English words 或 null>",
    "recommended_order": ["<candidate field>", "..."],
    "exclude": ["<other visible object>", "..."],
    "reason": "<简短中文语义理由，不发送给 SAM3>"
  },
  "receiver": {
    "status": "ok" | "no_clear_seed",
    "seed_frame_id": <原始 frame id 或 null>,
    "category_query": "<1-4 lowercase English words 或 null>",
    "color_category_query": "<1-4 lowercase English words 或 null>",
    "shape_category_query": "<1-4 lowercase English words 或 null>",
    "general_fallback_query": "<1-4 lowercase English words 或 null>",
    "recommended_order": ["<candidate field>", "..."],
    "exclude": ["<other visible object>", "..."],
    "reason": "<简短中文语义理由，不发送给 SAM3>"
  }
}
```

字段约束补充说明：`category_query` 在 `status=ok` 时必填；其余 query 可以为 `null`，且在
无法形成有独立视觉依据的不同短语时必须为 `null`，不能复制其他候选。
`status=no_clear_seed` 时四个 query 均为 `null`，`recommended_order` 为空。候选顺序由
Qwen 根据证据给出，但当前小实验的执行策略固定为使用顺序第一项作为
`primary_query`；其余候选只写入 artifact 供追溯，不根据非空像素数自动切换，也不把多个
候选的 mask 做并集。未来若要启用回退，必须通过配置显式打开，而不能隐式改变语义。

Qwen 偶尔会把同一个短语复制到多个可选字段。client 在严格校验前只做一项窄规范化：对
空格归一化后完全相同的候选保留一次；若重复组含 `category_query`，固定保留该必填字段，
否则保留 `recommended_order` 中最靠前的字段，并同步去重字段顺序。该操作不生成新文本、
不改变原首选 query 的实际字符串，也不读取 SAM mask；不同文本的候选仍不会被自动替换。

### 5.4 Qwen 输出：`SemanticPlan`

```json
{
  "target": {
    "status": "ok",
    "seed_frame_id": 0,
    "category_query": "bottle",
    "color_category_query": "orange bottle",
    "shape_category_query": "plastic bottle",
    "general_fallback_query": "container",
    "recommended_order": [
      "category_query",
      "color_category_query",
      "shape_category_query",
      "general_fallback_query"
    ],
    "exclude": ["blue square pad", "black robot gripper", "table shadow"],
    "reason": "该物体在后续帧中被夹爪抓取并移动；颜色和类别在多帧中稳定可见。"
  },
  "receiver": {
    "status": "ok",
    "seed_frame_id": 0,
    "category_query": "pad",
    "color_category_query": "blue square pad",
    "shape_category_query": "square pad",
    "general_fallback_query": "mat",
    "recommended_order": [
      "color_category_query",
      "shape_category_query",
      "category_query",
      "general_fallback_query"
    ],
    "exclude": ["white pill bottle", "black robot gripper", "table shadow"],
    "reason": "该区域是 target 最终被放置的位置；蓝色和方形是稳定的区分线索。"
  }
}
```

`primary_query` 是 `recommended_order[0]` 对应的候选值，运行时动态解析，不写入通用代码。
`exclude` 和 `reason` 只作为语义记录，不拼接到 SAM3 的正向 text prompt。候选 bank 的
字段名与 v4.2 实验保持一致；若组合短语（如 `blue square pad`）比拆开的候选更稳定，允许
将其放在首位，但仍须满足 1–4 词合同。

### 5.5 episode 7152 合法 seed 矩阵

实现 Stage 3 时，旧 v4.2 表格使用的是 frame 58；它不属于当前 Stage 1 给出的无遮挡 seed
候选。为避免把旧帧上的结论误套到新 pipeline，单独在合法候选 `[0, 17, 34, 51]` 上运行了
text-only 短 query 矩阵。以下像素数只表示 SAM3 非空；结合 centroid 与 overlay 检查后，非空
结果分别落在正确的瓶子和 pad 上：

| role / query | f0 | f17 | f34 | f51 |
|---|---:|---:|---:|---:|
| target `bottle` | 2539 | 2496 | 0 | 0 |
| target `orange bottle` | 0 | 0 | 0 | 0 |
| target 其余已测短候选 | 0 | 0 | 0 | 0 |
| receiver `blue square pad` | 1070 | 1100 | 1066 | 1068 |
| receiver `blue pad` | 0 | 0 | 0 | 1066 |
| receiver `square pad` | 1068 | 0 | 0 | 0 |
| receiver `pad` / `mat` | 0 | 0 | 0 | 0 |

事前排序仍决定候选生成顺序，但正式运行会对 `recommended_order` 的前若干个 query 分别生成
SAM seed mask，再由独立 Qwen mask-QC prompt 比较实际结果。Python 中不写死 `bottle`、
`brown bottle` 或 `blue square pad`，也不会因为 mask 非空或面积较大而自动接受候选。

### 5.6 Box 的处理

当前主流程不要求 Qwen 返回 bbox：

- v4.1 实验显示 Qwen initial bbox 可能过紧、偏移或只覆盖局部；
- `text_box` 可能裁掉正确物体，扩框又可能引入背景；
- v4.1 历史 ep0 的 target 和 receiver 都以 `text_only` 作为有效 seed。

因此 Stage 2 的正式 schema 不包含 bbox。未来遇到多个视觉上相同的实例时，可以另行增加可选 coarse ROI，但不作为当前主路径。

---

## 6. 阶段三：SAM Mask and Propagation

Stage 3 使用 Qwen 的 `SemanticPlan`。Qwen 仅在 seed 候选生成后执行角色实例 QC，不逐帧
重新分割或修补 mask。

### 6.1 3A：Seed Mask

对 target 和 receiver 分别执行：

1. 读取 Qwen 选择的 `seed_frame_id`；
2. 取 `recommended_order` 的前 `mask.qc_max_candidates` 个非空 query；
3. 在 seed frame 对每个 query 生成独立 SAM3 candidate mask；
4. 先检查空 mask、异常面积，并按 IoU 去除近重复候选，再生成只画轮廓、不遮盖物体纹理的
   A/B/C 图；
5. Qwen 结合任务文本、候选图和动作上下文返回 `accept / reject_all / ambiguous`；
6. 只有 `qc_status=passed` 的候选才能成为实际 seed 并进入 native tracking。

如果语义 Qwen 返回 `no_clear_seed`、所有候选为空、QC Qwen 响应无法解析、置信度不足、
返回 `reject_all/ambiguous`，则显式记录 `rejected/ambiguous/error` 并停止该角色传播，不生成
默认 box 或静默接受更宽泛候选。

### 6.2 3B：Native Video Propagation

使用 seed mask 进行 SAM3 native-mask tracking：

- target 从 seed frame 跟踪到 `t_close_done`；
- receiver 从 seed frame 跟踪到 `t_open_done`；
- receiver 可以从动作前 seed 开始跟踪，但只在 `[t_close_done, t_open_done]` 写出结果。

native tracking 的作用是保持实例身份，不是补全被遮挡像素。

批量运行使用 `sam-batch`：一个 worker 在配置指定的一张 GPU 上顺序处理 episode，整个 batch
只初始化和关闭一次 `Sam3Adapter`，候选 query 也复用同一 episode 的视频 session。每个
episode 的 session 和临时帧目录仍独立清理；CUDA 初始化或 launch 级故障会 fail fast，普通
episode 失败则记录后继续。这个常驻范围是一次 batch 进程，不额外部署长期运行的 SAM3 服务。

### 6.3 3C：Role-window Composition

最终 mask 采用：

```text
final_role_mask[t] = native_track[t], t in role_output_window
final_role_mask[t] = empty,           otherwise
```

规则：

- 时间窗外为空；
- 时间窗内直接保留 native tracker 的实例 mask；
- 不再逐帧创建 text-only session，因此单帧 text detection 失败不会造成闪烁；
- `canonical_envelope` 仅保存为 seed 几何诊断，不再裁剪后续帧；
- target 被夹爪遮挡的部分不补全；
- receiver 被 target 遮挡的部分不保留；
- 不使用静态复制完整 receiver 的旧策略。

### 6.4 Temporal QC

对每个角色的输出窗口计算：coverage、存在性切换、内部断帧、相邻 IoU、质心跳变、面积比例
跳变以及相对 seed 的最大质心距离。单一异常或连续遮挡只标记为 `review`；IoU、质心、面积
三类严重跳变中至少两类同时越界时标记为 `quarantined`，对应像素不进入发布 NPZ。QC 只判断
传播连续性，稳定传播到错误实例仍需独立的 identity review。

候选 mask QC 在传播前判断 seed 的角色身份；temporal QC 在传播后判断轨迹连续性。两者分别
覆盖不同失败模式，均不引入人工审批。

### 6.5 episode 7152 真实 smoke

使用 Qwen 动态生成的 `bottle@frame0` 与 `blue square pad@frame0` 完成了真实 GPU smoke：

- target native track 在活动窗口 `[4,67]` 的 64 帧全部非空，最终 target 为 64/64 帧非空；
- receiver native track 在 `[0,132]` 全部非空；活动窗口 `[67,132]` 的 66 帧全部有最终
  visible mask；瓶子放到 pad 后，只保留未被瓶子遮挡的 pad 边缘；
- 两个角色的存在性切换和内部断帧均为 0；
- 四通道 `masks.npz` 中两个 gripper 通道全零且 metadata 明确为 `not_annotated`；
- overlay 抽查确认 target 没有漂到夹爪，receiver 没有保留瓶子覆盖区域。

这是候选 QC 引入前的历史 smoke，不宣称存在像素级 ground truth；新流程需要重新生成
`mask_qc.json` 后才视为已验证结果。

---

## 7. 阶段输出和产物

### 7.1 阶段产物

```text
Stage 1 → loop.json
Stage 2 → semantic_plan.json
Stage 3 → candidate seed masks + mask_qc.json + selected seed + tracks + masks.npz
```

### 7.2 建议目录

```text
artifacts/
  runs/<run_id>/
    <task>/
      episode_<id>/
        <camera>/
          loop.json
          semantic_plan.json
          target_0/
            seed.rgb.png
            seed.mask.png
            canonical_envelope.png
            native_track.npz
            temporal_qc.json
          receiver_0/
            seed.rgb.png
            seed.mask.png
            canonical_envelope.png
            native_track.npz
            temporal_qc.json
          masks.npz
          frame_provenance.json
          run_manifest.json
```

`semantic_plan.json` 保存 Qwen model、prompt hash、输入 frame ids、raw response 和解析后的结果。这样 prompt 可以修改，但每次运行仍然可追溯。

### 7.3 Mask channel

当前可以继续保持既有四通道 NPZ 兼容格式：

```text
channel 0: target_0       valid
channel 1: receiver_0     valid
channel 2: gripper_left   not_annotated
channel 3: gripper_right  not_annotated
```

manifest 必须记录 gripper 通道为 `not_annotated`。全零 gripper 通道不是负样本，当前下游不能把它当作真实标注。

### 7.4 测试数据集

本项目的真实数据测试集固定为：

```text
/DATA/disk8/xuran/add_mask_robotwin/dataset/move_pillbottle_pad_coverage20_original
```

当前目录中实际存在 20 个 `move_pillbottle_pad` episode：

```text
7152, 7156, 7157, 7163, 7168, 7179, 7181, 7185, 7187, 7188,
7274, 7317, 7335, 7367, 7424, 7464, 7571, 7621, 7673, 7674
```

每个测试 episode 应至少能找到：

```text
data/chunk-007/episode_<id>.parquet
videos/chunk-007/observation.images.cam_high/episode_<id>.mp4
sidecars/episode_<id>.hdf5
```

当前 20 条 AV1 视频都比 Parquet/state 多解码出 1 个尾帧。Stage 1 preflight 已将它记录为
dataset contract：可用帧数以 Parquet 的连续 `frame_index` 为准，Qwen 和 SAM 只消费
`[0, parquet_frame_count - 1]`，最后一个 video-only 帧不进入 pipeline。

前 10 个 episode 是 clean 样本，后 10 个是 randomized 样本。建议测试分两级：

| 测试级别 | 数据 | 用途 |
|---|---|---|
| smoke | `7152` | 每次改动后的快速端到端检查 |
| regression | 上述全部 20 个 episode | 阶段完成后的完整回归 |

原始视频和 Parquet 不复制到 Git。v2 project 通过 dataset config、固定 episode manifest 和 preflight 测试引用这份外部数据；这样既能把数据集纳入项目测试契约，又不会把二进制数据提交到仓库。

推荐配置：

```yaml
dataset:
  root: /DATA/disk8/xuran/add_mask_robotwin/dataset/move_pillbottle_pad_coverage20_original
  task: move_pillbottle_pad
  camera: cam_high
  smoke_episode_ids: [7152]
  regression_episode_ids: [7152, 7156, 7157, 7163, 7168, 7179, 7181, 7185, 7187, 7188,
                           7274, 7317, 7335, 7367, 7424, 7464, 7571, 7621, 7673, 7674]
```

运行时允许通过 `--dataset-root` 或环境变量覆盖绝对路径，但测试 manifest 中的 episode id 不应随意改变。

### 7.5 全长 overlay 视频

`scripts/render_coverage20_videos.py` 是 Stage 3 之后的只读可视化工具。它从每个 episode
已有的 `masks.npz` 中选择 target/receiver 有效角色数最多、同分时修改时间最新的 run，逐帧
叠加到原始 `cam_high` 视频。它不调用 Qwen 或 SAM3，也不修改 mask 数据。

默认渲染样式为：

- mask 内部按 `alpha=0.32` 半透明填充，保留物体纹理；
- mask 外侧 `3 px` 使用对应角色颜色绘制高亮轮廓；
- 从轮廓继续向外扩张到总计 `5 px`，使用黑色衬边保持复杂背景下的对比度；
- 彩色轮廓和黑色衬边都严格位于 mask 外部，不占用 mask 像素。

完整 coverage20 渲染命令：

```bash
.venv/bin/python scripts/render_coverage20_videos.py --overwrite
```

实验或发布检查应使用 `--run-id <exact_run>` 固定输入，避免选择器退回较旧但状态仍为 valid 的
错误 mask：

```bash
.venv/bin/python scripts/render_coverage20_videos.py \
  --run-id coverage20-sam3-native-v1 \
  --output-dir artifacts/rendered_videos/coverage20_sam3_native_v1
```

默认输出到 `artifacts/rendered_videos/coverage20_best_current/`。脚本原子替换每个 MP4，并在
全部 episode 成功后更新 `manifest.json`；manifest 记录输入 mask hash、输出视频 hash、编码
信息和实际渲染参数。`--alpha`、`--outline-radius`、`--halo-radius` 可以显式覆盖默认值，且
`halo-radius` 必须不小于 `outline-radius`。

overlay 只负责提高可见性。已有 mask 的断帧、漂移或目标身份错误会被原样显示，不能把轮廓
增强视为时序传播或目标选择的修复。

---

## 8. 配置

建议一个 pilot 配置包含数据、Qwen、SAM3 和输出设置：

```yaml
task: move_pillbottle_pad
camera: cam_high

dataset:
  root: /DATA/disk8/xuran/add_mask_robotwin/dataset/move_pillbottle_pad_coverage20_original
  smoke_episode_ids: [7152]
  regression_episode_ids: [7152, 7156, 7157, 7163, 7168, 7179, 7181, 7185, 7187, 7188,
                            7274, 7317, 7335, 7367, 7424, 7464, 7571, 7621, 7673, 7674]

qwen:
  endpoint: http://127.0.0.1:18086/v1/chat/completions
  model: qwen3.5-27b
  prompt_template: configs/prompts/target_receiver_semantic.txt
  timeout_seconds: 180
  max_tokens: 800
  query_selection: first_recommended
  allow_query_fallback: false

sam3:
  checkpoint: /path/to/sam3.pt
  gpus: [0]

mask:
  target_envelope_padding_px: 4
  receiver_envelope_padding_px: 4
  temporal_qc_min_adjacent_iou_p05: 0.5
  temporal_qc_max_centroid_jump_p95_px: 5.0
  temporal_qc_max_area_ratio_jump_p95: 0.4
  temporal_qc_quarantine_signal_count: 2
  qc_enabled: true
  qc_prompt_template: prompts/mask_candidate_qc.txt
  qc_max_candidates: 3
  qc_max_tokens: 160
  qc_max_attempts: 2
  qc_min_confidence: 0.70
  qc_min_area_fraction: 0.0001
  qc_max_area_fraction: 0.85
  qc_duplicate_iou_threshold: 0.98

output:
  root: artifacts
```

对象名称、颜色和视觉描述不放在 YAML 中；它们来自 Qwen 的 `SemanticPlan`。

---

## 9. 模块结构

当前不需要复杂的 class 层级。建议使用少量 dataclass 和 stage function：

```text
src/robotwin_annotation_v2/
  models/
    loop_context.py
    semantic_plan.py
    mask_qc.py
    mask_run.py

  pipeline/
    state_loop.py
    qwen_stage.py
    mask_qc.py
    sam_stage.py
    run_pipeline.py

  adapters/
    robotwin_dataset.py
    qwen_client.py
    sam3_adapter.py
    artifact_store.py

  config.py

configs/
  pilot_move_pillbottle_pad.yaml
  prompts/
    target_receiver_semantic.txt

scripts/
  serve_qwen.py
  run_target_receiver.py
```

`run_pipeline.py` 只做编排：

```python
loop = extract_loop(episode, config)
semantic = run_qwen(loop, config)
mask_run = run_sam(loop, semantic, config)
save_run(mask_run, config)
```

Qwen HTTP 细节、SAM3 session 细节和文件格式不进入编排函数。

---

## 10. 阶段间错误处理

本版本做必要的输入、运行错误处理和自动候选 mask QC，不建立人工评审流程。

### Stage 1

- state 缺失或无法形成合法 loop：保存失败原因，不调用 Qwen/SAM；
- 事件顺序非法：保存 `loop_failed`。

### Stage 2

- Qwen server 不可用：保存请求失败信息；
- JSON 无法解析：保存 raw response，不伪造语义计划；
- `seed_frame_id` 不在候选列表：拒绝该语义计划；
- `category_query` 缺失，或任一 query 不满足 1–4 个小写英文词：拒绝该角色的 seed 请求；
- 完全相同的重复候选先按 5.4 的规则规范化；规范化后 `recommended_order` 仍与非空候选
  不一致，或首项无法解析为 `primary_query`：拒绝该角色的 seed 请求。

### Stage 3

- 所有候选 seed mask 为空或面积异常：记录 `qc_status=rejected`；
- QC Qwen 返回 `reject_all`、`ambiguous` 或低于阈值：停止该角色传播；
- QC 服务或响应合同错误：记录 `qc_status=error`，不回退到第一候选；
- seed mask 为空：该角色没有可用 mask；
- tracking 无输出：保留已有阶段产物，不生成假的连续 mask；
- 多个独立时序信号同时严重异常：保留 native track 诊断，但发布 mask 标为 `quarantined`；
- 输出帧尺寸不一致：停止写出该 run。

所有失败和隔离原因都必须可追溯。

---

## 11. 测试计划

### Stage 1 测试

- state 输入能得到合法 `LoopContext`；
- target / receiver 输出窗口正确；
- seed candidate 和 context frame 的用途标签正确；
- 事件顺序异常时显式失败。

### Stage 2 测试

- prompt 模板变量能正确填充；
- 输入帧 id 和 purpose 与图像一一对应；
- Qwen JSON schema 能解析 target / receiver；
- 不接受 bbox 作为必填字段；
- 每条 query 都通过 1–4 个小写英文词及禁用词校验；
- `category_query` 必填、候选互异、general fallback 最后；
- `primary_query` 来自 Qwen 的 `recommended_order[0]`，而不是代码中的 task-specific 常量；
- query bank 保持语义排序，不在 Stage 2 按像素数自动切换或合并候选 mask。

### Stage 3 测试

- query bank 前若干项分别生成独立 seed 候选；
- Qwen 只能从 A/B/C 中选择或拒绝全部，且看不到候选 query 名；
- `rejected/ambiguous/error` 均 fail closed，不进入 propagation；
- `qc_status=passed` 时使用已选 seed 和 query；
- seed 使用 `text_only`；
- native tracking 的 seed frame 和方向正确；
- target / receiver 的输出窗口互不混淆；
- visible composition 原样保留窗口内 native mask；
- Stage 3 不调用逐帧 text observation；
- 时序 QC 区分单一 review 信号与多信号 quarantine；
- 窗口外始终为空；
- gripper 通道被标记为 `not_annotated`。

### 集成测试

先在测试数据集的 `episode_007152 / cam_high` 上完成一次完整运行，再扩展到 20 个 episode。检查：

```text
loop.json
semantic_plan.json
mask_qc.json
target/receiver qc_candidates
target/receiver seed masks
native tracks
final masks.npz
frame_provenance.json
```

---

## 12. 实施顺序

### P0：确定数据契约

1. 确定 `LoopContext`、`SemanticPlan`、`MaskRun` schema；
2. 确定 prompt 配置文件格式；
3. 确定 artifact 目录和 NPZ channel metadata。

### P1：实现 State Loop

1. 读取 episode state；
2. 检测 active arm 和事件边界；
3. 采样带用途标签的 Qwen 输入帧；
4. 保存 `loop.json`。

### P2：实现 Qwen Client / Server

1. 整理 server 启动入口和 health endpoint；
2. 实现 client 的 prompt 文件读取和变量填充；
3. 实现多帧请求和 JSON 解析；
4. 保存 `semantic_plan.json`；
5. 在 smoke episode `7152` 上确认 Qwen 生成合法的短 query bank，并正确解析
   `primary_query`。

### P3：实现 SAM seed 和传播

1. 使用 Qwen text-only 生成 seed mask；
2. 实现 native-mask tracking；
3. 按角色活动窗口裁剪 native track；
4. 计算 temporal QC 并隔离严重异常；
5. 导出 `masks.npz`、temporal QC 和 provenance。

### P4：端到端运行

1. 运行 smoke episode `007152`；
2. 检查三阶段中间产物；
3. 修正 prompt、时间窗口或 SAM 参数；
4. 通过 smoke 测试后运行全部 20 个 regression episode。

### 测试与 Git 保存节奏

每完成一个可独立验证的阶段，先运行对应测试，再保存一个小而清晰的 commit：

```text
P0 schema/config tests       → test pass → commit
P1 state-loop unit + smoke   → test pass → commit
P2 Qwen client contract      → test pass → commit
P2 Qwen smoke episode 7152   → test pass → commit
P3 SAM seed/propagation      → test pass → commit
P4 20-episode regression     → test pass → commit
```

commit 只包含代码、配置、测试和文档，不包含外部 dataset 的视频、Parquet 或 HDF5 原文件。

---

## 13. 未来扩展边界

后续增加 gripper mask 时，只增加新的 role 配置和 SAM 策略，不修改 Stage 1/Stage 2 的基本接口：

```text
LoopContext → SemanticPlan → MaskQCResult → MaskRun
```

后续如果需要 bbox，可以将 coarse ROI 作为 `SemanticPlan` 的可选字段，但不能让它重新成为默认的 mask 边界。

本架构定义自动候选 mask QC，但暂不定义人工审批或全任务通用化方案。

---

## 14. 与旧架构的迁移关系

旧架构中的以下概念不再属于当前主流程：

```text
KeyframeRequest
ReviewStatus
KeyframeReviewer
ApprovedSeed
HumanReview
VideoQCService
GripperAnchorPolicy
box_only / text_box candidate bank
```

可以继续复用的底层能力包括：

- RoboTwin episode / frame 读取；
- state timeline 的基础算法；
- SAM3 session 和 native mask prompt 的底层封装；
- 文件 hash、run manifest 和 artifact 存储；
- 测试框架与 fake adapter。

代码按本文 P0–P4 顺序迁移，每个阶段通过测试后单独提交。

---

## 15. 当前默认决策

本文暂按以下默认值编写：

1. Qwen 一次联合规划 target 和 receiver；
2. Qwen 默认不返回 bbox；
3. SAM3 seed 默认使用 `text_only`；
4. target 输出窗口为 `[move_start, close_done]`；
5. receiver 输出窗口为 `[close_done, open_done]`；
6. 继续保留四通道 NPZ 的兼容结构，但 gripper 通道标记为 `not_annotated`；
7. Qwen server 独立启动，client 只调用和解析；
8. prompt 模板可通过配置文件替换，具体 query bank 由 Qwen 动态生成；
9. 对 Qwen 排名前若干个 query 生成 seed 候选，由 Qwen 查看实际轮廓后选择；空 mask、面积和
   置信度只用于拒绝，不用于按像素数自动挑选。
