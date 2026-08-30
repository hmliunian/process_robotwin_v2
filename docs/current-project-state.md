# 当前项目状态、数据边界与改进建议

> 审计快照：2026-08-30。适用分支：feat/target-only-profiles-optimization（基于
> feat/unified-config-binding）；对照 master：0a28f99。

## 结论先行

| 问题 | 当前结论 |
| --- | --- |
| /DATA/disk8/xuran/robotwin2.0_full_aloha_rgbd_320_rgb_correct_v1 是否全量 | 是 RoboTwin 2.0 的 50 task × 550 episode 全量 RGB/state 数据；不是深度完整全量 |
| 深度完整性 | cam_high depth 26,304/27,500，缺 1,196；URDF 不能把它当作 27,500 条完整输入 |
| contact_press | 在当前 target-only 11-task 业务集合中只有 click_alarmclock、click_bell、press_stapler 三个任务，是 target semantic profile，不是独立 annotation mode |
| target-only | 按业务分类是“只要求 target 语义角色”的任务合同，不应与 contact_press 并列；最终仍发布固定四通道，当前代码以 AnnotationMode.TARGET_ONLY 作为兼容字段驱动时间线 |
| unified-config feature | 主要增加 PipelineProfile + DatasetBinding 和 path/collection discovery；不是对 master R0–R8 重构的替代实现 |
| 混合根目录 | resolver 会读取 `meta/episodes.jsonl` 的 `full_structured_tasks[0]`，在显式 `--task` 时按 task 过滤；未声明 task 且发现多个 task 时 fail-closed |

统一 profile 只解决“算法参数可复用”，不代表当前 pipeline 已经支持全量 50 个 coarse task。
按既有语义审计，当前“保守、可直接运行”的 pipeline-compatible P&P 子集是 10/50 个 task（5,500 条）；严格单臂
target-only 抽取是 11 个 task、220 条验证 slice。其余多对象、双臂、动态 receiver、工具接触
或 articulated 边界仍需单独合同和验证。

## 1. 全量数据目录核对

### 1.1 实测统计

核对目录：

~~~
/DATA/disk8/xuran/robotwin2.0_full_aloha_rgbd_320_rgb_correct_v1
~~~

meta/episodes.jsonl 共 27,500 行。每行的 full_structured_tasks[0] 作为 coarse task 名称时，
得到 50 个 task，且每个 task 恰好 550 条 episode；episode 全局编号为 0–27,499。

| 资源 | 数量 | 说明 |
| --- | ---: | --- |
| Parquet | 27,500 | data/chunk-*/episode_*.parquet |
| HDF5 sidecar | 27,500 | sidecars/episode_*.hdf5 |
| 四路 RGB MP4 | 27,500/路 | videos/chunk-*/observation.images.{cam_high,cam_left_wrist,cam_right_wrist,front_camera}/ |
| cam_high depth MKV | 26,304 | sidecars/videos/.../observation.depths.cam_high/ |
| 顶层 EXTRACT_MANIFEST.json | 无 | 这是原生混合布局，不是已按 task 拆好的 collection |

`meta/info.json` 的 LeRobot feature schema 只声明四路 RGB；但 sidecar 目录实际另有四路
部分存在的 FFV1 depth MKV。因此，“全量”的准确说法是：

- 对 RGB、Parquet、HDF5 和 metadata 而言，这是 50 task 的完整 27,500 episode 根目录；
- 对需要 depth 的 URDF backend 而言，只有 26,304 条满足文件发现合同，另外 1,196 条会被
  discover_episodes(..., require_depth=True) 排除；
- 对只需要 RGB 的 SAM/object-source 阶段，文件层面可以发现 27,500 条，但这不代表每条
  都满足当前 task 语义、state loop 或 mask QC 合同。
- 四个 depth camera（`cam_high`、`cam_left_wrist`、`cam_right_wrist`、`front_camera`）都只有
  26,304 条；缺失不是只发生在某一路 camera。每个 coarse task 的 metadata 区间连续占用
  550 个全局 episode id（例如 `move_pillbottle_pad` 为 7,150–7,699），但 depth 缺口分散在
  多个 task，不能把某一个 task 简单标记为 depth 完整。

meta/tasks.jsonl 主要是 task/text 索引；单条 episode 的 task authority 是
episodes.jsonl 中的 full_structured_tasks[0]，而 tasks[0] 是自然语言描述，不能把它
当作稳定的 task 名。

### 1.2 当前 resolver 的混合根目录行为

这个根目录没有 task 子目录，也没有顶层 manifest。resolver 会在发现 `meta/episodes.jsonl`
时把 `full_structured_tasks[0]` 作为 episode-to-task authority；显式 `--task` 只保留该 task
的完整文件条目：

~~~
请求 task=move_pillbottle_pad、camera=cam_high
返回 task=move_pillbottle_pad、episode 数=550（文件完整性仍按 camera/backend 再检查）
~~~

未指定 `--task` 且 metadata 含多个 task 时会要求显式选择；metadata 缺失或条目无法解析时，
resolver 不会凭任务名猜 profile。该行为修复了把其他 task 文本传给 Qwen/SAM 的 correctness
风险，但仍建议为长期批处理物化 task-level manifest。

当前建议做法：

1. 使用已经按 task 拆分、并带 task-level EXTRACT_MANIFEST.json 的目录，例如
   dataset/pick_place_20/<task> 或 dataset/target_only_20_v2/<task>；
2. 也可以直接使用当前 resolver 的 metadata 过滤，但必须显式传 `--task` 和 `--camera`，并
   先用少量 episode 做 smoke；
3. 生产批处理仍建议先生成带 task_kind/profile 和文件 hash 的 task-level manifest，便于审计。

原生目录有多个相机（cam_high、cam_left_wrist、cam_right_wrist、front_camera）。
当前无 manifest 的相机推断在多个候选时会报 ambiguous；使用 task-level/native 输入时建议
显式传 --camera cam_high。

另一个已复核的兼容边界是 `profile_compat_20`：实际任务位于
`<root>/pick_place/<task>` 和 `<root>/target_only_release/<task>` 这样的嵌套路径；当前
collection resolver 仍只按 `<root>/<task>` 查找。这类历史 collection 应先物化为当前
task-level manifest，或后续增加显式 nested-root resolver；历史 workflow alias 会在输入边界
规范化，但不会替 collection 猜测缺失的 task kind。

## 2. 业务分类、兼容字段与 contact_press

先按本次约定纠正一个容易混淆的层次：`contact_press` 是三个具体任务的 target 语义 profile；
`target-only` 是“只要求 target 语义角色”的任务/输出合同，不是和 `contact_press` 同层的第三种
语义模式。当前代码为了兼容既有 loop、artifact 和测试，仍保留
`AnnotationMode.TARGET_ONLY`；它在实现中承担时间线、required roles 和 canonical output 的
合同作用。因此下面明确区分“业务分类”和“当前字段”，不能把兼容字段误读成最终领域模型。

当前字段承担的责任：

| 概念 | 当前代码中的位置 | 作用 |
| --- | --- | --- |
| annotation_mode（兼容实现字段） | AnnotationMode.PICK_PLACE / TARGET_ONLY | State Loop、必需对象角色和 canonical 输出合同 |
| target_profile | TargetProfile.GRASP_MANIPULATION / CONTACT_PRESS / DOOR_OPEN | target 应解释为“被抓取的物体”、contact action site 还是 door-opening action site |
| task_kind | manifest 的 provenance 字段 | 描述 single_movable_target、contact_action_site、articulated_action_site 等任务类型，并用于路由 profile |
| gripper_backend | sam / urdf | gripper mask 的生产方式，与上述语义维度正交 |

当前项目的实际映射是：

| task | 当前兼容字段 `annotation.mode` | task_kind | target_profile |
| --- | --- | --- | --- |
| click_alarmclock | target_only | contact_action_site | contact_press |
| click_bell | target_only | contact_action_site | contact_press |
| press_stapler | target_only | contact_action_site | contact_press |
| open_laptop、open_microwave、turn_switch | target_only | articulated_action_site | 当前仍为 grasp_manipulation；显式 `door_open_action_site` 可切换 `door_open` |

如果目标是直接标注完整的 `contact_press` 三类任务，可从混合根目录按 metadata 的全局
episode id 物化以下三个 task-level 目录/manifest：

| task | 全局 episode id | 数量 | 四路 RGB + 四路 depth |
| --- | ---: | ---: | ---: |
| `click_alarmclock` | 2200–2749 | 550 | 550/550 |
| `click_bell` | 2750–3299 | 550 | 550/550 |
| `press_stapler` | 20350–20899 | 550 | 550/550 |
| **合计** | — | **1,650** | **1,650/1,650** |

这三个区间目前比混合根目录整体更适合作为首批 URDF/contact-press 输入；物化时仍应保留
原始 `source_dataset_root`、episode id 和文件校验信息，不要改写原始总根目录。

所以，在当前项目范围内，contact_press 确实只有三个任务；它不是第三种时间线，也不应被写成和
pick_place、target_only 并列的 annotation mode。当前 CLI 仍保留 `--mode contact_press`
这个兼容选择器，内部会映射成 `annotation.mode=target_only` 加
`annotation.profile=contact_press`；这只是入口命名，不能改变上述领域语义。

如果要把业务模型和实现字段彻底对齐，建议下一版 manifest 新增一个明确的任务/输出合同字段
（例如 `task_contract: target_only`，名称可在 schema 评审时确定），再把 `target_profile` 和
`task_kind` 保持为独立字段：

~~~yaml
# 建议的 canonical 业务字段（当前尚未实现）
task_contract: target_only
target_profile: contact_press
task_kind: contact_action_site

# 旧字段仅用于兼容读取，逐步停止写入新 manifest
annotation:
  mode: target_only
  profile: contact_press
~~~

迁移期间不要直接删除旧字段：先让 reader 同时接受新旧写法，写出新字段并做双向一致性校验，
待下游 artifact 和测试完成迁移后再移除 `AnnotationMode.TARGET_ONLY` 的领域含义。

### 2.1 当前命名与兼容规则

configs/process.yaml 已采用正交写法：modes.contact_press.annotation.mode 是
target_only，profile 是 contact_press；输入 resolver 和 `bind_dataset()` 现在都会把
`contact_press`、`door_open`、`origin` 等历史 alias 规范化，并校验 `target_profile` 与
`task_kind`/annotation profile 一致。自然写成 `profile: contact_press` 的 task 或 collection
可以在 `--target-only` 下正确绑定，不会回退到普通 origin prompt。缺少语义字段的旧 manifest
仍按显式 mode 兼容；未知 `task_kind`、冲突 profile 或路径身份则 fail-closed。

建议最终统一为：

~~~
annotation_mode: target_only
target_profile: contact_press
task_kind: contact_action_site
~~~

旧的 profile 字段保留为兼容读取时，应明确它是历史 alias，不能在新代码中同时表示
annotation mode 和 target profile。

## 3. 当前 feature 实际做了什么

当前分支相对 master 只有两个新增提交：

~~~
c898006 refactor: add dataset binding application port
ccb7b14 feat: unify pipeline profiles with dataset binding
~~~

主调用链是：

~~~
configs/process.yaml
    -> load_profile()
    -> resolve_dataset_input()
    -> DatasetTarget
    -> dataset_binding_from_target()
    -> bind_dataset()
    -> PipelineConfig
    -> DatasetPipeline / SamWorkflow / UrdfWorkflow
~~~

这解决了“每个 task 复制一份算法 YAML”的主要重复：算法 profile 可以复用，数据根、task、
camera 和 episode selection 在运行时绑定；旧的 task-bound YAML 仍由 load_config() 兼容。
同时这是一个 CLI 行为变化：`just process` 的默认配置已从 master 的 task-bound
`process_qwen38_api.yaml` 切换为不含 dataset block 的共享 profile；不带数据路径时不能再沿用
master 的隐含数据集，必须显式传 `--data-path`（或使用旧的 `--config` 兼容入口）。

它没有继续推进 master 文档中 R0–R8 的 canonical owner 重构，且新增编排后几个文件反而更厚：

| 文件 | master 行数 | 当前行数 | 主要负担 |
| --- | ---: | ---: | --- |
| src/robotwin_annotation_v2/config.py | 544 | 1,924 | profile overlay、legacy parser、binding、类型校验 |
| src/robotwin_annotation_v2/application/dataset_input.py | 167 | 1,412 | native/legacy/manifest/collection 三套解析 |
| src/robotwin_annotation_v2/application/dataset_runtime.py | 959 | 1,711 | CLI、兼容 seam、path mode、workflow dispatch |

这并不表示面向对象方向错误：domain 的 enum/dataclass、adapter、pipeline 和 workflow 边界
是清楚的；问题是 binding/provenance/legacy 兼容逻辑仍集中在 application/config 大文件中。

### 3.1 与 master R0–R8 的准确关系

feature 不是 R0–R8 的“下一阶段完成版”，而是横向增加配置与数据输入能力。按退出条件核对：

| master 阶段 | feature 的实际覆盖 | 结论 |
| --- | --- | --- |
| R0 行为/输入合同冻结 | 增加 profile、binding 和 focused parser/input tests；full-root、nested collection、frozen-source 的端到端合同仍未冻结 | 部分覆盖 |
| R1–R2 import 边界、canonical mask codec | 基本复用 master owner，没有实质推进 | 未改变 |
| R3 dataset runtime 拆分与 typed discovery | 增加 discovery/binding，但 `config.py`、`dataset_input.py`、`dataset_runtime.py` 变厚，仍有 dynamic re-bind | 部分覆盖且引入新债务 |
| R4 workflow / R5 publisher / R6 stage cohesion / R7 timeline | 主要复用 master 的 workflow、publisher、stage 和 detector | 未实质推进 |
| R8 shim 清理与静态质量 | Ruff/mypy 仍通过，但 legacy fallback、alias 和第二绑定路径增加 | 质量通过，shim 反而增多 |

因此，若目标是继续“精简、易读、面向对象”，下一轮应先收敛 feature 新增的 binding/provenance
入口，再继续 R0/R3/R8 的退出条件，而不是再添加一层 facade。

## 4. 可读性、冗余和性能审查

### 4.1 做得较好的地方

- AnnotationMode、TargetProfile、DatasetBinding、PipelineProfile 等类型让主要合同可被静态检查；
- Qwen、SAM、URDF、dataset I/O 和 canonical publisher 已有相对明确的 adapter/stage owner；
- 薄脚本只负责启动 package API，旧入口仍有兼容路径；
- 关键错误多数采用 fail-closed，而不是发布猜测 mask。

### 4.2 当前最值得改的重复

1. episode ID 校验在 config.py、dataset_binding.py、dataset_input.py、dataset_runtime.py、
   urdf_batch.py 各有一份；有的拒绝重复，有的自动去重，输入相同时行为不一致。
2. DatasetBinding 同时在 config.py 定义并从 application 模块 re-export；同时存在
   PipelineProfile.bind_dataset() 和全局 bind_dataset()，公开入口有两个。
3. collection summary 仍大量使用 ad-hoc dict，没有统一复用 typed process summary。
4. build_sam_dynamic_config() 与 DatasetBinding 仍是两条兼容入口；SamWorkflow 已在动态
   manifest、summary 和 per-episode artifact 边界补齐 task/profile/source provenance，但长期仍应
   收敛为单一 binding owner。
5. DatasetBinding 虽然是 frozen=True，其中的 manifest_data 仍是可变 dict；“immutable”
   目前主要是 DTO 引用不可替换，不是深层不可变。
6. `adapters.robotwin_dataset` 新增依赖 `config.validate_dataset_component`，adapter 与配置
   parser 形成反向耦合；共享的 task/camera value validator 应下沉到 domain 或独立边界模块。
7. collection summary 顶层 `profile` 仍表示 CLI selector，但每条 task record 现在记录实际
   `target_profile`，并在可用时记录不可变 `prompt_bundle`，可据此复现自动路由结果。
8. `_is_legacy_single_task_manifest` 目前主要按字段形状识别 legacy，可能把缺字段的现代
   manifest 当作兼容输入并静默补值；这会掩盖 manifest 错误，应该改为版本白名单 + 明确迁移提示。

### 4.3 可以直接节省的 I/O

- path runtime、SAM workflow、URDF workflow 会重复 discovery；
- preflight 对每个 episode 重读 Parquet 并完整解码视频，native manifest 缺失时
  frame_shape() 还会再次解码；
- discovery 只返回 episode 列表，没有把已测的 frame count、shape、video surplus 带给后续阶段。

建议把 DiscoveryResult 扩展为带 measurement/cache 的 typed 对象，从 resolver 一直传到
preflight/workflow；同一运行内只做一次文件存在性检查和必要的视频测量。这样既减少耗时，也
能避免不同阶段对“完整 episode”得出不同结论。

## 5. 第 4 点：root provenance 是什么

Manifest 的 `source_dataset_root`（历史文件有时叫 `dataset_root`）通常记录“生成
 manifest 时的原始路径”，例如：

~~~
{"source_dataset_root": "/old/machine/robotwin/task-a"}
~~~

数据搬到新机器后，实际访问路径可能是 /data/robotwin/task-a。当前 feature 为了让移动后
仍能运行，会把 runtime binding 的路径写回内存中的 `manifest_data["dataset_root"]`；而很多
真实 extract 只保留 `source_dataset_root`，并没有 `dataset_root`。好处是适配器不会因旧机器
绝对路径失效；代价是如果下游重建 dynamic manifest，原始来源字段可能被丢弃，审计时无法区分
“声明来源”和“本次实际访问位置”，也无法及时发现绑定到了错误的数据根。

建议保留两份值，不要覆盖：

~~~json
{
  "source_dataset_root": "/old/machine/robotwin/task-a",
  "declared_dataset_root": "/old/machine/robotwin/task-a",
  "runtime_dataset_root": "/data/robotwin/task-a",
  "dataset_root": "/data/robotwin/task-a",
  "root_relocated": true,
  "allow_relocated_root": true
}
~~~

推荐边界：

- （建议的未来行为）直接使用旧 task-bound config 或 adapter 时，严格校验 declared/source root；
- 明确经过 DatasetBinding 的运行允许 relocation，但必须写出上述 provenance；
- source run、输入文件和输出 manifest 中都保留 source/declared 与 runtime 两个值，便于复现和审计；
- dynamic manifest 应继承这些字段，不能用一次新的基础 dict 覆盖输入 manifest。

当前实现保留 `source_dataset_root`、`source_manifest_path` 和 `runtime_dataset_root`（并在
绑定 manifest 中更新运行时 `dataset_root`），但尚未写出 `root_relocated`，也不会校验 source
root 与 runtime root 是否确实为同一份数据；这仍是后续 provenance/identity 加强项。

这不是“路径能不能访问”的问题，而是数据身份和来源是否可追溯的问题。

## 6. 第 5 点：严格配置类型校验是什么

这是配置输入合同问题，不是 target 语义问题。当前 parser 在若干字段使用隐式转换：

~~~python
int(1.5)          # 变成 1
int("2")          # 被接受
bool("false")     # 变成 True
float("nope")     # 可能抛出裸 ValueError，而不是统一 ConfigError
~~~

因此 YAML 表面上看似合法，运行行为却可能完全不同：GPU、token 上限、QC 开关尤其危险。

当前分支已经增加统一的严格 parser：

- 整数字段只接受真正的 int，拒绝 bool、float 和数字字符串；
- 浮点字段只接受真正的 int/float，检查 finite、上下界和 NaN/Inf；
- boolean 字段只接受真正的 bool，不接受 "false"/"true"；
- list 字段先检查容器类型，再逐项检查元素类型；
- 所有解析错误统一包装成带字段路径的 ConfigError，例如
  mask.qc_max_attempts must be a positive integer；
- episode ID、GPU、task/camera 名称共用 canonical validator，并明确“重复值拒绝”还是
  “有序去重”，不要在不同入口采用不同策略。

profile/legacy 两条加载路径均使用这些检查，相关 unit tests 已覆盖字符串、浮点、bool、NaN/Inf
和超大整数。episode ID 校验仍有多个 owner，是下一轮可继续收敛的重复逻辑。

## 7. 当前入口的已知限制与安全用法

### 7.1 已验证可用的输入

带 per-task manifest 的抽取目录或 collection 可以走共享 profile：

~~~bash
# 普通 target-only collection；显式 camera 便于复现
just process /DATA/disk8/xuran/add_mask_robotwin/dataset/target_only_20_v2 \
  --camera cam_high

# contact-press 三任务 collection；按 task 选择更适合 smoke
just process /DATA/disk8/xuran/add_mask_robotwin/dataset/target_only_20_v2_contact_press \
  --task click_alarmclock --camera cam_high
~~~

当前 target_only_20_v2 是 11 个 task、220 条抽样 episode；其中只有上述三个
contact_action_site task 自动切到 contact_press，不是 11 个 task 都使用该 profile。

### 7.2 frozen-source 单任务入口

当前 runtime 支持单任务 `--data-path + --source-run-dir`。所以可直接使用：

~~~bash
just process DATASET_ROOT OUTPUT_ROOT --source-run-dir SOURCE_RUN ...
~~~

单任务目录会绑定共享 `configs/process.yaml`；collection 必须先加 `--task` 选择一个任务，
否则会因没有唯一 frozen-source 映射而拒绝。source run 的 task、camera、annotation mode、
target profile、prompt bundle、episode 和 lineage 都会交叉校验，不能跨 profile 复用。

### 7.3 API key 与远程数据边界

当前实现优先读环境变量 QWEN_API_KEY，否则读取被 Git 忽略的
secrets/qwen_api_key.txt；key 不写入 YAML、命令行或运行产物。当前 key 文件权限已核对为
600，这部分做法是合适的。

但共享 configs/process.yaml 默认 qwen.runtime=api，意味着普通入口会把图片和 prompt
发送到远程 MaaS，并可能产生费用或数据合规问题。建议：

1. 生产/共享仓库默认使用 local，远程 API 通过显式 profile 或 --allow-remote opt-in；
2. 在运行前显示 endpoint、数据是否出机和费用提示；
3. 本机被 Git 忽略的 `docs/qwen_3_8_maas_api.md` 已确认含明文凭据且权限为 640；
   `.gitignore` 不能撤销已经暴露的风险，应立即 revoke/rotate，删除文档中的明文，并把凭据
   只放在权限为 600 的 secrets 文件或受控 secret manager 中；
4. `QWEN_API_KEY_FILE` 可以指向任意本地文件，当前代码不校验 owner/权限；建议启动前检查
   文件为单行、仅当前用户可读（通常 `chmod 600`），并避免把路径指向共享目录。
5. 若数据含敏感内容，先确认 MaaS 的保留/训练/跨境策略，再运行 API profile。

## 8. 建议实施顺序

### P0：先修正确性和安全边界

1. 混合 native root：继续为 metadata 缺行、重复 episode 和 task/file 交叉一致性补 integration
   覆盖；当前显式 `--task` 已按 `full_structured_tasks[0]` 过滤；
2. 继续统一 annotation contract、`target_profile` 与历史 `profile` alias 的 manifest/schema
   版本规则；当前主要路径已完成一致性校验；
3. 收敛 dynamic manifest 与 DatasetBinding 两条兼容入口，并为 frozen-source positional CLI
   和 collection summary provenance 补端到端测试；
4. 远程 API 改为显式 opt-in，并清理/轮换本地明文 key。

### P1：再收敛合同与可读性

1. 提取一个 canonical episode-selection validator；
2. 用不可变 provenance DTO 替代可变 manifest_data dict，或明确其只读边界；
3. 将 DatasetBinding、collection summary 和 dynamic manifest 的 owner 收敛到单一路径；
4. 为 full-root、nested collection、contact profile 和 frozen-source 各补 integration test。

### P2：最后做性能和拆分

1. 缓存 discovery/视频测量结果，避免同一 batch 重复解码；
2. 将 config.py 拆成 profile parser、legacy parser、binding/provenance 三个 cohesive
   模块；
3. 将 dataset_input.py 拆成 native discovery、manifest adapter、collection resolver；
4. 将 dataset_runtime.py 保持为窄 CLI compatibility layer，把业务编排留在 workflow。

## 9. 验证记录

当前工作区验证结果（以最终命令输出为准）：

~~~
unit + integration: 972 passed, 1 skipped
ruff:               passed
mypy src:           passed (71 source files)
git diff --check:   passed
~~~

现有 fixture 覆盖 alias、mixed metadata、full-root task filtering、profile routing 和
frozen-source provenance；真实 27,500 episode full-root 的 GPU/Qwen 全链路仍未运行，不能仅凭
单元测试宣称像素质量或远程服务合同已经冻结。
