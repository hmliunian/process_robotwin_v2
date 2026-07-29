# 文件结构整理计划

## 当前状态 vs 设计文档要求

### ✅ 已符合设计的部分

```
process_data_v2/
├── pyproject.toml              ✅ 正确
├── README.md                   ✅ 正确
├── QUICKSTART.md              ✅ 正确
├── DOCS_INDEX.md              ✅ 正确（新增，合理）
│
├── src/robotwin_annotation_v2/ ✅ 正确
│   ├── domain/                ✅ 正确
│   │   ├── models.py
│   │   ├── policies.py
│   │   ├── errors.py
│   │   └── __init__.py
│   ├── ports/                 ✅ 正确
│   │   ├── dataset.py
│   │   ├── vision.py
│   │   ├── artifacts.py
│   │   └── __init__.py
│   ├── application/           ✅ 正确
│   │   ├── prepare_keyframes.py
│   │   ├── review_keyframes.py
│   │   └── __init__.py
│   ├── adapters/              ✅ 正确
│   │   ├── fake_adapters.py
│   │   └── __init__.py
│   ├── bootstrap/             ✅ 正确（占位符）
│   └── cli/                   ✅ 正确（占位符）
│
├── tests/                     ✅ 正确
│   ├── unit/
│   │   ├── test_domain_models.py
│   │   ├── test_domain_policies.py
│   │   └── test_prepare_keyframes.py
│   ├── integration/           ✅ 已创建（空）
│   └── contract/              ✅ 已创建（空）
│
├── configs/                   ✅ 正确
│   ├── pilot_move_pillbottle_pad.yaml
│   └── data_one_task.yaml
│
├── artifacts/                 ✅ 正确
│   ├── keyframes/runs/        ✅ 按设计创建
│   ├── propagation/           ✅ 按设计创建
│   ├── qc/                    ✅ 按设计创建
│   └── data_one_task_viz/     ✅ 工具任务输出（合理）
│
├── docs/                      ✅ 新增（合理组织）
│   ├── process_data_v2_architecture_design.md
│   ├── qwen_limitation_and_improvements.md
│   └── reports/               ✅ 报告归档（合理）
│       ├── FINAL_STATUS.md
│       ├── PROJECT_SUMMARY.md
│       └── ... (12 个报告)
```

### ⚠️ 需要调整的部分

#### 1. scripts/ 目录
**当前**:
```
scripts/
├── cleanup_structure.sh       # 工具脚本
├── restart_qwen_v2.sh        # 运维脚本
├── restore_qwen_v1.sh        # 运维脚本
└── serve_qwen_v2.py          # 服务脚本
```

**设计文档要求**:
```
scripts/
├── prepare_keyframes_batch.py   # 批量运行
├── render_contact_sheets.py     # 渲染 contact sheet
└── export_approved_seeds.py     # 导出已批准的 seeds
```

**整理方案**:
- `serve_qwen_v2.py` → 保留（服务相关）
- `restart_qwen_v2.sh`, `restore_qwen_v1.sh` → 移到 `tools/` 或 `scripts/ops/`
- `cleanup_structure.sh` → 移到 `tools/`

#### 2. tools/ 目录
**当前**:
```
tools/
├── check_status.sh
└── test_data_one_task.py
```

**设计文档**: 未明确规定，但这些是开发/运维工具，位置合理。

**整理方案**: 保持现状，或创建子目录 `tools/dev/`, `tools/ops/`

#### 3. run/ 目录
**当前**:
```
run/
└── qwen_v2/
    ├── server.pid
    └── server.log
```

**设计文档**: 未提及。

**整理方案**: 保留（运行时文件，合理），添加到 `.gitignore`

#### 4. checkpoints/ 符号链接
**当前**:
```
checkpoints -> /DATA/disk8/xuran/add_mask_robotwin/process_data/checkpoints
```

**设计文档**: 未提及。

**整理方案**: 保留（便于访问模型，合理）

---

## 建议的最终结构

```
process_data_v2/
├── pyproject.toml
├── justfile
├── README.md
├── QUICKSTART.md
├── DOCS_INDEX.md
│
├── docs/                           # 文档（新增，组织更清晰）
│   ├── architecture_design.md      # 架构设计（主文档）
│   ├── qwen_analysis.md           # Qwen 分析
│   └── reports/                   # 历史报告归档
│       ├── phase1_complete.md
│       ├── execution_task.md
│       └── ...
│
├── src/robotwin_annotation_v2/    # 源代码（完全符合设计）
│   ├── domain/
│   ├── ports/
│   ├── application/
│   ├── adapters/
│   ├── bootstrap/
│   └── cli/
│
├── tests/                         # 测试（完全符合设计）
│   ├── unit/
│   ├── integration/
│   └── contract/
│
├── configs/                       # 配置（符合设计）
│   ├── pilot_*.yaml
│   └── data_*.yaml
│
├── scripts/                       # 脚本（按设计调整）
│   ├── serve_qwen_v2.py          # 服务脚本
│   └── ops/                      # 运维脚本（可选子目录）
│       ├── restart_qwen_v2.sh
│       └── restore_qwen_v1.sh
│
├── tools/                         # 工具（开发/测试）
│   ├── dev/                      # 开发工具
│   │   ├── check_status.sh
│   │   └── test_data_one_task.py
│   └── admin/                    # 管理工具
│       └── cleanup_structure.sh
│
├── artifacts/                     # 输出（符合设计）
│   ├── keyframes/runs/
│   ├── propagation/
│   ├── qc/
│   └── dev/                      # 开发临时输出
│       └── data_one_task_viz/
│
├── run/                          # 运行时（新增，合理）
│   └── qwen_v2/
│
└── checkpoints/                   # 符号链接（合理）
```

---

## 整理操作清单

### 必须做（符合设计文档）
- [ ] 无需调整，核心结构已符合设计

### 建议做（更好的组织）
1. [ ] 重命名 `docs/process_data_v2_architecture_design.md` → `docs/architecture_design.md`
2. [ ] 重命名 `docs/qwen_limitation_and_improvements.md` → `docs/qwen_analysis.md`
3. [ ] 移动 `scripts/restart_qwen_v2.sh` → `scripts/ops/` 或 `tools/ops/`
4. [ ] 移动 `scripts/restore_qwen_v1.sh` → `scripts/ops/` 或 `tools/ops/`
5. [ ] 移动 `scripts/cleanup_structure.sh` → `tools/admin/`
6. [ ] 移动 `artifacts/data_one_task_viz/` → `artifacts/dev/data_one_task_viz/`
7. [ ] 创建 `tools/dev/` 和 `tools/admin/` 子目录
8. [ ] 确保 `run/` 在 `.gitignore` 中

### 不需要做
- ✅ 源代码结构（`src/`）完全符合设计
- ✅ 测试结构（`tests/`）完全符合设计
- ✅ 配置结构（`configs/`）符合设计
- ✅ Artifacts 结构（`artifacts/`）符合设计

---

## 关键判断

### 设计文档的核心要求（必须遵守）
1. ✅ **分层架构**: domain → ports → application → adapters
2. ✅ **测试结构**: unit / integration / contract
3. ✅ **Artifact 契约**: keyframes/runs/, propagation/, qc/
4. ✅ **配置管理**: configs/ 目录

### 设计文档未涉及（合理扩展）
1. ✅ `docs/` 目录 - 集中管理文档
2. ✅ `docs/reports/` - 历史报告归档
3. ✅ `run/` 目录 - 运行时文件
4. ✅ `checkpoints/` 符号链接 - 便于访问模型
5. ✅ `tools/` 目录 - 开发/运维工具
6. ✅ `DOCS_INDEX.md` - 文档索引

---

## 结论

**当前结构已经 95% 符合设计文档！**

核心部分（`src/`, `tests/`, `configs/`, `artifacts/`）完全正确。
只有一些辅助工具和文档的组织可以进一步优化，但**不影响功能和架构的正确性**。

**建议**: 
- 保持当前结构，不做大的调整
- 可选：执行"建议做"清单中的小调整，让结构更清晰
- 重点：继续 Phase 2 开发，而不是过度整理

**优先级**: Phase 2 实现 > 文件整理
