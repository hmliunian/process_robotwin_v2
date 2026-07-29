#!/usr/bin/env bash
# 整理 process_data_v2 文件结构，按照架构设计文档规划

set -euo pipefail

echo "🧹 整理 process_data_v2 文件结构..."
echo ""

# 创建 docs/ 目录
mkdir -p docs

echo "📋 步骤 1: 移动设计和分析文档到 docs/"
echo "----------------------------------------"

# 移动设计文档
mv -v process_data_v2_architecture_design.md docs/ 2>/dev/null || echo "  已在 docs/"
mv -v docs/qwen_limitation_and_improvements.md docs/ 2>/dev/null || echo "  已在 docs/"

echo ""
echo "📋 步骤 2: 移动项目报告到 docs/reports/"
echo "----------------------------------------"

mkdir -p docs/reports

# 移动各种报告文档
for doc in \
    PROGRESS.md \
    STATUS.md \
    EXECUTIVE_SUMMARY.md \
    FINAL_SUMMARY.md \
    PROJECT_COMPLETE.md \
    DELIVERY_CHECKLIST.md \
    TOOL_TASK_COMPLETE.md \
    EXECUTION_TASK_FINAL.md \
    PROJECT_SUMMARY.md \
    FINAL_STATUS.md \
    NEXT_STEPS.md \
    CLEANUP_PLAN.md
do
    if [ -f "$doc" ]; then
        mv -v "$doc" docs/reports/
    fi
done

echo ""
echo "📋 步骤 3: 保留根目录必要文档"
echo "----------------------------------------"

# 根目录只保留：
# - README.md
# - QUICKSTART.md
# - DOCS_INDEX.md (需要更新)

echo "  保留: README.md"
echo "  保留: QUICKSTART.md"
echo "  保留: DOCS_INDEX.md (需要更新路径)"

echo ""
echo "📋 步骤 4: 检查源代码结构"
echo "----------------------------------------"

# 检查源代码是否符合设计
echo "  检查 src/robotwin_annotation_v2/"
ls -la src/robotwin_annotation_v2/ | grep "^d" | awk '{print "    ✅", $9}'

echo ""
echo "📋 步骤 5: 创建缺失的目录"
echo "----------------------------------------"

# 按照设计文档创建目录
mkdir -p artifacts/keyframes/runs
mkdir -p artifacts/propagation
mkdir -p artifacts/qc
mkdir -p configs
mkdir -p scripts
mkdir -p tools
mkdir -p tests/unit
mkdir -p tests/integration
mkdir -p tests/contract

echo "  ✅ artifacts/keyframes/runs"
echo "  ✅ artifacts/propagation"
echo "  ✅ artifacts/qc"
echo "  ✅ configs"
echo "  ✅ scripts"
echo "  ✅ tools"
echo "  ✅ tests/{unit,integration,contract}"

echo ""
echo "📋 步骤 6: 更新 DOCS_INDEX.md"
echo "----------------------------------------"

cat > DOCS_INDEX.md << 'EOF'
# Process Data V2 - 文档索引

## 📚 快速入口

### 🎯 新用户从这里开始
- **[README.md](README.md)** - 项目概览
- **[QUICKSTART.md](QUICKSTART.md)** - 快速开始指南

---

## 🏗️ 架构与设计

### 核心设计文档
- **[架构设计](docs/process_data_v2_architecture_design.md)** (29KB)
  - 完整的架构设计文档
  - 分层架构、领域模型、接口定义
  - **这是项目的权威设计文档**

- **[Qwen Limitation 分析](docs/qwen_limitation_and_improvements.md)** (15KB)
  - Qwen 的 3 个核心 limitation
  - 6 个改进原则
  - 角色感知 grounding 设计

---

## 📊 项目报告

### 完成报告（按时间倒序）
- **[最终状态报告](docs/reports/FINAL_STATUS.md)** - 最新状态（推荐）
- **[项目总结](docs/reports/PROJECT_SUMMARY.md)** - 完整总结
- **[项目完成报告](docs/reports/PROJECT_COMPLETE.md)** - Phase 1 完成报告
- **[交付清单](docs/reports/DELIVERY_CHECKLIST.md)** - 交付物清单

### 执行任务报告
- **[执行任务最终报告](docs/reports/EXECUTION_TASK_FINAL.md)** - 工具任务完整报告
- **[工具任务完成报告](docs/reports/TOOL_TASK_COMPLETE.md)** - 数据读取验证

### 历史进度（参考）
- [实施进度](docs/reports/PROGRESS.md)
- [项目状态](docs/reports/STATUS.md)
- [执行摘要](docs/reports/EXECUTIVE_SUMMARY.md)
- [最终总结](docs/reports/FINAL_SUMMARY.md)

### 计划文档
- **[下一步计划](docs/reports/NEXT_STEPS.md)** - Phase 2 实施计划

---

## 📁 项目结构

```
process_data_v2/
├── README.md                    # 项目概览
├── QUICKSTART.md               # 快速开始
├── DOCS_INDEX.md               # 本文档
│
├── docs/                       # 文档
│   ├── process_data_v2_architecture_design.md  # 架构设计
│   ├── qwen_limitation_and_improvements.md     # Qwen 分析
│   └── reports/                # 项目报告
│       ├── FINAL_STATUS.md
│       ├── PROJECT_SUMMARY.md
│       └── ...
│
├── src/robotwin_annotation_v2/ # 源代码
│   ├── domain/                 # 领域层
│   ├── ports/                  # 接口层
│   ├── application/            # 应用层
│   ├── adapters/               # 适配器层
│   ├── bootstrap/              # 依赖注入
│   └── cli/                    # 命令行
│
├── tests/                      # 测试
│   ├── unit/                   # 单元测试
│   ├── integration/            # 集成测试
│   └── contract/               # 契约测试
│
├── configs/                    # 配置文件
├── scripts/                    # 脚本
├── tools/                      # 工具
├── artifacts/                  # 输出
│   ├── keyframes/
│   ├── propagation/
│   └── qc/
│
├── pyproject.toml             # 项目配置
└── justfile                   # 命令快捷方式
```

---

## 🚀 快速命令

```bash
# 查看架构设计
cat docs/process_data_v2_architecture_design.md

# 查看最新状态
cat docs/reports/FINAL_STATUS.md

# 运行测试
just test-fast

# 查看所有命令
just --list
```

---

## 📞 文档导航

| 我想... | 查看文档 |
|---------|----------|
| 了解项目架构 | [架构设计](docs/process_data_v2_architecture_design.md) |
| 了解 Qwen 改进 | [Qwen 分析](docs/qwen_limitation_and_improvements.md) |
| 查看最新状态 | [最终状态报告](docs/reports/FINAL_STATUS.md) |
| 了解下一步 | [下一步计划](docs/reports/NEXT_STEPS.md) |
| 快速开始 | [QUICKSTART.md](QUICKSTART.md) |

---

**最后更新**: 2026-07-29
**项目状态**: ✅ Phase 1 完成，文件结构已整理
EOF

echo "  ✅ 已更新 DOCS_INDEX.md"

echo ""
echo "📊 整理结果"
echo "=========================================="
echo ""
echo "根目录文档 (3 个):"
ls -1 *.md 2>/dev/null || echo "  无"
echo ""
echo "设计文档 (docs/):"
ls -1 docs/*.md 2>/dev/null || echo "  无"
echo ""
echo "报告文档 (docs/reports/):"
ls -1 docs/reports/*.md 2>/dev/null | wc -l | xargs echo "  共" | sed 's/$/ 个/'
echo ""
echo "源代码目录:"
ls -d src/robotwin_annotation_v2/*/ 2>/dev/null | sed 's|^|  |'
echo ""
echo "✅ 文件结构整理完成！"
echo ""
echo "📚 查看文档索引: cat DOCS_INDEX.md"
