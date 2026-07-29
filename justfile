# process_data_v2/justfile
# 快速命令入口

# 默认：显示帮助
default:
    @just --list

# === 环境管理 ===

# 安装阶段 1 依赖（关键帧标注）
install:
    uv sync --extra phase1 --extra dev

# 安装完整依赖（包括阶段 2 视频传播）
install-full:
    uv sync --extra phase2 --extra dev

# 临时方案：链接到 process_data 环境进行开发测试
link-dev-env:
    @echo "⚠️  临时方案：链接到 ../process_data/.venv"
    @echo "    生产环境请使用 'just install'"
    ln -sf ../process_data/.venv .venv
    @echo "✅ 已链接，可运行测试"

# 清理环境
clean:
    rm -rf .venv
    rm -rf build dist *.egg-info
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true

# === 测试 ===

# 运行所有测试
test:
    uv run pytest -v

# 快速测试（只跑 unit，不跑 integration）
test-fast:
    uv run pytest tests/unit -v

# 测试覆盖率
test-cov:
    uv run pytest --cov=src/robotwin_annotation_v2 --cov-report=term --cov-report=html

# 类型检查
typecheck:
    uv run mypy src/robotwin_annotation_v2

# Lint
lint:
    uv run ruff check src tests

# Format
fmt:
    uv run ruff format src tests

# === 阶段 1：关键帧标注 ===

# 准备关键帧（pilot: move_pillbottle_pad）
prepare-keyframes episode_id="007152":
    uv run python -m robotwin_annotation_v2.cli.keyframes prepare \
        --task move_pillbottle_pad \
        --episode {{episode_id}} \
        --camera cam_high

# 审查关键帧（打开 contact sheet）
review-keyframes run_id:
    uv run python -m robotwin_annotation_v2.cli.keyframes review \
        --run-id {{run_id}}

# 批准 seed
approve-seed run_id episode_id slot candidate_id:
    uv run python -m robotwin_annotation_v2.cli.keyframes approve \
        --run-id {{run_id}} \
        --episode {{episode_id}} \
        --slot {{slot}} \
        --candidate {{candidate_id}}

# === Pilot 批量运行 ===

# 运行 pilot（10 episodes）
run-pilot:
    #!/usr/bin/env bash
    set -euo pipefail
    episodes=(007152 007157 011016 011018 012183 012290 013758 013771 022575 024753)
    for ep in "${episodes[@]}"; do
        echo "Processing episode $ep..."
        just prepare-keyframes "$ep"
    done

# === 清理与工具 ===

# 清理 artifacts（保留最近 N 个 run）
clean-artifacts keep="5":
    #!/usr/bin/env bash
    cd artifacts/keyframes/runs
    ls -t | tail -n +$(({{keep}} + 1)) | xargs -r rm -rf

# 检查 GPU 可用性
check-gpu:
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv

# 生成配置模板
init-config:
    mkdir -p configs
    uv run python -m robotwin_annotation_v2.bootstrap.init_config

# === 开发工具 ===

# 交互式 Python shell（加载 domain 模型）
shell:
    uv run python

# 查看最近的 run
list-runs:
    ls -lht artifacts/keyframes/runs/ | head -10

# 查看某个 run 的 manifest
inspect-run run_id:
    cat artifacts/keyframes/runs/{{run_id}}/run_manifest.json | uv run python -m json.tool
