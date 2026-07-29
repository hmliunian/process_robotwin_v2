#!/usr/bin/env bash
# 重启 Qwen 服务，使用 v2 逻辑

set -euo pipefail

echo "🔄 重启 Qwen v2 服务..."
echo ""

# 配置
MODEL_PATH="/DATA/disk8/xuran/add_mask_robotwin/process_data/checkpoints/Qwen/Qwen3.5-27B"
PORT=18086
DEVICE="cuda:0"
PID_FILE="/DATA/disk8/xuran/add_mask_robotwin/process_data_v2/run/qwen_v2/server.pid"
LOG_FILE="/DATA/disk8/xuran/add_mask_robotwin/process_data_v2/run/qwen_v2/server.log"

# 创建 run 目录
mkdir -p "$(dirname "$PID_FILE")"

# 1. 停止旧服务
echo "📋 步骤 1: 停止旧服务"
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "   停止进程 $OLD_PID..."
        kill "$OLD_PID" || true
        sleep 2

        # 如果还活着，强制 kill
        if ps -p "$OLD_PID" > /dev/null 2>&1; then
            echo "   强制停止..."
            kill -9 "$OLD_PID" || true
        fi
    fi
    rm -f "$PID_FILE"
fi

# 检查端口是否被占用
if lsof -i :$PORT > /dev/null 2>&1; then
    echo "⚠️  端口 $PORT 仍被占用，尝试清理..."
    OLD_PID=$(lsof -t -i :$PORT)
    kill "$OLD_PID" || true
    sleep 2
fi

echo "   ✅ 旧服务已停止"
echo ""

# 2. 启动 v2 服务
echo "📋 步骤 2: 启动 Qwen v2 服务"
echo "   模型: $MODEL_PATH"
echo "   端口: $PORT"
echo "   设备: $DEVICE"
echo ""

# 使用 process_data 的 qwen 环境
VENV_PYTHON="/DATA/disk8/xuran/add_mask_robotwin/process_data/.venv-qwen35/bin/python"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "❌ Qwen 环境不存在: $VENV_PYTHON"
    exit 1
fi

# 启动服务（后台）
nohup $VENV_PYTHON scripts/serve_qwen_v2.py \
    --model "$MODEL_PATH" \
    --served-model-name "qwen3.5-27b" \
    --host "127.0.0.1" \
    --port $PORT \
    --device "$DEVICE" \
    --dtype bfloat16 \
    --pid-file "$PID_FILE" \
    > "$LOG_FILE" 2>&1 &

# 等待启动
echo "   等待服务启动..."
sleep 5

# 3. 验证服务
echo ""
echo "📋 步骤 3: 验证服务"

MAX_RETRIES=10
for i in $(seq 1 $MAX_RETRIES); do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "   ✅ 服务启动成功！"
        echo ""

        # 显示状态
        echo "📊 服务状态:"
        curl -s http://localhost:$PORT/health | python3 -m json.tool
        echo ""

        # 显示端点
        echo "🌐 可用端点:"
        echo "   GET  http://localhost:$PORT/health"
        echo "   GET  http://localhost:$PORT/v1/models"
        echo "   POST http://localhost:$PORT/v1/chat/completions  (v1 兼容)"
        echo "   POST http://localhost:$PORT/v2/ground            (v2 角色感知 grounding)"
        echo ""

        # 显示日志位置
        echo "📝 日志文件: $LOG_FILE"
        echo "   查看日志: tail -f $LOG_FILE"
        echo ""

        echo "✅ Qwen v2 服务重启完成！"
        exit 0
    fi

    echo "   等待中... ($i/$MAX_RETRIES)"
    sleep 2
done

echo ""
echo "❌ 服务启动失败"
echo "   查看日志: cat $LOG_FILE"
exit 1
