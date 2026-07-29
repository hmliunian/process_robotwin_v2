#!/usr/bin/env bash
# 恢复 Qwen v1 服务（临时方案，直到 v2 CUDA 问题解决）

set -euo pipefail

echo "🔄 恢复 Qwen v1 服务..."
echo ""

# 配置
MODEL_PATH="/DATA/disk8/xuran/add_mask_robotwin/process_data/checkpoints/Qwen/Qwen3.5-27B"
PORT=18086
DEVICE="cuda:0"
PID_FILE="/DATA/disk8/xuran/add_mask_robotwin/process_data/run/qwen_device0/server.pid"
LOG_FILE="/DATA/disk8/xuran/add_mask_robotwin/process_data/run/qwen_device0/server.log"

# 切换到 process_data 目录
cd /DATA/disk8/xuran/add_mask_robotwin/process_data

# 创建 run 目录
mkdir -p "$(dirname "$PID_FILE")"

# 使用原来的 venv-qwen35 环境和 serve_qwen.py
VENV_PYTHON=".venv-qwen35/bin/python"

echo "📋 启动 Qwen v1 服务"
echo "   模型: $MODEL_PATH"
echo "   端口: $PORT"
echo "   设备: $DEVICE"
echo ""

# 启动服务（后台）
nohup $VENV_PYTHON -u scripts/serve_qwen.py \
    --model "$MODEL_PATH" \
    --served-model-name "qwen3.5-27b" \
    --host "127.0.0.1" \
    --port $PORT \
    --device "$DEVICE" \
    --dtype bfloat16 \
    --pid-file "$PID_FILE" \
    > "$LOG_FILE" 2>&1 &

# 等待启动
echo "   等待服务启动（模型加载需要时间）..."
sleep 10

# 验证服务
echo ""
echo "📋 验证服务"

MAX_RETRIES=20
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
        echo "   POST http://localhost:$PORT/v1/chat/completions"
        echo ""

        echo "📝 日志: tail -f $LOG_FILE"
        echo ""

        echo "✅ Qwen v1 服务恢复完成！"
        exit 0
    fi

    echo "   等待中... ($i/$MAX_RETRIES)"
    sleep 3
done

echo ""
echo "❌ 服务启动失败"
echo "   查看日志: tail -50 $LOG_FILE"
exit 1
