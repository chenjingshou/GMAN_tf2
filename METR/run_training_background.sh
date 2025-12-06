#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=""  # 禁用GPU，使用CPU

# 创建日志目录
mkdir -p logs
mkdir -p models

# 获取当前时间作为标识
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/training_${TIMESTAMP}.log"
MODEL_FILE="models/GMAN_${TIMESTAMP}"

# 后台运行训练，输出重定向到日志文件
nohup python -u train.py \
    --max_epoch 50 \
    --batch_size 32 \
    --patience 5 \
    --model_file ${MODEL_FILE} \
    --log_file ${LOG_FILE} \
    > logs/nohup_${TIMESTAMP}.out 2>&1 &

# 保存进程ID
PID=$!
echo ${PID} > logs/training_${TIMESTAMP}.pid

echo "Training started in background"
echo "Process ID: ${PID}"
echo "Log file: ${LOG_FILE}"
echo "Nohup output: logs/nohup_${TIMESTAMP}.out"
echo "PID file: logs/training_${TIMESTAMP}.pid"
echo ""
echo "Monitor training with:"
echo "  tail -f logs/nohup_${TIMESTAMP}.out"
echo ""
echo "Stop training with:"
echo "  kill ${PID}"