#!/bin/bash

# 默认参数
START_PORT=8000
INSTANCE_COUNT=2
GPU_START_ID=0

# 解析命令行参数
for arg in "$@"
do
    case $arg in
        --start_port=*)
        START_PORT="${arg#*=}"
        shift
        ;;
        --instance_count=*)
        INSTANCE_COUNT="${arg#*=}"
        shift
        ;;
        --gpu_start=*)
        GPU_START_ID="${arg#*=}"
        shift
        ;;
        *)
        # 未知参数
        echo "Unknown parameter: $arg"
        echo "Usage: $0 [--start_port=port_number] [--instance_count=count] [--gpu_start=start_gpu_id]"
        exit 1
        ;;
    esac
done

# 模型路径
MODEL_PATH=/your/path/to/Qwen/QwQ-32B

# 根据参数生成实例配置
declare -A INSTANCE_CONFIGS
PIDS=()
for ((i=0; i<INSTANCE_COUNT; i++)); do
  port=$((START_PORT + i))
  gpu1=$((GPU_START_ID + i * 2))
  gpu2=$((GPU_START_ID + i * 2 + 1))
  INSTANCE_CONFIGS["instance$((i+1))"]="$port:$gpu1,$gpu2"
done

# 模型并行化参数
TENSOR_PARALLEL_SIZE=2
PIPELINE_PARALLEL_SIZE=1

# 暴露 sleep 和 wake_up 接口
export VLLM_SERVER_DEV_MODE=1

# 捕获 SIGINT (Ctrl+C) 信号以停止所有实例
trap "echo 'Stopping instances...'; for pid in \"${PIDS[@]}\"; do kill \$pid; done; exit 0" SIGINT

# 打印配置信息
echo "Starting vLLM instances with configuration:"
echo "Start Port: $START_PORT"
echo "Instance Count: $INSTANCE_COUNT"
echo "GPU Start ID: $GPU_START_ID"
echo "GPU Devices per Instance: 2"
echo "Total GPU Devices Used: $((INSTANCE_COUNT * 2))"

# 启动所有实例
for instance_name in "${!INSTANCE_CONFIGS[@]}"; do
  # 解析配置
  config="${INSTANCE_CONFIGS[$instance_name]}"
  port="${config%%:*}"
  gpu_devices="${config#*:}"
  echo "Starting $instance_name on port $port using GPUs $gpu_devices"
  
  # 启动服务实例并记录它的 PID
  CUDA_VISIBLE_DEVICES="$gpu_devices" \
  vllm serve "$MODEL_PATH" \
    --port "$port" \
    --enable_sleep_mode \
    --gpu_memory_utilization 0.9 \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --pipeline_parallel_size "$PIPELINE_PARALLEL_SIZE" &
  PIDS+=($!)
done

echo "All instances started in background. Press Ctrl+C to stop."

# 等待所有后台进程完成
wait
