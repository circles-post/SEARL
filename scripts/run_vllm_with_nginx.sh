#!/bin/bash

# 默认参数
VLLM_START_PORT=8000
VLLM_INSTANCE_COUNT=2
VLLM_GPU_START_ID=0
NGINX_LISTEN_PORT=9000

# 解析命令行参数
for arg in "$@"
do
    case $arg in
        --vllm_start_port=*)
        VLLM_START_PORT="${arg#*=}"
        shift
        ;;
        --vllm_instance_count=*)
        VLLM_INSTANCE_COUNT="${arg#*=}"
        shift
        ;;
        --vllm_gpu_start=*)
        VLLM_GPU_START_ID="${arg#*=}"
        shift
        ;;
        --nginx_port=*)
        NGINX_LISTEN_PORT="${arg#*=}"
        shift
        ;;
        *)
        # 未知参数
        echo "Unknown parameter: $arg"
        echo "Usage: $0 [--vllm_start_port=port] [--vllm_instance_count=count] [--vllm_gpu_start=id] [--nginx_port=port]"
        exit 1
        ;;
    esac
done

# 1. 启动 vLLM 服务器
echo "Starting vLLM servers..."
bash scripts/vllm_server.sh \
    --start_port=$VLLM_START_PORT \
    --instance_count=$VLLM_INSTANCE_COUNT \
    --gpu_start=$VLLM_GPU_START_ID &

# 等待服务器启动
sleep 10

# 2. 准备 Nginx 上游端口列表
UPSTREAM_PORTS=()
for ((i=0; i<VLLM_INSTANCE_COUNT; i++)); do
    UPSTREAM_PORTS+=($((VLLM_START_PORT + i)))
done

# 将端口数组转换为逗号分隔的字符串
UPSTREAM_PORTS_STR=$(IFS=, ; echo "${UPSTREAM_PORTS[*]}")

# 3. 安装并配置 Nginx
echo "Setting up Nginx load balancer..."
bash scripts/install_nginx.sh \
    --upstream=$UPSTREAM_PORTS_STR \
    --port=$NGINX_LISTEN_PORT

# 4. 打印最终信息
echo ""
echo "============================================"
echo "vLLM servers and Nginx load balancer started"
echo "vLLM servers:"
for port in "${UPSTREAM_PORTS[@]}"; do
    echo " - http://localhost:$port"
done
echo "Nginx load balancer: http://localhost:$NGINX_LISTEN_PORT"
echo "============================================"
echo ""

# 捕获 SIGINT (Ctrl+C) 信号以停止所有服务
trap "echo 'Stopping all services...'; pkill -f 'vllm serve'; $HOME/nginx/sbin/nginx -s stop; exit 0" SIGINT

# 保持脚本运行
wait
