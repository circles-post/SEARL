#!/bin/bash

# 设置默认参数
CONCURRENCY=10
NUM_SAMPLES=100
MODEL_NAME="/your/path/to/Qwen/QwQ-32B"
PORT=9000
OUTPUT_FILE="async_results.csv"

# 显示帮助信息
show_help() {
    echo "异步并行生成器测试脚本"
    echo "用法: $0 [选项]"
    echo "选项:"
    echo "  -c, --concurrency  设置并发请求数 (默认: 10)"
    echo "  -s, --samples      设置样本数量 (默认: 100)"
    echo "  -m, --model        设置模型名称 (默认: local)"
    echo "  --port             设置API端口 (默认: 8080)"
    echo "  -o, --output       设置输出文件路径 (默认: async_results.csv)"
    echo "  -i, --input        设置输入文件路径 (可选)"
    echo "  -l, --limit        限制处理的样本数量 (可选)"
    echo "  -t, --timeout      设置请求超时时间 (默认: 60.0)"
    echo "  -h, --help         显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 -c 20 -s 500 -m local --port 8080 -o results.csv"
    echo "  $0 -c 10 -i input_queries.csv -o results.csv"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -c|--concurrency)
            CONCURRENCY="$2"
            shift 2
            ;;
        -s|--samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        -m|--model)
            MODEL_NAME="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -i|--input)
            INPUT_FILE="$2"
            shift 2
            ;;
        -l|--limit)
            LIMIT="$2"
            shift 2
            ;;
        -t|--timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

# 构建命令行
CMD="python3 async_generator_test.py --concurrency $CONCURRENCY --model_name $MODEL_NAME --port $PORT --output $OUTPUT_FILE"

# 添加可选参数
if [ ! -z "$INPUT_FILE" ]; then
    CMD="$CMD --input_file $INPUT_FILE"
else
    CMD="$CMD --num_samples $NUM_SAMPLES"
fi

if [ ! -z "$LIMIT" ]; then
    CMD="$CMD --limit $LIMIT"
fi

if [ ! -z "$TIMEOUT" ]; then
    CMD="$CMD --timeout $TIMEOUT"
fi

# 显示将要执行的命令
echo "执行命令: $CMD"
echo "开始测试..."

# 执行命令
cd "$(dirname "$0")"
eval $CMD

# 检查执行结果
if [ $? -eq 0 ]; then
    echo "测试完成，结果已保存到 $OUTPUT_FILE"
else
    echo "测试执行出错，请检查日志"
fi