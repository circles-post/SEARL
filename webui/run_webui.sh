#!/bin/bash

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 安装依赖
# echo "正在安装依赖..."
# pip install -r requirements.txt

# 启动 WebUI
echo "正在启动 SEARL WebUI..."
echo "服务将在 http://localhost:7860 启动"
echo "按 Ctrl+C 停止服务"

python3 app.py 
