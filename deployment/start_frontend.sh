#!/bin/bash

# GRPO微调模型前端启动脚本

echo "=========================================="
echo "启动GRPO微调模型前端服务"
echo "=========================================="

# 检查是否在正确的目录
if [ ! -f "frontend/index.html" ]; then
    echo "错误: 请在项目根目录运行此脚本"
    exit 1
fi

# 进入前端目录
cd frontend

# 检查Node.js环境（可选）
if command -v node &> /dev/null; then
    echo "检测到Node.js环境，使用npm启动..."
    if [ -f "package.json" ]; then
        npm install
        npm start
    else
        echo "使用Python HTTP服务器启动..."
        python -m http.server 3000
    fi
else
    echo "使用Python HTTP服务器启动..."
    python -m http.server 3000
fi

echo "前端服务地址: http://localhost:3000"
echo "按 Ctrl+C 停止服务"
echo "==========================================" 