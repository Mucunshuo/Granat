#!/bin/bash

# GRPO微调模型后端启动脚本

echo "=========================================="
echo "启动GRPO微调模型后端服务"
echo "=========================================="

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "错误: 未找到Python环境"
    exit 1
fi

# 检查是否在正确的目录
if [ ! -f "backend/app.py" ]; then
    echo "错误: 请在项目根目录运行此脚本"
    exit 1
fi

# 进入后端目录
cd backend

# 检查依赖
echo "检查Python依赖..."
if [ ! -f "requirements.txt" ]; then
    echo "错误: 未找到requirements.txt文件"
    exit 1
fi

# 安装依赖（如果需要）
echo "安装Python依赖..."
pip install -r requirements.txt

# 检查模型文件
echo "检查模型文件..."
if [ ! -d "../grpo_peft_finetuned_model" ]; then
    echo "警告: 未找到微调模型文件"
    echo "请确保已完成模型微调"
fi

# 启动服务
echo "启动后端服务..."
echo "服务地址: http://localhost:8000"
echo "API文档: http://localhost:8000/docs"
echo ""
echo "按 Ctrl+C 停止服务"
echo "=========================================="

python app.py 