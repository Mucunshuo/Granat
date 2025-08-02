@echo off
chcp 65001 >nul

echo ==========================================
echo 启动GRPO微调模型后端服务
echo ==========================================

REM 检查Python环境
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python环境
    pause
    exit /b 1
)

REM 检查是否在正确的目录
if not exist "backend\app.py" (
    echo 错误: 请在项目根目录运行此脚本
    pause
    exit /b 1
)

REM 进入后端目录
cd backend

REM 检查依赖
echo 检查Python依赖...
if not exist "requirements.txt" (
    echo 错误: 未找到requirements.txt文件
    pause
    exit /b 1
)

REM 安装依赖
echo 安装Python依赖...
pip install -r requirements.txt

REM 检查模型文件
echo 检查模型文件...
if not exist "..\grpo_peft_finetuned_model" (
    echo 警告: 未找到微调模型文件
    echo 请确保已完成模型微调
)

REM 启动服务
echo 启动后端服务...
echo 服务地址: http://localhost:8000
echo API文档: http://localhost:8000/docs
echo.
echo 按 Ctrl+C 停止服务
echo ==========================================

python app.py

pause 