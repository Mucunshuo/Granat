@echo off
chcp 65001 >nul

echo ==========================================
echo 启动GRPO微调模型前端服务
echo ==========================================

REM 检查是否在正确的目录
if not exist "frontend\index.html" (
    echo 错误: 请在项目根目录运行此脚本
    pause
    exit /b 1
)

REM 进入前端目录
cd frontend

REM 检查Node.js环境（可选）
node --version >nul 2>&1
if not errorlevel 1 (
    echo 检测到Node.js环境，使用npm启动...
    if exist "package.json" (
        npm install
        npm start
    ) else (
        echo 使用Python HTTP服务器启动...
        python -m http.server 3000
    )
) else (
    echo 使用Python HTTP服务器启动...
    python -m http.server 3000
)

echo 前端服务地址: http://localhost:3000
echo 按 Ctrl+C 停止服务
echo ==========================================

pause 