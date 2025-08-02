#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRPO微调模型完整部署启动脚本
同时启动后端API服务和前端Web界面
"""

import os
import sys
import time
import subprocess
import threading
import signal
import webbrowser
from pathlib import Path

class GRPODeplyment:
    def __init__(self):
        self.backend_process = None
        self.frontend_process = None
        self.running = True
        
        # 设置信号处理
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """处理退出信号"""
        print("\n正在关闭服务...")
        self.running = False
        self.stop_services()
        sys.exit(0)
    
    def check_environment(self):
        """检查运行环境"""
        print("=" * 60)
        print("GRPO微调模型部署检查")
        print("=" * 60)
        
        # 检查Python版本
        python_version = sys.version_info
        if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
            print("❌ Python版本过低，需要Python 3.8+")
            return False
        print(f"✅ Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # 检查必要文件
        required_files = [
            "backend/app.py",
            "backend/requirements.txt",
            "frontend/index.html",
            "grpo_peft_finetuned_model"
        ]
        
        for file_path in required_files:
            if os.path.exists(file_path):
                print(f"✅ {file_path}")
            else:
                print(f"❌ {file_path} - 文件不存在")
                if file_path == "grpo_peft_finetuned_model":
                    print("   请先完成模型微调")
                return False
        
        print("✅ 环境检查通过")
        return True
    
    def install_backend_dependencies(self):
        """安装后端依赖"""
        print("\n安装后端依赖...")
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", "backend/requirements.txt"
            ], check=True, capture_output=True, text=True)
            print("✅ 后端依赖安装完成")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ 后端依赖安装失败: {e}")
            return False
    
    def start_backend(self):
        """启动后端服务"""
        print("\n启动后端API服务...")
        try:
            self.backend_process = subprocess.Popen([
                sys.executable, "backend/app.py"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # 等待服务启动
            time.sleep(5)
            
            if self.backend_process.poll() is None:
                print("✅ 后端服务启动成功 (http://localhost:8000)")
                return True
            else:
                stdout, stderr = self.backend_process.communicate()
                print(f"❌ 后端服务启动失败: {stderr}")
                return False
                
        except Exception as e:
            print(f"❌ 启动后端服务时出错: {e}")
            return False
    
    def start_frontend(self):
        """启动前端服务"""
        print("\n启动前端Web服务...")
        try:
            # 切换到前端目录
            os.chdir("frontend")
            
            self.frontend_process = subprocess.Popen([
                sys.executable, "-m", "http.server", "3000"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # 等待服务启动
            time.sleep(3)
            
            if self.frontend_process.poll() is None:
                print("✅ 前端服务启动成功 (http://localhost:3000)")
                return True
            else:
                stdout, stderr = self.frontend_process.communicate()
                print(f"❌ 前端服务启动失败: {stderr}")
                return False
                
        except Exception as e:
            print(f"❌ 启动前端服务时出错: {e}")
            return False
    
    def open_browser(self):
        """打开浏览器"""
        print("\n正在打开浏览器...")
        try:
            webbrowser.open("http://localhost:3000")
            print("✅ 浏览器已打开")
        except Exception as e:
            print(f"⚠️ 无法自动打开浏览器: {e}")
            print("请手动访问: http://localhost:3000")
    
    def monitor_services(self):
        """监控服务状态"""
        print("\n" + "=" * 60)
        print("服务运行中...")
        print("后端API: http://localhost:8000")
        print("前端界面: http://localhost:3000")
        print("API文档: http://localhost:8000/docs")
        print("按 Ctrl+C 停止所有服务")
        print("=" * 60)
        
        while self.running:
            # 检查后端服务
            if self.backend_process and self.backend_process.poll() is not None:
                print("❌ 后端服务已停止")
                break
            
            # 检查前端服务
            if self.frontend_process and self.frontend_process.poll() is not None:
                print("❌ 前端服务已停止")
                break
            
            time.sleep(5)
    
    def stop_services(self):
        """停止所有服务"""
        print("\n正在停止服务...")
        
        if self.frontend_process:
            self.frontend_process.terminate()
            try:
                self.frontend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.frontend_process.kill()
            print("✅ 前端服务已停止")
        
        if self.backend_process:
            self.backend_process.terminate()
            try:
                self.backend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.backend_process.kill()
            print("✅ 后端服务已停止")
    
    def run(self):
        """运行完整部署"""
        print("🚀 启动GRPO微调模型完整部署")
        
        # 检查环境
        if not self.check_environment():
            print("\n❌ 环境检查失败，请检查上述问题")
            return False
        
        # 安装依赖
        if not self.install_backend_dependencies():
            print("\n❌ 依赖安装失败")
            return False
        
        # 启动后端
        if not self.start_backend():
            print("\n❌ 后端启动失败")
            return False
        
        # 启动前端
        if not self.start_frontend():
            print("\n❌ 前端启动失败")
            self.stop_services()
            return False
        
        # 打开浏览器
        self.open_browser()
        
        # 监控服务
        try:
            self.monitor_services()
        except KeyboardInterrupt:
            pass
        
        return True

def main():
    """主函数"""
    deployment = GRPODeplyment()
    
    try:
        success = deployment.run()
        if success:
            print("\n✅ 部署完成")
        else:
            print("\n❌ 部署失败")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ 部署过程中出现错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 