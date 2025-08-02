#!/usr/bin/env python3
"""
简化的GRPO微调启动脚本
"""

import os
import sys
import subprocess
import json

def check_environment():
    """检查环境"""
    print("检查环境...")
    
    # 检查Python版本
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python版本过低，需要Python 3.8+")
        return False
    print(f"✅ Python版本: {version.major}.{version.minor}.{version.micro}")
    
    # 检查依赖包
    required_packages = ['torch', 'transformers', 'peft', 'datasets', 'accelerate']
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} 未安装")
            return False
    
    # 检查CUDA
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ CUDA可用: {gpu_count}个GPU")
            print(f"   GPU 0: {gpu_name}")
            print(f"   显存: {gpu_memory:.1f}GB")
        else:
            print("⚠️  CUDA不可用，将使用CPU训练（不推荐）")
    except Exception as e:
        print(f"❌ CUDA检查失败: {e}")
    
    # 检查文件
    required_files = ['training_config.json', 's1k_grpo_format.jsonl', 'grpo_finetune_simple.py']
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} 不存在")
            return False
    
    return True

def load_config():
    """加载配置"""
    try:
        with open('training_config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        print("✅ 配置文件加载成功")
        return config
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        return None

def start_finetune():
    """启动微调"""
    print("\n" + "="*50)
    print("开始简化GRPO微调")
    print("="*50)
    
    # 构建命令
    cmd = [
        sys.executable, 'grpo_finetune_simple.py',
        '--config', 'training_config.json',
        '--train_file', 's1k_grpo_format.jsonl',
        '--output_dir', './grpo_simple_finetuned_model',
        '--model_name', 'Qwen/Qwen2-1.5B'
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    print("\n开始训练...")
    
    try:
        # 运行微调
        result = subprocess.run(cmd, check=True)
        print("\n✅ 微调完成！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 微调失败: {e}")
        return False

def main():
    """主函数"""
    print("简化GRPO微调环境检查")
    print("="*50)
    
    # 检查环境
    if not check_environment():
        print("\n环境检查失败，请解决上述问题后重试")
        return
    
    # 加载配置
    config = load_config()
    if not config:
        return
    
    # 显示配置信息
    print(f"\n模型: {config.get('model_name', 'Unknown')}")
    print(f"学习率: {config['training_config']['learning_rate']}")
    print(f"批次大小: {config['training_config']['batch_size']}")
    print(f"最大步数: {config['training_config']['max_steps']}")
    
    # 询问是否开始训练
    print("\n环境检查完成！")
    response = input("是否开始微调？(Y/n): ")
    if response.lower() in ['', 'y', 'yes']:
        start_finetune()
    else:
        print("取消训练")

if __name__ == "__main__":
    main() 