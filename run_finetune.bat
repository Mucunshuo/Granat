@echo off
chcp 65001 >nul

echo ==========================================
echo GRPO微调脚本 - Qwen3-1.7B on s1K-1.1
echo ==========================================

REM 检查CUDA是否可用
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo ✓ CUDA可用
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
) else (
    echo ⚠ CUDA不可用，将使用CPU训练（不推荐）
)

REM 检查必要的依赖
echo 检查依赖...
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import transformers; print(f'Transformers版本: {transformers.__version__}')"
python -c "import peft; print(f'PEFT版本: {peft.__version__}')"

REM 设置环境变量
set CUDA_VISIBLE_DEVICES=0
set TOKENIZERS_PARALLELISM=false

REM 创建输出目录
if not exist "grpo_peft_finetuned_model" mkdir grpo_peft_finetuned_model
if not exist "grpo_finetuned_model" mkdir grpo_finetuned_model

echo ==========================================
echo 开始GRPO微调 (PEFT/LoRA版本)
echo ==========================================

REM 运行PEFT微调
python grpo_finetune_peft.py --config training_config.json --train_file s1k_grpo_format.jsonl --output_dir ./grpo_peft_finetuned_model --model_name Qwen/Qwen3-1.7B

if %errorlevel% equ 0 (
    echo ✓ PEFT微调完成
    
    echo ==========================================
    echo 测试微调后的模型
    echo ==========================================
    
    REM 测试模型
    python test_finetuned_model.py --model_path ./grpo_peft_finetuned_model --base_model Qwen/Qwen3-1.7B --test_file s1k_grpo_format.jsonl --num_samples 3
    
) else (
    echo ✗ PEFT微调失败
    pause
    exit /b 1
)

echo ==========================================
echo 微调完成！
echo ==========================================
echo 模型保存在: ./grpo_peft_finetuned_model
echo 可以使用以下命令测试模型:
echo python test_finetuned_model.py --model_path ./grpo_peft_finetuned_model

pause 