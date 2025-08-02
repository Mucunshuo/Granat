# GRPO微调指南 - Qwen3-1.7B on s1K-1.1

本项目实现了在s1K-1.1数据集上使用GRPO（Group Relative Policy Optimization）方法微调Qwen3-1.7B模型。

## 文件说明

### 数据文件
- `s1k_grpo_format.jsonl` - GRPO格式的训练数据（主要格式）
- `s1k_qwen_format.jsonl` - Qwen专用格式的训练数据
- `s1k_dpo_format.jsonl` - DPO对比学习格式的训练数据
- `s1k_grpo_alternative.jsonl` - 替代GRPO格式

### 微调脚本
- `grpo_finetune.py` - 基础GRPO微调脚本（全参数微调）
- `grpo_finetune_peft.py` - PEFT/LoRA版本的GRPO微调脚本（推荐）
- `test_finetuned_model.py` - 测试微调后模型的脚本

### 配置文件
- `training_config.json` - 训练配置文件
- `requirements.txt` - Python依赖包列表

### 运行脚本
- `run_finetune.sh` - Linux/macOS运行脚本
- `run_finetune.bat` - Windows运行脚本

## 环境要求

### 硬件要求
- GPU: 至少8GB显存（推荐16GB+）
- RAM: 至少16GB内存
- 存储: 至少10GB可用空间

### 软件要求
- Python 3.8+
- CUDA 11.8+ (如果使用GPU)
- PyTorch 2.0+

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 使用Windows批处理文件（推荐）

```cmd
run_finetune.bat
```

### 2. 使用Linux/macOS脚本

```bash
chmod +x run_finetune.sh
./run_finetune.sh
```

### 3. 手动运行

```bash
# PEFT/LoRA微调（推荐）
python grpo_finetune_peft.py \
    --config training_config.json \
    --train_file s1k_grpo_format.jsonl \
    --output_dir ./grpo_peft_finetuned_model \
    --model_name Qwen/Qwen3-1.7B

# 测试模型
python test_finetuned_model.py \
    --model_path ./grpo_peft_finetuned_model \
    --base_model Qwen/Qwen3-1.7B \
    --test_file s1k_grpo_format.jsonl \
    --num_samples 5
```

## 训练配置说明

### 主要参数
- `learning_rate`: 2e-5 (学习率)
- `batch_size`: 2 (批次大小)
- `gradient_accumulation_steps`: 8 (梯度累积步数)
- `max_steps`: 2000 (最大训练步数)
- `warmup_steps`: 200 (预热步数)

### LoRA配置
- `lora_r`: 16 (LoRA秩)
- `lora_alpha`: 32 (LoRA缩放参数)
- `target_modules`: ["q_proj", "v_proj", "k_proj", "o_proj"] (目标模块)
- `lora_dropout`: 0.05 (LoRA dropout)

## 数据格式说明

### GRPO格式
```json
{
    "prompt": "问题内容",
    "responses": ["回答1", "回答2", "回答3", "回答4"],
    "scores": [1.0, 0.8, 0.5, 0.2],
    "metadata": {
        "cot_type": "math",
        "source_type": "AI-MO/NuminaMath-CoT/aops_forum",
        "original_index": 0
    }
}
```

### 分数说明
- 1.0: 高质量回答（标准答案或高分回答）
- 0.5: 中等质量回答（变体或中等分数）
- 0.0: 低质量回答（被拒绝的回答）

## 训练过程监控

训练过程中会显示：
- 训练损失
- 学习率变化
- GPU内存使用情况
- 训练进度

## 模型保存

训练完成后，模型会保存在：
- `./grpo_peft_finetuned_model/` - PEFT/LoRA模型
- `./grpo_finetuned_model/` - 全参数微调模型

## 测试模型

使用测试脚本验证微调效果：

```bash
python test_finetuned_model.py \
    --model_path ./grpo_peft_finetuned_model \
    --base_model Qwen/Qwen3-1.7B \
    --test_file s1k_grpo_format.jsonl \
    --num_samples 5
```

## 性能优化建议

### 内存优化
1. 使用PEFT/LoRA而不是全参数微调
2. 启用8位量化 (`load_in_8bit=True`)
3. 使用梯度检查点 (`gradient_checkpointing=True`)
4. 调整批次大小和梯度累积步数

### 训练优化
1. 使用混合精度训练 (`fp16=True`)
2. 启用梯度裁剪 (`max_grad_norm=1.0`)
3. 使用余弦学习率调度器
4. 设置适当的权重衰减

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减少批次大小
   - 增加梯度累积步数
   - 使用8位量化

2. **训练速度慢**
   - 检查GPU使用率
   - 调整数据加载器工作进程数
   - 使用混合精度训练

3. **模型不收敛**
   - 调整学习率
   - 增加训练步数
   - 检查数据质量

### 日志查看
训练日志会显示在控制台，包括：
- 训练损失
- 验证损失（如果有验证集）
- 学习率变化
- 内存使用情况

## 扩展功能

### 添加验证集
在配置文件中添加验证集路径：
```json
{
    "data_config": {
        "validation_file": "validation_data.jsonl"
    }
}
```

### 使用WandB监控
安装wandb并登录：
```bash
pip install wandb
wandb login
```

### 自定义LoRA配置
修改配置文件中的LoRA参数：
```json
{
    "peft_config": {
        "lora_r": 32,
        "lora_alpha": 64,
        "target_modules": ["q_proj", "v_proj"]
    }
}
```

## 许可证

本项目遵循MIT许可证。

## 贡献

欢迎提交Issue和Pull Request来改进项目。 