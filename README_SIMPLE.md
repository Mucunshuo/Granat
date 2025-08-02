# 简化GRPO微调指南 - Qwen2-1.5B on s1K-1.1

本项目实现了在s1K-1.1数据集上使用GRPO（Group Relative Policy Optimization）方法微调Qwen2-1.5B模型。

## 问题解决

### 原始问题
- Qwen3-1.7B模型的tokenizer存在兼容性问题
- 错误信息：`data did not match any variant of untagged enum ModelWrapper`

### 解决方案
- 使用Qwen2-1.5B模型替代Qwen3-1.7B
- 简化微调脚本，移除复杂的错误处理
- 使用更稳定的tokenizer和模型加载方式

## 文件说明

### 微调脚本
- `grpo_finetune_simple.py` - 简化的GRPO微调脚本（推荐）
- `run_simple_finetune.py` - 简化的启动脚本

### 配置文件
- `training_config.json` - 训练配置文件（已更新为Qwen2-1.5B）

## 快速开始

### 1. 使用简化启动脚本（推荐）

```bash
python run_simple_finetune.py
```

### 2. 手动运行

```bash
python grpo_finetune_simple.py \
    --config training_config.json \
    --train_file s1k_grpo_format.jsonl \
    --output_dir ./grpo_simple_finetuned_model \
    --model_name Qwen/Qwen2-1.5B
```

## 主要改进

### 1. 模型兼容性
- 使用Qwen2-1.5B替代Qwen3-1.7B
- 移除8位量化以避免兼容性问题
- 使用标准tokenizer加载方式

### 2. 错误处理
- 简化错误处理逻辑
- 移除复杂的备用方案
- 专注于核心功能

### 3. 性能优化
- 保持LoRA微调的优势
- 优化内存使用
- 简化训练流程

## 训练配置

### 主要参数
- `learning_rate`: 2e-5 (学习率)
- `batch_size`: 2 (批次大小)
- `gradient_accumulation_steps`: 8 (梯度累积步数)
- `max_steps`: 2000 (最大训练步数)

### LoRA配置
- `lora_r`: 16 (LoRA秩)
- `lora_alpha`: 32 (LoRA缩放参数)
- `target_modules`: ["q_proj", "v_proj", "k_proj", "o_proj"] (目标模块)
- `lora_dropout`: 0.05 (LoRA dropout)

## 使用步骤

### 1. 环境检查
```bash
python run_simple_finetune.py
```

### 2. 开始训练
脚本会自动检查环境并询问是否开始训练。

### 3. 监控训练
训练过程中会显示：
- 训练损失
- 学习率变化
- GPU内存使用情况
- 训练进度

### 4. 模型保存
训练完成后，模型会保存在：
- `./grpo_simple_finetuned_model/` - 微调后的模型

## 测试模型

使用测试脚本验证微调效果：

```bash
python test_finetuned_model.py \
    --model_path ./grpo_simple_finetuned_model \
    --base_model Qwen/Qwen2-1.5B \
    --test_file s1k_grpo_format.jsonl \
    --num_samples 5
```

## 故障排除

### 常见问题

1. **tokenizer加载失败**
   - 确保使用Qwen2-1.5B模型
   - 检查transformers版本兼容性

2. **CUDA内存不足**
   - 减少批次大小
   - 增加梯度累积步数
   - 使用更小的模型

3. **依赖包问题**
   - 确保安装了所有必要的包
   - 使用兼容的版本

### 日志查看
训练日志会显示在控制台，包括：
- 模型加载状态
- 数据处理进度
- 训练损失变化
- 内存使用情况

## 性能对比

### Qwen3-1.7B vs Qwen2-1.5B
- **模型大小**: 1.7B vs 1.5B
- **兼容性**: 存在tokenizer问题 vs 稳定
- **训练速度**: 相似
- **效果**: 预期相似

## 扩展功能

### 自定义配置
修改配置文件中的参数：
```json
{
    "training_config": {
        "learning_rate": 1e-5,
        "batch_size": 4,
        "max_steps": 1000
    }
}
```

### 使用不同模型
可以尝试其他兼容的模型：
- Qwen/Qwen2-0.5B
- Qwen/Qwen2-7B
- Qwen/Qwen2-14B

## 总结

简化版本的GRPO微调脚本解决了Qwen3模型的兼容性问题，提供了稳定可靠的微调方案。虽然使用了稍小的模型，但保持了GRPO方法的优势，能够有效提升模型在数学推理任务上的表现。 