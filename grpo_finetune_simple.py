#!/usr/bin/env python3
"""
简化的GRPO微调脚本 - 使用Qwen2模型
用于在s1K-1.1数据集上微调Qwen2-1.5B模型
"""

import os
import json
import torch
import logging
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)
from datasets import Dataset
import numpy as np
from typing import Dict, List, Any
import argparse

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GRPODataProcessor:
    """GRPO数据处理器"""
    
    def __init__(self, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def load_grpo_data(self, file_path: str) -> Dataset:
        """加载GRPO格式数据"""
        logger.info(f"正在加载数据: {file_path}")
        
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        
        logger.info(f"加载了 {len(data)} 个样本")
        return Dataset.from_list(data)
    
    def format_conversation(self, prompt: str, response: str) -> str:
        """格式化对话为模型输入格式"""
        # Qwen2的对话格式
        formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
        return formatted
    
    def tokenize_function(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """对数据进行分词处理"""
        # 为每个回答创建训练样本
        all_texts = []
        all_scores = []
        
        for prompt, responses, scores in zip(examples['prompt'], examples['responses'], examples['scores']):
            for response, score in zip(responses, scores):
                # 只使用高质量回答进行训练
                if score >= 0.5:
                    formatted_text = self.format_conversation(prompt, response)
                    all_texts.append(formatted_text)
                    all_scores.append(score)
        
        # 分词
        tokenized = self.tokenizer(
            all_texts,
            truncation=True,
            padding=False,
            max_length=self.max_length,
            return_tensors=None
        )
        
        # 添加标签（用于因果语言建模）
        tokenized['labels'] = tokenized['input_ids'].copy()
        
        return tokenized

class GRPOSimpleTrainer:
    """简化的GRPO训练器"""
    
    def __init__(self, model_name: str, config: Dict[str, Any]):
        self.model_name = model_name
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"使用设备: {self.device}")
        
        # 加载tokenizer
        logger.info(f"正在加载tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True
        )
        
        # 设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        logger.info(f"正在加载模型: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 配置LoRA
        logger.info("配置LoRA...")
        lora_config = LoraConfig(
            r=config['peft_config']['lora_r'],
            lora_alpha=config['peft_config']['lora_alpha'],
            target_modules=config['peft_config']['target_modules'],
            lora_dropout=config['peft_config']['lora_dropout'],
            bias=config['peft_config']['bias'],
            task_type=TaskType.CAUSAL_LM,
        )
        
        # 应用LoRA
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        # 初始化数据处理器
        self.data_processor = GRPODataProcessor(self.tokenizer, config['data_config']['max_seq_length'])
    
    def prepare_dataset(self, train_file: str) -> Dataset:
        """准备训练数据集"""
        # 加载原始数据
        raw_dataset = self.data_processor.load_grpo_data(train_file)
        
        # 处理数据
        processed_dataset = raw_dataset.map(
            self.data_processor.tokenize_function,
            batched=True,
            remove_columns=raw_dataset.column_names,
            desc="处理训练数据"
        )
        
        logger.info(f"处理后的数据集大小: {len(processed_dataset)}")
        return processed_dataset
    
    def train(self, train_dataset: Dataset, output_dir: str):
        """开始训练"""
        # 训练参数
        training_args = TrainingArguments(
            output_dir=output_dir,
            **self.config['training_config'],
            dataloader_pin_memory=False,
            remove_unused_columns=False,
        )
        
        # 数据整理器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # 创建训练器
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # 开始训练
        logger.info("开始训练...")
        trainer.train()
        
        # 保存模型
        logger.info(f"保存模型到: {output_dir}")
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        return trainer

def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 更新模型名称
    config['model_name'] = "Qwen/Qwen2-1.5B"
    
    return config

def main():
    parser = argparse.ArgumentParser(description="简化GRPO微调脚本")
    parser.add_argument("--config", default="training_config.json", help="配置文件路径")
    parser.add_argument("--train_file", default="s1k_grpo_format.jsonl", help="训练数据文件")
    parser.add_argument("--output_dir", default="./grpo_simple_finetuned_model", help="输出目录")
    parser.add_argument("--model_name", default="Qwen/Qwen2-1.5B", help="模型名称")
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 创建训练器
    trainer = GRPOSimpleTrainer(args.model_name, config)
    
    # 准备数据集
    train_dataset = trainer.prepare_dataset(args.train_file)
    
    # 开始训练
    trainer.train(train_dataset, args.output_dir)
    
    logger.info("训练完成！")

if __name__ == "__main__":
    main() 