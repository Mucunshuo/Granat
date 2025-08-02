#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试微调后的GRPO模型
"""

import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GRPOModelTester:
    def __init__(self, base_model_name, peft_model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"使用设备: {self.device}")
        
        # 加载tokenizer
        logger.info(f"正在加载tokenizer: {base_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            use_fast=False
        )
        
        # 设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # 加载基础模型
        logger.info(f"正在加载基础模型: {base_model_name}")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 加载PEFT适配器
        logger.info(f"正在加载PEFT适配器: {peft_model_path}")
        self.model = PeftModel.from_pretrained(self.base_model, peft_model_path)
        self.model.eval()
        
        logger.info("模型加载完成")
    
    def generate_response(self, prompt, max_length=512, temperature=0.7, top_p=0.9):
        """生成回复"""
        # 构建输入
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # 转换为Qwen格式
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 编码
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # 解码
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取回复部分
        response = generated_text[len(text):].strip()
        return response
    
    def test_sample_questions(self):
        """测试一些样本问题"""
        test_questions = [
            "什么是机器学习？",
            "解释一下神经网络的工作原理",
            "如何实现一个简单的排序算法？",
            "什么是递归？请举例说明",
            "解释一下面向对象编程的概念"
        ]
        
        print("=" * 60)
        print("微调模型测试结果")
        print("=" * 60)
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n问题 {i}: {question}")
            print("-" * 40)
            
            try:
                response = self.generate_response(question)
                print(f"回答: {response}")
            except Exception as e:
                print(f"生成失败: {e}")
            
            print()

def main():
    # 配置
    base_model_name = "Qwen/Qwen3-1.7B"
    peft_model_path = "./grpo_peft_finetuned_model"
    
    try:
        # 创建测试器
        tester = GRPOModelTester(base_model_name, peft_model_path)
        
        # 运行测试
        tester.test_sample_questions()
        
        # 交互式测试
        print("\n" + "=" * 60)
        print("交互式测试模式 (输入 'quit' 退出)")
        print("=" * 60)
        
        while True:
            user_input = input("\n请输入问题: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
                
            if user_input:
                try:
                    response = tester.generate_response(user_input)
                    print(f"\n回答: {response}")
                except Exception as e:
                    print(f"生成失败: {e}")
    
    except Exception as e:
        logger.error(f"测试过程中出现错误: {e}")
        raise

if __name__ == "__main__":
    main() 