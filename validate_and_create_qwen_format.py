import json
import pandas as pd
from collections import Counter

def validate_grpo_format(file_path):
    """验证GRPO格式文件的正确性"""
    print(f"正在验证文件: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"总行数: {len(lines)}")
    
    # 验证每行都是有效的JSON
    valid_samples = 0
    total_responses = 0
    score_distribution = Counter()
    
    for i, line in enumerate(lines):
        try:
            sample = json.loads(line.strip())
            
            # 检查必要字段
            required_fields = ['prompt', 'responses', 'scores']
            if all(field in sample for field in required_fields):
                valid_samples += 1
                total_responses += len(sample['responses'])
                
                # 统计分数分布
                for score in sample['scores']:
                    score_distribution[score] += 1
            else:
                print(f"第 {i+1} 行缺少必要字段")
                
        except json.JSONDecodeError as e:
            print(f"第 {i+1} 行JSON解析错误: {e}")
    
    print(f"有效样本数: {valid_samples}")
    print(f"总回答数: {total_responses}")
    print(f"平均每个样本的回答数: {total_responses/valid_samples:.2f}")
    print(f"分数分布: {dict(score_distribution)}")
    
    return valid_samples == len(lines)

def create_qwen_format(input_file, output_file):
    """创建适用于Qwen1.7微调的格式"""
    print(f"正在创建Qwen格式: {output_file}")
    
    qwen_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line.strip())
            
            # Qwen格式：每个回答作为一个独立的训练样本
            prompt = sample['prompt']
            responses = sample['responses']
            scores = sample['scores']
            
            # 为每个回答创建样本
            for i, (response, score) in enumerate(zip(responses, scores)):
                # 根据分数决定是否作为正样本
                if score >= 0.5:  # 高质量回答
                    qwen_sample = {
                        "messages": [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": response}
                        ],
                        "score": score,
                        "metadata": {
                            "original_index": sample.get('metadata', {}).get('original_index', 0),
                            "response_index": i,
                            "cot_type": sample.get('metadata', {}).get('cot_type', ''),
                            "source_type": sample.get('metadata', {}).get('source_type', '')
                        }
                    }
                    qwen_data.append(qwen_sample)
    
    # 保存Qwen格式
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in qwen_data:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"Qwen格式保存完成: {len(qwen_data)} 个样本")
    
    # 显示统计信息
    scores = [sample['score'] for sample in qwen_data]
    print(f"分数统计:")
    print(f"  平均分: {sum(scores)/len(scores):.3f}")
    print(f"  最高分: {max(scores)}")
    print(f"  最低分: {min(scores)}")
    
    return qwen_data

def create_dpo_format(input_file, output_file):
    """创建DPO格式（用于对比学习）"""
    print(f"正在创建DPO格式: {output_file}")
    
    dpo_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line.strip())
            
            prompt = sample['prompt']
            responses = sample['responses']
            scores = sample['scores']
            
            # 找到最高分和最低分的回答
            if len(responses) >= 2:
                # 按分数排序
                response_score_pairs = list(zip(responses, scores))
                response_score_pairs.sort(key=lambda x: x[1], reverse=True)
                
                # 选择最高分和最低分的回答
                chosen_response = response_score_pairs[0][0]
                rejected_response = response_score_pairs[-1][0]
                
                # 只有当分数差异足够大时才创建样本
                if response_score_pairs[0][1] - response_score_pairs[-1][1] >= 0.3:
                    dpo_sample = {
                        "prompt": prompt,
                        "chosen": chosen_response,
                        "rejected": rejected_response,
                        "chosen_score": response_score_pairs[0][1],
                        "rejected_score": response_score_pairs[-1][1],
                        "metadata": sample.get('metadata', {})
                    }
                    dpo_data.append(dpo_sample)
    
    # 保存DPO格式
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in dpo_data:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"DPO格式保存完成: {len(dpo_data)} 个样本")
    return dpo_data

def create_training_config():
    """创建训练配置文件"""
    config = {
        "model_name": "Qwen/Qwen1.5-7B-Chat",
        "dataset_format": "GRPO",
        "training_config": {
            "learning_rate": 5e-5,
            "batch_size": 4,
            "gradient_accumulation_steps": 4,
            "max_steps": 1000,
            "warmup_steps": 100,
            "save_steps": 200,
            "eval_steps": 200,
            "logging_steps": 10,
            "save_total_limit": 3,
            "remove_unused_columns": False,
            "push_to_hub": False
        },
        "data_config": {
            "train_file": "s1k_grpo_format.jsonl",
            "validation_file": None,
            "max_seq_length": 2048,
            "preprocessing_num_workers": 4
        },
        "lora_config": {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj"],
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM"
        }
    }
    
    with open('training_config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print("训练配置文件已创建: training_config.json")

if __name__ == "__main__":
    # 验证转换结果
    print("=== 验证GRPO格式 ===")
    is_valid = validate_grpo_format('s1k_grpo_format.jsonl')
    
    if is_valid:
        print("✓ GRPO格式验证通过")
        
        # 创建Qwen格式
        print("\n=== 创建Qwen微调格式 ===")
        qwen_data = create_qwen_format('s1k_grpo_format.jsonl', 's1k_qwen_format.jsonl')
        
        # 创建DPO格式
        print("\n=== 创建DPO格式 ===")
        dpo_data = create_dpo_format('s1k_grpo_format.jsonl', 's1k_dpo_format.jsonl')
        
        # 创建训练配置
        print("\n=== 创建训练配置 ===")
        create_training_config()
        
        print("\n=== 转换完成 ===")
        print("生成的文件:")
        print("1. s1k_grpo_format.jsonl - 原始GRPO格式")
        print("2. s1k_qwen_format.jsonl - Qwen微调格式")
        print("3. s1k_dpo_format.jsonl - DPO对比学习格式")
        print("4. training_config.json - 训练配置文件")
        
    else:
        print("✗ GRPO格式验证失败") 