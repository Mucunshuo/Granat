import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
try:
    from transformers import Qwen3Tokenizer
except ImportError:
    from transformers.models.qwen3 import Qwen3Tokenizer
from peft import PeftModel
from typing import Optional, Dict, Any
import time

logger = logging.getLogger(__name__)

class GRPOModelLoader:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        self.is_loaded = False
        
    def load_model(self) -> bool:
        """加载模型和tokenizer"""
        try:
            logger.info("开始加载模型...")
            start_time = time.time()
            
            # 加载tokenizer
            logger.info(f"加载tokenizer: {self.config['base_model_name']}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config['base_model_name'],
                trust_remote_code=True,
                use_fast=False
            )
            
            # 设置pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 加载基础模型
            logger.info(f"加载基础模型: {self.config['base_model_name']}")
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.config['base_model_name'],
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # 加载PEFT适配器
            logger.info(f"加载PEFT适配器: {self.config['peft_model_path']}")
            self.model = PeftModel.from_pretrained(
                self.base_model, 
                self.config['peft_model_path']
            )
            self.model.eval()
            
            load_time = time.time() - start_time
            logger.info(f"模型加载完成，耗时: {load_time:.2f}秒")
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return False
    
    def generate_response(self, prompt: str, history: Optional[list] = None) -> str:
        """生成回复"""
        if not self.is_loaded:
            raise RuntimeError("模型未加载")
        
        try:
            # 构建完整的对话历史
            messages = []
            if history:
                messages.extend(history)
            messages.append({"role": "user", "content": prompt})
            
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
                    max_new_tokens=self.config['max_length'],
                    temperature=self.config['temperature'],
                    top_p=self.config['top_p'],
                    do_sample=self.config['do_sample'],
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # 解码
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取回复部分
            response = generated_text[len(text):].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"生成回复失败: {e}")
            return f"抱歉，生成回复时出现错误: {str(e)}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        if not self.is_loaded:
            return {"status": "not_loaded"}
        
        try:
            # 计算模型参数
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            return {
                "status": "loaded",
                "device": self.device,
                "total_params": total_params,
                "trainable_params": trainable_params,
                "model_name": self.config['base_model_name'],
                "peft_path": self.config['peft_model_path']
            }
        except Exception as e:
            logger.error(f"获取模型信息失败: {e}")
            return {"status": "error", "message": str(e)}
    
    def unload_model(self):
        """卸载模型"""
        if self.model is not None:
            del self.model
            del self.base_model
            self.model = None
            self.base_model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        self.is_loaded = False
        
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("模型已卸载")

# 全局模型实例
_model_loader: Optional[GRPOModelLoader] = None

def get_model_loader(config: Dict[str, Any]) -> GRPOModelLoader:
    """获取全局模型加载器实例"""
    global _model_loader
    if _model_loader is None:
        _model_loader = GRPOModelLoader(config)
    return _model_loader

def initialize_model(config: Dict[str, Any]) -> bool:
    """初始化模型"""
    global _model_loader
    _model_loader = GRPOModelLoader(config)
    return _model_loader.load_model()

def cleanup_model():
    """清理模型资源"""
    global _model_loader
    if _model_loader is not None:
        _model_loader.unload_model()
        _model_loader = None