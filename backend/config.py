import os
from typing import Optional

class Config:
    # 模型配置
    BASE_MODEL_NAME: str = "Qwen/Qwen3-1.7B"
    PEFT_MODEL_PATH: str = "../grpo_peft_finetuned_model"
    
    # 服务配置
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    
    # 模型推理配置
    MAX_LENGTH: int = 512
    TEMPERATURE: float = 0.7
    TOP_P: float = 0.9
    DO_SAMPLE: bool = True
    
    # 并发配置
    MAX_CONCURRENT_REQUESTS: int = 10
    REQUEST_TIMEOUT: int = 30
    
    # 日志配置
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[str] = "app.log"
    
    # CORS配置
    ALLOWED_ORIGINS: list = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8080",
        "http://127.0.0.1:8080"
    ]
    
    @classmethod
    def get_model_config(cls):
        return {
            "base_model_name": cls.BASE_MODEL_NAME,
            "peft_model_path": cls.PEFT_MODEL_PATH,
            "max_length": cls.MAX_LENGTH,
            "temperature": cls.TEMPERATURE,
            "top_p": cls.TOP_P,
            "do_sample": cls.DO_SAMPLE
        }
    
    @classmethod
    def get_server_config(cls):
        return {
            "host": cls.HOST,
            "port": cls.PORT,
            "debug": cls.DEBUG
        } 