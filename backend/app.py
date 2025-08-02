#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRPO微调模型后端API服务
"""

import logging
import time
from typing import List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from config import Config
from model_loader import initialize_model, get_model_loader, cleanup_model

# 配置日志
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Config.LOG_FILE) if Config.LOG_FILE else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="GRPO微调模型API",
    description="基于Qwen3-1.7B的GRPO微调模型API服务",
    version="1.0.0"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求模型
class ChatRequest(BaseModel):
    message: str
    history: Optional[List[dict]] = []

# 响应模型
class ChatResponse(BaseModel):
    response: str
    status: str
    timestamp: float

class ModelInfoResponse(BaseModel):
    status: str
    info: dict
    timestamp: float

# 全局变量
model_loader = None
is_initialized = False

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化模型"""
    global model_loader, is_initialized
    
    logger.info("正在启动GRPO微调模型API服务...")
    
    try:
        # 初始化模型
        config = Config.get_model_config()
        is_initialized = initialize_model(config)
        
        if is_initialized:
            model_loader = get_model_loader(config)
            logger.info("模型初始化成功")
        else:
            logger.error("模型初始化失败")
            
    except Exception as e:
        logger.error(f"启动时发生错误: {e}")
        is_initialized = False

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时清理资源"""
    logger.info("正在关闭API服务...")
    cleanup_model()
    logger.info("API服务已关闭")

@app.get("/")
async def root():
    """根路径"""
    return {
        "message": "GRPO微调模型API服务",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "model_loaded": is_initialized,
        "timestamp": time.time()
    }

@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """获取模型信息"""
    if not is_initialized or model_loader is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    try:
        info = model_loader.get_model_info()
        return ModelInfoResponse(
            status="success",
            info=info,
            timestamp=time.time()
        )
    except Exception as e:
        logger.error(f"获取模型信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取模型信息失败: {str(e)}")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """聊天接口"""
    if not is_initialized or model_loader is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    try:
        start_time = time.time()
        
        # 生成回复
        response = model_loader.generate_response(
            prompt=request.message,
            history=request.history
        )
        
        generation_time = time.time() - start_time
        
        logger.info(f"生成回复完成，耗时: {generation_time:.2f}秒")
        
        return ChatResponse(
            response=response,
            status="success",
            timestamp=time.time()
        )
        
    except Exception as e:
        logger.error(f"生成回复失败: {e}")
        raise HTTPException(status_code=500, detail=f"生成回复失败: {str(e)}")

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """流式聊天接口（实验性）"""
    if not is_initialized or model_loader is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    # 这里可以实现流式响应
    # 暂时返回普通响应
    try:
        response = model_loader.generate_response(
            prompt=request.message,
            history=request.history
        )
        
        return ChatResponse(
            response=response,
            status="success",
            timestamp=time.time()
        )
        
    except Exception as e:
        logger.error(f"流式生成回复失败: {e}")
        raise HTTPException(status_code=500, detail=f"生成回复失败: {str(e)}")

@app.post("/model/reload")
async def reload_model(background_tasks: BackgroundTasks):
    """重新加载模型"""
    global model_loader, is_initialized
    
    try:
        # 清理旧模型
        cleanup_model()
        
        # 在后台重新加载模型
        def reload():
            global model_loader, is_initialized
            config = Config.get_model_config()
            is_initialized = initialize_model(config)
            if is_initialized:
                model_loader = get_model_loader(config)
        
        background_tasks.add_task(reload)
        
        return {
            "message": "模型重新加载已开始",
            "status": "reloading"
        }
        
    except Exception as e:
        logger.error(f"重新加载模型失败: {e}")
        raise HTTPException(status_code=500, detail=f"重新加载模型失败: {str(e)}")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """全局异常处理器"""
    logger.error(f"未处理的异常: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "服务器内部错误",
            "timestamp": time.time()
        }
    )

if __name__ == "__main__":
    server_config = Config.get_server_config()
    
    logger.info(f"启动服务器: {server_config['host']}:{server_config['port']}")
    
    uvicorn.run(
        "app:app",
        host=server_config['host'],
        port=server_config['port'],
        reload=server_config['debug'],
        log_level=Config.LOG_LEVEL.lower()
    ) 