# GRPO微调模型部署指南

## 项目概述

本项目基于Qwen3-1.7B模型，使用GRPO（Group Relative Policy Optimization）方法在s1K-1.1数据集上进行微调，并提供了完整的Web部署方案。

## 项目结构

```
data/
├── README_GRPO_Deployment.md          # 部署指南
├── grpo_finetune_peft.py              # 微调脚本
├── training_config.json               # 训练配置
├── s1k_grpo_format.jsonl              # 训练数据
├── grpo_peft_finetuned_model/         # 微调后的模型
├── backend/
│   ├── app.py                         # FastAPI后端服务
│   ├── model_loader.py                # 模型加载器
│   ├── requirements.txt               # 后端依赖
│   └── config.py                      # 配置文件
├── frontend/
│   ├── index.html                     # 主页面
│   ├── style.css                      # 样式文件
│   ├── script.js                      # 前端逻辑
│   ├── assets/                        # 静态资源
│   │   ├── dog-avatar.png             # 小狗头像
│   │   └── favicon.ico                # 网站图标
│   └── package.json                   # 前端依赖
└── deployment/
    ├── start_backend.sh               # 启动后端脚本
    ├── start_frontend.sh              # 启动前端脚本
    └── docker-compose.yml             # Docker部署配置
```

## 部署步骤

### 1. 环境准备

确保已安装以下依赖：
- Python 3.8+
- Node.js 16+
- CUDA支持的GPU（推荐）

### 2. 后端部署

#### 2.1 安装后端依赖
```bash
cd backend
pip install -r requirements.txt
```

#### 2.2 启动后端服务
```bash
python app.py
```

后端将在 `http://localhost:8000` 启动

### 3. 前端部署

#### 3.1 安装前端依赖
```bash
cd frontend
npm install
```

#### 3.2 启动前端服务
```bash
npm start
```

前端将在 `http://localhost:3000` 启动

### 4. 使用Docker部署（可选）

```bash
docker-compose up -d
```

## 功能特性

### 后端API
- 模型推理接口
- 聊天历史管理
- 错误处理和日志记录
- 并发请求支持

### 前端界面
- 可爱的虚拟卷毛小狗形象
- 实时聊天界面
- 响应式设计
- 消息历史记录
- 打字机效果

## API接口

### POST /chat
发送消息并获取回复

**请求体：**
```json
{
    "message": "用户消息",
    "history": [
        {"role": "user", "content": "之前的问题"},
        {"role": "assistant", "content": "之前的回答"}
    ]
}
```

**响应：**
```json
{
    "response": "模型回复",
    "status": "success"
}
```

## 配置说明

### 模型配置
- 基础模型：Qwen/Qwen3-1.7B
- 微调方法：GRPO + LoRA
- 可训练参数：约1.7M（占总参数的1%）

### 服务配置
- 后端端口：8000
- 前端端口：3000
- 最大并发：10
- 超时时间：30秒

## 性能优化

1. **模型优化**
   - 使用LoRA减少内存占用
   - 半精度推理
   - 模型缓存

2. **服务优化**
   - 异步处理
   - 连接池
   - 负载均衡

## 监控和日志

- 请求日志记录
- 错误监控
- 性能指标
- 资源使用统计

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减少batch_size
   - 启用gradient_checkpointing
   - 使用CPU推理

2. **模型加载失败**
   - 检查模型路径
   - 验证依赖版本
   - 清理缓存

3. **API请求超时**
   - 增加超时时间
   - 优化模型推理
   - 检查网络连接

## 扩展功能

### 计划中的功能
- 多轮对话优化
- 情感分析
- 语音交互
- 多语言支持
- 用户认证

### 自定义开发
- 添加新的对话模式
- 集成外部API
- 自定义UI主题
- 数据导出功能

## 贡献指南

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 创建Pull Request

## 许可证

MIT License

## 联系方式

如有问题，请提交Issue或联系开发团队。 