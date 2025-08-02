# 🚀 GRPO微调模型快速启动指南

## 📋 前置条件

- ✅ 已完成模型微调（`grpo_peft_finetuned_model` 目录存在）
- ✅ Python 3.8+ 环境
- ✅ 足够的GPU内存（推荐8GB+）

## 🎯 一键启动（推荐）

```bash
# 在项目根目录运行
python start_all.py
```

这个脚本会自动：
1. 检查环境
2. 安装依赖
3. 启动后端API服务
4. 启动前端Web服务
5. 打开浏览器

## 🔧 分步启动

### 1. 启动后端API服务

**Linux/macOS:**
```bash
chmod +x deployment/start_backend.sh
./deployment/start_backend.sh
```

**Windows:**
```cmd
deployment\start_backend.bat
```

**手动启动:**
```bash
cd backend
pip install -r requirements.txt
python app.py
```

### 2. 启动前端Web服务

**Linux/macOS:**
```bash
chmod +x deployment/start_frontend.sh
./deployment/start_frontend.sh
```

**Windows:**
```cmd
deployment\start_frontend.bat
```

**手动启动:**
```bash
cd frontend
python -m http.server 3000
```

## 🌐 访问地址

- **前端界面**: http://localhost:3000
- **后端API**: http://localhost:8000
- **API文档**: http://localhost:8000/docs

## 🐕 使用说明

1. 打开浏览器访问 http://localhost:3000
2. 看到可爱的卷毛小狗AI助手界面
3. 在输入框中输入问题
4. 点击发送或按回车键
5. 等待小狗思考并回复

## 🔍 功能特性

- 🎨 可爱的卷毛小狗形象
- 💬 实时聊天对话
- ⌨️ 打字机效果回复
- 📱 响应式设计
- 🔄 自动重连
- 📝 对话历史记录
- 🧹 一键清空对话

## 🛠️ 故障排除

### 常见问题

1. **端口被占用**
   ```bash
   # 查看端口占用
   netstat -tulpn | grep :8000
   netstat -tulpn | grep :3000
   
   # 杀死进程
   kill -9 <PID>
   ```

2. **模型加载失败**
   - 检查 `grpo_peft_finetuned_model` 目录是否存在
   - 确保模型文件完整
   - 检查GPU内存是否足够

3. **依赖安装失败**
   ```bash
   # 升级pip
   pip install --upgrade pip
   
   # 重新安装依赖
   pip install -r backend/requirements.txt --force-reinstall
   ```

4. **CORS错误**
   - 确保前端访问的是正确的后端地址
   - 检查 `backend/config.py` 中的CORS配置

### 日志查看

- **后端日志**: 查看控制台输出
- **前端日志**: 打开浏览器开发者工具查看Console

## 📞 技术支持

如果遇到问题，请检查：
1. Python版本是否为3.8+
2. 是否已安装所有依赖
3. 模型文件是否完整
4. GPU内存是否足够

## 🎉 开始使用

现在你可以和可爱的卷毛小狗AI助手聊天了！它已经通过GRPO方法在s1K-1.1数据集上进行了微调，可以回答各种问题。

祝您使用愉快！🐕✨ 