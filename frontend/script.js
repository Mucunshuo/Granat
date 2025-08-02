// 配置
const CONFIG = {
    API_BASE_URL: 'http://localhost:8000',
    TYPING_SPEED: 50, // 打字机效果速度
    MAX_RETRIES: 3,
    RETRY_DELAY: 1000
};

// 全局变量
let chatHistory = [];
let isTyping = false;
let currentRequest = null;

// DOM元素
const elements = {
    chatMessages: document.getElementById('chatMessages'),
    messageInput: document.getElementById('messageInput'),
    sendButton: document.getElementById('sendButton'),
    clearButton: document.getElementById('clearButton'),
    charCount: document.getElementById('charCount'),
    statusDot: document.getElementById('statusDot'),
    statusText: document.getElementById('statusText'),
    loadingOverlay: document.getElementById('loadingOverlay'),
    errorToast: document.getElementById('errorToast'),
    errorMessage: document.getElementById('errorMessage'),
    welcomeTime: document.getElementById('welcomeTime')
};

// 初始化
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    setupEventListeners();
    checkServerStatus();
    setWelcomeTime();
});

// 初始化应用
function initializeApp() {
    console.log('初始化卷毛小狗AI助手...');
    
    // 设置输入框自动调整高度
    autoResizeTextarea();
    
    // 设置字符计数
    updateCharCount();
    
    // 检查输入框状态
    checkInputState();
}

// 设置事件监听器
function setupEventListeners() {
    // 发送按钮点击
    elements.sendButton.addEventListener('click', sendMessage);
    
    // 输入框回车发送
    elements.messageInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // 输入框内容变化
    elements.messageInput.addEventListener('input', function() {
        updateCharCount();
        checkInputState();
        autoResizeTextarea();
    });
    
    // 清空对话按钮
    elements.clearButton.addEventListener('click', clearChat);
    
    // 错误提示自动隐藏
    elements.errorToast.addEventListener('click', hideErrorToast);
}

// 检查服务器状态
async function checkServerStatus() {
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/health`);
        const data = await response.json();
        
        if (data.status === 'healthy' && data.model_loaded) {
            updateStatus('connected', '已连接');
        } else {
            updateStatus('disconnected', '模型未加载');
        }
    } catch (error) {
        console.error('检查服务器状态失败:', error);
        updateStatus('disconnected', '连接失败');
    }
}

// 更新状态显示
function updateStatus(status, text) {
    elements.statusDot.className = `status-dot ${status}`;
    elements.statusText.textContent = text;
}

// 设置欢迎消息时间
function setWelcomeTime() {
    const now = new Date();
    const timeString = now.toLocaleTimeString('zh-CN', {
        hour: '2-digit',
        minute: '2-digit'
    });
    elements.welcomeTime.textContent = timeString;
}

// 发送消息
async function sendMessage() {
    const message = elements.messageInput.value.trim();
    if (!message || isTyping) return;
    
    // 添加用户消息
    addMessage(message, 'user');
    
    // 清空输入框
    elements.messageInput.value = '';
    updateCharCount();
    checkInputState();
    autoResizeTextarea();
    
    // 显示加载动画
    showLoading();
    
    try {
        // 准备请求数据
        const requestData = {
            message: message,
            history: chatHistory
        };
        
        // 发送请求
        const response = await fetch(`${CONFIG.API_BASE_URL}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        if (data.status === 'success') {
            // 添加机器人回复
            addMessage(data.response, 'bot');
            
            // 更新聊天历史
            chatHistory.push(
                { role: 'user', content: message },
                { role: 'assistant', content: data.response }
            );
        } else {
            throw new Error(data.detail || '生成回复失败');
        }
        
    } catch (error) {
        console.error('发送消息失败:', error);
        showError(`发送失败: ${error.message}`);
        addMessage('抱歉，我遇到了一些问题，请稍后再试。', 'bot');
    } finally {
        hideLoading();
    }
}

// 添加消息到聊天界面
function addMessage(content, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    
    const time = new Date().toLocaleTimeString('zh-CN', {
        hour: '2-digit',
        minute: '2-digit'
    });
    
    if (sender === 'bot') {
        messageDiv.innerHTML = `
            <div class="message-avatar">
                <div class="mini-dog-avatar">
                    <div class="mini-dog-face">
                        <div class="mini-dog-eyes">
                            <div class="mini-eye"></div>
                            <div class="mini-eye"></div>
                        </div>
                        <div class="mini-dog-nose"></div>
                    </div>
                </div>
            </div>
            <div class="message-content">
                <div class="message-text" id="bot-message-${Date.now()}"></div>
                <div class="message-time">${time}</div>
            </div>
        `;
    } else {
        messageDiv.innerHTML = `
            <div class="message-content">
                <div class="message-text">${escapeHtml(content)}</div>
                <div class="message-time">${time}</div>
            </div>
        `;
    }
    
    elements.chatMessages.appendChild(messageDiv);
    scrollToBottom();
    
    // 如果是机器人消息，使用打字机效果
    if (sender === 'bot') {
        const messageText = messageDiv.querySelector('.message-text');
        typeWriter(messageText, content);
    }
}

// 打字机效果
function typeWriter(element, text) {
    isTyping = true;
    let i = 0;
    
    function type() {
        if (i < text.length) {
            element.textContent += text.charAt(i);
            i++;
            scrollToBottom();
            setTimeout(type, CONFIG.TYPING_SPEED);
        } else {
            isTyping = false;
        }
    }
    
    type();
}

// 清空对话
function clearChat() {
    if (confirm('确定要清空所有对话吗？')) {
        // 保留欢迎消息
        const welcomeMessage = elements.chatMessages.querySelector('.bot-message');
        elements.chatMessages.innerHTML = '';
        elements.chatMessages.appendChild(welcomeMessage);
        
        // 清空聊天历史
        chatHistory = [];
        
        // 重置时间
        setWelcomeTime();
        
        console.log('对话已清空');
    }
}

// 显示加载动画
function showLoading() {
    elements.loadingOverlay.classList.add('show');
}

// 隐藏加载动画
function hideLoading() {
    elements.loadingOverlay.classList.remove('show');
}

// 显示错误提示
function showError(message) {
    elements.errorMessage.textContent = message;
    elements.errorToast.classList.add('show');
    
    // 3秒后自动隐藏
    setTimeout(hideErrorToast, 3000);
}

// 隐藏错误提示
function hideErrorToast() {
    elements.errorToast.classList.remove('show');
}

// 滚动到底部
function scrollToBottom() {
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
}

// 更新字符计数
function updateCharCount() {
    const length = elements.messageInput.value.length;
    elements.charCount.textContent = `${length}/1000`;
    
    // 超过限制时变红
    if (length > 900) {
        elements.charCount.style.color = '#ff6b6b';
    } else {
        elements.charCount.style.color = '#999';
    }
}

// 检查输入框状态
function checkInputState() {
    const message = elements.messageInput.value.trim();
    const isEmpty = !message;
    const isTooLong = message.length > 1000;
    
    elements.sendButton.disabled = isEmpty || isTooLong || isTyping;
}

// 自动调整文本框高度
function autoResizeTextarea() {
    const textarea = elements.messageInput;
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
}

// HTML转义
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// 工具函数：防抖
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// 工具函数：节流
function throttle(func, limit) {
    let inThrottle;
    return function() {
        const args = arguments;
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    }
}

// 定期检查服务器状态
setInterval(checkServerStatus, 30000); // 每30秒检查一次

// 窗口大小变化时重新调整
window.addEventListener('resize', debounce(function() {
    scrollToBottom();
}, 100));

// 页面可见性变化时检查状态
document.addEventListener('visibilitychange', function() {
    if (!document.hidden) {
        checkServerStatus();
    }
});

// 错误处理
window.addEventListener('error', function(e) {
    console.error('页面错误:', e.error);
    showError('页面出现错误，请刷新重试');
});

// 未处理的Promise拒绝
window.addEventListener('unhandledrejection', function(e) {
    console.error('未处理的Promise拒绝:', e.reason);
    showError('网络请求失败，请检查连接');
}); 