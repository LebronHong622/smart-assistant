from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional
import logging
from datetime import datetime

from agent.qa_agent import create_qa_agent

# 配置日志
logger = logging.getLogger(__name__)

# 创建路由器实例
router = APIRouter()

# 应用状态管理
class AppState:
    """应用状态管理类"""
    def __init__(self):
        self.qa_agent = None

# 创建应用状态实例
app_state = AppState()

# 依赖项：获取QA代理
async def get_qa_agent():
    """获取QA代理实例"""
    if not app_state.qa_agent:
        raise HTTPException(status_code=503, detail="服务未初始化完成")
    return app_state.qa_agent

# 定义请求模型
class ChatRequest(BaseModel):
    """聊天请求模型"""
    message: str = Field(..., description="用户消息", min_length=1, max_length=1000)

# 定义响应模型
class ChatResponse(BaseModel):
    """聊天响应模型"""
    message: str = Field(..., description="助手响应")
    timestamp: Optional[str] = Field(None, description="响应时间戳")
    session_id: Optional[str] = Field(None, description="会话ID")
    success: Optional[bool] = Field(None, description="请求是否成功")

# 初始化QA代理
def init_qa_agent():
    """初始化QA代理"""
    try:
        logger.info("正在初始化QA代理...")
        app_state.qa_agent = create_qa_agent()
        logger.info("QA代理初始化完成")
        return app_state.qa_agent
    except Exception as e:
        logger.error(f"QA代理初始化失败: {str(e)}")
        raise

# 清理QA代理
def cleanup_qa_agent():
    """清理QA代理"""
    logger.info("清理QA代理...")
    app_state.qa_agent = None
    logger.info("QA代理清理完成")

# 根路径
@router.get("/")
async def root():
    """根路径，返回API信息"""
    return {
        "message": "智能助手API服务运行中",
        "docs": "/docs",
        "redoc": "/redoc",
        "version": "1.0.0"
    }

# 健康检查端点
@router.get("/health")
async def health_check():
    """健康检查端点"""
    status = "healthy"
    if not app_state.qa_agent:
        status = "unhealthy"
    
    return {
        "status": status,
        "service": "智能助手API",
        "timestamp": datetime.now().isoformat()
    }

# 聊天端点
@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    qa_agent = Depends(get_qa_agent)
):
    """处理用户聊天请求"""
    try:
        logger.info(f"接收到用户消息: {request.message[:100]}..." if len(request.message) > 100 else f"接收到用户消息: {request.message}")
        response = qa_agent.chat(request.message)
        
        # 处理响应格式
        if isinstance(response, dict):
            message = response.get('response', str(response))
            session_id = response.get('session_id')
        else:
            message = str(response)
            session_id = None
        
        logger.info(f"生成助手响应: {message[:100]}..." if len(message) > 100 else f"生成助手响应: {message}")
        
        return ChatResponse(
            message=message,
            success=True,
            timestamp=datetime.now().isoformat(),
            session_id=session_id
        )
    except Exception as e:
        error_msg = f"处理请求时发生错误: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return ChatResponse(
            message=error_msg,
            success=False,
            timestamp=datetime.now().isoformat(),
            session_id=None
        )

# 信息端点
@router.get("/info")
async def get_info():
    """获取服务信息"""
    return {
        "name": "智能助手API",
        "version": "1.0.0",
        "description": "多任务问答助手的FastAPI接口",
        "endpoints": {
            "root": "/",
            "health": "/health",
            "chat": "/chat",
            "info": "/info"
        }
    }
