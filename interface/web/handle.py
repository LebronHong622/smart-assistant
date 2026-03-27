"""
Agentic RAG API 路由 - 集成到 Web 应用中
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from uuid import uuid4

from interface.container import container
from domain.shared.ports.logger_port import LoggerPort

# 创建路由器实例
router = APIRouter()

# 会话存储（生产环境建议使用Redis）
active_agents = {}


# 依赖项：获取日志记录器
def get_logger() -> LoggerPort:
    """获取日志记录器"""
    return container.get_logger()


# 定义请求模型
class ChatRequest(BaseModel):
    """聊天请求模型"""
    message: str = Field(..., description="用户消息", min_length=1, max_length=1000)
    session_id: Optional[str] = Field(None, description="会话ID")


# 定义响应模型
class ChatResponse(BaseModel):
    """聊天响应模型"""
    message: str = Field(..., description="助手响应")
    timestamp: Optional[str] = Field(None, description="响应时间戳")
    session_id: Optional[str] = Field(None, description="会话ID")
    success: Optional[bool] = Field(None, description="请求是否成功")


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    logger: LoggerPort = Depends(get_logger)
):
    """
    发送聊天消息到Agentic RAG代理
    """
    try:
        logger.info(f"收到聊天请求: {request.message[:100]}..." if len(request.message) > 100 else f"收到聊天请求: {request.message}")

        # 获取或创建会话
        session_id = request.session_id or str(uuid4())
        if session_id not in active_agents:
            active_agents[session_id] = container.get_agentic_rag_agent(session_id=session_id)
            logger.info(f"创建新会话: {session_id}")

        agent = active_agents[session_id]
        answer, documents = agent.chat_with_documents(request.message)

        return ChatResponse(
            message=answer,
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            success=True
        )

    except Exception as e:
        logger.error(f"聊天请求处理失败: {str(e)}", exc_info=True)
        return ChatResponse(
            message=f"处理请求时发生错误: {str(e)}",
            success=False,
            timestamp=datetime.now().isoformat(),
            session_id=request.session_id
        )


@router.get("/")
async def root():
    """根路径，返回API信息"""
    return {
        "message": "智能助手API服务运行中（Agentic RAG架构）",
        "docs": "/docs",
        "redoc": "/redoc",
        "version": "1.0.0"
    }


@router.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "service": "智能助手API（Agentic RAG）",
        "timestamp": datetime.now().isoformat()
    }


@router.get("/info")
async def get_info():
    """获取服务信息"""
    return {
        "name": "智能助手API",
        "version": "1.0.0",
        "description": "基于Agentic RAG架构的多任务问答助手API",
        "architecture": "Agentic RAG",
        "features": [
            "智能工具选择",
            "多轮对话记忆",
            "动态工作流编排",
            "自适应检索策略"
        ],
        "endpoints": {
            "root": "/",
            "health": "/health",
            "chat": "/chat",
            "info": "/info"
        }
    }
