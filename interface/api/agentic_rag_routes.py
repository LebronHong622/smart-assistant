"""
Agentic RAG API 路由
独立于现有API路由，前缀 /agentic-rag
"""
from typing import Optional, List, Dict
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from uuid import uuid4

from application.agent.agentic_rag_agent import create_agentic_rag_agent
from infrastructure.log import app_logger

router = APIRouter(prefix="/agentic-rag", tags=["Agentic RAG"])

# 会话存储（生产环境建议使用Redis）
active_agents = {}


class ChatRequest(BaseModel):
    """聊天请求模型"""
    query: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    """聊天响应模型"""
    answer: str
    session_id: str
    trace_id: Optional[str] = None


class SessionHistoryResponse(BaseModel):
    """会话历史响应模型"""
    session_id: str
    history: List[Dict]


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    发送聊天消息
    """
    try:
        app_logger.info(f"收到Agentic RAG聊天请求: {request.query}")

        # 获取或创建会话
        session_id = request.session_id or str(uuid4())
        if session_id not in active_agents:
            active_agents[session_id] = create_agentic_rag_agent(session_id=session_id)
            app_logger.info(f"创建新会话: {session_id}")

        agent = active_agents[session_id]
        answer = agent.chat(request.query)

        return ChatResponse(
            answer=answer,
            session_id=session_id
        )

    except Exception as e:
        app_logger.error(f"聊天请求处理失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"处理请求失败: {str(e)}")


@router.get("/history/{session_id}", response_model=SessionHistoryResponse)
async def get_session_history(session_id: str):
    """
    获取会话历史
    """
    try:
        if session_id not in active_agents:
            raise HTTPException(status_code=404, detail=f"会话 {session_id} 不存在")

        agent = active_agents[session_id]
        history = agent.get_session_history()

        return SessionHistoryResponse(
            session_id=session_id,
            history=history
        )

    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"获取会话历史失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取会话历史失败: {str(e)}")


@router.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """
    清空并删除会话
    """
    try:
        if session_id in active_agents:
            del active_agents[session_id]
            app_logger.info(f"会话已删除: {session_id}")

        return {"status": "success", "message": "会话已删除"}

    except Exception as e:
        app_logger.error(f"删除会话失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"删除会话失败: {str(e)}")
