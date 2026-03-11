"""
Agentic RAG DTO 模型
"""
from typing import Optional, List, Dict
from pydantic import BaseModel


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


class ActiveSessionsResponse(BaseModel):
    """活跃会话响应模型"""
    active_sessions: List[str]
    count: int


__all__ = [
    "ChatRequest",
    "ChatResponse", 
    "SessionHistoryResponse",
    "ActiveSessionsResponse"
]