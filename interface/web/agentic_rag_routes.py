"""
Agentic RAG API 路由 - 集成到 Web 应用中
"""
from fastapi import APIRouter, HTTPException
from uuid import uuid4

from application.agent.agentic_rag_agent import create_agentic_rag_agent
from interface.container import container
from interface.web.dto import ChatRequest, ChatResponse, SessionHistoryResponse, ActiveSessionsResponse

router = APIRouter()

# 会话存储（生产环境建议使用Redis）
active_agents = {}


@router.post("/chat", response_model=ChatResponse, summary="Agentic RAG 聊天", description="使用智能RAG代理进行聊天")
async def chat(request: ChatRequest):
    """
    发送聊天消息到Agentic RAG代理
    """
    try:
        logger = container.get_logger()
        logger.info(f"收到Agentic RAG聊天请求: {request.query}")

        # 获取或创建会话
        session_id = request.session_id or str(uuid4())
        if session_id not in active_agents:
            active_agents[session_id] = create_agentic_rag_agent(session_id=session_id)
            logger.info(f"创建新Agentic RAG会话: {session_id}")

        agent = active_agents[session_id]
        answer = agent.chat(request.query)

        return ChatResponse(
            answer=answer,
            session_id=session_id
        )

    except Exception as e:
        logger = container.get_logger()
        logger.error(f"Agentic RAG聊天请求处理失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"处理请求失败: {str(e)}")


@router.get("/history/{session_id}", response_model=SessionHistoryResponse, summary="获取会话历史", description="获取Agentic RAG会话的历史记录")
async def get_session_history(session_id: str):
    """
    获取Agentic RAG会话历史
    """
    try:
        if session_id not in active_agents:
            raise HTTPException(status_code=404, detail=f"Agentic RAG会话 {session_id} 不存在")

        agent = active_agents[session_id]
        history = agent.get_session_history()

        return SessionHistoryResponse(
            session_id=session_id,
            history=history
        )

    except HTTPException:
        raise
    except Exception as e:
        logger = container.get_logger()
        logger.error(f"获取Agentic RAG会话历史失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取会话历史失败: {str(e)}")


@router.delete("/session/{session_id}", summary="删除会话", description="清空并删除Agentic RAG会话")
async def clear_session(session_id: str):
    """
    清空并删除Agentic RAG会话
    """
    try:
        if session_id in active_agents:
            del active_agents[session_id]
            logger = container.get_logger()
            logger.info(f"Agentic RAG会话已删除: {session_id}")

        return {"status": "success", "message": "Agentic RAG会话已删除"}

    except Exception as e:
        logger = container.get_logger()
        logger.error(f"删除Agentic RAG会话失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"删除会话失败: {str(e)}")


@router.get("/sessions", response_model=ActiveSessionsResponse, summary="列出活跃会话", description="获取所有活跃的Agentic RAG会话")
async def list_active_sessions():
    """
    获取所有活跃的Agentic RAG会话
    """
    return ActiveSessionsResponse(
        active_sessions=list(active_agents.keys()),
        count=len(active_agents)
    )