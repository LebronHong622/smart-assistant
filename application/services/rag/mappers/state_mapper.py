"""State mapping utilities"""
from typing import Dict, Any, List, Optional
from domain.vo.conversation.rag_conversation_state import RAGConversationState
from ..workflow.state import AgentState


class StateMapper:
    """状态映射工具类"""

    @staticmethod
    def to_agent_state(
        query: str,
        session_id: str,
        chat_history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        将输入参数转换为 AgentState 初始状态

        Args:
            query: 用户查询
            session_id: 会话ID
            chat_history: 对话历史

        Returns:
            AgentState 初始状态字典
        """
        return {
            "query": query,
            "rewritten_query": None,
            "retrieved_documents": [],
            "relevant_documents": [],
            "needs_retrieval": False,
            "rewrite_count": 0,
            "answer": None,
            "documents": None,
            "session_id": session_id,
            "chat_history": chat_history or [],
            "messages": []
        }

    @staticmethod
    def to_rag_conversation_state(
        result: Dict[str, Any],
        session_id: str,
        query: str,
        chat_history: List[Dict]
    ) -> RAGConversationState:
        """
        将工作流结果转换为 RAGConversationState

        Args:
            result: 工作流执行结果
            session_id: 会话ID
            query: 原始查询
            chat_history: 对话历史

        Returns:
            RAGConversationState 值对象
        """
        return RAGConversationState(
            session_id=session_id,
            query=query,
            chat_history=chat_history,
            retrieved_documents=result.get("retrieved_documents", []),
            relevant_documents=result.get("relevant_documents", []),
            rewritten_query=result.get("rewritten_query"),
            rewrite_count=result.get("rewrite_count", 0),
            answer=result.get("answer"),
            documents=result.get("documents"),
            needs_retrieval=result.get("needs_retrieval", False)
        )

    @staticmethod
    def to_error_rag_conversation_state(
        session_id: str,
        query: str,
        chat_history: List[Dict],
        error: str
    ) -> RAGConversationState:
        """
        创建错误的 RAGConversationState

        Args:
            session_id: 会话ID
            query: 原始查询
            chat_history: 对话历史
            error: 错误信息

        Returns:
            包含错误信息的 RAGConversationState
        """
        return RAGConversationState(
            session_id=session_id,
            query=query,
            chat_history=chat_history,
            error=error
        )
