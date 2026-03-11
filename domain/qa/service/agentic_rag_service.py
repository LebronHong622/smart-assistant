from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from domain.qa.value_object.rag_state import RagState


class AgenticRagService(ABC):
    """
    Agentic RAG 领域服务接口
    定义核心业务能力，不包含任何技术实现细节
    """

    @abstractmethod
    def execute_workflow(self, query: str, session_id: str, chat_history: Optional[List[Dict]] = None) -> RagState:
        """
        执行完整的 Agentic RAG 工作流
        Args:
            query: 用户查询
            session_id: 会话ID
            chat_history: 历史对话消息
        Returns:
            工作流执行完成后的状态
        """
        pass

    @abstractmethod
    def grade_documents_relevance(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        评估文档与查询的相关性
        Args:
            query: 用户查询
            documents: 待评估的文档列表
        Returns:
            标记了相关性的文档列表
        """
        pass

    @abstractmethod
    def rewrite_query(self, query: str, chat_history: Optional[List[Dict]] = None) -> str:
        """
        重写用户查询以提升检索效果
        Args:
            query: 原始用户查询
            chat_history: 历史对话消息
        Returns:
            重写后的查询
        """
        pass

    @abstractmethod
    def generate_answer(self, query: str, relevant_documents: List[Dict[str, Any]], chat_history: Optional[List[Dict]] = None) -> str:
        """
        基于相关文档生成最终回答
        Args:
            query: 用户查询
            relevant_documents: 相关文档列表
            chat_history: 历史对话消息
        Returns:
            生成的回答内容
        """
        pass

    @abstractmethod
    def should_retrieve(self, query: str, chat_history: Optional[List[Dict]] = None) -> bool:
        """
        判断是否需要进行文档检索
        Args:
            query: 用户查询
            chat_history: 历史对话消息
        Returns:
            True: 需要检索，False: 可以直接回答
        """
        pass
