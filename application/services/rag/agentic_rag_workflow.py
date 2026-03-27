from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from domain.vo.conversation.rag_conversation_state import RAGConversationState


class AgenticRAGWorkflow(ABC):
    """
    Agentic RAG 工作流接口
    定义应用层的 RAG 工作流编排能力

    职责边界说明：
    - 该接口属于应用层，负责编排领域服务完成 RAG 工作流
    - 不包含领域业务规则，只负责协调和组合各领域服务
    - 实现类使用 LangGraph 等工作流引擎编排执行流程
    """

    @abstractmethod
    def execute_workflow(
        self,
        query: str,
        session_id: str,
        chat_history: Optional[List[Dict]] = None
    ) -> RAGConversationState:
        """
        执行完整的 Agentic RAG 工作流

        工作流步骤：
        1. 意图识别：判断是否需要检索
        2. 查询重写：优化查询以提升检索效果
        3. 文档检索：根据查询获取相关文档
        4. 相关性评分：筛选相关文档
        5. 答案生成：基于相关文档生成回答

        Args:
            query: 用户查询
            session_id: 会话ID
            chat_history: 历史对话消息

        Returns:
            工作流执行完成后的状态
        """
        pass

    @abstractmethod
    def grade_documents_relevance(
        self,
        query: str,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
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
    def rewrite_query(
        self,
        query: str,
        chat_history: Optional[List[Dict]] = None
    ) -> str:
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
    def generate_answer(
        self,
        query: str,
        relevant_documents: List[Dict[str, Any]],
        chat_history: Optional[List[Dict]] = None
    ) -> str:
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
    def should_retrieve(
        self,
        query: str,
        chat_history: Optional[List[Dict]] = None
    ) -> bool:
        """
        判断是否需要进行文档检索

        Args:
            query: 用户查询
            chat_history: 历史对话消息

        Returns:
            True: 需要检索，False: 可以直接回答
        """
        pass
