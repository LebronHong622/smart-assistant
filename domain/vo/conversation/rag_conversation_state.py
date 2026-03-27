"""
RAG对话状态值对象
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from domain.vo.conversation.conversation_state import ConversationState


@dataclass
class RAGConversationState(ConversationState):
    """
    RAG 对话状态值对象（子类）
    继承 ConversationState，添加 RAG 特有的状态属性
    """
    # 检索相关
    retrieved_documents: List[Dict[str, Any]] = field(default_factory=list)
    relevant_documents: List[Dict[str, Any]] = field(default_factory=list)
    rewritten_query: Optional[str] = None
    rewrite_count: int = 0

    # 控制流
    needs_retrieval: bool = False
    needs_rewrite: bool = False

    # 原始文档（包含所有字段）
    documents: Optional[List[Dict[str, Any]]] = None
    retrieval_attempts: int = 0

    def increment_rewrite_count(self) -> None:
        """增加重写次数"""
        self.rewrite_count += 1

    def increment_retrieval_attempts(self) -> None:
        """增加检索尝试次数"""
        self.retrieval_attempts += 1

    def add_retrieved_documents(self, documents: List[Dict[str, Any]]) -> None:
        """添加检索到的文档"""
        self.retrieved_documents.extend(documents)

    def set_relevant_documents(self, documents: List[Dict[str, Any]]) -> None:
        """设置相关文档"""
        self.relevant_documents = documents

    def set_rewritten_query(self, query: str) -> None:
        """设置重写后的查询"""
        self.rewritten_query = query

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        base_dict = super().to_dict()
        base_dict.update({
            "retrieved_documents": self.retrieved_documents,
            "relevant_documents": self.relevant_documents,
            "rewritten_query": self.rewritten_query,
            "rewrite_count": self.rewrite_count,
            "needs_retrieval": self.needs_retrieval,
            "needs_rewrite": self.needs_rewrite,
            "documents": self.documents,
            "retrieval_attempts": self.retrieval_attempts
        })
        return base_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RAGConversationState":
        """从字典创建实例"""
        return cls(**data)
