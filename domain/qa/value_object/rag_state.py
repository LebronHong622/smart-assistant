from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from uuid import uuid4


@dataclass
class RagState:
    """
    RAG 工作流状态值对象
    包含工作流执行过程中的所有状态信息
    """
    # 基础信息
    session_id: str
    query: str
    chat_history: List[Dict] = field(default_factory=list)

    # 检索相关
    retrieved_documents: List[Dict[str, Any]] = field(default_factory=list)
    relevant_documents: List[Dict[str, Any]] = field(default_factory=list)
    rewritten_query: Optional[str] = None
    rewrite_count: int = 0

    # 控制流
    needs_retrieval: bool = False
    needs_rewrite: bool = False

    # 结果
    answer: Optional[str] = None
    error: Optional[str] = None
    documents: Optional[List[Dict[str, Any]]] = None

    # 元数据
    trace_id: str = field(default_factory=lambda: str(uuid4()))
    total_steps: int = 0
    retrieval_attempts: int = 0

    def is_valid(self) -> bool:
        """
        验证状态是否有效
        """
        return all([
            self.session_id is not None,
            self.query is not None,
            self.rewrite_count >= 0,
            self.retrieval_attempts >= 0
        ])

    def increment_rewrite_count(self) -> None:
        """
        增加重写次数
        """
        self.rewrite_count += 1

    def increment_retrieval_attempts(self) -> None:
        """
        增加检索尝试次数
        """
        self.retrieval_attempts += 1

    def increment_total_steps(self) -> None:
        """
        增加总步骤数
        """
        self.total_steps += 1

    def add_retrieved_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        添加检索到的文档
        """
        self.retrieved_documents.extend(documents)

    def set_relevant_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        设置相关文档
        """
        self.relevant_documents = documents

    def set_rewritten_query(self, query: str) -> None:
        """
        设置重写后的查询
        """
        self.rewritten_query = query

    def set_answer(self, answer: str) -> None:
        """
        设置最终回答
        """
        self.answer = answer

    def set_error(self, error: str) -> None:
        """
        设置错误信息
        """
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式
        """
        return {
            "session_id": self.session_id,
            "query": self.query,
            "chat_history": self.chat_history,
            "retrieved_documents": self.retrieved_documents,
            "relevant_documents": self.relevant_documents,
            "rewritten_query": self.rewritten_query,
            "rewrite_count": self.rewrite_count,
            "needs_retrieval": self.needs_retrieval,
            "needs_rewrite": self.needs_rewrite,
            "answer": self.answer,
            "error": self.error,
            "documents": self.documents,
            "trace_id": self.trace_id,
            "total_steps": self.total_steps,
            "retrieval_attempts": self.retrieval_attempts
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RagState":
        """
        从字典创建实例
        """
        return cls(**data)
