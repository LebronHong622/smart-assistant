"""
对话状态值对象
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from uuid import uuid4
from datetime import datetime


@dataclass
class ConversationState:
    """
    对话状态值对象（父类）
    包含对话执行过程中的基础状态信息
    """
    # 基础信息
    session_id: str
    query: str
    chat_history: List[Dict] = field(default_factory=list)

    # 结果
    answer: Optional[str] = None
    error: Optional[str] = None

    # 元数据
    trace_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    total_steps: int = 0

    def is_valid(self) -> bool:
        """验证状态是否有效"""
        return all([
            self.session_id is not None,
            self.query is not None,
            self.total_steps >= 0
        ])

    def increment_total_steps(self) -> None:
        """增加总步骤数"""
        self.total_steps += 1

    def set_answer(self, answer: str) -> None:
        """设置最终回答"""
        self.answer = answer

    def set_error(self, error: str) -> None:
        """设置错误信息"""
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "session_id": self.session_id,
            "query": self.query,
            "chat_history": self.chat_history,
            "answer": self.answer,
            "error": self.error,
            "trace_id": self.trace_id,
            "created_at": self.created_at.isoformat(),
            "total_steps": self.total_steps
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationState":
        """从字典创建实例"""
        return cls(**data)
