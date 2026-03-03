from typing import Optional, List
from abc import ABC, abstractmethod

from domain.qa.entity.qa_conversation import QAConversation

class QAConversationRepository(ABC):
    """问答对话仓库接口"""

    @abstractmethod
    def save(self, conversation: QAConversation) -> None:
        """保存对话"""
        pass

    @abstractmethod
    def get_by_session_id(self, session_id: str) -> Optional[QAConversation]:
        """根据会话ID获取对话"""
        pass

    @abstractmethod
    def get_all(self) -> List[QAConversation]:
        """获取所有对话"""
        pass

    @abstractmethod
    def delete(self, conversation: QAConversation) -> None:
        """删除对话"""
        pass
