"""
对话仓库接口
"""

from typing import Optional, List
from abc import ABC, abstractmethod

from domain.entity.conversation.conversation import Conversation


class ConversationRepository(ABC):
    """对话仓库接口"""

    @abstractmethod
    def save(self, conversation: Conversation) -> None:
        """保存对话"""
        pass

    @abstractmethod
    def get_by_session_id(self, session_id: str) -> Optional[Conversation]:
        """根据会话ID获取对话"""
        pass

    @abstractmethod
    def get_all(self) -> List[Conversation]:
        """获取所有对话"""
        pass

    @abstractmethod
    def delete(self, conversation: Conversation) -> None:
        """删除对话"""
        pass
