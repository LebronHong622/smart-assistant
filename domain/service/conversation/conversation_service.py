"""
对话领域服务接口
领域层只定义抽象接口，不包含具体技术实现
具体实现在应用层
"""

from abc import ABC, abstractmethod
from domain.entity.conversation.conversation import Conversation
from domain.vo.conversation.message import QueryMessage, ResponseMessage


class ConversationService(ABC):
    """
    对话领域服务接口
    定义处理用户查询的核心业务接口
    """

    @abstractmethod
    def process_query(
        self,
        conversation: Conversation,
        query: QueryMessage
    ) -> ResponseMessage:
        """
        处理用户查询

        Args:
            conversation: 对话实体
            query: 查询消息

        Returns:
            响应消息
        """
        pass
