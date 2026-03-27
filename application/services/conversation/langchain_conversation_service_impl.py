"""
LangChain 对话服务实现
应用层具体技术实现，不放在领域层符合DDD原则
"""

from typing import Optional
from langchain.agents import create_agent
from langchain.messages import HumanMessage

from domain.entity.conversation.conversation import Conversation
from domain.vo.conversation.message import QueryMessage, ResponseMessage
from domain.shared.ports.tool_port import ToolPort
from domain.shared.ports.model_port import ModelPort
from domain.shared.ports.memory_port import MemoryPort
from domain.shared.ports.logger_port import LoggerPort
from domain.service.conversation.conversation_service import ConversationService


class LangchainConversationServiceImpl(ConversationService):
    """
    基于 LangChain 的对话服务实现
    技术实现细节放在应用层，符合DDD原则
    """

    def __init__(
        self,
        logger: LoggerPort,
        tool_provider: ToolPort,
        model_provider: ModelPort,
        memory_provider: MemoryPort
    ):
        self.logger = logger
        self.tools = tool_provider.init_tools()
        self.model = model_provider.get_default_model()
        self.memory_manager = memory_provider
        self.middleware = self._get_middleware()
        self.agent = self._create_agent()

    def _create_agent(self):
        """创建 LangChain 智能代理"""
        return create_agent(
            model=self.model,
            tools=self.tools,
            system_prompt="你是一个专业的问答助手，能够根据用户问题调用工具获取答案。",
            checkpointer=self.memory_manager.get_saver(),
            middleware=self.middleware
        )

    def _get_middleware(self) -> list:
        """获取内存管理中间件"""
        middleware = []
        middleware.append(self.memory_manager.get_overflow_memory_middleware())
        return middleware

    def process_query(self, conversation: Conversation, query: QueryMessage) -> ResponseMessage:
        """处理用户查询"""
        self.logger.info(f"处理查询: {query.content}")

        # 使用代理处理用户查询
        query_message = HumanMessage(content=query.content)
        prompt_message = {
            "messages": [query_message]
        }

        response = self.agent.invoke(
            prompt_message,
            self.memory_manager.get_thread_memory_config(conversation.session_id),
        )

        message = response["messages"][-1].content
        self.logger.info(f"查询完成，响应长度: {len(message)}")

        return ResponseMessage(content=message)
