from typing import Optional
from langchain.agents import create_agent
from langchain.messages import HumanMessage

from domain.qa.entity.qa_conversation import QAConversation
from domain.qa.value_object.qa_query import QAQuery
from domain.qa.value_object.qa_response import QAResponse
from domain.port.tool_port import ToolPort
from domain.port.model_port import ModelPort
from domain.port.memory_port import MemoryPort
from domain.port.logger_port import LoggerPort


class QAService:
    """问答领域服务"""

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

    def process_query(self, conversation: QAConversation, query: QAQuery) -> QAResponse:
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

        return QAResponse(content=message)
