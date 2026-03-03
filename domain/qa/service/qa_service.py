from typing import Optional
from langchain.agents import create_agent
from langchain.messages import HumanMessage

from domain.qa.entity.qa_conversation import QAConversation
from domain.qa.value_object.qa_query import QAQuery
from domain.qa.value_object.qa_response import QAResponse
from infrastructure.tool import tool_manager
from infrastructure.model import model_manager
from infrastructure.memory import MemoryManager
from infrastructure.log import app_logger

class QAService:
    """问答领域服务"""

    def __init__(self):
        self.tools = tool_manager.init_tools()
        self.model = model_manager.get_default_model()
        self.memory_manager = MemoryManager()
        self.middleware = self._get_middleware()
        self.agent = self._create_agent()

    def _create_agent(self):
        """创建 LangChain 智能代理"""
        return create_agent(
            model=self.model,
            tools=self.tools,
            system_prompt="你是一个专业的问答助手，能够根据用户问题调用工具获取答案。",
            checkpointer=self.memory_manager.get_in_memory(),
            middleware=self.middleware
        )

    def _get_middleware(self) -> list:
        """获取内存管理中间件"""
        middleware = []
        middleware.append(self.memory_manager.get_overflow_memory_middleware())
        return middleware

    def process_query(self, conversation: QAConversation, query: QAQuery) -> QAResponse:
        """处理用户查询"""
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

        return QAResponse(content=message)
